"""
Clean Qwen3 Forward Pass with TurboQuant KV Cache Compression.

This implementation supports:
- MSE-only compression (reconstruction-based)
- Full QJL (unbiased inner product estimation for attention scores)

QJL Inner Product Estimator (TurboQuant Definition 1):
<y, x> ≈ <y, x_mse> + ||r|| * sqrt(π/2)/d * <S@y, sign(S@r)>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant import TurboQuantMSE, TurboQuantProd


# ============================================================================
# Basic Components
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# ============================================================================
# Chunk-based KV Cache with Full QJL Support
# ============================================================================

class ChunkedKVCacheQJL:
    """
    KV cache with chunk-based compression and QJL inner product support.

    For compressed keys, stores:
    - x_mse: MSE reconstruction
    - qjl_signs: sign(S @ r) for QJL correction
    - residual_norm: ||r|| for QJL correction
    - shape: original tensor shape

    For values, only stores MSE reconstruction (values don't need inner products).

    QJL Inner Product Formula:
    <q, k> ≈ <q, k_mse> + ||r|| * sqrt(π/2)/d * <S@q, sign(S@r)>
    """

    def __init__(self, num_layers: int, head_dim: int, bits: int, keep_recent: int = 32):
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.bits = bits
        self.keep_recent = keep_recent

        # Per-layer storage for keys
        # Each key_cache entry has:
        # - 'chunks': list of dicts {'x_mse', 'qjl_signs', 'residual_norm', 'shape'}
        # - 'raw': tensor for recent tokens (no compression, no QJL needed)
        self.key_caches = [{'chunks': [], 'raw': None} for _ in range(num_layers)]

        # Per-layer storage for values (MSE only)
        # Each value_cache entry has:
        # - 'chunks': list of (x_mse, shape) tuples
        # - 'raw': tensor for recent tokens
        self.value_caches = [{'chunks': [], 'raw': None} for _ in range(num_layers)]

        # Quantizers - use TurboQuantProd for keys (MSE + QJL)
        self.key_quantizers = [TurboQuantProd(head_dim, bits, seed=i) for i in range(num_layers)]
        self.value_quantizers = [TurboQuantMSE(head_dim, bits, seed=i + 1000) for i in range(num_layers)]

    def _compress_keys(self, tensor: torch.Tensor, quantizer: TurboQuantProd) -> Dict[str, Any]:
        """Compress keys with full QJL data."""
        shape = tensor.shape
        flat = tensor.reshape(-1, self.head_dim).float()

        # TurboQuantProd.quantize returns full QJL data
        compressed = quantizer.quantize(flat)

        return {
            'x_mse': compressed['x_mse'],
            'qjl_signs': compressed['qjl_signs'],
            'residual_norm': compressed['residual_norm'],
            'shape': shape
        }

    def _compress_values(self, tensor: torch.Tensor, quantizer: TurboQuantMSE) -> Tuple[torch.Tensor, tuple]:
        """Compress values (MSE only)."""
        shape = tensor.shape
        flat = tensor.reshape(-1, self.head_dim).float()
        x_mse, _, _ = quantizer.quantize(flat, return_indices=True)
        return x_mse, shape

    def _reconstruct_key_chunks(self, chunks: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct key data from compressed chunks.

        Returns:
            x_mse_concat: (B, H, S_compressed, D) MSE reconstruction
            qjl_signs_concat: (B, H, S_compressed, D) QJL signs
            residual_norm_concat: (B, H, S_compressed) residual norms
        """
        if not chunks:
            return None, None, None

        x_mse_parts = [c['x_mse'].reshape(c['shape']) for c in chunks]
        qjl_signs_parts = [c['qjl_signs'].reshape(c['shape']) for c in chunks]
        # residual_norm shape is (B*H*S,), need to reshape to (B, H, S)
        residual_norm_parts = []
        for c in chunks:
            shape = c['shape']  # (B, H, S, D)
            B, H, S, D = shape
            rn = c['residual_norm'].reshape(B, H, S)
            residual_norm_parts.append(rn)

        return (
            torch.cat(x_mse_parts, dim=2),
            torch.cat(qjl_signs_parts, dim=2),
            torch.cat(residual_norm_parts, dim=2)
        )

    def _reconstruct_value_chunks(self, chunks: List[Tuple]) -> torch.Tensor:
        """Reconstruct values from compressed chunks (MSE only)."""
        if not chunks:
            return None
        parts = [x_mse.reshape(shape) for x_mse, shape in chunks]
        return torch.cat(parts, dim=2)

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache and return full K, V (MSE reconstruction only for backward compatibility)."""
        cache_k = self.key_caches[layer_idx]
        cache_v = self.value_caches[layer_idx]

        current_raw_k = cache_k['raw']
        current_raw_v = cache_v['raw']

        if current_raw_k is None:
            # First call
            cache_k['raw'] = new_k.clone()
            cache_v['raw'] = new_v.clone()
            return new_k, new_v

        # Append new to current raw
        combined_k = torch.cat([current_raw_k, new_k.clone()], dim=2)
        combined_v = torch.cat([current_raw_v, new_v.clone()], dim=2)

        raw_len = combined_k.shape[2]

        if raw_len > self.keep_recent:
            compress_len = raw_len - self.keep_recent

            to_compress_k = combined_k[:, :, :compress_len, :].clone()
            to_compress_v = combined_v[:, :, :compress_len, :].clone()

            # Compress and store
            k_compressed = self._compress_keys(to_compress_k, self.key_quantizers[layer_idx])
            v_compressed = self._compress_values(to_compress_v, self.value_quantizers[layer_idx])

            cache_k['chunks'].append(k_compressed)
            cache_v['chunks'].append(v_compressed)

            # Keep recent as raw
            cache_k['raw'] = combined_k[:, :, compress_len:, :].clone()
            cache_v['raw'] = combined_v[:, :, compress_len:, :].clone()
        else:
            cache_k['raw'] = combined_k
            cache_v['raw'] = combined_v

        return self.get_kv(layer_idx)

    def get_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get full K, V (MSE reconstruction only, for backward compatibility)."""
        cache_k = self.key_caches[layer_idx]
        cache_v = self.value_caches[layer_idx]

        raw_k = cache_k['raw']
        raw_v = cache_v['raw']

        # Reconstruct compressed keys (MSE only)
        if cache_k['chunks']:
            k_mse_comp, _, _ = self._reconstruct_key_chunks(cache_k['chunks'])
            k = torch.cat([k_mse_comp, raw_k], dim=2)
        else:
            k = raw_k

        # Reconstruct compressed values
        if cache_v['chunks']:
            v_comp = self._reconstruct_value_chunks(cache_v['chunks'])
            v = torch.cat([v_comp, raw_v], dim=2)
        else:
            v = raw_v

        return k, v

    def get_qjl_data(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get full QJL data for computing unbiased inner products.

        Returns dict with:
            - 'x_mse': (B, H, S_total, D) MSE reconstruction of all keys
            - 'qjl_signs': (B, H, S_compressed, D) QJL signs (zeros for raw part)
            - 'residual_norm': (B, H, S_compressed) residual norms (zeros for raw part)
            - 'compressed_len': number of compressed tokens
            - 'raw_len': number of raw tokens
        """
        cache_k = self.key_caches[layer_idx]

        raw_k = cache_k['raw']
        chunks = cache_k['chunks']

        if not chunks:
            # No compression, return raw only (QJL data is zeros)
            B, H, S, D = raw_k.shape
            device = raw_k.device
            return {
                'x_mse': raw_k.float(),
                'qjl_signs': torch.zeros(B, H, S, D, device=device),
                'residual_norm': torch.zeros(B, H, S, device=device),
                'compressed_len': 0,
                'raw_len': S
            }

        # Reconstruct compressed part
        k_mse_comp, qjl_signs_comp, residual_norm_comp = self._reconstruct_key_chunks(chunks)

        # Concatenate with raw (raw part has no QJL correction)
        B, H, S_comp, D = k_mse_comp.shape
        S_raw = raw_k.shape[2]

        # For raw tokens, QJL correction is zero (they're exact)
        qjl_signs_raw = torch.zeros(B, H, S_raw, D, device=raw_k.device, dtype=qjl_signs_comp.dtype)
        residual_norm_raw = torch.zeros(B, H, S_raw, device=raw_k.device, dtype=residual_norm_comp.dtype)

        x_mse_full = torch.cat([k_mse_comp, raw_k.float()], dim=2)
        qjl_signs_full = torch.cat([qjl_signs_comp, qjl_signs_raw], dim=2)
        residual_norm_full = torch.cat([residual_norm_comp, residual_norm_raw], dim=2)

        return {
            'x_mse': x_mse_full,
            'qjl_signs': qjl_signs_full,
            'residual_norm': residual_norm_full,
            'compressed_len': S_comp,
            'raw_len': S_raw
        }

    def get_seq_length(self) -> int:
        """Get total sequence length."""
        if self.key_caches[0]['raw'] is None and len(self.key_caches[0]['chunks']) == 0:
            return 0

        total = 0
        for chunk in self.key_caches[0]['chunks']:
            total += chunk['shape'][2]  # shape[2] is seq_len
        if self.key_caches[0]['raw'] is not None:
            total += self.key_caches[0]['raw'].shape[2]
        return total

    def get_stats(self) -> dict:
        """Get compression statistics."""
        total_seq = self.get_seq_length()
        if total_seq == 0:
            return {'total_seq': 0, 'compressed_seq': 0, 'raw_seq': 0, 'ratio': 1.0}

        compressed_seq = sum(chunk['shape'][2] for chunk in self.key_caches[0]['chunks'])
        raw_seq = total_seq - compressed_seq

        # Memory calculation
        if self.key_caches[0]['raw'] is not None:
            num_kv = self.key_caches[0]['raw'].shape[1]
        else:
            num_kv = self.key_caches[0]['chunks'][0]['shape'][1]

        fp16_bits = 2 * self.num_layers * num_kv * total_seq * self.head_dim * 16
        compressed_bits = 2 * self.num_layers * num_kv * compressed_seq * self.head_dim * self.bits
        raw_bits = 2 * self.num_layers * num_kv * raw_seq * self.head_dim * 16
        total_bits = compressed_bits + raw_bits

        return {
            'total_seq': total_seq,
            'compressed_seq': compressed_seq,
            'raw_seq': raw_seq,
            'fp16_bits': fp16_bits,
            'compressed_bits': total_bits,
            'ratio': fp16_bits / total_bits if total_bits > 0 else 1.0
        }

    def clear(self):
        for i in range(self.num_layers):
            self.key_caches[i] = {'chunks': [], 'raw': None}
            self.value_caches[i] = {'chunks': [], 'raw': None}


# ============================================================================
# QJL Attention Score Computation
# ============================================================================

def compute_qjl_attention_scores(
    query: torch.Tensor,
    qjl_data: Dict[str, torch.Tensor],
    quantizer: TurboQuantProd,
    scale: float
) -> torch.Tensor:
    """
    Compute attention scores using QJL unbiased inner product estimator.

    Formula: <q, k> ≈ <q, k_mse> + ||r|| * sqrt(π/2)/d * <S@q, sign(S@r)>

    Args:
        query: (B, num_heads, S_q, D) query tensor
        qjl_data: dict from get_qjl_data() containing:
            - x_mse: (B, H, S_total, D)
            - qjl_signs: (B, H, S_total, D)
            - residual_norm: (B, H, S_total)
        quantizer: TurboQuantProd instance (has S matrix)
        scale: attention scale factor

    Returns:
        scores: (B, num_heads, S_q, S_total) attention scores
    """
    B, num_heads, S_q, D = query.shape
    x_mse = qjl_data['x_mse']  # (B, H_kv, S_total, D)
    qjl_signs = qjl_data['qjl_signs']  # (B, H_kv, S_total, D)
    residual_norm = qjl_data['residual_norm']  # (B, H_kv, S_total)

    H_kv = x_mse.shape[1]

    # Ensure float for computation
    q_float = query.float()

    # Term 1: Standard inner product <q, k_mse>
    # Need to handle GQA: if num_heads > H_kv, repeat k_mse
    if num_heads != H_kv:
        n_rep = num_heads // H_kv
        x_mse = repeat_kv(x_mse, n_rep)  # (B, num_heads, S_total, D)
        qjl_signs = repeat_kv(qjl_signs, n_rep)
        residual_norm = repeat_kv(residual_norm.unsqueeze(-1), n_rep).squeeze(-1)  # (B, num_heads, S_total)

    # Term 1: <q, k_mse>
    term1 = torch.matmul(q_float, x_mse.transpose(-2, -1))  # (B, num_heads, S_q, S_total)

    # Term 2: QJL correction
    # ||r|| * sqrt(π/2)/d * <S@q, sign(S@r)>
    S = quantizer.S.to(q_float.device, torch.float32)

    # Project query: S@q -> q @ S.T
    q_projected = torch.matmul(q_float, S.T)  # (B, num_heads, S_q, D)

    # Inner product <q_projected, qjl_signs>
    qjl_ip = torch.matmul(q_projected, qjl_signs.transpose(-2, -1))  # (B, num_heads, S_q, S_total)

    # Correction scale
    correction_scale = math.sqrt(math.pi / 2) / D

    # Apply residual norm and correction
    term2 = correction_scale * qjl_ip * residual_norm.unsqueeze(2)  # (B, num_heads, S_q, S_total)

    # Final scores
    scores = (term1 + term2) * scale

    return scores


# ============================================================================
# Qwen3 Forward with TurboQuant (Full QJL Support)
# ============================================================================

class Qwen3ForwardWithTurboQuant:
    """Qwen3 forward pass with TurboQuant KV cache compression and QJL inner product."""

    def __init__(self, model, bits: int = 4, use_qjl: bool = True, keep_recent: int = 32):
        self.model = model
        self.config = model.config

        # Dimensions
        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.num_kv_heads = model.config.num_key_value_heads
        self.head_dim = getattr(model.config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_layers = model.config.num_hidden_layers
        self.n_rep = self.num_heads // self.num_kv_heads
        self.vocab_size = model.config.vocab_size

        # Extract weights
        self.embed_tokens = model.model.embed_tokens.weight
        self.lm_head = model.lm_head.weight
        self.norm_weight = model.model.norm.weight

        self.layers = []
        for i in range(self.num_layers):
            layer = model.model.layers[i]
            self.layers.append({
                'input_layernorm_weight': layer.input_layernorm.weight,
                'post_attention_layernorm_weight': layer.post_attention_layernorm.weight,
                'q_proj_weight': layer.self_attn.q_proj.weight,
                'k_proj_weight': layer.self_attn.k_proj.weight,
                'v_proj_weight': layer.self_attn.v_proj.weight,
                'o_proj_weight': layer.self_attn.o_proj.weight,
                'q_norm_weight': layer.self_attn.q_norm.weight,
                'k_norm_weight': layer.self_attn.k_norm.weight,
                'gate_proj_weight': layer.mlp.gate_proj.weight,
                'up_proj_weight': layer.mlp.up_proj.weight,
                'down_proj_weight': layer.mlp.down_proj.weight,
            })

        self.rotary_emb = model.model.rotary_emb

        # KV cache
        self.bits = bits
        self.use_qjl = use_qjl
        self.keep_recent = keep_recent
        self.kv_cache = None

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return (weight * x).to(input_dtype)

    def _attention(self, hidden_states: torch.Tensor, layer_idx: int,
                   position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                   attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        layer = self.layers[layer_idx]

        # Project Q, K, V
        q = F.linear(hidden_states, layer['q_proj_weight'])
        k = F.linear(hidden_states, layer['k_proj_weight'])
        v = F.linear(hidden_states, layer['v_proj_weight'])

        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # QK norm
        q = self._rms_norm(q, layer['q_norm_weight'])
        k = self._rms_norm(k, layer['k_norm_weight'])

        # Transpose to (batch, heads, seq, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Update cache
        self.kv_cache.update(layer_idx, k, v)

        # Compute attention scores
        scale = self.head_dim ** -0.5

        if self.use_qjl:
            # Use QJL unbiased inner product estimator
            qjl_data = self.kv_cache.get_qjl_data(layer_idx)
            attn_weights = compute_qjl_attention_scores(
                q, qjl_data, self.kv_cache.key_quantizers[layer_idx], scale
            )
        else:
            # Standard attention with MSE reconstruction
            k, v = self.kv_cache.get_kv(layer_idx)
            k = k.to(q.dtype)
            v = v.to(q.dtype)
            k_rep = repeat_kv(k, self.n_rep)
            v_rep = repeat_kv(v, self.n_rep)
            attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) * scale

        # Apply attention mask (causal)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(q.dtype if not self.use_qjl else torch.float32)

        # Get values for weighted sum
        v_full = self.kv_cache.get_kv(layer_idx)[1]  # Get V
        v_rep = repeat_kv(v_full.to(attn_weights.dtype), self.n_rep)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, v_rep)

        # Reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        return F.linear(attn_output.to(hidden_states.dtype), layer['o_proj_weight'])

    def _mlp(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        layer = self.layers[layer_idx]
        gate = F.linear(hidden_states, layer['gate_proj_weight'])
        up = F.linear(hidden_states, layer['up_proj_weight'])
        return F.linear(F.silu(gate) * up, layer['down_proj_weight'])

    def _decoder_layer(self, hidden_states: torch.Tensor, layer_idx: int,
                       position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        layer = self.layers[layer_idx]

        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, layer['input_layernorm_weight'])
        hidden_states = self._attention(hidden_states, layer_idx, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, layer['post_attention_layernorm_weight'])
        hidden_states = self._mlp(hidden_states, layer_idx)
        return residual + hidden_states

    def forward(self, input_ids: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            past_len = self.kv_cache.get_seq_length() if self.kv_cache else 0
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embedding
        hidden_states = F.embedding(input_ids, self.embed_tokens)

        # RoPE
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Causal mask
        if seq_len > 1:
            total_seq = self.kv_cache.get_seq_length() + seq_len if self.kv_cache else seq_len
            causal_mask = torch.triu(
                torch.full((seq_len, total_seq), float('-inf'), device=input_ids.device),
                diagonal=total_seq - seq_len + 1
            ).unsqueeze(0).unsqueeze(0)
        else:
            causal_mask = None

        # Layers
        for layer_idx in range(self.num_layers):
            hidden_states = self._decoder_layer(hidden_states, layer_idx, position_embeddings, causal_mask)

        # Final norm and LM head
        hidden_states = self._rms_norm(hidden_states, self.norm_weight)
        return F.linear(hidden_states, self.lm_head)

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_p: float = 1.0,
                 do_sample: bool = True) -> torch.Tensor:
        # Initialize cache
        self.kv_cache = ChunkedKVCacheQJL(
            num_layers=self.num_layers,
            head_dim=self.head_dim,
            bits=self.bits,
            keep_recent=self.keep_recent
        )

        generated = []
        cur_ids = input_ids

        for step in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(cur_ids)

            next_logits = logits[:, -1, :]

            if do_sample:
                next_logits = next_logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    next_logits[sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)] = float('-inf')
                next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated.append(next_token)
            cur_ids = next_token

        return torch.cat([input_ids] + generated, dim=-1)

    def get_compression_stats(self) -> dict:
        return self.kv_cache.get_stats() if self.kv_cache else {}


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_PATH = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    prompt = "Who are you?\nAnswer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("=" * 60)
    print("Testing Qwen3 Forward with TurboQuant (Full QJL Support)")
    print("=" * 60)

    print("\n=== FP16 Baseline ===")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, use_cache=True, do_sample=False)
        print(tokenizer.decode(out[0], skip_special_tokens=True))

    # Test QJL mode
    print("\n=== TurboQuant 4-bit with QJL (keep_recent=16) ===")
    tq_qjl = Qwen3ForwardWithTurboQuant(model, bits=4, use_qjl=True, keep_recent=16)

    with torch.no_grad():
        output_qjl = tq_qjl.generate(inputs["input_ids"], max_new_tokens=50, do_sample=False)
        text_qjl = tokenizer.decode(output_qjl[0], skip_special_tokens=True)
    print(text_qjl)
    stats_qjl = tq_qjl.get_compression_stats()
    print(f"Stats: total={stats_qjl['total_seq']}, compressed={stats_qjl['compressed_seq']}, "
          f"raw={stats_qjl['raw_seq']}, ratio={stats_qjl['ratio']:.2f}x")

    # Test MSE-only mode (for comparison)
    print("\n=== TurboQuant 4-bit MSE-only (keep_recent=16) ===")
    tq_mse = Qwen3ForwardWithTurboQuant(model, bits=4, use_qjl=False, keep_recent=16)

    with torch.no_grad():
        output_mse = tq_mse.generate(inputs["input_ids"], max_new_tokens=50, do_sample=False)
        text_mse = tokenizer.decode(output_mse[0], skip_special_tokens=True)
    print(text_mse)
    stats_mse = tq_mse.get_compression_stats()
    print(f"Stats: total={stats_mse['total_seq']}, compressed={stats_mse['compressed_seq']}, "
          f"raw={stats_mse['raw_seq']}, ratio={stats_mse['ratio']:.2f}x")

    # Test with larger keep_recent
    print("\n=== TurboQuant 4-bit with QJL (keep_recent=32) ===")
    tq_qjl32 = Qwen3ForwardWithTurboQuant(model, bits=4, use_qjl=True, keep_recent=32)

    with torch.no_grad():
        output_qjl32 = tq_qjl32.generate(inputs["input_ids"], max_new_tokens=50, do_sample=False)
        text_qjl32 = tokenizer.decode(output_qjl32[0], skip_special_tokens=True)
    print(text_qjl32)
    stats_qjl32 = tq_qjl32.get_compression_stats()
    print(f"Stats: total={stats_qjl32['total_seq']}, compressed={stats_qjl32['compressed_seq']}, "
          f"raw={stats_qjl32['raw_seq']}, ratio={stats_qjl32['ratio']:.2f}x")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)