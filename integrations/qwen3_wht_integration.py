"""
WHT-based Qwen3 Forward Pass with TurboQuant KV Cache Compression.

Uses Walsh-Hadamard Transform matching llama.cpp exactly.
Now supports QJL in attention computation.
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

from turboquant_wht import TurboQuantWHT, serial_wht


class WHTKVCache:
    """
    WHT-based KV cache with proper QJL support.

    Stores compressed key data (indices, norms, qjl_signs) for later use
    in attention computation via compute_attention_scores().
    """

    def __init__(self, num_layers: int, head_dim: int, bits: int, use_qjl: bool = False):
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.bits = bits
        self.use_qjl = use_qjl
        self.quantizers = [TurboQuantWHT(head_dim, bits) for _ in range(num_layers)]

        # Store compressed key data per layer
        self.key_data = [None] * num_layers  # List of dicts
        self.value_data = [None] * num_layers

        # Also store reconstructed values for non-QJL attention
        self.key_cache = [None] * num_layers  # Reconstructed keys (B, H, S, D)
        self.value_cache = [None] * num_layers  # Reconstructed values

        self.total_tokens = 0

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """
        Compress and store new K, V.

        Args:
            new_k: (B, H, S, D) new keys
            new_v: (B, H, S, D) new values

        Returns:
            For MSE mode: reconstructed (k_cache, v_cache)
            For QJL mode: (key_data, value_data) dicts
        """
        B, H, S, D = new_k.shape
        quantizer = self.quantizers[layer_idx]

        # Process each head separately
        # Flatten: (B, H, S, D) -> (B*H*S, D)
        k_flat = new_k.reshape(-1, D)
        v_flat = new_v.reshape(-1, D)

        # Quantize keys
        k_data = quantizer.quantize_key(k_flat, use_qjl=self.use_qjl)

        # Quantize values (simpler, no QJL)
        v_data = quantizer.quantize_value(v_flat)

        # Reconstruct for non-QJL attention
        k_recon = quantizer.reconstruct_key(k_data).reshape(B, H, S, D)
        v_recon = quantizer.reconstruct_key(v_data).reshape(B, H, S, D)

        # Concatenate with existing cache
        if self.key_data[layer_idx] is not None:
            # Merge key data dicts
            old_k_data = self.key_data[layer_idx]
            self.key_data[layer_idx] = {
                'vec_norm': torch.cat([old_k_data['vec_norm'], k_data['vec_norm']], dim=0),
                'indices': torch.cat([old_k_data['indices'], k_data['indices']], dim=0),
                'centroids_wht': torch.cat([old_k_data['centroids_wht'], k_data['centroids_wht']], dim=0),
            }
            if self.use_qjl and 'qjl_signs' in k_data:
                self.key_data[layer_idx]['qjl_signs'] = torch.cat([old_k_data['qjl_signs'], k_data['qjl_signs']], dim=0)
                self.key_data[layer_idx]['d_qjl'] = torch.cat([old_k_data['d_qjl'], k_data['d_qjl']], dim=0)

            # Merge value data
            old_v_data = self.value_data[layer_idx]
            self.value_data[layer_idx] = {
                'vec_norm': torch.cat([old_v_data['vec_norm'], v_data['vec_norm']], dim=0),
                'indices': torch.cat([old_v_data['indices'], v_data['indices']], dim=0),
                'centroids_wht': torch.cat([old_v_data['centroids_wht'], v_data['centroids_wht']], dim=0),
            }

            # Merge reconstructed caches
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k_recon], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v_recon], dim=2)
        else:
            self.key_data[layer_idx] = k_data
            self.value_data[layer_idx] = v_data
            self.key_cache[layer_idx] = k_recon
            self.value_cache[layer_idx] = v_recon

        self.total_tokens += S

        return self.key_data[layer_idx], self.value_data[layer_idx], \
               self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_layer_data(self, layer_idx: int):
        """Get compressed data for a layer."""
        return self.key_data[layer_idx], self.value_data[layer_idx]

    def get_layer_cache(self, layer_idx: int):
        """Get reconstructed cache for a layer."""
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_stats(self):
        return {
            'total_seq': self.total_tokens,
            'ratio': 16.0 / self.bits
        }


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    B, H, S, D = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)


class Qwen3ForwardWithWHT:
    """Qwen3 forward pass with WHT-based KV cache compression and QJL support."""

    def __init__(self, model, bits: int = 4, use_qjl: bool = False, keep_recent: int = 0):
        self.model = model
        self.config = model.config
        self.bits = bits
        self.use_qjl = use_qjl
        self.keep_recent = keep_recent

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.num_kv_heads = model.config.num_key_value_heads
        self.head_dim = model.config.hidden_size // self.num_heads
        self.num_layers = model.config.num_hidden_layers
        self.n_rep = self.num_heads // self.num_kv_heads
        self.vocab_size = model.config.vocab_size

        # Extract weights
        self.embed_tokens = model.model.embed_tokens.weight
        self.lm_head = model.lm_head.weight
        self.norm_weight = model.model.norm.weight

        self.layers = []
        for layer in model.model.layers:
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

        # Create quantizer for query transformation
        self.query_quantizer = TurboQuantWHT(self.head_dim, bits)

        self.kv_cache = None

    def _rms_norm(self, x, weight, eps=1e-6):
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return (weight * x).to(input_dtype)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _attention_qjl(self, query, key_cache, key_data, value_cache, scale, attention_mask, n_rep):
        """
        Compute attention using QJL unbiased estimator (vectorized version).

        Strategy:
        1. Compute base attention scores using reconstructed keys (matmul)
        2. Add QJL correction term using vectorized operations
        """
        B, H_q, S_q, D = query.shape
        total_seq = key_cache.shape[2]  # (B, H_kv, total_seq, D)
        bs = self.query_quantizer.block_size  # WHT block size

        # Step 1: Base attention scores with reconstructed keys (same as MSE)
        k_rep = repeat_kv(key_cache, n_rep)  # (B, H_q, total_seq, D)
        v_rep = repeat_kv(value_cache, n_rep)  # (B, H_q, total_seq, D)

        q_float = query.float()  # (B, H_q, S_q, D)
        k_rep_float = k_rep.float()
        v_rep_float = v_rep.float()

        base_scores = torch.matmul(q_float, k_rep_float.transpose(-2, -1)) * scale  # (B, H_q, S_q, total_seq)

        # Step 2: Add QJL correction
        # For each query head, we need to compute correction with its KV head's key_data

        # Prepare query for QJL: apply TBQ signs + WHT
        signs = self.query_quantizer.tbq_signs.to(query.device)

        # Pad query if necessary
        if self.query_quantizer.padded:
            q_padded = F.pad(q_float, (0, bs - D))
        else:
            q_padded = q_float

        # Apply signs and WHT: (B, H_q, S_q, bs)
        q_signed = q_padded * signs.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        q_wht = serial_wht(q_signed.reshape(-1, bs)).reshape(B, H_q, S_q, bs)

        # Apply QJL signs and second WHT for QJL correction
        qjl_s = self.query_quantizer.qjl_signs.to(query.device)
        q_qjl_signed = q_wht * qjl_s.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        q_qjl_wht = serial_wht(q_qjl_signed.reshape(-1, bs)).reshape(B, H_q, S_q, bs)

        # QJL scale factor: scale * sqrt(pi/2) / D^2
        qjl_factor = math.sqrt(math.pi / 2)
        qjl_scale = scale * qjl_factor / (bs * bs)

        # Compute QJL correction for each head
        # key_data['qjl_signs'] has shape (B * H_kv * total_seq, bs)
        # key_data['d_qjl'] has shape (B * H_kv * total_seq,)
        qjl_signs_flat = key_data['qjl_signs']  # (B * H_kv * total_seq, bs)
        d_qjl_flat = key_data['d_qjl']  # (B * H_kv * total_seq,)

        # Reshape to (B, H_kv, total_seq, bs) and (B, H_kv, total_seq)
        qjl_signs_3d = qjl_signs_flat.reshape(B, self.num_kv_heads, total_seq, bs)
        d_qjl_3d = d_qjl_flat.reshape(B, self.num_kv_heads, total_seq)

        # Compute correction: d_qjl * (q_qjl_wht · qjl_signs)
        # For each KV head, all Q heads sharing it use the same correction

        correction = torch.zeros(B, H_q, S_q, total_seq, device=query.device, dtype=torch.float32)

        for h_q in range(H_q):
            h_kv = h_q // n_rep  # Which KV head this Q head uses

            # q_qjl_wht for this head: (B, S_q, bs)
            q_h = q_qjl_wht[:, h_q]  # (B, S_q, bs)

            # qjl_signs for this KV head: (B, total_seq, bs)
            k_signs = qjl_signs_3d[:, h_kv]  # (B, total_seq, bs)

            # d_qjl for this KV head: (B, total_seq)
            d_h = d_qjl_3d[:, h_kv]  # (B, total_seq)

            # Inner product: (B, S_q, bs) @ (B, bs, total_seq) -> (B, S_q, total_seq)
            # Only use original dim columns for inner product
            ip = torch.matmul(q_h[:, :, :D], k_signs[:, :, :D].transpose(-1, -2))  # (B, S_q, total_seq)

            # Correction: d_qjl * ip * qjl_scale
            correction[:, h_q] = d_h.unsqueeze(1) * ip * qjl_scale

        # Add correction to base scores
        attn_weights = base_scores + correction

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum with values
        attn_output = torch.matmul(attn_weights, v_rep_float)

        return attn_output

    def _attention_mse(self, query, key_cache, value_cache, scale, attention_mask, n_rep):
        """
        Compute attention using reconstructed keys (MSE mode).
        Standard attention with reconstructed KV cache.
        """
        B, H_q, S_q, D = query.shape

        # Repeat K, V for GQA
        k_rep = repeat_kv(key_cache, n_rep)  # (B, H_q, total_seq, D)
        v_rep = repeat_kv(value_cache, n_rep)  # (B, H_q, total_seq, D)

        # Attention scores
        q_float = query.float()
        k_rep_float = k_rep.float()
        v_rep_float = v_rep.float()

        attn_weights = torch.matmul(q_float, k_rep_float.transpose(-2, -1)) * scale

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, v_rep_float)

        return attn_output

    def _attention(self, hidden_states, layer_idx, position_embeddings, attention_mask):
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
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        # Update cache with compression
        key_data, value_data, key_cache, value_cache = self.kv_cache.update(layer_idx, k, v)

        # Compute attention
        scale = 1.0 / math.sqrt(self.head_dim)

        if self.use_qjl:
            attn_output = self._attention_qjl(q, key_cache, key_data, value_cache, scale, attention_mask, self.n_rep)
        else:
            attn_output = self._attention_mse(q, key_cache, value_cache, scale, attention_mask, self.n_rep)

        # Reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        return F.linear(attn_output.to(hidden_states.dtype), layer['o_proj_weight'])

    def _mlp(self, hidden_states, layer_idx):
        layer = self.layers[layer_idx]
        gate = F.linear(hidden_states, layer['gate_proj_weight'])
        up = F.linear(hidden_states, layer['up_proj_weight'])
        return F.linear(F.silu(gate) * up, layer['down_proj_weight'])

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            past_len = self.kv_cache.total_tokens if self.kv_cache else 0
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embedding
        hidden_states = F.embedding(input_ids, self.embed_tokens)

        # RoPE
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Causal mask
        if seq_len > 1:
            total_seq = self.kv_cache.total_tokens + seq_len if self.kv_cache else seq_len
            causal_mask = torch.triu(
                torch.full((seq_len, total_seq), float('-inf'), device=input_ids.device),
                diagonal=total_seq - seq_len + 1
            ).unsqueeze(0).unsqueeze(0)
        else:
            causal_mask = None

        # Layers
        for layer_idx in range(self.num_layers):
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.layers[layer_idx]['input_layernorm_weight'])
            hidden_states = self._attention(hidden_states, layer_idx, position_embeddings, causal_mask)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.layers[layer_idx]['post_attention_layernorm_weight'])
            hidden_states = self._mlp(hidden_states, layer_idx)
            hidden_states = residual + hidden_states

        # Final norm and LM head
        hidden_states = self._rms_norm(hidden_states, self.norm_weight)
        return F.linear(hidden_states, self.lm_head)

    def get_compression_stats(self):
        return self.kv_cache.get_stats() if self.kv_cache else {}


# Keep old SimpleWHTKVCache for backward compatibility
class SimpleWHTKVCache(WHTKVCache):
    """Alias for backward compatibility."""
    pass