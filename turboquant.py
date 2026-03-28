"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

Based on paper: https://arxiv.org/abs/2504.19874

TurboQuant combines two techniques:
1. MSE Quantizer: Random rotation + Lloyd-Max scalar quantization (b-1 bits)
2. QJL: 1-bit residual error correction using QJL transform

This achieves:
- ~5x memory compression (down to 3 bits total)
- Near-optimal distortion bound
- No training or fine-tuning required

Algorithm 2 from TurboQuant paper:
Quantprod(x):
  idx ← Quantmse(x)              # Lloyd-Max with b-1 bits
  r ← x - DeQuantmse(idx)        # Residual vector
  qjl ← sign(S · r)              # QJL sketch of residual (d sign bits)
  output: (idx, qjl, ||r||_2)

For attention: <q, x̃> = <q, x̃_mse> + sqrt(π/2)/d · ||r|| · <qS, sign(Sr)>

Storage breakdown for b bits total:
- MSE indices: (b-1) bits/element → (b-1)*d bits per vector
- QJL signs: 1 bit/element → d bits per vector
- Residual norm: 16 bits per vector (FP16)
- Total: b*d bits per vector + 16 bits for norm (negligible for large sequences)
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Dict, Any

# Use relative imports when in package, absolute imports when standalone
try:
    from .polarquant import PolarQuant, RandomRotation, solve_lloyd_max
    from .qjl import QJL
except ImportError:
    from polarquant import PolarQuant, RandomRotation, solve_lloyd_max
    from qjl import QJL


class TurboQuantMSE:
    """
    Stage 1: MSE-optimal quantizer.

    As per TurboQuant paper Algorithm 1 and Section 3.1:
    1. Store vector norms separately
    2. Normalize vectors to unit length
    3. Apply random rotation Π
    4. Apply Lloyd-Max optimal scalar quantizer per coordinate
    """

    def __init__(self, dim: int, bits: int, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.n_levels = 2 ** bits

        # Random rotation matrix (orthogonal)
        self.rotation = RandomRotation(dim, seed)

        # Lloyd-Max optimal centroids for N(0, 1/d)
        self.centroids = solve_lloyd_max(dim, bits)

    def quantize(self, x: torch.Tensor, return_indices: bool = False) -> Tuple:
        """
        Quantize vectors using Lloyd-Max optimal quantizer.

        Args:
            x: Input tensor of shape (..., dim)
            return_indices: Whether to return indices for storage

        Returns:
            x_reconstructed: Reconstructed tensor
            vec_norms: (optional) Vector norms
            indices: (optional) Lloyd-Max indices
        """
        original_shape = x.shape
        dim = x.shape[-1]

        # Step 1: Compute and store vector norms
        vec_norms = torch.norm(x, p=2, dim=-1, keepdim=True)

        # Step 2: Normalize to unit length
        x_norm = x / (vec_norms + 1e-8)

        # Step 3: Apply random rotation
        x_rotated = self.rotation.apply(x_norm)  # x_rotated = x_norm @ Q

        # Step 4: Lloyd-Max quantization (find nearest centroid for each coordinate)
        centroids = self.centroids.to(x_rotated.device, x_rotated.dtype)
        diffs = x_rotated.unsqueeze(-1) - centroids  # (..., dim, n_levels)
        indices = diffs.abs().argmin(dim=-1)  # (..., dim)

        # Step 5: Dequantize (lookup centroids)
        x_rotated_quantized = centroids[indices]  # (..., dim)

        # Step 6: Inverse rotation
        x_unrotated = self.rotation.apply(x_rotated_quantized, inverse=True)

        # Step 7: Rescale to original magnitude
        x_reconstructed = x_unrotated * vec_norms

        if return_indices:
            return x_reconstructed, vec_norms.squeeze(-1), indices
        return x_reconstructed

    def dequantize(self, vec_norms: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Dequantize from stored norms and indices.

        Args:
            vec_norms: Vector norms, shape (...,)
            indices: Lloyd-Max indices, shape (..., dim)

        Returns:
            Reconstructed tensor
        """
        centroids = self.centroids.to(indices.device)
        x_rotated = centroids[indices.long()]

        # Inverse rotation
        x_unrotated = self.rotation.apply(x_rotated, inverse=True)

        # Rescale
        return x_unrotated * vec_norms.unsqueeze(-1)


class TurboQuantProd:
    """
    Stage 1 + Stage 2: Unbiased inner product quantizer.

    As per TurboQuant Algorithm 2:
    - Uses (b-1)-bit MSE quantizer + 1-bit QJL on residuals
    - Provides unbiased inner product estimates

    Inner product estimator from Definition 1:
    <y, x> ≈ <y, x_mse> + ||r|| * sqrt(π/2)/m * <S@y, sign(S@r)>
    """

    def __init__(self, dim: int, bits: int, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)

        # Stage 1: MSE quantizer with (bits-1) bits
        self.mse = TurboQuantMSE(dim, self.mse_bits, seed=seed)

        # Stage 2: QJL projection matrix S ∈ R^{d×d}
        if seed is not None:
            torch.manual_seed(seed + 10000)
        self.S = torch.randn(dim, dim)

    def quantize(self, x: torch.Tensor) -> dict:
        """
        Full TurboQuant quantization.

        Returns dict with:
            - 'x_mse': MSE reconstruction (for term1 in inner product)
            - 'mse_indices': (..., dim) Lloyd-Max indices
            - 'qjl_signs': (..., dim) sign bits of QJL-projected residual
            - 'residual_norm': (...,) L2 norm of residual
        """
        # Convert to float for computation
        x_float = x.float()

        # Stage 1: MSE quantize
        x_mse, vec_norms, mse_indices = self.mse.quantize(x_float, return_indices=True)

        # Compute residual in original space
        residual = x_float - x_mse
        residual_norm = torch.norm(residual, dim=-1)  # (...,)

        # Stage 2: QJL - project residual and take sign
        S = self.S.to(residual.device)
        projected = residual @ S.T  # (..., dim)
        qjl_signs = (projected >= 0).float() * 2 - 1  # {-1, +1}

        return {
            "x_mse": x_mse,  # Store MSE reconstruction directly (for term1)
            "mse_indices": mse_indices,
            "qjl_signs": qjl_signs,
            "residual_norm": residual_norm,
        }

    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """
        Compute unbiased inner product estimate.

        From TurboQuant Definition 1:
        <y, x> ≈ <y, x_mse> + ||r|| * sqrt(π/2)/m * <S@y, sign(S@r)>

        Args:
            y: query vectors (..., dim) - NOT quantized
            compressed: dict from quantize() containing x_mse of shape (n_vectors, dim)

        Returns:
            Estimated inner products (..., n_vectors)
        """
        # Flatten x_mse to 2D if needed
        x_mse = compressed["x_mse"]
        original_x_shape = x_mse.shape
        x_mse_flat = x_mse.reshape(-1, self.dim).float()  # (n_vectors, dim)

        # Ensure y is float
        y_float = y.float()

        # y: (batch_q, dim) -> term1: (batch_q, n_vectors)
        # Use mT for proper transpose
        term1 = torch.matmul(y_float, x_mse_flat.mT)

        # Term 2: QJL correction
        S = self.S.to(y.device, torch.float32)
        y_projected = y_float @ S.T  # (batch_q, dim)

        # Flatten qjl_signs to 2D and ensure float
        qjl_signs_flat = compressed["qjl_signs"].reshape(-1, self.dim).float()
        qjl_ip = torch.matmul(y_projected, qjl_signs_flat.mT)  # (batch_q, n_vectors)

        m = self.dim
        correction_scale = math.sqrt(math.pi / 2) / m
        # Flatten residual_norm
        residual_norm_flat = compressed["residual_norm"].reshape(-1).float()
        term2 = correction_scale * qjl_ip * residual_norm_flat.unsqueeze(0)

        return term1 + term2


class TurboQuantKVCache:
    """
    KV cache wrapper that uses TurboQuant to compress keys and values.

    Keys use TurboQuantProd (for unbiased inner products via QJL).
    Values use TurboQuantMSE (for MSE reconstruction only).

    Two usage modes:
    1. QJL mode (recommended): Use compute_attention() to get unbiased estimates
    2. MSE-only mode: Use get_reconstructed_keys/values() for simple reconstruction
    """

    def __init__(self, head_dim: int, bits: int = 3, seed: int = 42, num_heads: int = None):
        self.head_dim = head_dim
        self.bits = bits
        self.num_heads = num_heads

        # Use TurboQuantProd for keys (need inner products for attention)
        self.key_quantizer = TurboQuantProd(head_dim, bits, seed=seed)
        # Use TurboQuantMSE for values (need MSE reconstruction, not inner products)
        self.value_quantizer = TurboQuantMSE(head_dim, bits, seed=seed + 100)

        # Storage - accumulated across all append calls
        self.key_cache = {
            "x_mse": [],      # list of tensors
            "qjl_signs": [],  # list of tensors
            "residual_norm": [],  # list of tensors
            "shapes": [],     # original shapes for reconstruction
        }
        self.value_cache = {
            "indices": [],
            "vec_norms": [],
            "shapes": [],
        }

    def append(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Append new key-value pairs to cache (incremental).

        Args:
            keys: (B, H, S_new, D) - new keys to append
            values: (B, H, S_new, D) - new values to append

        Note: Call update() to sync with model's DynamicCache after appending.
        """
        orig_shape = keys.shape  # (B, H, S, D)

        # Flatten to (B*H*S, D) for quantization
        k_flat = keys.reshape(-1, self.head_dim)
        v_flat = values.reshape(-1, self.head_dim)

        # Compress keys with TurboQuantProd
        k_compressed = self.key_quantizer.quantize(k_flat)
        self.key_cache["x_mse"].append(k_compressed["x_mse"])
        self.key_cache["qjl_signs"].append(k_compressed["qjl_signs"])
        self.key_cache["residual_norm"].append(k_compressed["residual_norm"])
        self.key_cache["shapes"].append(orig_shape)

        # Compress values with TurboQuantMSE
        v_recon, v_norms, v_indices = self.value_quantizer.quantize(v_flat, return_indices=True)
        self.value_cache["indices"].append(v_indices)
        self.value_cache["vec_norms"].append(v_norms)
        self.value_cache["shapes"].append(orig_shape)

    def compute_attention_qjl(self, query: torch.Tensor, scale: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention using QJL's unbiased inner product estimation.

        Args:
            query: (B, H, D) or (B, D) - query vectors
                   H should match num_kv_heads of the cache
            scale: attention scale factor (default: 1/sqrt(head_dim))

        Returns:
            output: (B, H, D) - attention output
            weights: (B, H, S) - attention weights (after softmax)
        """
        import torch.nn.functional as F

        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        # Handle 2D query (single head)
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (B, D) -> (B, 1, D)

        B, H, D = query.shape

        # Get total sequence length from cache
        if not self.key_cache["shapes"]:
            raise ValueError("Cache is empty. Call append() first.")

        total_seq = sum(s[2] for s in self.key_cache["shapes"])
        cache_shape = self.key_cache["shapes"][0]  # (B, H_cache, S, D)
        H_cache = cache_shape[1]

        # Concat all stored data
        x_mse_all = torch.cat(self.key_cache["x_mse"], dim=0)  # (B*H_cache*S_total, D)
        qjl_signs_all = torch.cat(self.key_cache["qjl_signs"], dim=0)
        residual_norm_all = torch.cat(self.key_cache["residual_norm"], dim=0)

        # Reshape to (B, H_cache, S_total, D)
        x_mse_reshaped = x_mse_all.reshape(B, H_cache, total_seq, D)
        qjl_signs_reshaped = qjl_signs_all.reshape(B, H_cache, total_seq, D)
        residual_norm_reshaped = residual_norm_all.reshape(B, H_cache, total_seq)

        # If query has fewer heads than cache, broadcast or use first H query heads
        if H < H_cache:
            # Use first H cache heads
            H_use = H
        elif H > H_cache:
            raise ValueError(f"Query has {H} heads but cache has {H_cache}. H should match or be <= H_cache.")
        else:
            H_use = H

        # Compute attention scores per head using QJL inner product
        all_scores = []
        for h in range(min(H, H_cache)):
            q_h = query[:, h, :]  # (B, D)

            # Build compressed dict for head h
            k_h_mse = x_mse_reshaped[:, h, :, :].reshape(B * total_seq, D)
            k_h_qjl = qjl_signs_reshaped[:, h, :, :].reshape(B * total_seq, D)
            k_h_norm = residual_norm_reshaped[:, h, :].reshape(B * total_seq)

            k_compressed_h = {
                "x_mse": k_h_mse,
                "qjl_signs": k_h_qjl,
                "residual_norm": k_h_norm,
            }

            # QJL inner product
            scores_h = self.key_quantizer.inner_product(q_h, k_compressed_h)  # (B, B*S)

            # Extract diagonal: for batch i query, we want batch i keys
            scores_h = scores_h.reshape(B, B, total_seq)
            scores_h = scores_h.diagonal(dim1=0, dim2=1).T  # (B, S)

            all_scores.append(scores_h)

        # Stack: (B, H, S)
        scores = torch.stack(all_scores, dim=1)

        # Apply scale
        scores = scores * scale

        # Softmax
        weights = F.softmax(scores, dim=-1)

        # Weighted sum with reconstructed values
        v_recon = self.get_values()  # (B, H_cache, S, D)

        # Select values for the heads we computed
        if H <= H_cache:
            v_recon = v_recon[:, :H, :, :]  # (B, H, S, D)

        # weights @ V: (B, H, S) @ (B, H, S, D) -> (B, H, D)
        output = torch.matmul(weights.unsqueeze(2), v_recon).squeeze(2)

        return output, weights

    def get_keys(self) -> torch.Tensor:
        """Reconstruct all cached keys (MSE reconstruction only, no QJL)."""
        keys = []
        for i in range(len(self.key_cache["x_mse"])):
            k = self.key_cache["x_mse"][i]
            shape = self.key_cache["shapes"][i]
            keys.append(k.reshape(shape))
        return torch.cat(keys, dim=2) if keys else torch.tensor([])  # concat along seq dim

    def get_values(self) -> torch.Tensor:
        """Reconstruct all cached values."""
        values = []
        for i in range(len(self.value_cache["indices"])):
            v = self.value_quantizer.dequantize(
                self.value_cache["vec_norms"][i],
                self.value_cache["indices"][i]
            )
            shape = self.value_cache["shapes"][i]
            values.append(v.reshape(shape))
        return torch.cat(values, dim=2) if values else torch.tensor([])  # concat along seq dim

    def get_reconstructed(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get MSE-reconstructed keys and values (QJL not used)."""
        return self.get_keys(), self.get_values()

    def clear(self):
        """Clear the cache."""
        self.key_cache = {"x_mse": [], "qjl_signs": [], "residual_norm": [], "shapes": []}
        self.value_cache = {"indices": [], "vec_norms": [], "shapes": []}

    def compress(self, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One-shot compress and reconstruct (MSE-only, QJL not used).
        Convenience method for simple usage.
        """
        self.clear()
        self.append(keys, values)
        return self.get_reconstructed()

    def memory_usage_bits(self) -> dict:
        """Estimate memory usage in bits."""
        n_keys = sum(k.numel() for k in self.key_cache["x_mse"]) if self.key_cache["x_mse"] else 0
        n_norms = sum(n.numel() for n in self.key_cache["residual_norm"]) if self.key_cache["residual_norm"] else 0
        n_values = sum(v.numel() for v in self.value_cache["indices"]) if self.value_cache["indices"] else 0

        # Key storage: (b-1)*d bits for MSE + d bits for QJL + 16 bits per vector for residual norm
        key_bits = n_keys * self.key_quantizer.mse_bits + n_keys * 1 + n_norms * 16
        x_mse_bits = n_keys * 16  # x_mse stored in FP16

        n_vec_norms = sum(n.numel() for n in self.value_cache["vec_norms"]) if self.value_cache["vec_norms"] else 0
        value_bits = n_values * self.bits + n_vec_norms * 16

        fp16_equivalent = (n_keys + n_values) * 16

        return {
            "key_bits": key_bits,
            "x_mse_bits": x_mse_bits,
            "value_bits": value_bits,
            "total_bits": key_bits + x_mse_bits + value_bits,
            "fp16_bits": fp16_equivalent,
            "compression_ratio": fp16_equivalent / (key_bits + value_bits) if (key_bits + value_bits) > 0 else 0,
        }

    def __len__(self):
        return sum(s[2] for s in self.key_cache["shapes"]) if self.key_cache["shapes"] else 0


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("Testing TurboQuant (MSE + QJL)")
    print("=" * 60)

    dim = 128
    n_bits = 3

    # Test TurboQuantProd
    quantizer = TurboQuantProd(dim=dim, bits=n_bits, seed=42)

    # Generate test vectors
    torch.manual_seed(0)
    x = torch.randn(100, dim) + 2  # shifted mean to test bias
    y = torch.randn(10, dim) + 2

    # Quantize
    compressed = quantizer.quantize(x)

    print(f"\nCompressed storage:")
    print(f"  x_mse shape: {compressed['x_mse'].shape}")
    print(f"  mse_indices shape: {compressed['mse_indices'].shape}")
    print(f"  qjl_signs shape: {compressed['qjl_signs'].shape}")
    print(f"  residual_norm shape: {compressed['residual_norm'].shape}")

    # Compute MSE reconstruction error
    x_mse = compressed['x_mse']
    mse_error = torch.norm(x - x_mse) / torch.norm(x)
    print(f"\nMSE reconstruction relative error: {mse_error.item():.4f}")

    # Test inner product estimation
    true_inner = torch.matmul(y, x.T)  # (10, 100)
    estimated_inner = quantizer.inner_product(y, compressed)  # (10, 100)

    # Check correlation
    correlation = torch.corrcoef(torch.stack([true_inner.flatten(), estimated_inner.flatten()]))[0, 1]
    print(f"Correlation: {correlation.item():.4f}")

    # Check unbiasedness
    a = torch.dot(estimated_inner.flatten(), true_inner.flatten()) / torch.dot(true_inner.flatten(), true_inner.flatten())
    bias = abs((1 - a).item()) * 100
    print(f"Inner product bias (1-a): {bias:.2f}%")

    # Relative error
    rel_error = torch.norm(true_inner - estimated_inner) / torch.norm(true_inner)
    print(f"Inner product relative error: {rel_error.item():.4f}")

    # Test KV Cache (simplified single batch/head case first)
    print("\n" + "=" * 60)
    print("Testing TurboQuantKVCache (simplified)")
    print("=" * 60)

    # Single batch, single head for clarity
    seq_len = 256
    head_dim = 128

    keys = torch.randn(seq_len, head_dim)
    values = torch.randn(seq_len, head_dim)
    query = torch.randn(10, head_dim)  # 10 queries

    kv_cache = TurboQuantKVCache(head_dim=head_dim, bits=3, seed=42)
    kv_cache.append(keys.unsqueeze(0).unsqueeze(0), values.unsqueeze(0).unsqueeze(0))

    # True attention
    true_scores = torch.matmul(query, keys.T) / math.sqrt(head_dim)  # (10, 256)
    true_weights = torch.softmax(true_scores, dim=-1)
    true_output = torch.matmul(true_weights, values)  # (10, 128)

    # TurboQuant attention
    est_scores = kv_cache.attention_scores(query) / math.sqrt(head_dim)  # (10, 256)
    est_weights = torch.softmax(est_scores, dim=-1)

    reconstructed_values = kv_cache.get_values()  # (256, 128)
    est_output = torch.matmul(est_weights, reconstructed_values)  # (10, 128)

    # Errors
    output_error = torch.norm(est_output - true_output) / torch.norm(true_output)
    weights_error = torch.norm(est_weights - true_weights) / torch.norm(true_weights)
    scores_error = torch.norm(est_scores - true_scores) / torch.norm(true_scores)

    print(f"\nAttention scores relative error: {scores_error.item():.4f}")
    print(f"Attention weights relative error: {weights_error.item():.4f}")
    print(f"Attention output relative error: {output_error.item():.4f}")

    # Memory usage
    stats = kv_cache.memory_usage_bits()
    print(f"\nMemory usage:")
    print(f"  Key bits: {stats['key_bits']}")
    print(f"  x_mse bits (FP16): {stats['x_mse_bits']}")
    print(f"  Value bits: {stats['value_bits']}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")

    print("\n" + "=" * 60)
    print("TurboQuant test completed!")
    print("=" * 60)