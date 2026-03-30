"""
WHT-based TurboQuant following llama.cpp implementation exactly.

Key insight from llama.cpp:
- Keys are stored in WHT domain (quantized)
- Query is transformed with WHT on-the-fly during attention
- Values are stored in WHT domain, IWHT applied after attention
- Scale factor: WHT(query) is scaled by scale/D

This matches the fused kernel approach in fattn-vec.cuh and fattn-common.cuh.
"""

import torch
import math
from typing import Tuple, Optional


# Sign patterns from llama.cpp cpy-utils.cuh (256 bits for 256 elements)
TBQ_SIGNS = bytes([
    0xa7, 0x3b, 0x91, 0xf4, 0x6d, 0xc2, 0x58, 0x0e,
    0xb3, 0x7f, 0x24, 0xd6, 0x89, 0x45, 0xea, 0x1c,
    0x63, 0xaf, 0xd8, 0x52, 0x97, 0x0b, 0xe1, 0x3d,
    0x76, 0xc4, 0x19, 0xfe, 0x4a, 0x85, 0x2c, 0xdb,
])

# Independent sign pattern for QJL SRHT (from llama.cpp)
QJL_SIGNS = bytes([
    0xd3, 0x4e, 0xa8, 0x17, 0x9c, 0x5b, 0xe6, 0x31,
    0x72, 0xb9, 0x0d, 0xf5, 0x43, 0x8a, 0x6e, 0xc7,
    0x58, 0x2f, 0x94, 0xe1, 0xb6, 0x3d, 0x0a, 0x7c,
    0xc5, 0x61, 0xd8, 0x4f, 0xa3, 0x97, 0x1e, 0x85,
])


def get_sign(signs: bytes, idx: int) -> float:
    """Get sign (-1.0 or +1.0) at index idx from sign bytes."""
    return -1.0 if ((signs[idx >> 3] >> (idx & 7)) & 1) else 1.0


def serial_wht(x: torch.Tensor) -> torch.Tensor:
    """
    Serial Walsh-Hadamard Transform (in-place butterfly algorithm).

    For d=256, this is equivalent to H @ x where H is the Hadamard matrix.
    Note: WHT is orthogonal up to scaling: H @ H^T = d * I
    So inverse is H @ x / d

    Args:
        x: tensor of shape (..., d) where d must be a power of 2

    Returns:
        transformed tensor of same shape (not normalized)
    """
    shape = x.shape
    d = shape[-1]

    # Flatten to process last dimension
    x_flat = x.reshape(-1, d).clone()

    # Butterfly algorithm: for len = 1, 2, 4, 8, ..., d/2
    length = 1
    while length < d:
        for i in range(0, d, 2 * length):
            for j in range(length):
                u = x_flat[:, i + j].clone()
                v = x_flat[:, i + j + length].clone()
                x_flat[:, i + j] = u + v
                x_flat[:, i + j + length] = u - v
        length *= 2

    return x_flat.reshape(shape)


def inverse_wht(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse Walsh-Hadamard Transform.
    Since H @ H = d * I, the inverse is H @ x / d.
    """
    return serial_wht(x) / x.shape[-1]


# Lloyd-Max boundaries and centroids for N(0, 1) distribution
# From llama.cpp cpy-utils.cuh

LM_BOUNDARIES = {
    2: torch.tensor([-0.9816, 0.0, 0.9816]),  # 3 boundaries for 2-bit (4 levels)
    3: torch.tensor([-1.7480, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7480]),  # 7 boundaries
    4: torch.tensor([-2.4008, -1.8435, -1.4371, -1.0993, -0.7996, -0.5225, -0.2583,
                     0.0, 0.2583, 0.5225, 0.7996, 1.0993, 1.4371, 1.8435, 2.4008]),  # 15 boundaries
}

LM_CENTROIDS = {
    2: torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104]),  # 4 centroids for 2-bit
    3: torch.tensor([-2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520]),  # 8 centroids
    4: torch.tensor([-2.70, -2.12, -1.64, -1.27, -0.96, -0.67, -0.39, -0.13,
                     0.13, 0.39, 0.67, 0.96, 1.27, 1.64, 2.12, 2.70]),  # 16 centroids
}


def lloyd_max_quantize(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lloyd-Max quantization for N(0, 1) distribution.

    Args:
        x: tensor of shape (..., d), assumed to follow N(0, 1) after WHT
        bits: number of bits (2, 3, or 4)

    Returns:
        indices: quantized indices
        centroids: reconstructed values (same shape as x)
    """
    n_levels = 2 ** bits
    boundaries = LM_BOUNDARIES[bits].to(x.device)
    centroids = LM_CENTROIDS[bits].to(x.device)

    # Find index for each element
    indices = torch.zeros_like(x, dtype=torch.long)
    for i, b in enumerate(boundaries):
        indices = indices + (x >= b).long()

    # Clamp to valid range
    indices = indices.clamp(0, n_levels - 1)

    # Get centroids
    reconstructed = centroids[indices]

    return indices, reconstructed


class TurboQuantWHT:
    """
    WHT-based TurboQuant matching llama.cpp exactly.

    The key difference from random rotation:
    - Uses deterministic WHT (Walsh-Hadamard Transform)
    - Uses fixed sign patterns (TBQ_SIGNS for MSE, QJL_SIGNS for QJL)
    - Block size must be 256 (WHT requires power of 2)
    - Stores keys in WHT domain, transforms query on-the-fly
    """

    def __init__(self, dim: int, bits: int, block_size: Optional[int] = None):
        """
        Args:
            dim: dimension of vectors
            bits: bits per element (2, 3, or 4)
            block_size: WHT block size (default: next power of 2 >= dim)
        """
        assert bits in [2, 3, 4], f"bits must be 2, 3, or 4"

        self.dim = dim
        self.bits = bits

        # Determine block size (must be power of 2)
        if block_size is None:
            block_size = 1
            while block_size < dim:
                block_size *= 2

        self.block_size = block_size
        self.padded = block_size != dim

        # Precompute sign tensors (for block_size)
        self.tbq_signs = torch.tensor(
            [get_sign(TBQ_SIGNS, i % 256) for i in range(block_size)],
            dtype=torch.float32
        )
        self.qjl_signs = torch.tensor(
            [get_sign(QJL_SIGNS, i % 256) for i in range(block_size)],
            dtype=torch.float32
        )

    def quantize_key(self, key: torch.Tensor, use_qjl: bool = False) -> dict:
        """
        Quantize key vectors for KV cache storage.

        Following llama.cpp's quantize_f32_tbq*_block:
        1. Compute L2 norm
        2. Normalize
        3. Apply TBQ signs
        4. Apply WHT
        5. Lloyd-Max quantize

        Args:
            key: tensor of shape (..., dim)
            use_qjl: if True, also compute QJL residual data

        Returns:
            dict with quantized data for attention computation
        """
        original_shape = key.shape
        key_float = key.float()
        d = self.dim
        bs = self.block_size

        # Reshape to (n_vectors, dim)
        key_flat = key_float.reshape(-1, d)
        n_vectors = key_flat.shape[0]

        # Pad if necessary
        if self.padded:
            key_flat = F.pad(key_flat, (0, bs - d))

        # Step 1: Compute L2 norm per vector (on original dim)
        vec_norms = torch.norm(key_flat[:, :d], dim=-1)  # (n_vectors,)

        # Step 2: Normalize
        key_norm = key_flat / (vec_norms.unsqueeze(-1) + 1e-10)

        # Step 3: Apply TBQ signs
        signs = self.tbq_signs.to(key.device)
        key_signed = key_norm * signs.unsqueeze(0)

        # Step 4: Apply WHT
        key_wht = serial_wht(key_signed)

        # Step 5: Lloyd-Max quantization
        indices, centroids = lloyd_max_quantize(key_wht, self.bits)

        # Store: indices, norm
        result = {
            'vec_norm': vec_norms,  # (n_vectors,)
            'indices': indices,  # (n_vectors, block_size)
            'centroids_wht': centroids,  # (n_vectors, block_size) - centroids in WHT domain
            'key_wht_norm': key_wht,  # For debugging
        }

        if use_qjl:
            # QJL: compute residual in WHT domain
            residual = key_wht - centroids  # (n_vectors, block_size)

            # Residual norm per vector
            residual_norm = torch.norm(residual, dim=-1)  # (n_vectors,)

            # Apply QJL signs and WHT (SRHT = D2 @ H @ D1)
            # Matching llama.cpp: sign(WHT(qjl_signs * residual))
            qjl_s = self.qjl_signs.to(key.device)
            residual_qjl_signed = residual * qjl_s.unsqueeze(0)
            residual_qjl_wht = serial_wht(residual_qjl_signed)

            # Take sign (1 for >= 0, -1 for < 0)
            qjl_sign_bits = (residual_qjl_wht >= 0).float() * 2 - 1

            # d_qjl = gamma * norm (matching llama.cpp line 325)
            d_qjl = residual_norm * vec_norms

            result['qjl_signs'] = qjl_sign_bits  # (n_vectors, block_size)
            result['residual_norm'] = residual_norm  # (n_vectors,)
            result['d_qjl'] = d_qjl  # (n_vectors,)

        return result

    def quantize_value(self, value: torch.Tensor) -> dict:
        """
        Quantize value vectors for KV cache storage.

        Same as key quantization but simpler (no QJL needed).
        Values are stored in WHT domain for efficiency.
        """
        original_shape = value.shape
        value_float = value.float()
        d = self.dim
        bs = self.block_size

        value_flat = value_float.reshape(-1, d)
        n_vectors = value_flat.shape[0]

        # Pad if necessary
        if self.padded:
            value_flat = F.pad(value_flat, (0, bs - d))

        # L2 norm (on original dim)
        vec_norms = torch.norm(value_flat[:, :d], dim=-1)

        # Normalize + signs + WHT
        value_norm = value_flat / (vec_norms.unsqueeze(-1) + 1e-10)
        signs = self.tbq_signs.to(value.device)
        value_signed = value_norm * signs.unsqueeze(0)
        value_wht = serial_wht(value_signed)

        # Lloyd-Max quantize
        indices, centroids = lloyd_max_quantize(value_wht, self.bits)

        return {
            'vec_norm': vec_norms,
            'indices': indices,
            'centroids_wht': centroids,
        }

    def compute_attention_scores(
        self,
        query: torch.Tensor,
        key_data: dict,
        use_qjl: bool = False,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute attention scores <query, keys> using WHT-based method.

        Following llama.cpp's vec_dot_fattn_vec_KQ_tbq*:
        1. Apply TBQ signs to query
        2. Apply WHT to query
        3. Compute inner product with centroids in WHT domain
        4. Multiply by key norm

        Args:
            query: (batch, dim) query vectors
            key_data: dict from quantize_key()
            use_qjl: if True, add QJL correction
            scale: attention scale (default 1/sqrt(dim))

        Returns:
            scores: (batch, n_keys) attention scores
        """
        if scale is None:
            scale = 1.0 / math.sqrt(self.dim)

        d = self.dim
        bs = self.block_size
        query_float = query.float()
        batch = query_float.shape[0]
        n_keys = key_data['vec_norm'].shape[0]

        # Pad query if necessary
        if self.padded:
            query_float = F.pad(query_float, (0, bs - d))

        # Step 1-2: Apply signs + WHT to query
        signs = self.tbq_signs.to(query.device)
        query_signed = query_float * signs.unsqueeze(0)
        query_wht = serial_wht(query_signed)

        # Step 3: Inner product with centroids in WHT domain
        centroids_wht = key_data['centroids_wht']  # (n_keys, block_size)
        key_norms = key_data['vec_norm']  # (n_keys,)

        # Query scaled by scale/D (from llama.cpp line 271-272)
        # Note: D here is block_size for WHT scaling
        query_scaled = query_wht * scale / bs

        # Inner product (only on original dim columns)
        scores = torch.matmul(query_scaled[:, :d], centroids_wht[:, :d].T)  # (batch, n_keys)

        # Multiply by key norms
        scores = scores * key_norms.unsqueeze(0)

        if not use_qjl:
            return scores

        # QJL correction (matching llama.cpp fattn-common.cuh lines 689-691)
        # Formula: d_qjl * (Q_qjl · qjl_signs)
        # Where Q_qjl = WHT(Q_wht * qjl_signs) * scale * sqrt(pi/2) / D^2

        # Apply QJL signs to query_wht and do second WHT
        qjl_s = self.qjl_signs.to(query.device)
        query_qjl_signed = query_wht * qjl_s.unsqueeze(0)  # Apply signs
        query_qjl_wht = serial_wht(query_qjl_signed)  # Second WHT

        # QJL inner product with stored signs
        qjl_signs = key_data['qjl_signs']  # (n_keys, block_size)
        d_qjl = key_data['d_qjl']  # (n_keys,)

        # QJL correction: d_qjl * (query_qjl_wht · qjl_signs)
        # Scaling: scale * sqrt(pi/2) / D^2 (from llama.cpp line 318)
        qjl_factor = math.sqrt(math.pi / 2)
        qjl_scale = scale * qjl_factor / (bs * bs)

        # Only on original dim columns
        qjl_ip = torch.matmul(query_qjl_wht[:, :d], qjl_signs[:, :d].T)  # (batch, n_keys)

        # Add correction
        scores = scores + d_qjl.unsqueeze(0) * qjl_ip * qjl_scale

        return scores

    def reconstruct_key(self, key_data: dict) -> torch.Tensor:
        """
        Reconstruct key vectors from quantized data (for debugging/testing).

        Args:
            key_data: dict from quantize_key()

        Returns:
            reconstructed keys: (n_keys, dim)
        """
        centroids_wht = key_data['centroids_wht']  # (n_keys, block_size)
        vec_norms = key_data['vec_norm']
        bs = self.block_size
        d = self.dim

        # Inverse WHT to get back to original domain
        signs = self.tbq_signs.to(centroids_wht.device)
        centroids_orig = inverse_wht(centroids_wht) * signs.unsqueeze(0)

        # Rescale by norm and take only original dim
        reconstructed = centroids_orig[:, :d] * vec_norms.unsqueeze(-1)

        return reconstructed


# Test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from turboquant import TurboQuantMSE, TurboQuantProd

    print("=" * 70)
    print("WHT-based TurboQuant Test")
    print("=" * 70)

    dim = 256
    bits = 3

    torch.manual_seed(42)
    keys = torch.randn(100, dim)
    query = torch.randn(10, dim)

    # True inner products
    true_ip = torch.matmul(query, keys.T)

    # WHT version
    wht_quant = TurboQuantWHT(dim, bits)

    # Test MSE-only (scale=1 for raw inner product comparison)
    key_data_mse = wht_quant.quantize_key(keys, use_qjl=False)
    scores_mse = wht_quant.compute_attention_scores(query, key_data_mse, use_qjl=False, scale=1.0)

    # Test with QJL
    key_data_qjl = wht_quant.quantize_key(keys, use_qjl=True)
    scores_qjl = wht_quant.compute_attention_scores(query, key_data_qjl, use_qjl=True, scale=1.0)

    # Random rotation version for comparison
    mse_quant = TurboQuantMSE(dim, bits, seed=42)
    prod_quant = TurboQuantProd(dim, bits, seed=42)

    mse_recon, _, _ = mse_quant.quantize(keys, return_indices=True)
    prod_result = prod_quant.quantize(keys)

    mse_ip = torch.matmul(query, mse_recon.T)
    prod_ip = prod_quant.inner_product(query, prod_result)

    def metrics(est, true):
        mse = torch.mean((est - true) ** 2).item()
        bias = torch.mean(est - true).item() / torch.mean(torch.abs(true)).item() * 100
        var = torch.var(est - true).item()
        return mse, bias, var

    print(f"\n{'Method':<30} {'MSE':>12} {'Bias%':>10} {'Variance':>12}")
    print("-" * 65)

    print(f"{'Random MSE (3b)':<30} {metrics(mse_ip, true_ip)[0]:>12.4f} {metrics(mse_ip, true_ip)[1]:>+10.2f} {metrics(mse_ip, true_ip)[2]:>12.2f}")
    print(f"{'Random QJL (2b+1b)':<30} {metrics(prod_ip, true_ip)[0]:>12.4f} {metrics(prod_ip, true_ip)[1]:>+10.2f} {metrics(prod_ip, true_ip)[2]:>12.2f}")
    print(f"{'WHT MSE (3b)':<30} {metrics(scores_mse, true_ip)[0]:>12.4f} {metrics(scores_mse, true_ip)[1]:>+10.2f} {metrics(scores_mse, true_ip)[2]:>12.2f}")
    print(f"{'WHT QJL (2b+1b)':<30} {metrics(scores_qjl, true_ip)[0]:>12.4f} {metrics(scores_qjl, true_ip)[1]:>+10.2f} {metrics(scores_qjl, true_ip)[2]:>12.2f}")

    # Test reconstruction
    print("\n" + "=" * 70)
    print("Key Reconstruction Test")
    print("=" * 70)

    key_recon = wht_quant.reconstruct_key(key_data_mse)
    recon_mse = torch.mean((key_recon - keys) ** 2).item()
    print(f"Reconstruction MSE: {recon_mse:.6f}")
    print(f"Original key norm (avg): {torch.mean(torch.norm(keys, dim=-1)).item():.4f}")
    print(f"Reconstructed key norm (avg): {torch.mean(torch.norm(key_recon, dim=-1)).item():.4f}")

    print("\n" + "=" * 70)