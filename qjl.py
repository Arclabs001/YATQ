"""
QJL: 1-Bit Quantized Johnson-Lindenstrauss Transform for KV Cache Quantization

Based on TurboQuant paper: https://arxiv.org/abs/2504.19874
QJL paper: https://arxiv.org/abs/2406.03482

Core Algorithm (from TurboQuant Definition 1):
1. S ∈ R^{d×d} is a random matrix with i.i.d. N(0,1) entries
2. Qqjl(x) = sign(S · x) outputs d sign bits
3. Q^{-1}_qjl(z) = sqrt(π/2) / d * S^T * z

Storage: d bits per vector (1 bit per element)
Correction factor: sqrt(π/2) / d
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class QJL:
    """
    QJL: 1-Bit Quantized Johnson-Lindenstrauss Transform.

    As defined in TurboQuant paper Definition 1:
    - Qqjl(x) := sign(S · x) where S ∈ R^{d×d} has i.i.d. N(0,1) entries
    - Output is {−1, +1}^d, i.e., d sign bits
    - Unbiased inner product estimator with correction sqrt(π/2)/d
    """

    def __init__(self, dim: int, seed: Optional[int] = None):
        """
        Args:
            dim: Dimension of vectors (d in paper)
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)

        # Generate random projection matrix S ∈ R^{d×d} with i.i.d. N(0,1) entries
        # As per TurboQuant Definition 1 and Algorithm 2
        self.projection_matrix = torch.randn(dim, dim)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply projection: x @ S

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Projected tensor of shape (..., dim)
        """
        original_shape = x.shape
        x = x.reshape(-1, self.dim)

        S = self.projection_matrix.to(x.device, x.dtype)
        x_proj = x @ S

        return x_proj.reshape(original_shape[:-1] + (self.dim,))

    def quantize_to_sign(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize to 1-bit signs: sign(x)."""
        return (x > 0).float() * 2 - 1

    def sketch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create QJL sketch of input vectors.

        As per TurboQuant Algorithm 2 line 7:
        qjl ← sign(S · r)

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            sign_bits: 1-bit quantized sketch, shape (..., dim)
            norms: L2 norms of original vectors
        """
        norms = torch.norm(x, p=2, dim=-1)
        x_proj = self.project(x)
        sign_bits = self.quantize_to_sign(x_proj)
        return sign_bits, norms

    def estimate_inner_product(
        self,
        query: torch.Tensor,
        key_sketch: torch.Tensor,
        key_norms: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate inner product using asymmetric estimator.

        As per TurboQuant Definition 1:
        <y, Q^{-1}_qjl(Qqjl(x))> = sqrt(π/2) / d * <y @ S, sign(S @ x)>

        Args:
            query: Query vector (NOT quantized), shape (..., dim)
            key_sketch: QJL sketch of key (sign bits), shape (..., dim)
            key_norms: L2 norms of keys

        Returns:
            Estimated inner product
        """
        # Project query (unquantized)
        query_proj = self.project(query)

        # Compute dot product in projected space
        query_flat = query_proj.reshape(-1, self.dim)
        key_flat = key_sketch.reshape(-1, self.dim)

        proj_dot = torch.sum(query_flat * key_flat, dim=-1)

        # Correction factor: sqrt(π/2) / d (from TurboQuant Definition 1)
        correction = math.sqrt(math.pi / 2) / self.dim

        estimated = proj_dot * correction

        # Scale by key norms
        if key_norms is not None:
            key_norms_flat = key_norms.reshape(-1)
            estimated = estimated * key_norms_flat

        return estimated

    def dequantize(self, sketch: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """
        Dequantize from QJL sketch.

        As per TurboQuant Definition 1:
        Q^{-1}_qjl(z) = sqrt(π/2) / d * γ * S^T · z

        Args:
            sketch: Sign bits, shape (..., dim)
            norm: L2 norm of original vector (γ in paper)

        Returns:
            Reconstructed vector
        """
        original_shape = sketch.shape
        sketch_flat = sketch.reshape(-1, self.dim)

        S = self.projection_matrix.to(sketch.device, sketch.dtype)
        reconstructed = math.sqrt(math.pi / 2) / self.dim * sketch_flat @ S.T

        # Scale by norm
        reconstructed = reconstructed * norm.reshape(-1).unsqueeze(-1)

        return reconstructed.reshape(original_shape)


class QJLEncoder:
    """
    Encoder for QJL compression.
    """

    def __init__(self, dim: int, seed: Optional[int] = None):
        self.qjl = QJL(dim=dim, seed=seed)

    def encode(self, x: torch.Tensor) -> dict:
        """Encode vectors to QJL sketch."""
        sketch, norms = self.qjl.sketch(x)
        return {
            'sketch': sketch,
            'norms': norms,
            'dtype': x.dtype
        }

    def compression_stats(self, original_shape: tuple) -> dict:
        """Calculate compression statistics."""
        original_elements = math.prod(original_shape)
        original_bits = original_elements * 16  # FP16

        # Sketch: 1 bit per element
        compressed_bits = original_elements * 1

        # Add norm overhead (one FP16 per vector)
        n_vectors = original_elements // self.qjl.dim
        compressed_bits += n_vectors * 16

        return {
            'original_bits': original_bits,
            'compressed_bits': compressed_bits,
            'compression_ratio': original_bits / compressed_bits,
            'bits_per_element': compressed_bits / original_elements
        }


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("Testing QJL (TurboQuant Definition 1)")
    print("=" * 60)

    dim = 128
    n_vectors = 100

    # Create QJL
    qjl = QJL(dim=dim, seed=42)

    # Generate test vectors
    torch.manual_seed(0)
    vectors = torch.randn(n_vectors, dim)

    # Create sketches
    sketches, norms = qjl.sketch(vectors)
    print(f"\nOriginal shape: {vectors.shape}")
    print(f"Sketch shape: {sketches.shape} (d={dim} sign bits)")
    print(f"Sketch values are ±1: {(sketches.abs() == 1).all().item()}")

    # Test inner product estimation
    query = torch.randn(10, dim)
    true_inner = torch.matmul(query, vectors.T)  # (10, 100)

    # Estimate using QJL
    estimated_inner = torch.zeros(10, n_vectors)
    for i in range(10):
        q = query[i:i+1]
        estimated_inner[i] = qjl.estimate_inner_product(q, sketches, norms)

    # Compare
    relative_error = torch.norm(true_inner - estimated_inner) / torch.norm(true_inner)
    print(f"\nInner product estimation relative error: {relative_error.item():.4f}")

    # Test correlation
    correlation = torch.corrcoef(torch.stack([true_inner.flatten(), estimated_inner.flatten()]))[0, 1]
    print(f"Correlation between true and estimated: {correlation.item():.4f}")

    # Test unbiasedness
    mean_error = (estimated_inner - true_inner).mean()
    print(f"Mean error (bias): {mean_error.item():.4f}")

    # Compression stats
    encoder = QJLEncoder(dim=dim, seed=42)
    stats = encoder.compression_stats(vectors.shape)
    print(f"\nCompression ratio: {stats['compression_ratio']:.2f}x")
    print(f"Bits per element (QJL only): {stats['bits_per_element']:.2f}")
    print(f"Note: QJL uses {dim} bits per vector + 16 bits for norm")

    print("\n" + "=" * 60)
    print("QJL test completed!")
    print("=" * 60)