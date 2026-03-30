"""
PolarQuant: Quantizing KV Caches with Polar Transformation

Based on paper: https://arxiv.org/abs/2502.02617
Also referenced in TurboQuant: https://arxiv.org/abs/2504.19874

Core Algorithm:
1. Normalize input vector and store norm separately
2. Apply random rotation (preconditioning) using random orthogonal matrix
3. After rotation, each coordinate follows Beta distribution (approx N(0, 1/d))
4. Apply Lloyd-Max optimal scalar quantizer for each coordinate

Key Insight from TurboQuant paper Section 3.1:
"We find optimal scalar quantizers for random variables with Beta distributions
by solving a continuous 1-dimensional k-means problem using the Max-Lloyd algorithm."

After random rotation of a unit vector, each coordinate follows:
f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1-x^2)^((d-3)/2)

For large d, this converges to N(0, 1/d).
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RandomRotation:
    """
    Random orthogonal rotation matrix.

    Generated via QR decomposition of a random Gaussian matrix.
    As per TurboQuant paper: "We can generate Π by applying QR decomposition
    on a random matrix with i.i.d Normal entries."
    """

    def __init__(self, dim: int, seed: Optional[int] = None):
        self.dim = dim
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)

        # Generate random orthogonal matrix via QR decomposition
        random_matrix = torch.randn(dim, dim)
        Q, R = torch.linalg.qr(random_matrix)
        # Ensure proper orthogonal matrix (det = +1) by flipping one column if needed
        if torch.det(Q) < 0:
            Q[:, 0] *= -1
        self.rotation_matrix = Q

    def apply(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """
        Apply random rotation.

        Forward: y = x @ Q  (each row is rotated)
        Inverse: x = y @ Q.T

        Args:
            x: Input tensor of shape (..., dim)
            inverse: If True, apply inverse rotation

        Returns:
            Rotated tensor of same shape
        """
        original_shape = x.shape
        dim = x.shape[-1]

        x_flat = x.reshape(-1, dim)
        rot = self.rotation_matrix.to(x.device, x.dtype)
        if inverse:
            x_flat = x_flat @ rot.T  # Inverse
        else:
            x_flat = x_flat @ rot    # Forward
        return x_flat.reshape(original_shape)


def solve_lloyd_max(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10) -> torch.Tensor:
    """
    Solve Lloyd-Max optimal quantizer for the coordinate distribution.

    After random rotation of a d-dimensional unit vector, each coordinate
    follows approximately N(0, 1/d).

    Lloyd-Max algorithm:
    1. Initialize centroids uniformly
    2. Compute boundaries as midpoints between centroids
    3. Update centroids as E[X | X in partition]
    4. Repeat until convergence

    Args:
        d: Vector dimension
        bits: Number of quantization bits
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Optimal centroids tensor of shape (2^bits,)
    """
    from scipy import integrate

    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)

    # PDF of N(0, 1/d)
    def pdf(x):
        return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-x * x / (2 * sigma ** 2))

    # Initialize centroids uniformly in [-3.5*sigma, 3.5*sigma]
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    # Lloyd-Max iterations
    for _ in range(max_iter):
        # Step 1: Compute boundaries (midpoints between adjacent centroids)
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]

        # Step 2: Update centroids as conditional expectations
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]

            # Numerical integration for E[X | a < X < b]
            numerator, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            denominator, _ = integrate.quad(pdf, a, b)

            if denominator > 1e-15:
                new_centroids.append(numerator / denominator)
            else:
                new_centroids.append(centroids[i])

        # Check convergence
        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids

        if max_shift < tol:
            break

    return torch.tensor(centroids, dtype=torch.float32)


class PolarQuant:
    """
    PolarQuant: MSE-optimal quantization using Lloyd-Max algorithm.

    The algorithm (from TurboQuant Algorithm 1):
    1. Store vector norms separately
    2. Normalize vectors to unit length
    3. Apply random rotation
    4. Apply Lloyd-Max optimal scalar quantizer per coordinate
    5. Unrotate and rescale

    This matches the official implementation in turboquant-pytorch.
    """

    def __init__(
        self,
        dim: int,
        n_bits: int = 2,
        seed: Optional[int] = None
    ):
        """
        Args:
            dim: Dimension of input vectors
            n_bits: Number of bits for quantization
            seed: Random seed for reproducible rotation
        """
        self.dim = dim
        self.n_bits = n_bits
        self.seed = seed

        # Number of quantization levels
        self.n_levels = 2 ** n_bits

        # Random rotation matrix
        self.rotation = RandomRotation(dim, seed)

        # Precompute Lloyd-Max optimal centroids
        self.centroids = solve_lloyd_max(dim, n_bits)

    def quantize(
        self,
        x: torch.Tensor,
        return_indices: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Full quantization pipeline with Lloyd-Max optimal quantizer.

        Args:
            x: Input tensor of shape (..., dim)
            return_indices: Whether to return quantization indices

        Returns:
            quantized_x: Reconstructed tensor
            vec_norms: Vector norms (for storage)
            indices: (optional) Quantization indices
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

    def dequantize(
        self,
        vec_norms: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantize from stored norms and indices.

        Args:
            vec_norms: Vector norms, shape (...,)
            indices: Quantization indices, shape (..., dim)

        Returns:
            Reconstructed tensor
        """
        # Lookup centroids
        centroids = self.centroids.to(indices.device)
        x_rotated = centroids[indices.long()]

        # Inverse rotation
        x_unrotated = self.rotation.apply(x_rotated, inverse=True)

        # Rescale
        return x_unrotated * vec_norms.unsqueeze(-1)


class PolarQuantKVCache:
    """
    PolarQuant applied to KV Cache for LLM inference.
    """

    def __init__(
        self,
        head_dim: int,
        n_bits: int = 2,
        seed: Optional[int] = None
    ):
        self.head_dim = head_dim
        self.n_bits = n_bits

        # Separate quantizers for K and V
        self.k_quantizer = PolarQuant(dim=head_dim, n_bits=n_bits, seed=seed)
        self.v_quantizer = PolarQuant(dim=head_dim, n_bits=n_bits, seed=seed + 1 if seed else None)

    def compress_keys(self, keys: torch.Tensor) -> dict:
        """Compress key cache."""
        batch, heads, seq_len, head_dim = keys.shape
        keys_flat = keys.reshape(-1, head_dim)
        _, vec_norms, indices = self.k_quantizer.quantize(keys_flat, return_indices=True)

        return {
            'vec_norms': vec_norms.reshape(batch, heads, seq_len),
            'indices': indices.reshape(batch, heads, seq_len, head_dim),
            'shape': keys.shape
        }

    def compress_values(self, values: torch.Tensor) -> dict:
        """Compress value cache."""
        batch, heads, seq_len, head_dim = values.shape
        values_flat = values.reshape(-1, head_dim)
        _, vec_norms, indices = self.v_quantizer.quantize(values_flat, return_indices=True)

        return {
            'vec_norms': vec_norms.reshape(batch, heads, seq_len),
            'indices': indices.reshape(batch, heads, seq_len, head_dim),
            'shape': values.shape
        }

    def decompress_keys(self, compressed: dict) -> torch.Tensor:
        """Decompress key cache."""
        shape = compressed['shape']
        vec_norms = compressed['vec_norms'].reshape(-1)
        indices = compressed['indices'].reshape(-1, self.head_dim)
        return self.k_quantizer.dequantize(vec_norms, indices).reshape(shape)

    def decompress_values(self, compressed: dict) -> torch.Tensor:
        """Decompress value cache."""
        shape = compressed['shape']
        vec_norms = compressed['vec_norms'].reshape(-1)
        indices = compressed['indices'].reshape(-1, self.head_dim)
        return self.v_quantizer.dequantize(vec_norms, indices).reshape(shape)

    def compression_ratio(self) -> float:
        """Calculate theoretical compression ratio."""
        # Storage: n_bits per coordinate + 16 bits per vector for norm
        # For large seq_len, norm overhead is negligible
        return 16.0 / self.n_bits


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("Testing PolarQuant with Lloyd-Max Optimal Quantizer")
    print("=" * 60)

    dim = 128
    n_bits = 3

    quantizer = PolarQuant(dim=dim, n_bits=n_bits, seed=42)

    # Print centroids
    print(f"\nLloyd-Max centroids for d={dim}, {n_bits} bits:")
    print(f"  Centroids: {quantizer.centroids.tolist()}")
    print(f"  Range: [{quantizer.centroids.min().item():.4f}, {quantizer.centroids.max().item():.4f}]")
    print(f"  Expected range (±3.5σ): [{-3.5/math.sqrt(dim):.4f}, {3.5/math.sqrt(dim):.4f}]")

    # Test quantization
    torch.manual_seed(0)
    x = torch.randn(100, dim) + 2  # shifted mean to test bias
    y = torch.randn(10, dim) + 2

    x_quantized, vec_norms, indices = quantizer.quantize(x, return_indices=True)

    # Compute error
    error = torch.norm(x - x_quantized) / torch.norm(x)
    print(f"\nRelative quantization error: {error.item():.4f}")

    # Test inner product bias
    true_ip = torch.matmul(y, x.T)
    est_ip = torch.matmul(y, x_quantized.T)

    a = torch.dot(est_ip.flatten(), true_ip.flatten()) / torch.dot(true_ip.flatten(), true_ip.flatten())
    print(f"Inner product bias (1-a): {abs((1-a).item())*100:.2f}%")

    print(f"\nCompression ratio: {16 / n_bits:.1f}x")
    print("=" * 60)