"""
Lloyd-Max Optimal Quantization Parameter Calculator

This script computes the optimal decision boundaries and reconstruction
levels (centroids) for Lloyd-Max quantization of a Gaussian N(0,1) distribution.

The Lloyd-Max algorithm alternates between two steps until convergence:
1. Update centroids: c_i = E[X | X in interval i] (conditional expectation)
2. Update boundaries: b_i = (c_{i-1} + c_i) / 2 (midpoint of adjacent centroids)

Reference: Lloyd, S.P. (1957) "Least Squares Quantization in PCM"
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import torch


def lloyd_max_optimal(n_levels: int, max_iter: int = 1000, tol: float = 1e-10) -> tuple:
    """
    Compute Lloyd-Max optimal quantization parameters for N(0,1) distribution.

    Args:
        n_levels: Number of quantization levels (2^bits)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance for boundaries

    Returns:
        boundaries: Decision boundaries (n_levels - 1 values)
        centroids: Reconstruction levels (n_levels values)
    """
    # Initialize with uniform spacing
    boundaries = np.linspace(-3, 3, n_levels + 1)[1:-1]
    centroids = np.zeros(n_levels)

    for iteration in range(max_iter):
        # Step 1: Compute centroids as conditional expectations
        for i in range(n_levels):
            if i == 0:
                b_low, b_high = -np.inf, boundaries[0]
            elif i == n_levels - 1:
                b_low, b_high = boundaries[-1], np.inf
            else:
                b_low, b_high = boundaries[i - 1], boundaries[i]

            # E[X | X in (b_low, b_high)] = integral(x * phi(x)) / integral(phi(x))
            numerator, _ = quad(lambda x: x * norm.pdf(x), b_low, b_high)
            denominator, _ = quad(norm.pdf, b_low, b_high)

            if denominator > 0:
                centroids[i] = numerator / denominator

        # Step 2: Update boundaries as midpoints of adjacent centroids
        new_boundaries = (centroids[:-1] + centroids[1:]) / 2

        # Check convergence
        if np.max(np.abs(new_boundaries - boundaries)) < tol:
            boundaries = new_boundaries
            break

        boundaries = new_boundaries

    return boundaries, centroids


def format_for_torch(values: np.ndarray, precision: int = 4) -> str:
    """
    Format numpy array as torch.tensor string for code generation.

    Args:
        values: Array to format
        precision: Decimal precision

    Returns:
        Formatted string representation
    """
    formatted = np.array2string(values, precision=precision, suppress_small=True,
                                max_line_width=80, separator=', ')
    return f"torch.tensor({formatted})"


def compute_all_bits(bits_list: list = [2, 3, 4, 6, 8]) -> dict:
    """
    Compute Lloyd-Max parameters for all specified bit depths.

    Args:
        bits_list: List of bit depths to compute

    Returns:
        Dictionary with 'boundaries' and 'centroids' for each bit depth
    """
    results = {'boundaries': {}, 'centroids': {}}

    for bits in bits_list:
        n_levels = 2 ** bits
        print(f"Computing {bits}-bit ({n_levels} levels)...")

        boundaries, centroids = lloyd_max_optimal(n_levels)

        results['boundaries'][bits] = boundaries
        results['centroids'][bits] = centroids

        # Verify symmetry (for even n_levels)
        if n_levels % 2 == 0:
            mid = n_levels // 2
            asymmetry = np.max(np.abs(centroids[:mid] + centroids[mid:][::-1]))
            print(f"  Symmetry check: {asymmetry:.2e}")
            assert asymmetry < 1e-6, f"Asymmetric result for {bits}-bit"

    return results


def generate_code_output(results: dict) -> None:
    """
    Print the results in a format suitable for copying into Python code.

    Args:
        results: Dictionary from compute_all_bits()
    """
    print("\n" + "=" * 80)
    print("LM_BOUNDARIES = {")
    for bits in sorted(results['boundaries'].keys()):
        b = results['boundaries'][bits]
        print(f"    {bits}: {format_for_torch(b)},")
    print("}")

    print("\nLM_CENTROIDS = {")
    for bits in sorted(results['centroids'].keys()):
        c = results['centroids'][bits]
        print(f"    {bits}: {format_for_torch(c)},")
    print("}")


if __name__ == "__main__":
    print("=" * 80)
    print("Lloyd-Max Optimal Quantization for N(0,1) Distribution")
    print("=" * 80)

    # Compute all bit depths
    results = compute_all_bits([2, 3, 4, 6, 8])

    # Print detailed results
    print("\n" + "=" * 80)
    print("Detailed Results")
    print("=" * 80)

    for bits in sorted(results['boundaries'].keys()):
        n_levels = 2 ** bits
        b = results['boundaries'][bits]
        c = results['centroids'][bits]

        print(f"\n{bits}-bit ({n_levels} levels):")
        print(f"  Boundaries ({len(b)}):")
        print(f"    {np.array2string(b, precision=4, suppress_small=True, max_line_width=80)}")
        print(f"  Centroids ({len(c)}):")
        print(f"    {np.array2string(c, precision=4, suppress_small=True, max_line_width=80)}")

    # Generate code output
    generate_code_output(results)