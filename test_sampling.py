#!/usr/bin/env python
"""Test QJL with different sampling sizes"""
import sys
sys.path.insert(0, '.')

from qjl import QJL
import torch
import math

print("=" * 60)
print("Testing QJL with Different Sampling Sizes")
print("=" * 60)

dim = 128
torch.manual_seed(42)

# Create test vectors
q = torch.randn(1, dim)
k = torch.randn(1, dim)
true_inner = torch.dot(q[0], k[0])
k_norm = torch.norm(k)

print(f"||q|| = {torch.norm(q):.4f}")
print(f"||k|| = {torch.norm(k):.4f}")
print(f"<q, k> = {true_inner:.4f}")

for s in [16, 32, 64, 128]:
    qjl = QJL(input_dim=dim, sparse_sign_count=s, seed=42)

    q_sampled = qjl.project_and_sample(q)
    k_sampled = qjl.project_and_sample(k)
    k_sign = (k_sampled > 0).float() * 2 - 1

    sampled_inner = torch.dot(q_sampled[0], k_sampled[0])
    sampled_with_sign = torch.dot(q_sampled[0], k_sign[0])

    # Correction factor d/s
    correction = dim / s

    # Estimate without ||k||
    estimated_no_norm = sampled_with_sign * correction

    # Estimate with ||k||
    estimated_with_norm = sampled_with_sign * correction * k_norm

    # TurboQuant formula: sqrt(pi/2)/d * ||k|| * sampled * d = sqrt(pi/2) * ||k|| * sampled
    estimated_tq = math.sqrt(math.pi/2) * k_norm * sampled_with_sign

    print(f"\n--- s = {s} ---")
    print(f"Sampled inner <q,k>: {sampled_inner:.4f} (expected ~{s/dim * true_inner:.4f})")
    print(f"Sampled with sign: {sampled_with_sign:.4f}")
    print(f"Estimate (d/s * sampled): {estimated_no_norm:.4f}")
    print(f"Estimate (d/s * ||k|| * sampled): {estimated_with_norm:.4f}")
    print(f"Estimate (sqrt(pi/2) * ||k|| * sampled): {estimated_tq:.4f}")
    print(f"True <q,k>: {true_inner:.4f}")
    print(f"Errors: no_norm={abs(estimated_no_norm - true_inner):.2f}, with_norm={abs(estimated_with_norm - true_inner):.2f}, tq={abs(estimated_tq - true_inner):.2f}")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)