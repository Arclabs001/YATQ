#!/usr/bin/env python
"""Test script for TurboQuant"""
import sys
sys.path.insert(0, '.')

from turboquant import TurboQuant
import torch
import math

print("=" * 60)
print("Testing TurboQuant (PolarQuant Mode)")
print("=" * 60)

# Parameters
dim = 128
n_bits = 3
batch_size = 2
seq_len = 256
n_heads = 8

# Create TurboQuant (default: use_residual_correction=False)
tq = TurboQuant(dim=dim, n_bits=n_bits, seed=42)

# Generate test data
torch.manual_seed(0)
x = torch.randn(batch_size, n_heads, seq_len, dim)

print(f"\nInput shape: {x.shape}")
print(f"Target bits: {n_bits}")
print(f"Mode: PolarQuant only (recommended)")

# Quantize
x_quantized, metadata = tq.quantize(x.reshape(-1, dim), return_metadata=True)
x_quantized = x_quantized.reshape(x.shape)

# Compute error
error = torch.norm(x - x_quantized) / torch.norm(x)
print(f"\nRelative quantization error: {error.item():.4f}")
print(f"Compression ratio: {tq.compression_ratio():.1f}x")

# Test KV cache compression
print("\n" + "=" * 60)
print("Testing KV Cache Compression")
print("=" * 60)

keys = torch.randn(batch_size, n_heads, seq_len, dim)
values = torch.randn(batch_size, n_heads, seq_len, dim)
query = torch.randn(batch_size, n_heads, 1, dim)

# Compare with full precision attention
true_attention = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(dim)
true_weights = torch.softmax(true_attention, dim=-1)
true_output = torch.matmul(true_weights, values)

# Test with PolarQuant only (recommended)
print("\n--- PolarQuant 3-bit (Recommended) ---")
compressed_kv = tq.quantize_kv_cache(keys, values)

output, weights = tq.compute_attention_with_compressed(query, compressed_kv)

output_error = torch.norm(output - true_output) / torch.norm(true_output)
weights_error = torch.norm(weights - true_weights) / torch.norm(true_weights)

print(f"Attention output relative error: {output_error.item():.4f}")
print(f"Attention weights relative error: {weights_error.item():.4f}")

# Also test 4-bit for comparison
print("\n--- PolarQuant 4-bit ---")
tq_4bit = TurboQuant(dim=dim, n_bits=4, seed=42)
compressed_kv_4bit = tq_4bit.quantize_kv_cache(keys, values)
output_4bit, weights_4bit = tq_4bit.compute_attention_with_compressed(query, compressed_kv_4bit)

output_error_4bit = torch.norm(output_4bit - true_output) / torch.norm(true_output)
weights_error_4bit = torch.norm(weights_4bit - true_weights) / torch.norm(true_weights)

print(f"Attention output relative error: {output_error_4bit.item():.4f}")
print(f"Attention weights relative error: {weights_error_4bit.item():.4f}")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"3-bit PolarQuant: {output_error.item():.4f} output error, 5.3x compression")
print(f"4-bit PolarQuant: {output_error_4bit.item():.4f} output error, 4x compression")
print("\nRecommendation: Use 3-bit or 4-bit PolarQuant for KV cache compression.")