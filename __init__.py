"""
TurboQuant: KV Cache Compression with Near-optimal Distortion Rate

A PyTorch implementation of TurboQuant algorithm for LLM KV cache compression.

Paper: https://arxiv.org/abs/2504.19874

Components:
- TurboQuantMSE: MSE-optimal quantization (Lloyd-Max + random rotation)
- TurboQuantProd: MSE + QJL for unbiased inner product estimation
- TurboQuantKVCache: High-level wrapper for KV cache compression

For model integrations, see:
- integrations/qwen3_integration.py: Qwen3 with full QJL support
- integrations/hf_integration.py: HuggingFace models (MSE-only)

Usage:
    from turboquant import TurboQuantMSE, TurboQuantProd

    # Basic quantization
    quantizer = TurboQuantMSE(dim=128, bits=4)
    x_recon, norms, indices = quantizer.quantize(x, return_indices=True)

    # For Qwen3 integration with QJL:
    from integrations.qwen3_integration import Qwen3ForwardWithTurboQuant
"""

from .turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache

__all__ = [
    # Core components
    'TurboQuantMSE',
    'TurboQuantProd',
    'TurboQuantKVCache',
]

__version__ = '0.1.0'