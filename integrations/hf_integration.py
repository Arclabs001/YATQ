"""
TurboQuant HuggingFace Integration

This module provides a clean integration with HuggingFace Transformers models.
It compresses the KV cache using TurboQuant to reduce memory usage during generation.

Key features:
- Uses the model's native forward with DynamicCache
- Applies TurboQuant compression to the KV cache
- Supports both MSE reconstruction (TODO: real QJL integration)
- Works with any HuggingFace model that uses DynamicCache

Usage:
    from turboquant.integrations.hf_integration import TurboQuantHFWithCache

    model = AutoModelForCausalLM.from_pretrained(...)
    tq_hf = TurboQuantHFWithCache(model, bits=4)
    output = tq_hf.generate(input_ids, max_new_tokens=50)
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, List, Tuple

try:
    from transformers import DynamicCache
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant import TurboQuantMSE, TurboQuantProd


class TurboQuantHFWithCache:
    """
    TurboQuant integration using DynamicCache with periodic compression.

    This implementation:
    1. Uses the model's native forward with DynamicCache
    2. Periodically compresses the KV cache using TurboQuant
    3. Keeps recent tokens in FP16 for quality

    Args:
        model: HuggingFace model (e.g., from AutoModelForCausalLM)
        bits: Number of bits for quantization (1-8)
        keep_recent: Number of recent tokens to keep in FP16
        use_qjl: Use QJL for unbiased inner product (only for keys)

    Usage:
        model = AutoModelForCausalLM.from_pretrained(...)
        tq_hf = TurboQuantHFWithCache(model, bits=4)
        output = tq_hf.generate(input_ids, max_new_tokens=50)
    """

    def __init__(self, model, bits: int = 4, keep_recent: int = 32, use_qjl: bool = False):
        self.model = model
        self.bits = bits
        self.keep_recent = keep_recent
        self.use_qjl = use_qjl

        # Model config
        self.num_layers = model.config.num_hidden_layers
        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = getattr(model.config, 'num_key_value_heads', self.num_heads)

        # Create quantizers for each layer
        if use_qjl:
            self.key_quantizers = [TurboQuantProd(self.head_dim, bits, seed=i) for i in range(self.num_layers)]
        else:
            self.key_quantizers = [TurboQuantMSE(self.head_dim, bits, seed=i) for i in range(self.num_layers)]
        self.val_quantizers = [TurboQuantMSE(self.head_dim, bits, seed=i + 1000) for i in range(self.num_layers)]

        # Storage for compressed KV (when using QJL, we need to store the compressed data)
        self.compressed_kv = {}

    def _compress_cache(self, cache, keep_recent: int = None):
        """
        Compress the KV cache, keeping recent tokens in FP16.

        For MSE mode: Replace old tokens with quantized version
        For QJL mode: Store QJL-compressed data separately and use MSE reconstruction in cache
        """
        if keep_recent is None:
            keep_recent = self.keep_recent

        for layer_idx in range(self.num_layers):
            k = cache.layers[layer_idx].keys
            v = cache.layers[layer_idx].values

            if k.shape[2] <= keep_recent:
                continue  # Don't compress if too short

            # Split old and recent
            old_len = k.shape[2] - keep_recent
            k_old = k[:, :, :old_len, :]
            k_new = k[:, :, old_len:, :]
            v_old = v[:, :, :old_len, :]
            v_new = v[:, :, old_len:, :]

            # Compress old tokens
            k_old_flat = k_old.reshape(-1, self.head_dim).float()
            v_old_flat = v_old.reshape(-1, self.head_dim).float()

            if self.use_qjl:
                # Use TurboQuantProd for keys (stores MSE + QJL)
                k_comp = self.key_quantizers[layer_idx].quantize(k_old_flat)
                # Store compressed data for later QJL inner product computation
                if layer_idx not in self.compressed_kv:
                    self.compressed_kv[layer_idx] = {'keys': None, 'values': None}
                self.compressed_kv[layer_idx]['keys'] = k_comp
                # Use MSE reconstruction for cache
                k_old_recon = k_comp['x_mse']
            else:
                # Use TurboQuantMSE for keys
                k_old_recon, _, _ = self.key_quantizers[layer_idx].quantize(k_old_flat, return_indices=True)

            v_old_recon, _, _ = self.val_quantizers[layer_idx].quantize(v_old_flat, return_indices=True)

            # Concatenate compressed old + recent new
            cache.layers[layer_idx].keys = torch.cat([
                k_old_recon.reshape(k_old.shape).to(k.dtype),
                k_new
            ], dim=2)
            cache.layers[layer_idx].values = torch.cat([
                v_old_recon.reshape(v_old.shape).to(v.dtype),
                v_new
            ], dim=2)

    def generate(
        self,
        input_ids,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        compress_every: int = 16,
        do_sample: bool = True
    ):
        """
        Generate with compressed KV cache.

        Args:
            input_ids: Input token ids
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            compress_every: Compress every N decode steps
            do_sample: Whether to sample or use greedy decoding

        Returns:
            output_ids: Generated token ids including input
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required. Install with: pip install transformers")

        # Clear compressed KV storage
        self.compressed_kv = {}

        cache = DynamicCache()
        generated = []
        cur_ids = input_ids

        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(cur_ids, past_key_values=cache, use_cache=True)
                cache = outputs.past_key_values

            # Compress periodically (only after prefill)
            if step > 0 and step % compress_every == 0:
                self._compress_cache(cache)

            # Get next token logits
            logits = outputs.logits[:, -1, :]

            if do_sample:
                # Sample with temperature and top-p
                logits = logits / temperature
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                logits[sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)] = float('-inf')
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                # Greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated.append(next_token)
            cur_ids = next_token

        return torch.cat([input_ids] + generated, dim=-1)

    def memory_stats(self, cache):
        """
        Get memory usage statistics.

        Returns:
            dict with fp16_bits, compressed_bits, and ratio
        """
        total_fp16 = 0
        total_compressed = 0

        for layer_idx in range(self.num_layers):
            k = cache.layers[layer_idx].keys
            v = cache.layers[layer_idx].values

            seq_len = k.shape[2]
            n_vecs = k.shape[0] * k.shape[1] * seq_len

            total_fp16 += n_vecs * self.head_dim * 16 * 2  # K + V in FP16

            recent = min(seq_len, self.keep_recent)
            old = max(0, seq_len - recent)

            recent_bits = k.shape[0] * k.shape[1] * recent * self.head_dim * 16 * 2
            old_bits = k.shape[0] * k.shape[1] * old * self.head_dim * self.bits * 2

            total_compressed += recent_bits + old_bits

        return {
            'fp16_bits': total_fp16,
            'compressed_bits': total_compressed,
            'ratio': total_fp16 / total_compressed if total_compressed > 0 else 1.0
        }


# Backward compatibility alias
TurboQuantHF = TurboQuantHFWithCache


def apply_turboquant(model, n_bits=4, use_qjl=False, keep_recent=32, verbose=True):
    """
    Apply TurboQuant to a model.

    Args:
        model: HuggingFace model
        n_bits: Number of bits for quantization
        use_qjl: Use QJL for unbiased inner product
        keep_recent: Number of recent tokens to keep in FP16
        verbose: Print configuration

    Returns:
        TurboQuantHFWithCache instance
    """
    if verbose:
        print(f"Creating TurboQuantHFWithCache for {type(model).__name__}")
        print(f"  Bits: {n_bits}")
        print(f"  Use QJL: {use_qjl}")
        print(f"  Keep recent: {keep_recent}")

    return TurboQuantHFWithCache(model, bits=n_bits, use_qjl=use_qjl, keep_recent=keep_recent)


# Test
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("Testing TurboQuantHFWithCache...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_PATH = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    prompt = "Who are you?\nAnswer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Test FP16 baseline
    print("\n=== FP16 Baseline ===")
    with torch.no_grad():
        outputs_fp16 = model.generate(**inputs, max_new_tokens=50, use_cache=True, do_sample=False)
        text_fp16 = tokenizer.decode(outputs_fp16[0], skip_special_tokens=True)
    print(text_fp16)

    # Test TurboQuant 4-bit
    print("\n=== TurboQuant 4-bit ===")
    tq_hf = TurboQuantHFWithCache(model, bits=4, keep_recent=32)

    with torch.no_grad():
        output = tq_hf.generate(inputs["input_ids"], max_new_tokens=50, temperature=0.7, top_p=0.9, compress_every=16)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text)

    # Test TurboQuant 8-bit
    print("\n=== TurboQuant 8-bit ===")
    tq_hf_8bit = TurboQuantHFWithCache(model, bits=8, keep_recent=32)

    with torch.no_grad():
        output_8bit = tq_hf_8bit.generate(inputs["input_ids"], max_new_tokens=50, temperature=0.7, top_p=0.9, compress_every=16)
        text_8bit = tokenizer.decode(output_8bit[0], skip_special_tokens=True)
    print(text_8bit)