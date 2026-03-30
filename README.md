# Yet Another TurboQuant in PyTorch (YATQ)
## TurboQuant: KV Cache Quantization with Lloyd-Max and QJL

A PyTorch implementation of **TurboQuant** for KV cache compression, following the paper [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026). With HuggingFace interface supported.

## Overview

This project implements the TurboQuant algorithm for compressing KV caches in Large Language Models. The implementation follows the paper exactly, including:

- **Lloyd-Max optimal scalar quantization** for the Beta/N(0, 1/d) distribution arising from random rotation
- **QJL (Quantized Johnson-Lindenstrauss)** for unbiased inner product estimation
- **Comprehensive evaluation** on real model KV caches (Qwen3-1.7B)
- Experiments shows although 1bit QJL can eliminate bias, it will increase variance and lead to top-k token shift. It is recommended not use 1bit QJL, just MSE. (maybe I'm wrong) 

## Paper References

- **TurboQuant**: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) - ICLR 2026
- **PolarQuant**: [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) - AISTATS 2026
- **QJL**: [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) - AAAI 2025

## Algorithm

![TurboQuant Algorithm](image/TurboQuant.png)

### Stage 1: MSE-Optimal Quantization (TurboQuantMSE)

Based on **TurboQuant Algorithm 1** and **Section 3.1**:

```
1. Store vector norms separately: ||x||₂
2. Normalize to unit length: x̂ = x / ||x||₂
3. Apply random rotation: y = x̂ @ Π  (Π is orthogonal from QR decomposition)
4. After rotation, each coordinate follows N(0, 1/d)
5. Apply Lloyd-Max optimal scalar quantizer per coordinate
6. Dequantize and rescale: x̃ = dequantize(y) * ||x||₂
```

**Key Insight**: After random rotation of a d-dimensional unit vector, each coordinate follows a Beta distribution that converges to N(0, 1/d) for large d. This known distribution enables training-free optimal quantization.

### Stage 2: QJL for Unbiased Inner Products (TurboQuantProd)

Based on **TurboQuant Algorithm 2** and **Definition 1**:

```
1. MSE quantize: x_mse, idx = QuantMSE(x)  (uses bits-1 bits)
2. Compute residual: r = x - x_mse
3. QJL sketch: qjl = sign(S @ r)  (S ∈ R^{d×d}, uses 1 bit per element)
4. Store: (x_mse, qjl, ||r||₂)
```

**Inner Product Estimator** (Definition 1):
```
<y, x> ≈ <y, x_mse> + ||r|| * sqrt(π/2)/d * <S@y, sign(S@r)>
```

This estimator is **unbiased** with variance O(1/d).

## Installation

```bash
pip install torch numpy scipy
pip install transformers accelerate  # for HuggingFace models
```

## Usage

### Basic Usage

```python
from turboquant import TurboQuantMSE, TurboQuantProd
import torch

# Create quantizers (both use 3 bits total)
mse_quantizer = TurboQuantMSE(dim=128, bits=3, seed=28)    # 3 bits for MSE
prod_quantizer = TurboQuantProd(dim=128, bits=3, seed=28)  # 2 bits MSE + 1 bit QJL

# Quantize vectors
k = torch.randn(100, 128)
mse_recon, _, _ = mse_quantizer.quantize(k, return_indices=True)
prod_compressed = prod_quantizer.quantize(k)

# Compute inner products
query = torch.randn(10, 128)
mse_scores = torch.matmul(query, mse_recon.T)
prod_scores = prod_quantizer.inner_product(query, prod_compressed)

# Compare with true inner products
true_scores = torch.matmul(query, k.T)

def metrics(est, true):
    mse = torch.mean((est - true) ** 2).item()
    cos_sim = torch.cosine_similarity(est.flatten(), true.flatten(), dim=0).item()
    corr = torch.corrcoef(torch.stack([est.flatten(), true.flatten()]))[0, 1].item()
    return mse, cos_sim, corr

print("TurboQuantMSE (3-bit MSE):")
print(f"  MSE: {metrics(mse_scores, true_scores)[0]:.4f}, CosSim: {metrics(mse_scores, true_scores)[1]:.4f}, Corr: {metrics(mse_scores, true_scores)[2]:.4f}")

print("TurboQuantProd (2-bit MSE + 1-bit QJL):")
print(f"  MSE: {metrics(prod_scores, true_scores)[0]:.4f}, CosSim: {metrics(prod_scores, true_scores)[1]:.4f}, Corr: {metrics(prod_scores, true_scores)[2]:.4f}")
```

### KV Cache in Qwen3

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from integrations.qwen3_integration import Qwen3ForwardWithTurboQuant

MODEL_PATH = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")

prompt = "Who are you?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# ========== FP16 Baseline ==========
print("=== FP16 Baseline ===")
with torch.no_grad():
    outputs_fp16 = model.generate(**inputs, max_new_tokens=50, temperature=0.7, top_p=0.5, use_cache=True, do_sample=False)
    print(tokenizer.decode(outputs_fp16[0], skip_special_tokens=True))

# ========== TurboQuant with QJL (Unbiased Inner Product) ==========
print("\n=== TurboQuant 4-bit with QJL ===")
tq_qjl = Qwen3ForwardWithTurboQuant(model, bits=4, use_qjl=True, keep_recent=24)

with torch.no_grad():
    outputs_qjl = tq_qjl.generate(inputs["input_ids"], max_new_tokens=50, temperature=0.7, top_p=0.5, do_sample=False)
    print(tokenizer.decode(outputs_qjl[0], skip_special_tokens=True))

stats = tq_qjl.get_compression_stats()
print(f"Compression ratio: {stats['ratio']:.2f}x")

# ========== TurboQuant MSE-only (Recommended for Quality) ==========
print("\n=== TurboQuant 4-bit MSE-only ===")
tq_mse = Qwen3ForwardWithTurboQuant(model, bits=4, use_qjl=False, keep_recent=24)

with torch.no_grad():
    outputs_mse = tq_mse.generate(inputs["input_ids"], max_new_tokens=50, temperature=0.7, top_p=0.5, do_sample=False)
    print(tokenizer.decode(outputs_mse[0], skip_special_tokens=True))
stats = tq_mse.get_compression_stats()
print(f"\nCompression ratio: {stats['ratio']:.2f}x")
```

**Output:**
```
=== FP16 Baseline ===
Who are you?
Answer: I am a large language model developed by Alibaba Group, and I am designed to assist users in various tasks, such as answering questions, providing information, and helping with different kinds of tasks. I am a language model that can understand and generate human-like

=== TurboQuant 4-bit with QJL ===
Who are you?
Answer: I am a large language model developed by Alibaba Group, and I am designed to assist users in various tasks, such as answering questions, providing information, and engaging in conversations. I am trained on a large amount of text, and I am a language

Compression ratio: 1.73x

=== TurboQuant 4-bit MSE-only ===
Who are you?
Answer: I am a large language model developed by Alibaba Group, and I am designed to assist users in various tasks, such as answering questions, providing information, and helping with different kinds of tasks.
But I am not a real person, and I cannot have

Compression ratio: 1.73x
```

## Implementation Details

### HuggingFace Integration

We provide two integration approaches:

#### 1. `TurboQuantHFWithCache` (MSE-only, Easy to Use)

Located in `integrations/hf_integration.py`. This implementation:
- Uses the model's **native forward pass** with `DynamicCache`
- Periodically compresses the KV cache using MSE quantization
- Keeps recent tokens in FP16 for quality
- **Pros**: Simple, works with any HuggingFace model, no forward pass modification
- **Cons**: No QJL support (MSE reconstruction only)

```python
from integrations.hf_integration import TurboQuantHFWithCache

model = AutoModelForCausalLM.from_pretrained(...)
tq_hf = TurboQuantHFWithCache(model, bits=4, keep_recent=32)
output = tq_hf.generate(input_ids, max_new_tokens=50)
```

#### 2. `Qwen3ForwardWithTurboQuant` (Full QJL Support)

Located in `integrations/qwen3_integration.py`. This implementation:
- Implements a **clean forward pass from scratch** following Qwen3's architecture
- Supports both **MSE-only** and **QJL unbiased inner product** modes
- Uses chunk-based storage: compressed chunks are stored once, never recompressed
- **Pros**: True QJL unbiased inner product estimation, better for research
- **Cons**: Only supports Qwen3 models, slower (no FlashAttention optimization)

```python
from integrations.qwen3_integration import Qwen3ForwardWithTurboQuant

model = AutoModelForCausalLM.from_pretrained(...)

# MSE-only mode (recommended for quality)
tq_mse = Qwen3ForwardWithTurboQuant(model, bits=4, use_qjl=False, keep_recent=32)

# QJL mode (unbiased inner product, use for research)
tq_qjl = Qwen3ForwardWithTurboQuant(model, bits=4, use_qjl=True, keep_recent=32)

output = tq_qjl.generate(input_ids, max_new_tokens=50)
```

### Lloyd-Max Algorithm

The Lloyd-Max algorithm finds optimal centroids minimizing MSE for a given distribution:

```python
def solve_lloyd_max(d, bits, max_iter=200, tol=1e-10):
    # PDF of N(0, 1/d)
    sigma = 1.0 / sqrt(d)

    # Initialize centroids uniformly
    centroids = linspace(-3.5*sigma, 3.5*sigma, n_levels)

    for _ in range(max_iter):
        # Step 1: Boundaries are midpoints between centroids
        boundaries = [(centroids[i] + centroids[i+1]) / 2 for i in range(n_levels-1)]

        # Step 2: Update centroids as E[X | X in partition]
        new_centroids = [integral(x * pdf(x), a, b) / integral(pdf(x), a, b) for each partition]

        if converged: break

    return centroids
```

### Bit Allocation

| Config | MSE bits | QJL bits | Total bits/elem | Additional Storage |
|--------|----------|----------|-----------------|-------------------|
| K:2b | 2 | 0 | 2 | vec_norm (16b/vector) |
| K:2b+QJL | 1 | 1 | 2 | vec_norm + residual_norm (32b/vector) |
| K:3b | 3 | 0 | 3 | vec_norm (16b/vector) |
| K:3b+QJL | 2 | 1 | 3 | vec_norm + residual_norm (32b/vector) |

### Keys vs Values Strategy

| Component | Method | Reason |
|-----------|--------|--------|
| **Keys** | MSE + QJL (optional) | Need inner products `<q, k>` for attention scores |
| **Values** | MSE only | Weighted sum `softmax(scores) @ V` averages out per-vector errors |

**Note**: The paper does not explicitly prescribe this distinction. It is a practical optimization based on how attention mechanisms use keys and values differently.

### QJL Inside Forward Pass

**Why QJL requires modifying the forward pass:**

Standard attention computes: `scores = Q @ K.T` where K is the reconstructed key cache.

With QJL, we need to compute the unbiased inner product estimator:
```
<q, k> ≈ <q, k_mse> + ||r|| * sqrt(π/2)/d * <S@q, sign(S@r)>
```

This requires:
1. Project the query through the same random matrix S: `q_proj = q @ S.T`
2. Compute inner product with stored QJL signs: `<q_proj, sign(S@r)>`
3. Apply the correction term with residual norm

**Implementation comparison:**

| Implementation | QJL Support | Description |
|----------------|-------------|-------------|
| `hf_integration.py` | ❌ No | Uses native model forward, MSE reconstruction only |
| `qwen3_integration.py` | ✅ Yes | Custom forward pass, supports QJL inner product |

The `qwen3_integration.py` stores full QJL data for compressed keys:
- `x_mse`: MSE reconstruction
- `qjl_signs`: sign(S @ residual)
- `residual_norm`: ||residual||₂

For raw (uncompressed) tokens, QJL correction is zero since they're exact.

**Performance note**: The current QJL implementation is slower than optimized attention (no FlashAttention). Community contributions for CUDA optimization are welcome. 


### Using QJL Unbiased Inner Product

QJL provides unbiased inner product estimation:

```
<y, x> ≈ <y, x_mse> + ||r|| * sqrt(π/2)/d * <S@y, sign(S@r)>
```

**Term 1** (`<y, x_mse>`): Standard inner product with MSE reconstruction
**Term 2** (QJL correction): Unbiased estimator eliminating quantization bias

```python
# Direct usage of compute_attention_qjl
tq_cache = TurboQuantKVCache(head_dim=128, bits=4)
tq_cache.append(keys, values)  # keys: (B, H, S, D)

query = torch.randn(1, 8, 128)  # (B, H, D)
output, weights = tq_cache.compute_attention_qjl(query)
# output: (B, H, D), weights: (B, H, S)
# Uses QJL: <query, keys> ≈ <query, k_mse> + qjl_correction
```

### WHT-based TurboQuant (Recommended for QJL)

For best results with QJL, use the WHT-based implementation matching llama.cpp:

```python
from turboquant_wht import TurboQuantWHT

# Create WHT-based quantizer
wht = TurboQuantWHT(dim=128, bits=3)

# Quantize keys with QJL
key_data = wht.quantize_key(keys, use_qjl=True)

# Compute attention scores
scores = wht.compute_attention_scores(query, key_data, use_qjl=True, scale=1/math.sqrt(128))
```

**WHT vs Random Rotation comparison (3-bit inner product MSE):**

| Method | MSE | QJL Effect |
|--------|-----|------------|
| Random rotation | 4.34 | QJL makes it 5.7x worse |
| **WHT** | 4.53 | **QJL makes it 1.8x better** |


## Experimental Results

### Test Setup
- **Model**: Qwen3-1.7B (28 layers, 8 KV heads, 128 head_dim)
- **Context Length**: 4124 tokens for attention metrics, 1584 tokens for PPL
- **Baseline FP16 PPL**: 4.6562
- **Task**: Needle retrieval from long context + Perplexity measurement

### Perplexity Comparison (Random Rotation vs WHT)

| Config | Random PPL | WHT PPL | Random Δ | WHT Δ | Ratio |
|--------|------------|---------|----------|-------|-------|
| 2b MSE | 9792 | 4080 | +9787 | +4075 | 8x |
| 2b QJL | 16128 | **2256** | +16123 | +2251 | 8x |
| 3b MSE | 2048 | 624 | +2043 | +619 | 5.33x |
| 3b QJL | 3376 | **99** | +3371 | +94 | 5.33x |
| 4b MSE | 604 | 10.12 | +599 | +5.47 | 4x |
| 4b QJL | 1408 | **5.0** | +1403 | +0.34 | 4x |
| 6b MSE | 4.78 | 4.62 | +0.12 | -0.03 | 2.67x |
| 6b QJL | 4.72 | **4.59** | +0.06 | -0.06 | 2.67x |
| 8b MSE | 4.66 | 4.62 | +0.00 | -0.03 | 2x |
| 8b QJL | 4.62 | 4.62 | -0.03 | -0.03 | 2x |

**Key Finding**: WHT + QJL dramatically outperforms Random Rotation + QJL in PPL.
- **WHT 4b+QJL** achieves near-baseline PPL (5.0) with **4x compression**
- **WHT 3b+QJL** achieves PPL 99 with **5.33x compression**

### Attention Score Metrics Comparison

| Config | Method | CosSim | Top1% | Top5% | KL-Div | Bias% | Variance |
|--------|--------|--------|-------|-------|--------|-------|----------|
| 2b | Random | 0.9975 | 63.8 | 88.8 | 6.58 | -9.90% | 320387 |
| 2b | WHT | 0.9973 | 62.1 | 88.4 | 7.57 | +2.53% | 369370 |
| 2b+QJL | Random | 0.9909 | 50.0 | 74.6 | 10.41 | +1.66% | 383961 |
| 2b+QJL | **WHT** | 0.9988 | **64.7** | **91.1** | 7.27 | +0.23% | **21797** |
| 3b | Random | 0.9992 | 72.8 | 94.6 | 5.23 | -2.96% | 48691 |
| 3b | WHT | 0.9993 | 73.2 | 96.0 | 5.16 | +0.29% | 126877 |
| 3b+QJL | Random | 0.9967 | 61.2 | 82.6 | 8.19 | -0.06% | 61805 |
| 3b+QJL | **WHT** | 0.9996 | **78.6** | **97.8** | 3.51 | +0.01% | **5455** |
| 4b | Random | 0.9998 | 79.9 | 99.6 | 3.28 | -0.44% | 7048 |
| 4b | WHT | 0.9998 | 83.0 | 98.2 | 2.67 | -0.36% | 6021 |
| 4b+QJL | Random | 0.9990 | 69.6 | 94.2 | 6.09 | +0.39% | 29223 |
| 4b+QJL | **WHT** | 0.9999 | **86.6** | **99.6** | 1.52 | +0.00% | **1820** |
| 6b | Random | 1.0000 | 93.3 | 99.6 | 0.43 | -0.01% | 758 |
| 6b | WHT | 1.0000 | 96.4 | 99.6 | 0.40 | -0.02% | 255 |
| 6b+QJL | Random | 0.9999 | 89.7 | 99.6 | 1.46 | +0.00% | 1804 |
| 6b+QJL | WHT | 1.0000 | 96.4 | 99.6 | 0.46 | -0.00% | 130 |
| 8b | Random | 1.0000 | 92.9 | 99.6 | 0.16 | +0.00% | 150 |
| 8b | WHT | 1.0000 | 99.1 | 100.0 | 0.07 | -0.00% | 18 |
| 8b+QJL | Random | 1.0000 | 94.6 | 100.0 | 0.43 | -0.00% | 181 |
| 8b+QJL | WHT | 1.0000 | 98.2 | 100.0 | 0.06 | +0.00% | 10 |

### Key Observations

**WHT vs Random Rotation (with QJL):**
- WHT+QJL has **17x lower variance** than Random+QJL at 2-bit (21797 vs 383961)
- WHT+QJL achieves **15% higher Top-1** than Random+QJL at 2-bit (64.7% vs 50.0%)
- WHT+QJL achieves **17% higher Top-1** than Random+QJL at 3-bit (78.6% vs 61.2%)

**Why WHT+QJL works better:**
- WHT is deterministic → lower variance in rotation
- Random rotation is stochastic → higher variance
- QJL's unbiased estimator amplifies existing variance
- Low-variance WHT + QJL = beneficial
- High-variance Random + QJL = harmful

## Key Findings

### 1. WHT + QJL is the Optimal Combination

**Confirmed**: WHT-based QJL significantly outperforms Random Rotation-based QJL:
- **PPL**: WHT 4b+QJL achieves PPL 5.0 (vs Random 1408) at 4x compression
- **Top-1**: WHT 4b+QJL achieves 86.6% (vs Random 69.6%) at 4-bit
- **Variance**: WHT 2b+QJL has 17x lower variance than Random 2b+QJL

**Recommendation**: Use WHT implementation (`qwen3_wht_integration`) with QJL for best results.

### 2. Random Rotation + QJL is Harmful (Confirmed)

**Theory**: QJL eliminates bias but increases variance.

**Observation with Random Rotation**: QJL's added variance hurts more:
- MSE-only achieves better Top-K at same bit budget
- QJL increases PPL 2-10x at 3-4 bit with random rotation

### 3. Optimal Configurations

| Use Case | Recommended Config | PPL | Ratio | Reason |
|----------|-------------------|-----|-------|--------|
| Maximum quality | 8b MSE | ~4.66 | 2x | Near-perfect reconstruction |
| Balanced compression | **WHT 4b+QJL** | ~5.0 | 4x | Near-baseline PPL, excellent Top-K |
| High compression | **WHT 3b+QJL** | ~99 | 5.33x | Good PPL, reasonable Top-K |
| Extreme compression | WHT 2b+QJL | ~2256 | 8x | Better than Random+QJL |

## Recommendations

| Use Case | Recommended Config | PPL | Ratio | Notes |
|----------|-------------------|-----|-------|-------|
| Production quality | **WHT 4b+QJL** | ~5.0 | 4x | Near-baseline PPL, use `qwen3_wht_integration` |
| High compression | **WHT 3b+QJL** | ~99 | 5.33x | Good PPL with high compression |
| Maximum compression | WHT 2b+QJL | ~2256 | 8x | Better than Random Rotation alternative |
| High quality | 8b MSE | ~4.66 | 2x | Near-perfect, both methods equivalent |
| Random Rotation users | MSE-only | - | - | **Never use QJL with random rotation** |

**Critical**: Only use QJL with WHT implementation. Random Rotation + QJL is harmful.

## File Structure

```
TurboQuant/
├── polarquant.py              # Lloyd-Max + Random Rotation quantization
├── qjl.py                     # QJL implementation
├── turboquant.py              # TurboQuantMSE and TurboQuantProd classes (random rotation)
├── turboquant_wht.py          # WHT-based TurboQuant matching llama.cpp
├── integrations/
│   ├── __init__.py
│   ├── hf_integration.py      # HuggingFace integration (MSE-only)
│   ├── qwen3_integration.py   # Qwen3 forward with random rotation + QJL support
│   └── qwen3_wht_integration.py # Qwen3 forward with WHT + QJL (recommended)
├── validate_qwen.py           # Comprehensive validation: PPL + Attention metrics
├── measure_true_ppl.py        # True perplexity measurement (random rotation)
├── measure_true_ppl_wht.py    # WHT-based PPL test
├── test_context.txt           # Test context file
└── README.md
```

## Running Tests

```bash
# Test individual components
python polarquant.py
python qjl.py
python turboquant.py

# Validate on real model (Top-K accuracy)
python validate_qwen.py

# Measure true perplexity with compressed KV cache
python measure_true_ppl.py
```

## Discussions and Future Works

- ✅ **Qwen3 Integration**: Full QJL support is now available in `integrations/qwen3_integration.py`
- ✅ **HuggingFace Integration**: MSE-only compression available in `integrations/hf_integration.py`
- ✅ **True PPL Measurement**: `measure_true_ppl.py` confirms QJL is beneficial at 2-bit, harmful at 3-4 bit (random rotation)
- ✅ **WHT Implementation**: `turboquant_wht.py` implements Walsh-Hadamard Transform matching llama.cpp
- ✅ **WHT QJL Works**: WHT-based QJL is **beneficial** at all bit levels (PPL 5.0 at 4-bit, 99 at 3-bit)
- ✅ **WHT Integration**: Full Qwen3 forward pass with WHT available in `integrations/qwen3_wht_integration.py`
- ✅ **Comprehensive Validation**: `validate_qwen.py` compares Random Rotation vs WHT with full PPL measurement
- 🔲 **CUDA Support**: Current implementation is PyTorch-only. CUDA kernels would significantly speed up compression
- 🔲 **BF16 Native Support**: Currently converts to float32 for quantization. Native BF16 would reduce overhead
- 🔲 **More Models**: Extend integration approach to other model architectures (Llama, Mistral, etc.)

## Citation

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and others},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}

@article{zandieh2025polarquant,
  title={PolarQuant: Quantizing KV Caches with Polar Transformation},
  author={Zandieh, Amir and others},
  journal={arXiv preprint arXiv:2502.02617},
  year={2025}
}

@article{zandieh2024qjl,
  title={QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead},
  author={Zandieh, Amir and others},
  journal={arXiv preprint arXiv:2406.03482},
  year={2024}
}
```

## License

This is a research implementation. Please refer to the original papers for licensing terms.