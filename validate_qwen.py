"""
TurboQuant KV Cache Validation - Comprehensive Metrics
"""

import torch
import torch.nn.functional as F
import sys
import math

sys.path.insert(0, 'C:/Users/arcla/TurboQuant')

from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import TurboQuantProd, TurboQuantMSE

MODEL_PATH = "C:/Users/arcla/models/qwen3-1.7b"
CONTEXT_FILE = "test_context.txt"


def test_config(cache, num_layers, num_kv_heads, head_dim,
                key_bits, key_use_qjl, val_bits, val_use_qjl):
    """Test configuration with comprehensive metrics."""
    total_compressed_bits = 0
    total_uncompressed_bits = 0

    # Metrics
    top1_matches = 0
    top5_matches = 0
    cosine_sims = []
    score_mses = []
    kl_divs = []

    # Inner product metrics
    ip_true_list = []
    ip_est_list = []
    ip_errors = []

    n_checks = 0

    for layer_idx in range(num_layers):
        if hasattr(cache, 'layers'):
            keys = cache.layers[layer_idx].keys
            values = cache.layers[layer_idx].values
        else:
            keys = cache[layer_idx][0]
            values = cache[layer_idx][1]

        B, H, S, D = keys.shape
        total_uncompressed_bits += (keys.numel() + values.numel()) * 16

        # Create quantizers
        key_q = TurboQuantProd(dim=D, bits=key_bits, seed=layer_idx * 1000) if key_use_qjl else None
        key_mse = TurboQuantMSE(dim=D, bits=key_bits, seed=layer_idx * 1000) if not key_use_qjl else None
        val_q = TurboQuantMSE(dim=D, bits=val_bits, seed=layer_idx * 1000 + 500)

        # Compress values
        _, _, _ = val_q.quantize(values.reshape(-1, D), return_indices=True)

        # Calculate storage
        n_vecs = B * H * S
        if key_use_qjl:
            k_bits = n_vecs * D * (key_bits - 1) + n_vecs * D + n_vecs * 32
        else:
            k_bits = n_vecs * D * key_bits + n_vecs * 16
        v_bits = n_vecs * D * val_bits + n_vecs * 16
        total_compressed_bits += k_bits + v_bits

        # Test attention scores
        query = keys[:, :, -1:, :]
        real_scores = torch.matmul(query.float(), keys.float().transpose(-2, -1)).squeeze(-2)

        tq_scores = torch.zeros(B, H, S, device=keys.device)

        for h in range(H):
            q_h = query[0, h, 0, :].unsqueeze(0)
            k_h = keys[0, h, :, :]

            if key_use_qjl:
                comp_k = key_q.quantize(k_h)
                scores_h = key_q.inner_product(q_h, comp_k).squeeze(0)
            else:
                k_recon, _, _ = key_mse.quantize(k_h, return_indices=True)
                scores_h = torch.matmul(q_h, k_recon.T).squeeze(0)

            tq_scores[0, h, :] = scores_h

            # Collect inner products
            ip_true_list.extend(real_scores[0, h].tolist())
            ip_est_list.extend(scores_h.tolist())
            ip_errors.extend((scores_h - real_scores[0, h]).tolist())

        # Per-head metrics
        for h in range(H):
            rs = real_scores[0, h]
            ts = tq_scores[0, h]

            cosine_sims.append(F.cosine_similarity(rs.unsqueeze(0), ts.unsqueeze(0)).item())
            score_mses.append(F.mse_loss(rs, ts).item())

            # KL Divergence of attention weights
            p = F.softmax(rs, dim=0)
            q = F.softmax(ts, dim=0)
            # KL(p||q) = sum(p * log(p/q))
            kl = (p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).sum().item()
            kl_divs.append(kl)

            if rs.argmax().item() == ts.argmax().item():
                top1_matches += 1
            if rs.argmax().item() in ts.topk(5).indices.tolist():
                top5_matches += 1
            n_checks += 1

    # Compute metrics
    ratio = total_uncompressed_bits / total_compressed_bits
    avg_cos = sum(cosine_sims) / len(cosine_sims)
    avg_mse = sum(score_mses) / len(score_mses)
    avg_kl = sum(kl_divs) / len(kl_divs)
    top1_pct = 100 * top1_matches / n_checks
    top5_pct = 100 * top5_matches / n_checks

    # Inner product bias and variance
    ip_true = torch.tensor(ip_true_list)
    ip_est = torch.tensor(ip_est_list)
    ip_error = torch.tensor(ip_errors)

    # Bias: E[est - true] normalized
    ip_bias = (ip_est.mean() - ip_true.mean()).item()
    relative_bias = ip_bias / ip_true.mean().abs().item() * 100

    # Variance of inner product estimates (key metric for QJL trade-off)
    ip_variance = ip_error.var().item()
    ip_std = ip_error.std().item()

    # Relative error
    rel_error = torch.norm(ip_est - ip_true) / torch.norm(ip_true)

    return {
        'ratio': ratio,
        'cos_sim': avg_cos,
        'score_mse': avg_mse,
        'kl_div': avg_kl,
        'top1': top1_pct,
        'top5': top5_pct,
        'ip_bias': ip_bias,
        'relative_bias': relative_bias,
        'ip_variance': ip_variance,
        'ip_std': ip_std,
        'rel_error': rel_error.item(),
    }


def main():
    print("=" * 100)
    print("TurboQuant KV Cache Validation - Comprehensive Metrics")
    print("=" * 100)

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    num_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)

    print(f"Layers: {num_layers}, KV heads: {num_kv_heads}, Head dim: {head_dim}")

    with open(CONTEXT_FILE, 'r', encoding='utf-8') as f:
        context_text = f.read()

    # Find the secret code in context
    import re
    code_match = re.search(r'secret access code is[:\s]+([A-Z0-9-]+)', context_text, re.IGNORECASE)
    secret_code = None
    if code_match:
        secret_code = code_match.group(1)
        print(f"\nSecret code found in context: {secret_code}")
    else:
        print("\nWarning: Could not find secret code in context file")

    prompt = f"{context_text}\n\nQuestion: What is the secret access code?\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8000).to(model.device)
    print(f"Sequence length: {inputs['input_ids'].shape[1]} tokens")

    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    cache = outputs.past_key_values

    # Test if model can retrieve the secret code (baseline)
    print("\n" + "=" * 100)
    print("NEEDLE RETRIEVAL TEST (Baseline - FP16)")
    print("=" * 100)
    with torch.no_grad():
        gen_outputs = model.generate(**inputs, max_new_tokens=20, use_cache=True, do_sample=False)
        generated_text = tokenizer.decode(gen_outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Model answer: {generated_text.strip()}")
    if code_match and secret_code in generated_text:
        print(f"[SUCCESS] Model correctly retrieved the secret code!")
    elif code_match:
        print(f"[FAILED] Expected '{secret_code}' but got '{generated_text.strip()}'")

    # Test configurations
    print("\n" + "=" * 100)
    print("COMPRESSION RESULTS")
    print("=" * 100)

    # Header
    print(f"\n{'Config':<18} {'Ratio':>6} {'CosSim':>8} {'Top1%':>6} {'Top5%':>6} {'KL-Div':>8} {'Bias%':>10} {'Variance':>12}")
    print("-" * 90)

    results = []
    for bits in [2, 3, 4, 8]:
        for key_qjl in [False, True]:
            r = test_config(cache, num_layers, num_kv_heads, head_dim,
                           key_bits=bits, key_use_qjl=key_qjl,
                           val_bits=bits, val_use_qjl=False)

            label = f"K:{bits}b+QJL" if key_qjl else f"K:{bits}b"
            print(f"{label:<18} {r['ratio']:>6.2f} {r['cos_sim']:>8.4f} {r['top1']:>6.1f} {r['top5']:>6.1f} {r['kl_div']:>8.4f} {r['relative_bias']:>+10.2f}% {r['ip_variance']:>12.2f}")

            results.append({'bits': bits, 'qjl': key_qjl, **r})

    # Analysis
    print("\n" + "=" * 100)
    print("QJL vs MSE-only ANALYSIS")
    print("=" * 100)

    print(f"\n{'Bits':<6} {'Method':<12} {'CosSim':>8} {'Top1%':>7} {'KL-Div':>8} {'Bias%':>9} {'Variance':>12}")
    print("-" * 65)

    for bits in [2, 3, 4, 8]:
        mse = next(r for r in results if r['bits'] == bits and not r['qjl'])
        qjl = next(r for r in results if r['bits'] == bits and r['qjl'])

        print(f"{bits:<6} {'MSE-only':<12} {mse['cos_sim']:>8.4f} {mse['top1']:>7.1f} {mse['kl_div']:>8.4f} {mse['relative_bias']:>+9.2f} {mse['ip_variance']:>12.2f}")
        print(f"{'':<6} {'+QJL':<12} {qjl['cos_sim']:>8.4f} {qjl['top1']:>7.1f} {qjl['kl_div']:>8.4f} {qjl['relative_bias']:>+9.2f} {qjl['ip_variance']:>12.2f}")

        # Difference
        cos_diff = qjl['cos_sim'] - mse['cos_sim']
        top1_diff = qjl['top1'] - mse['top1']
        kl_diff = qjl['kl_div'] - mse['kl_div']
        bias_diff = qjl['relative_bias'] - mse['relative_bias']
        var_diff = qjl['ip_variance'] - mse['ip_variance']

        print(f"{'':<6} {'Diff':<12} {cos_diff:>+8.4f} {top1_diff:>+7.1f} {kl_diff:>+8.4f} {bias_diff:>+9.2f} {var_diff:>+12.2f}")
        print()

    # Summary of QJL trade-off
    print("=" * 100)
    print("QJL TRADE-OFF SUMMARY")
    print("=" * 100)
    print("\nTheory: QJL should reduce bias but increase variance")
    print("\nObservations:")
    for bits in [2, 3, 4]:
        mse = next(r for r in results if r['bits'] == bits and not r['qjl'])
        qjl = next(r for r in results if r['bits'] == bits and r['qjl'])
        bias_improve = mse['relative_bias'] - qjl['relative_bias']  # positive = QJL better
        var_cost = qjl['ip_variance'] - mse['ip_variance']  # positive = QJL worse

        print(f"  {bits}-bit: Bias change = {bias_improve:+.2f}%, Variance change = {var_cost:+.2f}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()