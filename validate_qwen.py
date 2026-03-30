"""
TurboQuant KV Cache Validation - Comprehensive Metrics with PPL

Tests:
1. Random Rotation (qwen3_integration)
2. WHT (qwen3_wht_integration)
"""

import torch
import torch.nn.functional as F
import sys
import math

sys.path.insert(0, 'C:/Users/arcla/YATQ')

from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import TurboQuantProd, TurboQuantMSE
from turboquant_wht import TurboQuantWHT
from integrations.qwen3_integration import Qwen3ForwardWithTurboQuant, ChunkedKVCacheQJL
from integrations.qwen3_wht_integration import Qwen3ForwardWithWHT, WHTKVCache

MODEL_PATH = "C:/Users/arcla/models/qwen3-1.7b"
CONTEXT_FILE = "test_context.txt"


def compute_ppl_with_forward(wrapper, tokenizer, text, device="cuda"):
    """Compute perplexity using wrapper's forward pass."""
    wrapper.model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        logits = wrapper.forward(input_ids)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))
    ppl = torch.exp(loss).item()

    stats = wrapper.get_compression_stats()
    return ppl, loss.item(), input_ids.shape[1], stats


def compute_ppl_baseline(model, tokenizer, text, device="cuda"):
    """Compute baseline FP16 perplexity."""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))
    ppl = torch.exp(loss).item()

    return ppl, loss.item(), input_ids.shape[1]


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

    ip_bias = (ip_est.mean() - ip_true.mean()).item()
    relative_bias = ip_bias / ip_true.mean().abs().item() * 100

    ip_variance = ip_error.var().item()
    ip_std = ip_error.std().item()

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


def test_config_wht(cache, num_layers, num_kv_heads, head_dim, bits, use_qjl):
    """Test WHT configuration with comprehensive metrics."""
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

        # Create WHT quantizer
        wht_q = TurboQuantWHT(dim=D, bits=bits)

        # Calculate storage
        n_vecs = B * H * S
        k_bits = n_vecs * D * bits + n_vecs * 16  # indices + norms
        if use_qjl:
            k_bits += n_vecs * D + n_vecs * 16  # qjl_signs + residual_norm
        v_bits = n_vecs * D * bits + n_vecs * 16
        total_compressed_bits += k_bits + v_bits

        # Test attention scores
        query = keys[:, :, -1:, :]
        real_scores = torch.matmul(query.float(), keys.float().transpose(-2, -1)).squeeze(-2)

        tq_scores = torch.zeros(B, H, S, device=keys.device)

        for h in range(H):
            q_h = query[0, h, 0, :].unsqueeze(0).float()
            k_h = keys[0, h, :, :].float()

            # WHT quantization
            k_data = wht_q.quantize_key(k_h, use_qjl=use_qjl)

            if use_qjl:
                # Use QJL estimator
                scores_h = wht_q.compute_attention_scores(q_h, k_data, use_qjl=True, scale=1.0).squeeze(0)
            else:
                # MSE reconstruction
                k_recon = wht_q.reconstruct_key(k_data)
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

            # KL Divergence
            p = F.softmax(rs, dim=0)
            q = F.softmax(ts, dim=0)
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

    ip_bias = (ip_est.mean() - ip_true.mean()).item()
    relative_bias = ip_bias / ip_true.mean().abs().item() * 100

    ip_variance = ip_error.var().item()
    ip_std = ip_error.std().item()

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
    print("=" * 100, flush=True)
    print("TurboQuant KV Cache Validation - Comprehensive Metrics with PPL", flush=True)
    print("=" * 100, flush=True)

    print("\nLoading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    num_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)

    print(f"Layers: {num_layers}, KV heads: {num_kv_heads}, Head dim: {head_dim}", flush=True)

    with open(CONTEXT_FILE, 'r', encoding='utf-8') as f:
        context_text = f.read()

    # Use a shorter text for PPL measurement (max 2048 tokens)
    ppl_text = context_text[:8000]  # Truncate for PPL test

    # Find the secret code in context
    import re
    code_match = re.search(r'secret access code is[:\s]+([A-Z0-9-]+)', context_text, re.IGNORECASE)
    secret_code = None
    if code_match:
        secret_code = code_match.group(1)
        print(f"\nSecret code found in context: {secret_code}", flush=True)
    else:
        print("\nWarning: Could not find secret code in context file", flush=True)

    prompt = f"{context_text}\n\nQuestion: What is the secret access code?\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8000).to(model.device)
    print(f"Sequence length: {inputs['input_ids'].shape[1]} tokens", flush=True)

    print("Running forward pass for cache extraction...", flush=True)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    cache = outputs.past_key_values

    # Test if model can retrieve the secret code (baseline)
    print("\n" + "=" * 100, flush=True)
    print("NEEDLE RETRIEVAL TEST (Baseline - FP16)", flush=True)
    print("=" * 100, flush=True)
    print("Generating response...", flush=True)
    with torch.no_grad():
        gen_outputs = model.generate(**inputs, max_new_tokens=20, use_cache=True, do_sample=False)
        generated_text = tokenizer.decode(gen_outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Model answer: {generated_text.strip()}", flush=True)
    if code_match and secret_code in generated_text:
        print(f"[SUCCESS] Model correctly retrieved the secret code!", flush=True)
    elif code_match:
        print(f"[FAILED] Expected '{secret_code}' but got '{generated_text.strip()}'", flush=True)

    # ==========================================================================
    # Baseline PPL
    # ==========================================================================
    print("\n" + "=" * 100, flush=True)
    print("BASELINE FP16 PPL", flush=True)
    print("=" * 100, flush=True)
    print("Computing baseline perplexity...", flush=True)

    ppl_fp16, loss_fp16, seq_len = compute_ppl_baseline(model, tokenizer, ppl_text)
    print(f"Sequence length: {seq_len} tokens", flush=True)
    print(f"Loss: {loss_fp16:.4f}, PPL: {ppl_fp16:.4f}", flush=True)

    # ==========================================================================
    # Random Rotation PPL (qwen3_integration)
    # ==========================================================================
    print("\n" + "=" * 100, flush=True)
    print("RANDOM ROTATION PPL (qwen3_integration)", flush=True)
    print("=" * 100, flush=True)

    print(f"\n{'Config':<15} {'Loss':>10} {'PPL':>10} {'Δ PPL':>10} {'Ratio':>8}", flush=True)
    print("-" * 55, flush=True)

    rr_ppl_results = []
    for bits in [2, 3, 4, 6, 8]:
        for use_qjl in [False, True]:
            config_name = f"{bits}b+QJL" if use_qjl else f"{bits}b"
            print(f"[{config_name}] Running...", end=" ", flush=True)

            wrapper = Qwen3ForwardWithTurboQuant(model, bits=bits, use_qjl=use_qjl, keep_recent=0)
            wrapper.kv_cache = ChunkedKVCacheQJL(
                num_layers=wrapper.num_layers,
                head_dim=wrapper.head_dim,
                bits=wrapper.bits,
                keep_recent=wrapper.keep_recent
            )

            ppl, loss, seq, stats = compute_ppl_with_forward(wrapper, tokenizer, ppl_text)
            delta = ppl - ppl_fp16
            ratio = stats.get('ratio', 1.0)
            print(f"\r{config_name:<15} {loss:>10.4f} {ppl:>10.4f} {delta:>+10.4f} {ratio:>8.2f}x", flush=True)

            rr_ppl_results.append({'bits': bits, 'use_qjl': use_qjl, 'ppl': ppl, 'loss': loss})

    # ==========================================================================
    # WHT PPL (qwen3_wht_integration)
    # ==========================================================================
    print("\n" + "=" * 100, flush=True)
    print("WHT PPL (qwen3_wht_integration)", flush=True)
    print("=" * 100, flush=True)

    print(f"\n{'Config':<15} {'Loss':>10} {'PPL':>10} {'Δ PPL':>10} {'Ratio':>8}", flush=True)
    print("-" * 55, flush=True)

    wht_ppl_results = []
    for bits in [2, 3, 4, 6, 8]:
        for use_qjl in [False, True]:
            config_name = f"{bits}b+QJL" if use_qjl else f"{bits}b"
            print(f"[{config_name}] Running...", end=" ", flush=True)

            wrapper = Qwen3ForwardWithWHT(model, bits=bits, use_qjl=use_qjl, keep_recent=0)
            wrapper.kv_cache = WHTKVCache(
                num_layers=wrapper.num_layers,
                head_dim=wrapper.head_dim,
                bits=bits,
                use_qjl=use_qjl
            )

            ppl, loss, seq, stats = compute_ppl_with_forward(wrapper, tokenizer, ppl_text)
            delta = ppl - ppl_fp16
            ratio = stats.get('ratio', 1.0)
            print(f"\r{config_name:<15} {loss:>10.4f} {ppl:>10.4f} {delta:>+10.4f} {ratio:>8.2f}x", flush=True)

            wht_ppl_results.append({'bits': bits, 'use_qjl': use_qjl, 'ppl': ppl, 'loss': loss})

    # ==========================================================================
    # Attention Score Metrics (using extracted cache)
    # ==========================================================================
    print("\n" + "=" * 100, flush=True)
    print("ATTENTION SCORE METRICS - Random Rotation", flush=True)
    print("=" * 100, flush=True)

    print(f"\n{'Config':<18} {'Ratio':>6} {'CosSim':>8} {'Top1%':>6} {'Top5%':>6} {'KL-Div':>8} {'Bias%':>10} {'Variance':>12}", flush=True)
    print("-" * 90, flush=True)

    rr_att_results = []
    for bits in [2, 3, 4, 6, 8]:
        for key_qjl in [False, True]:
            label = f"K:{bits}b+QJL" if key_qjl else f"K:{bits}b"
            print(f"[{label}] Computing...", end=" ", flush=True)

            r = test_config(cache, num_layers, num_kv_heads, head_dim,
                           key_bits=bits, key_use_qjl=key_qjl,
                           val_bits=bits, val_use_qjl=False)

            print(f"\r{label:<18} {r['ratio']:>6.2f} {r['cos_sim']:>8.4f} {r['top1']:>6.1f} {r['top5']:>6.1f} {r['kl_div']:>8.4f} {r['relative_bias']:>+10.2f}% {r['ip_variance']:>12.2f}", flush=True)

            rr_att_results.append({'bits': bits, 'qjl': key_qjl, **r})

    print("\n" + "=" * 100, flush=True)
    print("ATTENTION SCORE METRICS - WHT", flush=True)
    print("=" * 100, flush=True)

    print(f"\n{'Config':<18} {'Ratio':>6} {'CosSim':>8} {'Top1%':>6} {'Top5%':>6} {'KL-Div':>8} {'Bias%':>10} {'Variance':>12}", flush=True)
    print("-" * 90, flush=True)

    wht_att_results = []
    for bits in [2, 3, 4, 6, 8]:
        for use_qjl in [False, True]:
            label = f"K:{bits}b+QJL" if use_qjl else f"K:{bits}b"
            print(f"[{label}] Computing...", end=" ", flush=True)

            r = test_config_wht(cache, num_layers, num_kv_heads, head_dim, bits=bits, use_qjl=use_qjl)

            print(f"\r{label:<18} {r['ratio']:>6.2f} {r['cos_sim']:>8.4f} {r['top1']:>6.1f} {r['top5']:>6.1f} {r['kl_div']:>8.4f} {r['relative_bias']:>+10.2f}% {r['ip_variance']:>12.2f}", flush=True)

            wht_att_results.append({'bits': bits, 'qjl': use_qjl, **r})

    # ==========================================================================
    # Attention Metrics Comparison Table
    # ==========================================================================
    print("\n" + "=" * 100, flush=True)
    print("ATTENTION METRICS COMPARISON (Random Rotation vs WHT)", flush=True)
    print("=" * 100, flush=True)

    print(f"\n{'Config':<12} {'Method':<10} {'CosSim':>8} {'Top1%':>7} {'Top5%':>7} {'KL-Div':>8} {'Bias%':>9} {'Variance':>12}", flush=True)
    print("-" * 85, flush=True)

    for bits in [2, 3, 4, 6, 8]:
        for use_qjl in [False, True]:
            rr = next(r for r in rr_att_results if r['bits'] == bits and r['qjl'] == use_qjl)
            wht = next(r for r in wht_att_results if r['bits'] == bits and r['qjl'] == use_qjl)

            label = f"{bits}b+QJL" if use_qjl else f"{bits}b"

            print(f"{label:<12} {'Random':<10} {rr['cos_sim']:>8.4f} {rr['top1']:>7.1f} {rr['top5']:>7.1f} {rr['kl_div']:>8.4f} {rr['relative_bias']:>+9.2f} {rr['ip_variance']:>12.2f}", flush=True)
            print(f"{'':<12} {'WHT':<10} {wht['cos_sim']:>8.4f} {wht['top1']:>7.1f} {wht['top5']:>7.1f} {wht['kl_div']:>8.4f} {wht['relative_bias']:>+9.2f} {wht['ip_variance']:>12.2f}", flush=True)
        print("-" * 85, flush=True)

    # ==========================================================================
    # PPL Comparison Table
    # ==========================================================================
    print("\n" + "=" * 100, flush=True)
    print("PPL COMPARISON SUMMARY", flush=True)
    print("=" * 100, flush=True)

    print(f"\n{'Config':<12} {'Random PPL':>12} {'WHT PPL':>12} {'Random Δ':>10} {'WHT Δ':>10}", flush=True)
    print("-" * 58, flush=True)

    for bits in [2, 3, 4, 6, 8]:
        rr_mse = next(r for r in rr_ppl_results if r['bits'] == bits and not r['use_qjl'])
        rr_qjl = next(r for r in rr_ppl_results if r['bits'] == bits and r['use_qjl'])
        wht_mse = next(r for r in wht_ppl_results if r['bits'] == bits and not r['use_qjl'])
        wht_qjl = next(r for r in wht_ppl_results if r['bits'] == bits and r['use_qjl'])

        print(f"{bits}b MSE     {rr_mse['ppl']:>12.2f} {wht_mse['ppl']:>12.2f} {rr_mse['ppl']-ppl_fp16:>+10.2f} {wht_mse['ppl']-ppl_fp16:>+10.2f}", flush=True)
        print(f"{bits}b QJL     {rr_qjl['ppl']:>12.2f} {wht_qjl['ppl']:>12.2f} {rr_qjl['ppl']-ppl_fp16:>+10.2f} {wht_qjl['ppl']-ppl_fp16:>+10.2f}", flush=True)
        print("-" * 58, flush=True)

    print(f"\nBaseline FP16 PPL: {ppl_fp16:.4f}", flush=True)




if __name__ == "__main__":
    main()