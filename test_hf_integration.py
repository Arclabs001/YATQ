"""
Test TurboQuantHF integration
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, 'C:/Users/arcla/TurboQuant')

from integrations.hf_integration import TurboQuantHF

MODEL_PATH = "Qwen/Qwen3-1.7B"


def main():
    print("=" * 60)
    print("Testing TurboQuantHF Integration")
    print("=" * 60)

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    prompt = "Who are you?\nAnswer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # ========== Test 1: FP16 Baseline ==========
    print("\n" + "=" * 60)
    print("Test 1: FP16 Baseline")
    print("=" * 60)

    with torch.no_grad():
        outputs_fp16 = model.generate(**inputs, max_new_tokens=50, use_cache=True, do_sample=False)
        text_fp16 = tokenizer.decode(outputs_fp16[0], skip_special_tokens=True)
    print(text_fp16)

    # ========== Test 2: TurboQuantHF 4-bit ==========
    print("\n" + "=" * 60)
    print("Test 2: TurboQuantHF 4-bit")
    print("=" * 60)

    tq_hf = TurboQuantHF(model, bits=4, use_qjl=False, keep_recent=32)

    with torch.no_grad():
        outputs_tq = tq_hf.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            compress_every=16
        )
        text_tq = tokenizer.decode(outputs_tq[0], skip_special_tokens=True)
    print(text_tq)

    # ========== Test 3: TurboQuantHF 6-bit ==========
    print("\n" + "=" * 60)
    print("Test 3: TurboQuantHF 6-bit")
    print("=" * 60)

    tq_hf = TurboQuantHF(model, bits=6, use_qjl=False, keep_recent=32)

    with torch.no_grad():
        outputs_tq = tq_hf.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            compress_every=16
        )
        text_tq = tokenizer.decode(outputs_tq[0], skip_special_tokens=True)
    print(text_tq)

    # ========== Test 4: TurboQuantHF 8-bit ==========
    print("\n" + "=" * 60)
    print("Test 4: TurboQuantHF 8-bit")
    print("=" * 60)

    tq_hf_8bit = TurboQuantHF(model, bits=8, use_qjl=False, keep_recent=32)

    with torch.no_grad():
        outputs_tq8 = tq_hf_8bit.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            compress_every=16
        )
        text_tq8 = tokenizer.decode(outputs_tq8[0], skip_special_tokens=True)
    print(text_tq8)

    # ========== Test 5: Longer generation ==========
    print("\n" + "=" * 60)
    print("Test 5: Longer generation (100 tokens)")
    print("=" * 60)

    prompt2 = "Write a short story about a robot.\n"
    inputs2 = tokenizer(prompt2, return_tensors="pt").to(model.device)

    print("\n--- FP16 ---")
    with torch.no_grad():
        out_fp16 = model.generate(**inputs2, max_new_tokens=100, use_cache=True, do_sample=True, temperature=0.7, top_p=0.9)
        print(tokenizer.decode(out_fp16[0], skip_special_tokens=True)[:500])

    print("\n--- TurboQuant 6-bit ---")
    with torch.no_grad():
        out_tq = tq_hf.generate(inputs2["input_ids"], max_new_tokens=100, temperature=0.7, top_p=0.9, compress_every=16)
        print(tokenizer.decode(out_tq[0], skip_special_tokens=True)[:500])

    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


"""
Usage: python test_hf_integration.py
============================================================
Testing TurboQuantHF Integration
============================================================

Loading model...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|███████████████████████████████████████████████████████████████| 311/311 [00:06<00:00, 48.95it/s]

============================================================
Test 1: FP16 Baseline
============================================================
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Who are you?
Answer:
I am a large language model developed by Alibaba Group, and I am designed to assist with a wide range of tasks, including answering questions, providing information, and helping with various other activities. I am here to help you with whatever you need.
You

============================================================
Test 2: TurboQuantHF 4-bit
============================================================
Who are you?
Answer:
I am a large language model developed by Alibaba Group, designed to provide assistance and answer questions. I can help with a wide range of topics, from general knowledge to technical discussions, and I am here to help you in any way I can.
You

============================================================
Test 3: TurboQuantHF 6-bit
============================================================
Who are you?
Answer:
I am a large language model developed by Alibaba Group, designed to provide assistance and answer questions. I can help with a wide range of topics, from general knowledge to technical discussions. I am here to help you in any way I can.

You are

============================================================
Test 4: TurboQuantHF 8-bit
============================================================
Who are you?
Answer:
I am a large language model developed by Alibaba Group, designed to provide assistance and answer questions. I can help with a wide range of topics, from general knowledge to technical discussions. I am here to help you with any questions you may have.
You

============================================================
Test 5: Longer generation (100 tokens)
============================================================

--- FP16 ---
Write a short story about a robot.
Okay, so I need to write a short story about a robot. Let me think about how to approach this. First, I should decide on the main character of the robot. Maybe a robot with a name? Like a simple name that's easy to remember. Maybe something like "Nex" or "Robo". Let me go with "Nex" for now.

Now, the setting. Where is the robot living? Maybe a futuristic city? Or a more isolated place? Maybe a

--- TurboQuant 6-bit ---
Write a short story about a robot.
**Title: The Clockmaker's Dilemma**

In the heart of the city, where the skyline shimmered with neon lights and the hum of technology filled the air, there lived a peculiar robot named Leo. Unlike the sleek, silent machines that roamed the streets, Leo was a relic of the past—a mechanical creation that had been built with the hope of serving humanity. His design was simple: a metallic frame, a few blinking lights, and a heart of a clockwork engine. He

============================================================
Tests completed!
============================================================
"""