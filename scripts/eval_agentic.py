"""
eval_agentic.py
---------------
Agentic post-training evaluation of a causal LM covering:

  - Reasoning quality  : GSM8K — multi-step math word problems.
                         LLM-as-a-judge scoring (faithfulness, correctness,
                         multi-step reasoning).

  - Retrieval-Dependent Questions (RDQ proxy): HotpotQA (distractor setting)
                         — questions requiring multi-hop reasoning over
                         multiple documents. Scores: exact match, F1,
                         context grounding.

  - System metrics     : end-to-end latency, GPU memory usage.

Key design: context is injected directly for both tracks so bad retrieval
cannot pollute results, this isolates the model's own reasoning capability.

Hardware: A100 GPU (Goolge Colab Pro)

Usage:
    pip install transformers torch datasets psutil

    # Verify code works with Qwen
    python eval_agentic.py --model Qwen/Qwen3-0.6B

    # Evaluate trained checkpoint
    python eval_agentic.py \\
        --model Qwen/Qwen3-0.6B \\
        --ckpt_path ./checkpoints/<model_checkpoint>.pt \\
        --n_reasoning 50 \\
        --n_rdq 50 \\
        --output agentic_results.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import psutil
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── CLI args ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Agentic post-training evaluation (GSM8K + HotpotQA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="Optional path to a raw .pt checkpoint. Weights are injected into --model arch.",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Tokenizer name or path. Defaults to --model.",
    )
    parser.add_argument("--n_reasoning", type=int, default=50,
                        help="Number of GSM8K samples.")
    parser.add_argument("--n_rdq", type=int, default=50,
                        help="Number of HotpotQA samples.")
    parser.add_argument("--latency_repeats", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--output", type=str, default="agentic_results.json",
        help="Path to write JSON results.",
    )
    return parser.parse_args()


# ── Checkpoint loading ─────────────────────────
def load_checkpoint(ckpt_path: str, model):
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for k, v in state_dict.items():
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    model.load_state_dict(cleaned)
    model.eval()
    return model


# ── Generation ────────────────────────────────────────────────────────────────
def make_generate_fn(model, tokenizer, max_new_tokens):
    def generate(prompt: str, max_tokens: int = None) -> str:
        tokens = max_tokens or max_new_tokens
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=tokens, do_sample=False)
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
    return generate


# ── Reasoning eval — GSM8K ────────────────────────────────────────────────────
def extract_number(text: str):
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None

def llm_judge_reasoning(question, prediction, gold, generate):
    criteria = {
        "faithfulness": (
            "Does the answer stay strictly within what the question states "
            "with no invented facts? Answer YES or NO."
        ),
        "correctness": (
            f"The correct final answer is {gold}. "
            f"Is the prediction's final answer correct? Answer YES or NO."
        ),
        "multi_step": (
            "Does the answer show clear step-by-step reasoning with at least "
            "two explicit intermediate steps? Answer YES or NO."
        ),
    }
    scores = {}
    for key, criterion in criteria.items():
        prompt = (
            f"You are an expert evaluator.\n\n"
            f"Question: {question}\n"
            f"Prediction: {prediction}\n\n"
            f"Criterion: {criterion}"
        )
        response = generate(prompt, max_tokens=10).upper()
        scores[key] = 1 if "YES" in response else 0
    return scores

def run_reasoning_eval(gsm, n, generate):
    results = []
    for i, row in enumerate(gsm):
        question    = row["question"]
        gold_answer = row["answer"]
        gold_number = gold_answer.split("####")[-1].strip()

        prompt = (
            f"Solve this math problem step by step.\n\n"
            f"Problem: {question}\n\n"
            f"Solution:"
        )
        prediction = generate(prompt, max_tokens=200)

        pred_num    = extract_number(prediction)
        gold_num    = extract_number(gold_number)
        exact       = int(pred_num is not None and gold_num is not None and pred_num == gold_num)
        judge       = llm_judge_reasoning(question, prediction, gold_number, generate)

        results.append({
            "question":    question,
            "prediction":  prediction[:300],
            "gold":        gold_number,
            "exact_match": exact,
            **judge,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n} done")
    return results


# ── RDQ proxy — HotpotQA ─────────────────────────────────────────────────────
def f1_score(pred: str, gold: str) -> float:
    from collections import Counter
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def run_rdq_eval(hotpot, n, generate):
    results = []
    for i, row in enumerate(hotpot):
        question = row["question"]
        gold     = row["answer"]

        supporting_titles = row["supporting_facts"]["title"]
        context_passages  = [
            " ".join(row["context"]["sentences"][
                row["context"]["title"].index(t)
            ])
            for t in supporting_titles
            if t in row["context"]["title"]
        ]
        context = " ".join(context_passages)[:1500]

        prompt = (
            f"Answer the question using the context below. "
            f"Reason step by step across both passages.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        prediction = generate(prompt, max_tokens=80)

        em      = int(gold.lower().strip() == prediction.lower().strip().split("\n")[0])
        f1      = f1_score(prediction, gold)
        ctx_words  = set(context.lower().split())
        pred_words = set(prediction.lower().split())
        grounded   = int(len(ctx_words & pred_words) > 3)

        results.append({
            "question":    question,
            "prediction":  prediction[:200],
            "gold":        gold,
            "exact_match": em,
            "f1":          round(f1, 4),
            "grounded":    grounded,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n} done")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Load tokenizer
    tokenizer_name = args.tokenizer or args.model
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.ckpt_path:
        model = load_checkpoint(args.ckpt_path, model)
    else:
        model.eval()
    print(f"Model on device: {model.device}")

    generate = make_generate_fn(model, tokenizer, args.max_new_tokens)

    # ── Reasoning (GSM8K) ──
    print(f"\n[1/3] Loading GSM8K ...")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm = gsm.select(range(min(args.n_reasoning, len(gsm))))
    print("Running reasoning evaluation ...")
    reasoning_results = run_reasoning_eval(gsm, args.n_reasoning, generate)

    def avg(lst, key): return round(sum(r[key] for r in lst) / len(lst), 4)

    reasoning_summary = {
        "exact_match":  avg(reasoning_results, "exact_match"),
        "faithfulness": avg(reasoning_results, "faithfulness"),
        "correctness":  avg(reasoning_results, "correctness"),
        "multi_step":   avg(reasoning_results, "multi_step"),
    }
    print(f"Reasoning summary: {reasoning_summary}")

    # ── RDQ / HotpotQA ──
    print(f"\n[2/3] Loading HotpotQA (distractor) ...")
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    hotpot = hotpot.select(range(min(args.n_rdq, len(hotpot))))
    print("Running RDQ (HotpotQA) evaluation ...")
    rdq_results = run_rdq_eval(hotpot, args.n_rdq, generate)

    rdq_summary = {
        "exact_match": avg(rdq_results, "exact_match"),
        "f1":          avg(rdq_results, "f1"),
        "grounded":    avg(rdq_results, "grounded"),
        "n_evaluated": len(rdq_results),
    }
    print(f"RDQ (HotpotQA) summary: {rdq_summary}")

    # ── System metrics ──
    print(f"\n[3/3] Measuring latency over {args.latency_repeats} runs ...")
    BENCH_PROMPT = (
        "Solve step by step: A train travels 60 mph for 2.5 hours, "
        "then 80 mph for 1.5 hours. What is the total distance?"
    )
    latencies  = []
    proc       = psutil.Process()
    mem_before = proc.memory_info().rss / 1e6
    for _ in range(args.latency_repeats):
        t0 = time.perf_counter()
        generate(BENCH_PROMPT, max_tokens=150)
        latencies.append(time.perf_counter() - t0)
    mem_after = proc.memory_info().rss / 1e6

    system_summary = {
        "avg_latency_s":        round(sum(latencies) / len(latencies), 3),
        "min_latency_s":        round(min(latencies), 3),
        "max_latency_s":        round(max(latencies), 3),
        "cpu_memory_before_mb": round(mem_before, 1),
        "cpu_memory_after_mb":  round(mem_after, 1),
        "gpu_allocated_mb":     round(torch.cuda.memory_allocated() / 1e6, 1),
        "gpu_reserved_mb":      round(torch.cuda.memory_reserved() / 1e6, 1),
        "device":               str(model.device),
        "dtype":                "bfloat16",
    }
    print(f"System summary: {system_summary}")

    # ── Save ──
    output = {
        "model":        args.model,
        "ckpt_path":    args.ckpt_path,
        "n_reasoning":  args.n_reasoning,
        "n_rdq":        args.n_rdq,
        "reasoning":    reasoning_summary,
        "rdq_hotpotqa": rdq_summary,
        "system":       system_summary,
        "per_sample": {
            "reasoning": reasoning_results,
            "rdq":       rdq_results,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== Agentic Eval Results ===")
    print(f"  Reasoning    : {reasoning_summary}")
    print(f"  RDQ/HotpotQA : {rdq_summary}")
    print(f"  System       : {system_summary}")
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
