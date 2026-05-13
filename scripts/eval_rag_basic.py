"""
eval_rag_basic.py
-----------------
Basic RAG evaluation of a causal LM using MS MARCO v1.1.

Dataset : MS MARCO v1.1 — Microsoft's large-scale reading comprehension
          benchmark with real Bing queries and human-annotated answers.

Metrics (all self-contained, no external judge LLM required):
  - Exact Match      : strict normalized string match vs gold answer
  - Token F1         : token overlap between prediction and gold
  - Faithfulness     : word overlap between prediction and context (grounding proxy)
  - Answer Relevancy : cosine similarity between question & answer embeddings
                       via sentence-transformers/all-MiniLM-L6-v2

The original version used ragas LLM-as-judge metrics which cause
TimeoutErrors when backed by a slow local HF pipeline. ragas expects fast
API-based models (e.g. GPT-4). This version replaces those with fast,
self-contained equivalents.

Hardware: A100 GPU (Colab)

Usage:
    pip install transformers torch datasets sentence-transformers

    # Verify code works with Qwen
    python eval_rag_basic.py --model Qwen/Qwen3-0.6B

    # Evaluate trained checkpoint
    python eval_rag_basic.py \\
        --model Qwen/Qwen3-0.6B \\
        --ckpt_path ./checkpoints/<trained_model>.pt \\
        --output rag_results.json
"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── CLI args ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG evaluation on MS MARCO v1.1.",
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
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of MS MARCO validation samples to evaluate.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_input_length", type=int, default=768)
    parser.add_argument(
        "--output", type=str, default="rag_results.json",
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


# ── Metrics ───────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred.split("\n")[0]) == normalize(gold))

def token_f1(pred: str, gold: str) -> float:
    p_tok = normalize(pred).split()
    g_tok = normalize(gold).split()
    common = sum((Counter(p_tok) & Counter(g_tok)).values())
    if common == 0:
        return 0.0
    precision = common / len(p_tok) if p_tok else 0
    recall    = common / len(g_tok) if g_tok else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

def faithfulness_score(pred: str, context: str) -> float:
    """Fraction of prediction words that appear in the context."""
    ctx   = set(normalize(context).split())
    pred_ = set(normalize(pred).split())
    return len(ctx & pred_) / len(pred_) if pred_ else 0.0

def answer_relevancy_score(question: str, answer: str, embedder) -> float:
    """Cosine similarity between question and answer embeddings."""
    vecs = embedder.encode([question, answer], normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))


# ── Generation ────────────────────────────────────────────────────────────────
def generate_answer(question: str, context: str, model, tokenizer, args) -> str:
    prompt = (
        f"Answer the question using only the context below.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=args.max_input_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


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

    # Load embedding model for answer relevancy
    print("Loading sentence-transformers for answer relevancy ...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Load MS MARCO
    print("Loading MS MARCO v1.1 ...")
    raw = load_dataset("ms_marco", "v1.1", split="validation")

    def has_answer_and_passage(row):
        has_ans = len(row["answers"]) > 0 and row["answers"][0].strip() != ""
        has_ctx = any(p.strip() for p in row["passages"]["passage_text"])
        return has_ans and has_ctx

    raw = raw.filter(has_answer_and_passage)
    raw = raw.select(range(min(args.n_samples, len(raw))))
    print(f"Using {len(raw)} samples")

    # Evaluate
    print("Generating answers and scoring ...")
    t0 = time.perf_counter()
    samples = []

    for i, row in enumerate(raw):
        question = row["query"]
        context  = next((p for p in row["passages"]["passage_text"] if p.strip()), "")
        gold     = row["answers"][0]

        answer = generate_answer(question, context, model, tokenizer, args)

        samples.append({
            "question":         question,
            "answer":           answer[:300],
            "gold":             gold,
            "exact_match":      exact_match(answer, gold),
            "f1":               round(token_f1(answer, gold), 4),
            "faithfulness":     round(faithfulness_score(answer, context), 4),
            "answer_relevancy": round(answer_relevancy_score(question, answer, embedder), 4),
        })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(raw)} done")

    elapsed = time.perf_counter() - t0

    def avg(key): return round(sum(s[key] for s in samples) / len(samples), 4)

    summary = {
        "model":             args.model,
        "ckpt_path":         args.ckpt_path,
        "dataset":           "ms_marco v1.1",
        "n_samples":         len(samples),
        "exact_match":       avg("exact_match"),
        "f1":                avg("f1"),
        "faithfulness":      avg("faithfulness"),
        "answer_relevancy":  avg("answer_relevancy"),
        "generation_time_s": round(elapsed, 2),
        "device":            str(model.device),
        "per_sample":        samples,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Basic RAG Results ===")
    for k, v in summary.items():
        if k != "per_sample":
            print(f"  {k}: {v}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
