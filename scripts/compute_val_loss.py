"""Held-out validation loss / perplexity (Step 1 MVP — smoke test only).

What this script does *today*:
    Loads a model, runs a forward pass over a small hard-coded text sample,
    and prints overall cross-entropy loss + perplexity. The hard-coded text
    is only here so we can validate the plumbing (model loading, chunking,
    loss aggregation) end-to-end against a known pretrained model like
    Qwen2.5-0.5B **without depending on any external data or checkpoint**.

What this script will do *later* (Step 3 + 4 of the eval plan):
    Swap the hard-coded text for per-domain Arrow shards under
    `data/eval/ppl_v1/{web,code,math,math_reasoning,textbook}/` and report
    PPL per domain so G1 (data team) can read data-mix deltas.

Why a standalone script (vs. reusing run_eval.py):
    run_eval.py scores *downstream* tasks (MMLU, ARC, ...). During
    pretraining and at <1B tokens, downstream accuracy is near random —
    the signal we need is held-out loss / PPL vs. step. This script
    provides exactly that and nothing more.

Usage:
    # Smoke test with a pretrained HF model (no local ckpt needed):
    python scripts/compute_val_loss.py --model Qwen/Qwen2.5-0.5B --batch_size 2

    # On an HF checkpoint produced by pretrain_fineweb.py:
    python scripts/compute_val_loss.py \\
        --model ./output/my_run/checkpoint-5000 --batch_size 4

Expected smoke-test output on Qwen2.5-0.5B (CPU, fp32, max_length=256):
    PPL in ~15-40 range, loss ~2.7-3.7 nats/token. Exact numbers depend on
    max_length and batch_size, but a PPL in this range proves the pipeline
    is correct. NaN / inf / PPL > 1e4 indicates a bug.
"""

import argparse
import datetime
import json
import logging
from pathlib import Path

import torch

from eval_utils import load_model_and_tokenizer, resolve_device, score_texts

logger = logging.getLogger(__name__)


# Hard-coded mini eval set for Step 1. Three short English paragraphs, mixed
# domains, chosen so that total tokenized length comfortably exceeds the
# default max_length=256. Replace with per-domain Arrow shards in Step 3+4.
_MINI_TEXTS = [
    # web-like
    "The Great Lakes are a series of large interconnected freshwater lakes in "
    "the mid-east region of North America that connect to the Atlantic Ocean "
    "through the Saint Lawrence River. They consist of Lakes Superior, Michigan, "
    "Huron, Erie, and Ontario. Together they contain 21% of the world's surface "
    "fresh water by volume, and are bordered by the Canadian province of Ontario "
    "and the US states of Michigan, Wisconsin, Minnesota, Illinois, Indiana, Ohio, "
    "Pennsylvania, and New York. Shipping, fishing, and recreation have been "
    "central to the regional economy for more than two centuries, and the lakes "
    "also play a role in climate regulation for the surrounding areas.",
    # textbook-like
    "In a feed-forward neural network, each neuron in one layer is connected to "
    "every neuron in the next layer. Given an input vector x, the output of a "
    "single layer is computed as y = sigma(W x + b), where W is a learned weight "
    "matrix, b is a bias vector, and sigma is a non-linear activation function "
    "such as ReLU, sigmoid, or tanh. Stacking multiple such layers allows the "
    "network to approximate complex functions. Training is typically performed by "
    "minimizing a loss function over a dataset using stochastic gradient descent "
    "and the back-propagation algorithm, which computes gradients efficiently via "
    "the chain rule.",
    # code/math-like
    "def softmax(x):\n"
    "    # Numerically stable softmax along the last axis.\n"
    "    x = x - x.max(axis=-1, keepdims=True)\n"
    "    e = np.exp(x)\n"
    "    return e / e.sum(axis=-1, keepdims=True)\n\n"
    "# Cross-entropy between probabilities p and q (both sum to 1):\n"
    "#     H(p, q) = - sum_i p_i * log q_i\n"
    "# When p is a one-hot label vector, this reduces to -log q_{target},\n"
    "# which is the standard classification loss used throughout this repo.\n",
] * 4  # repeat 4x so we get enough tokens even at max_length=1024


def main():
    parser = argparse.ArgumentParser(
        description="Compute held-out loss / perplexity for an HF causal LM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B",
        help="HF model name or local checkpoint dir.",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="Optional raw .pt checkpoint to overlay on --model.",
    )
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="'auto' picks cuda if available.",
    )
    parser.add_argument(
        "--dtype", type=str, default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--output_file", type=str, default="./val_loss_results.jsonl",
        help="JSONL file; one record appended per invocation.",
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="Optional training step, passed through into the output record so "
             "downstream plotters can place the point on an x-axis.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)

    device = resolve_device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    logger.info("Device=%s dtype=%s", device, args.dtype)

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model,
        tokenizer_name_or_path=args.tokenizer,
        ckpt_path=args.ckpt_path,
        dtype=dtype,
        device=device,
    )

    stats = score_texts(
        model=model,
        tokenizer=tokenizer,
        texts=_MINI_TEXTS,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )

    nll_per_token = stats["sum_nll_nats"] / stats["total_pred_tokens"]
    ppl = float(torch.exp(torch.tensor(nll_per_token)).item())

    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model": args.model,
        "ckpt_path": args.ckpt_path,
        "step": args.step,
        "eval_set": "mini_builtin_v0",  # bumps to "ppl_v1" once real data is wired
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "n_chunks": stats["n_chunks"],
        "total_pred_tokens": stats["total_pred_tokens"],
        "nll_per_token": round(nll_per_token, 4),
        "ppl": round(ppl, 3),
    }

    print(json.dumps(record, indent=2))
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Appended result to %s", args.output_file)


if __name__ == "__main__":
    main()
