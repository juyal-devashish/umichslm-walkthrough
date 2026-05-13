"""Benchmark evaluation script for UMichSLM.

Evaluates a causal LM against standard NLP benchmarks using lm_eval and
appends results to a JSONL file for experiment tracking.

Supported benchmarks (via lm_eval):
    - MMLU          (5-shot)  — multi-subject knowledge, metric: acc,none
    - ARC-Challenge (0-shot)  — science reasoning,        metric: acc_norm,none
    - HellaSwag     (0-shot)  — commonsense completion,   metric: acc_norm,none
    - GSM8K         (4-shot)  — math word problems,       metric: exact_match,strict-match
    - TruthfulQA MC1/MC2 (0-shot) — factual accuracy,    metric: acc,none

Model loading (three modes):
    1. HF checkpoint dir  (--ckpt_path <dir with config.json>): loads full model directly.
    2. Raw .pt checkpoint (--ckpt_path <file> + --model <arch>): injects weights into arch.
    3. Pretrained only    (--model <hf name>): evaluates without any local checkpoint.

Output:
    - Results table printed to stdout.
    - One JSON record appended per run to --output_file (default: ./eval_results.jsonl).
    - All log messages written to stderr and optionally to --log_file.

Example usage:
    # Evaluate pretrained Qwen on full suite
    python scripts/run_eval.py --model Qwen/Qwen2.5-0.5B

    # Evaluate a trained HF checkpoint directory
    python scripts/run_eval.py --ckpt_path ./checkpoints/step-10000/

    # Evaluate a raw .pt checkpoint with specific tasks
    python scripts/run_eval.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --ckpt_path ./checkpoints/step-10000.pt \\
        --tasks hellaswag,mmlu:5,gsm8k:4 \\
        --output_file ./results/eval.jsonl \\
        --log_file ./results/eval.log
"""

import argparse
import json
from pathlib import Path
import logging
import datetime
import time
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM

from prepare_mix_dataset import setup_tokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_for_eval(args, model):
    """Load a raw PyTorch checkpoint into an already-initialized model."""
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Clean the state_dict in case of prefix from distributed training or FSDP
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict)
    model.eval()
    return model

def _primary_metric(results: dict, metric_priority: list[str]) -> tuple[str, float]:
    """Return (metric, value) for the best available metric."""
    for key in metric_priority:
        if key in results:
            return key, results[key]
    return "unknown", 0.0


def print_res_table(results: dict, metric_priority: list[str]) -> None:
    report_results = {}

    task_results = results.get("results", {})
    if not task_results:
        return

    for t in list(task_results.keys()):
        res = task_results.get(t, {})
        if res:
            _, val = _primary_metric(res, metric_priority)
            report_results[t] = round(val, 4)

    print(report_results)
    return report_results


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    seed,
    batch_size,
    max_length,
    metric_priority: list[str] = ["acc,none", "acc_norm,none", "exact_match,strict-match"],
    tasks: list | None = None,
    output_path: str | None = None,
):
    """Run benchmark evaluation and optionally append results to a JSONL file.

    Args:
        model: HuggingFace causal LM (already loaded, in eval mode).
        tokenizer: Corresponding tokenizer.
        seed: Random seed passed to lm_eval.
        batch_size: Inference batch size.
        max_length: Maximum sequence length.
        metric_priority: Ordered list of metric keys to display.
        tasks: List of (task_name, num_fewshot) tuples.  Defaults to the
               standard benchmark suite when None.
        output_path: If set, one JSON record per eval run is *appended* to
                     this file in JSONL format.  Existing lines are preserved.

    Returns:
        dict mapping task name → lm_eval metrics dict.
    """
    # Wrap the pre-loaded model for lm_eval
    lm_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, max_length=max_length)

    # Set up evaluation task configs
    if tasks is None:
        task_configs = [
            ("mmlu", 5),
            ("arc_challenge", 0),
            ("hellaswag", 0),
            ("gsm8k", 4), 
            ("truthfulqa_mc1", 0),
            ("truthfulqa_mc2", 0),
        ]
    else:
        task_configs = tasks

    # Start the eval
    start_eval_time = time.time()
    eval_results = {}

    for task, fewshot in task_configs:
        print(f"\nRunning {task} ({fewshot}-shot) ...")
        logger.info("Running %s (%d-shot) ...", task, fewshot)

        res = lm_eval.simple_evaluate(
            model=lm_model,
            tasks=[task],
            num_fewshot=fewshot,
            random_seed=seed,
            limit=50,
        )

        eval_results[task] = res["results"][task]
        metric, val = _primary_metric(eval_results[task], metric_priority)
        logger.info("  %s  %s = %.4f", task, metric, val)

    elapsed = time.time() - start_eval_time
    print(f"Ran eval in: {elapsed:.1f} seconds")
    logger.info("Eval finished in %.1f seconds", elapsed)

    print_res_table(results={"results": eval_results}, metric_priority=metric_priority)

    # Persist results to JSONL (one record per run, append mode)
    if output_path is not None:
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "elapsed_secs": round(elapsed, 2),
            "results": eval_results,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"Results appended to {output_path}")
        logger.info("Results appended to %s", output_path)

    return eval_results


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a causal LM on standard benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name or local path (defines architecture and weights).",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="Path to a raw PyTorch checkpoint (.pt). "
             "If set, these weights override the pretrained weights from --model.",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Tokenizer name or path. Defaults to --model.",
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated benchmark names, e.g. 'hellaswag,mmlu,gsm8k'. "
             "Defaults to the full suite (mmlu:5, arc_challenge:0, hellaswag:0, gsm8k:4).",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_file", type=str, default="./eval_results.jsonl",
        help="JSONL file to append results to. Each call appends one record.",
    )
    parser.add_argument(
        "--log_file", type=str, default=None,
        help="Optional log file path. Logs are always printed to stderr; "
             "this additionally writes them to a file.",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    args = parser.parse_args()

    # Configure logging — always to stderr, optionally also to a file
    handlers = [logging.StreamHandler()]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    # Quiet down noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("lm_eval").setLevel(logging.WARNING)

    # Load tokenizer
    tokenizer_name = args.tokenizer or args.model
    print(f"Loading tokenizer: {tokenizer_name}")
    logger.info("Loading tokenizer: %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model or checkpoints
    print(f"Loading model: {args.model}")
    logger.info("Loading model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    if args.ckpt_path is not None:
        print(f"Loading checkpoint: {args.ckpt_path}")
        logger.info("Loading checkpoint: %s", args.ckpt_path)
        model = load_model_for_eval(args, model)
    else:
        model.eval()

    # Parse tasks — accept "name" or "name:fewshot" tokens
    task_configs = None
    if args.tasks is not None:
        default_fewshot = {"mmlu": 5, "gsm8k": 4}
        task_configs = []
        for token in args.tasks.split(","):
            token = token.strip()
            if ":" in token:
                name, shots = token.split(":", 1)
                task_configs.append((name.strip(), int(shots)))
            else:
                task_configs.append((token, default_fewshot.get(token, 0)))

    evaluate(
        model=model,
        tokenizer=tokenizer,
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
        tasks=task_configs,
        output_path=args.output_file,
    )


if __name__ == "__main__":
    main()