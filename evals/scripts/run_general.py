"""Run all 5 general benchmarks with correct per-task num_fewshot settings.

Usage (local test):
    python evals/scripts/run_general.py \
        --model Qwen/Qwen3-0.6B \
        --limit 5 \
        --batch_size 1 \
        --output_path evals/results/test

Usage (full run):
    python evals/scripts/run_general.py \
        --model Qwen/Qwen3-0.6B \
        --batch_size auto \
        --device cuda \
        --output_path evals/results/qwen3-0.6b \
        --wandb
"""

import argparse
import json
import os
from pathlib import Path

import lm_eval
from lm_eval.loggers import WandbLogger

# Benchmark suite: task_name -> num_fewshot
GENERAL_BENCHMARKS = {
    "mmlu": 5,
    "arc_challenge": 25,
    "hellaswag": 10,
    "gsm8k": 4,
    "truthfulqa_mc2": 0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run general eval benchmarks")
    parser.add_argument("--model", type=str, required=True, help="HF model name or path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu, mps)")
    parser.add_argument("--batch_size", type=str, default="1", help="Batch size (int or 'auto')")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (testing only)")
    parser.add_argument("--output_path", type=str, default="evals/results", help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Log results to W&B")
    parser.add_argument("--wandb_project", type=str, default="umich-slm", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="umich-slm", help="W&B entity/team")
    parser.add_argument("--dtype", type=str, default="float32", help="Model dtype (float32 for CPU, bfloat16 for CUDA)")
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated subset of tasks to run (default: all 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Filter tasks if specified
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",")]
        benchmarks = {k: v for k, v in GENERAL_BENCHMARKS.items() if k in task_names}
        if not benchmarks:
            print(f"No matching tasks found. Available: {list(GENERAL_BENCHMARKS.keys())}")
            return
    else:
        benchmarks = GENERAL_BENCHMARKS

    model_slug = args.model.replace("/", "__")
    output_dir = Path(args.output_path) / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args = f"pretrained={args.model},dtype={args.dtype}"

    # Determine batch size
    batch_size = args.batch_size if args.batch_size == "auto" else int(args.batch_size)

    print(f"Model: {args.model}")
    print(f"Tasks: {list(benchmarks.keys())}")
    print(f"Output: {output_dir}")
    if args.limit:
        print(f"WARNING: --limit {args.limit} set, results are NOT meaningful")
    print()

    all_results = {}

    for task_name, num_fewshot in benchmarks.items():
        print(f"{'=' * 60}")
        print(f"Running {task_name} ({num_fewshot}-shot)")
        print(f"{'=' * 60}")

        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=args.device,
            limit=args.limit,
            log_samples=True,
        )

        # Save per-task results
        task_output = output_dir / f"{task_name}.json"
        # Remove non-serializable fields
        results_serializable = results.copy()
        results_serializable.pop("samples", None)
        with open(task_output, "w") as f:
            json.dump(results_serializable, f, indent=2, default=str)

        all_results[task_name] = results["results"]

        # Print task summary
        if task_name in results["results"]:
            metrics = results["results"][task_name]
            for metric, value in metrics.items():
                if not metric.endswith(",none") and not metric.startswith("alias"):
                    continue
                print(f"  {metric}: {value}")
        print()

    # Save combined summary
    summary = {
        "model": args.model,
        "benchmarks": {},
    }
    for task_name, task_results in all_results.items():
        if task_name in task_results:
            summary["benchmarks"][task_name] = {
                "num_fewshot": benchmarks[task_name],
                "metrics": task_results[task_name],
            }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # W&B logging
    if args.wandb:
        run_name = f"{model_slug}-general"
        wandb_logger = WandbLogger(
            wandb_args={
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "name": run_name,
                "job_type": "eval",
            }
        )
        # Log the last results set (contains all task info if run together)
        wandb_logger.post_init(results)
        wandb_logger.log_eval_result()
        wandb_logger.log_eval_samples(results["samples"])
        print(f"Results logged to W&B: {args.wandb_project}/{run_name}")

    # Print final summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Task':<20} {'n-shot':>6}  {'Metric':<12} {'Value':>8}")
    print(f"{'-' * 50}")

    primary_metrics = {
        "mmlu": "acc,none",
        "arc_challenge": "acc_norm,none",
        "hellaswag": "acc_norm,none",
        "gsm8k": "exact_match,flexible-extract",
        "truthfulqa_mc2": "acc,none",
    }

    for task_name in benchmarks:
        if task_name in all_results and task_name in all_results[task_name]:
            metrics = all_results[task_name][task_name]
            metric_key = primary_metrics.get(task_name, "acc,none")
            value = metrics.get(metric_key, "N/A")
            if isinstance(value, float):
                value = f"{value:.4f}"
            print(f"{task_name:<20} {benchmarks[task_name]:>6}  {metric_key:<12} {value:>8}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
