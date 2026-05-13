#!/usr/bin/env python3
"""Single-GPU smoke comparison: torchtitan vs LLaMA-Factory.

Drives the same model / dataset / hyperparameters through both training paths,
then emits a side-by-side report. Designed to answer **two** questions:

  1. *Correctness*: do the two paths produce equivalent loss trajectories?
     A divergence ≥ a few % at matched steps is a smoking gun for a config or
     wrapper bug (especially after architecture surgery like the stacking work).
  2. *Per-step trainer overhead*: HF Trainer's bookkeeping (callbacks, dataclass
     conversions, eval loop) vs. torchtitan's bare loop. The single-GPU number
     compounds at scale, so even a 1-GPU delta is informative.

Caveats: single-GPU MFU **does not** extrapolate to H100, and DDP scaling
overhead isn't tested here. See the conversation transcript or
``docs/throughput_comparison_v1.md`` for the multi-GPU follow-up.

Usage::

    python scripts/smoke_compare_frameworks.py \\
        --titan-toml configs/torchtitan/umichslm_pretrain_baseline.toml \\
        --lf-yaml   configs/llama_factory/umichslm_pretrain.yaml \\
        --output-dir ./outputs/smoke_compare \\
        --max-steps 50 \\
        --per-device-batch-size 4 \\
        --gradient-accumulation-steps 1 \\
        --seed 42

The script writes patched copies of the two configs into ``--output-dir``,
runs each framework as a subprocess (so a crash in one doesn't take the other
down), parses the resulting metrics, and prints a markdown table to stdout.
Pass ``--skip-titan`` or ``--skip-lf`` to run only one side.

Requires both frameworks installed: ``llamafactory-cli`` on PATH, and the
torchtitan launcher importable via ``torchtitan_train/launch.py`` (already in
this repo).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # py>=3.11
    import tomllib as _toml_read
except ModuleNotFoundError:  # pragma: no cover - py<3.11
    import tomli as _toml_read  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------- config patchers
def _toml_serialize(d: Dict[str, Any]) -> str:
    """Tiny TOML emitter for the dialect this repo's TitanConfig uses.

    We avoid taking a hard dependency on tomli-w; the schema is small enough
    to render by hand (one section header per top-level key, simple values).
    """

    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return f'"{v}"'
        if isinstance(v, list):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        if v is None:
            return '""'
        if isinstance(v, dict):
            return "{ " + ", ".join(f"{k} = {fmt(val)}" for k, val in v.items()) + " }"
        raise TypeError(f"Don't know how to TOML-serialize {type(v).__name__}: {v!r}")

    out = []
    for section, body in d.items():
        if not isinstance(body, dict):
            out.append(f"{section} = {fmt(body)}")
            continue
        out.append(f"\n[{section}]")
        for k, v in body.items():
            out.append(f"{k} = {fmt(v)}")
    return "\n".join(out) + "\n"


def patch_titan_toml(
    src_toml: Path,
    dst_toml: Path,
    *,
    output_dir: Path,
    run_name: str,
    max_steps: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    seed: int,
    tokenizer_path: Optional[str] = None,
    init_from: Optional[str] = None,
    dataset_dir: Optional[str] = None,
) -> None:
    with open(src_toml, "rb") as f:
        cfg = _toml_read.load(f)

    cfg.setdefault("job", {})
    cfg["job"]["output_dir"] = str(output_dir)
    cfg["job"]["run_name"] = run_name
    cfg["job"]["seed"] = seed

    cfg.setdefault("training", {})
    cfg["training"]["max_steps"] = max_steps
    cfg["training"]["per_device_batch_size"] = per_device_batch_size
    cfg["training"]["gradient_accumulation_steps"] = gradient_accumulation_steps

    cfg.setdefault("optimizer", {})
    cfg["optimizer"]["lr"] = learning_rate

    cfg.setdefault("checkpoint", {})
    # Keep checkpoints out of the smoke comparison's way; we only need metrics.
    cfg["checkpoint"]["save_every"] = max(max_steps, 1)
    cfg["checkpoint"]["keep_last"] = 1

    cfg.setdefault("logging", {})
    cfg["logging"]["log_every"] = max(1, max_steps // 25)
    cfg["logging"]["report_to"] = "none"  # Don't fight wandb in a smoke test.

    # Tokenizer / init_from / dataset overrides. Titan requires either
    # ``[data].tokenizer_path`` or ``[model].init_from`` (a checkpoint that
    # bundles a tokenizer). The CLI flags fill in whichever the user supplies
    # so the source config doesn't need to be edited by hand.
    cfg.setdefault("data", {})
    if tokenizer_path:
        cfg["data"]["tokenizer_path"] = str(tokenizer_path)
    if dataset_dir:
        cfg["data"]["dataset_dir"] = str(dataset_dir)
    cfg.setdefault("model", {})
    if init_from:
        cfg["model"]["init_from"] = str(init_from)

    # Disable any in-process staging — this script does single-stage runs only.
    if "staging" in cfg:
        cfg["staging"]["mode"] = "external"

    dst_toml.parent.mkdir(parents=True, exist_ok=True)
    dst_toml.write_text(_toml_serialize(cfg))


def patch_lf_yaml(
    src_yaml: Path,
    dst_yaml: Path,
    *,
    output_dir: Path,
    run_name: str,
    max_steps: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    seed: int,
    model_name_or_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
) -> None:
    """Patch an LF pretraining YAML with the smoke-test hyperparams.

    Reads/writes line-by-line to keep YAML comments and ordering intact,
    overwriting only the keys we care about. Unknown keys pass through.
    """
    src_lines = src_yaml.read_text().splitlines()
    overrides = {
        "output_dir": str(output_dir),
        "run_name": run_name,
        "max_steps": max_steps,
        "per_device_train_batch_size": per_device_batch_size,
        "per_device_eval_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "logging_steps": max(1, max_steps // 25),
        "save_steps": max(max_steps, 1),
        "save_total_limit": 1,
        "do_eval": False,
        "report_to": "none",
        "include_tokens_per_second": True,  # Required for the speed comparison.
        # Drop num_train_epochs (max_steps wins); LF will ignore both being set
        # but explicit -1 makes this clearer.
        "num_train_epochs": -1,
    }
    if model_name_or_path:
        overrides["model_name_or_path"] = model_name_or_path
    if dataset_dir:
        overrides["dataset_dir"] = dataset_dir
    seen_keys = set()
    new_lines: List[str] = []
    for line in src_lines:
        stripped = line.lstrip()
        # Detect "key:" or "key: value" at the top level (no leading dash, no nested indent).
        if (
            stripped
            and not stripped.startswith("#")
            and ":" in stripped
            and not line.startswith(" ")
            and not line.startswith("-")
        ):
            key = stripped.split(":", 1)[0].strip()
            if key in overrides:
                value = overrides[key]
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                else:
                    value_str = repr(value) if isinstance(value, (int, float)) else f"{value}"
                new_lines.append(f"{key}: {value_str}")
                seen_keys.add(key)
                continue
        new_lines.append(line)
    # Append any overrides that weren't already in the file.
    if seen_keys != set(overrides):
        new_lines.append("")
        new_lines.append("### Smoke-comparison overrides ###")
        for k, v in overrides.items():
            if k in seen_keys:
                continue
            value_str = (
                "true" if v is True
                else "false" if v is False
                else repr(v) if isinstance(v, (int, float))
                else f"{v}"
            )
            new_lines.append(f"{k}: {value_str}")

    dst_yaml.parent.mkdir(parents=True, exist_ok=True)
    dst_yaml.write_text("\n".join(new_lines) + "\n")


# ---------------------------------------------------------- subprocess runs
def run_titan(toml_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    """Invoke the torchtitan launcher as a subprocess; return exit code."""
    cmd = [sys.executable, str(REPO_ROOT / "torchtitan_train" / "launch.py"), str(toml_path)]
    print(f"\n[smoke] running torchtitan: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=REPO_ROOT, env=env)


def run_lf(yaml_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    """Invoke LF via the project's wrapper script; return exit code.

    Uses ``scripts/llamafactory_train.py train <yaml>`` so the UMichSLM
    architecture is registered before LF spawns its workers.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "llamafactory_train.py"),
        "train",
        str(yaml_path),
    ]
    print(f"\n[smoke] running LLaMA-Factory: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=REPO_ROOT, env=env)


# ---------------------------------------------------------- metrics parsers
def parse_titan_metrics(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Parse titan ``metrics.jsonl`` and return aggregate stats."""
    candidate = output_dir / "metrics.jsonl"
    if not candidate.exists():
        # The titan trainer also writes metrics inside the checkpoint dir.
        candidates = list(output_dir.rglob("metrics.jsonl"))
        if not candidates:
            return None
        candidate = candidates[0]
    rows = []
    with open(candidate) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return None
    losses = [r.get("loss") for r in rows if r.get("loss") is not None]
    step_times = [r.get("step_time_ms") for r in rows if r.get("step_time_ms") is not None]
    tokens_per_sec = [
        r.get("tokens_per_sec") for r in rows if r.get("tokens_per_sec") is not None
    ]
    mfu = [r.get("mfu_pct") for r in rows if r.get("mfu_pct") is not None]
    return {
        "framework": "torchtitan",
        "metrics_path": str(candidate),
        "n_rows": len(rows),
        "loss_first": losses[0] if losses else None,
        "loss_mid": losses[len(losses) // 2] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "step_ms_mean": (sum(step_times[1:]) / max(len(step_times) - 1, 1))
        if step_times else None,
        "tokens_per_sec_mean": (sum(tokens_per_sec[1:]) / max(len(tokens_per_sec) - 1, 1))
        if tokens_per_sec else None,
        "mfu_pct_mean": (sum(mfu[1:]) / max(len(mfu) - 1, 1)) if mfu else None,
    }


def parse_lf_metrics(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Parse LF's ``trainer_state.json`` and return aggregate stats."""
    candidate = output_dir / "trainer_state.json"
    if not candidate.exists():
        candidates = list(output_dir.rglob("trainer_state.json"))
        if not candidates:
            return None
        # Use the deepest checkpoint's trainer_state.
        candidates.sort(key=lambda p: int(
            p.parent.name.split("-")[-1]
        ) if p.parent.name.startswith("checkpoint-") and p.parent.name.split("-")[-1].isdigit() else 0)
        candidate = candidates[-1]
    state = json.loads(candidate.read_text())
    log_history = state.get("log_history", [])
    losses = [e.get("loss") for e in log_history if e.get("loss") is not None]
    tps = [
        e.get("train_tokens_per_second") for e in log_history
        if e.get("train_tokens_per_second") is not None
    ]
    runtime = state.get("train_runtime")
    n_steps = state.get("global_step")
    step_ms = (1000.0 * runtime / n_steps) if (runtime and n_steps) else None
    return {
        "framework": "llama-factory",
        "metrics_path": str(candidate),
        "n_rows": len(log_history),
        "loss_first": losses[0] if losses else None,
        "loss_mid": losses[len(losses) // 2] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "step_ms_mean": step_ms,
        "tokens_per_sec_mean": (sum(tps) / len(tps)) if tps else None,
        "mfu_pct_mean": None,  # LF doesn't compute MFU.
    }


# ---------------------------------------------------------- reporting
def render_report(
    titan: Optional[Dict[str, Any]],
    lf: Optional[Dict[str, Any]],
    *,
    titan_wall: Optional[float],
    lf_wall: Optional[float],
    loss_tolerance: float,
) -> str:
    """Return a markdown-flavored side-by-side comparison string."""

    def cell(d, key, fmt="{:.4f}"):
        if d is None:
            return "—"
        v = d.get(key)
        if v is None:
            return "—"
        try:
            return fmt.format(v)
        except (ValueError, TypeError):
            return str(v)

    rows = [
        ("loss[first]", "loss_first", "{:.4f}"),
        ("loss[mid]",   "loss_mid",   "{:.4f}"),
        ("loss[final]", "loss_final", "{:.4f}"),
        ("step_ms (mean)", "step_ms_mean", "{:.1f}"),
        ("tokens/sec (mean)", "tokens_per_sec_mean", "{:,.0f}"),
        ("MFU %", "mfu_pct_mean", "{:.1f}"),
        ("# log rows", "n_rows", "{}"),
    ]
    lines = []
    lines.append("| Metric | torchtitan | LLaMA-Factory |")
    lines.append("|--------|-----------|---------------|")
    for label, key, fmt in rows:
        lines.append(f"| {label} | {cell(titan, key, fmt)} | {cell(lf, key, fmt)} |")
    if titan_wall is not None or lf_wall is not None:
        titan_wall_s = f"{titan_wall:.1f}" if titan_wall is not None else "—"
        lf_wall_s = f"{lf_wall:.1f}" if lf_wall is not None else "—"
        lines.append(f"| wall time (s) | {titan_wall_s} | {lf_wall_s} |")

    # Correctness verdict.
    verdict_lines = []
    if titan and lf and titan.get("loss_final") and lf.get("loss_final"):
        delta = abs(float(titan["loss_final"]) - float(lf["loss_final"]))
        rel = delta / max(abs(float(lf["loss_final"])), 1e-9)
        verdict_lines.append(
            f"\n**loss[final] absolute Δ**: {delta:.4f} "
            f"(relative: {rel * 100:.2f}%) — "
            + ("PASS" if rel < loss_tolerance else f"FAIL (>{loss_tolerance * 100:.0f}%)")
        )
    return "\n".join(lines) + "\n".join(verdict_lines)


# ---------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--titan-toml", required=True,
                   help="Source torchtitan TOML config to base the smoke run on.")
    p.add_argument("--lf-yaml", required=True,
                   help="Source LLaMA-Factory YAML config to base the smoke run on.")
    p.add_argument("--output-dir", required=True,
                   help="Output dir for both runs (titan/ and lf/ subdirs are created).")
    p.add_argument("--max-steps", type=int, default=50,
                   help="Steps to run on each side. Default 50.")
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss-tolerance", type=float, default=0.05,
                   help="Relative tolerance for matching loss[final] across paths.")
    p.add_argument("--tokenizer-path", default=None,
                   help="Tokenizer path injected into [data].tokenizer_path on the patched "
                        "titan TOML. Required by titan unless --init-from points at a "
                        "checkpoint that already bundles a tokenizer.")
    p.add_argument("--init-from", default=None,
                   help="Optional [model].init_from for the titan TOML (a checkpoint dir "
                        "with bundled tokenizer + weights). When set, --tokenizer-path "
                        "may be omitted.")
    p.add_argument("--model-name-or-path", default=None,
                   help="Override LLaMA-Factory's model_name_or_path. LF needs a checkpoint "
                        "dir that bundles a tokenizer; usually the same value as --init-from.")
    p.add_argument("--dataset-dir", default=None,
                   help="Override [data].dataset_dir (titan) and dataset_dir (LF). Useful "
                        "when the source configs point at machine-specific paths.")
    p.add_argument("--skip-titan", action="store_true")
    p.add_argument("--skip-lf", action="store_true")
    p.add_argument("--report-out", default=None,
                   help="Optional path to also write the markdown report.")
    args = p.parse_args(argv)

    if not args.skip_titan and not (args.tokenizer_path or args.init_from):
        # The error the user hit: titan needs one of these. Fail fast with a
        # clearer message than the trainer's own ValueError.
        sys.stderr.write(
            "[smoke] error: titan requires either --tokenizer-path or --init-from "
            "(or pass --skip-titan to compare LF only).\n"
        )
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    titan_dir = output_dir / "titan"
    lf_dir = output_dir / "lf"

    titan_toml_patched = output_dir / "titan_patched.toml"
    lf_yaml_patched = output_dir / "lf_patched.yaml"

    patch_titan_toml(
        Path(args.titan_toml), titan_toml_patched,
        output_dir=titan_dir, run_name=f"smoke_titan_{args.seed}",
        max_steps=args.max_steps,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, seed=args.seed,
        tokenizer_path=args.tokenizer_path,
        init_from=args.init_from,
        dataset_dir=args.dataset_dir,
    )
    patch_lf_yaml(
        Path(args.lf_yaml), lf_yaml_patched,
        output_dir=lf_dir, run_name=f"smoke_lf_{args.seed}",
        max_steps=args.max_steps,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, seed=args.seed,
        model_name_or_path=args.model_name_or_path or args.init_from,
        dataset_dir=args.dataset_dir,
    )

    titan_metrics: Optional[Dict[str, Any]] = None
    lf_metrics: Optional[Dict[str, Any]] = None
    titan_wall: Optional[float] = None
    lf_wall: Optional[float] = None

    if not args.skip_titan:
        t0 = time.time()
        rc = run_titan(titan_toml_patched)
        titan_wall = time.time() - t0
        if rc != 0:
            print(f"[smoke] torchtitan failed with exit code {rc}", file=sys.stderr)
        titan_metrics = parse_titan_metrics(titan_dir)

    if not args.skip_lf:
        t0 = time.time()
        rc = run_lf(lf_yaml_patched)
        lf_wall = time.time() - t0
        if rc != 0:
            print(f"[smoke] LLaMA-Factory failed with exit code {rc}", file=sys.stderr)
        lf_metrics = parse_lf_metrics(lf_dir)

    report = render_report(
        titan_metrics, lf_metrics,
        titan_wall=titan_wall, lf_wall=lf_wall,
        loss_tolerance=args.loss_tolerance,
    )
    print("\n" + report)
    if args.report_out:
        Path(args.report_out).write_text(report)
        print(f"\n[smoke] report written to {args.report_out}")

    # Exit non-zero if either side failed to produce metrics or correctness failed.
    if titan_metrics is None and not args.skip_titan:
        return 2
    if lf_metrics is None and not args.skip_lf:
        return 2
    if titan_metrics and lf_metrics:
        delta = abs(float(titan_metrics["loss_final"]) - float(lf_metrics["loss_final"]))
        rel = delta / max(abs(float(lf_metrics["loss_final"])), 1e-9)
        if rel >= args.loss_tolerance:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
