#!/usr/bin/env python3
"""Head-to-head LLaMA-Factory vs torchtitan throughput comparison.

This is the headline deliverable from the torchtitan integration plan: given a
directory of LF run outputs (each with ``trainer_state.json``) and a directory
of titan run outputs (each with ``metrics.jsonl``), pair them by variant and
emit a markdown table + CSV. Optional matplotlib bar chart for slides.

Usage:

    python scripts/throughput_compare.py \\
        --lf_runs /shared/4/projects/umich-slm/models \\
        --titan_runs ./outputs/titan \\
        --out docs/throughput_comparison_v1.md

Each input directory should contain one subdirectory per run. Run subdirectory
names must include the variant name as a substring (e.g. ``small_mla_lr3e4``,
``titan_small_mla_lr3e4_ddp4``); recognized variants live in ``KNOWN_VARIANTS``
below. Pairs only get emitted when both LF and titan have a matching variant
*and* the apples-to-apples conditions (effective batch, seq_len, world_size,
precision) are satisfied — mismatches surface in the report's notes column.

Why two formats? LF writes HF-Trainer's ``trainer_state.json`` at end-of-run
and includes ``train_tokens_per_second`` only when the YAML sets
``include_tokens_per_second: true`` (already added to the project's
pretraining YAMLs). Titan writes ``metrics.jsonl`` continuously; this script
takes a stable tail-mean to dodge warmup outliers.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Make ``import umich_slm`` work when run directly from a checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


KNOWN_VARIANTS: Tuple[str, ...] = (
    # Order matters: longer names first so "deltanet_gated" matches before "deltanet".
    "deltanet_gated",
    "deltanet_kda",
    "zcrmsnorm",
    "swa512",
    "baseline",
    "mla",
    "moe",
    "mod",
    "mtp",
    "pope",
)


# ---------------------------------------------------------------- data model
@dataclass
class RunMetrics:
    """A single run's parsed throughput metrics."""

    framework: str  # "lf" | "titan"
    variant: str
    run_dir: Path
    world_size: Optional[int] = None
    seq_len: Optional[int] = None
    effective_batch: Optional[int] = None
    bf16: Optional[bool] = None
    tokens_per_sec: Optional[float] = None
    step_ms: Optional[float] = None
    mfu_pct: Optional[float] = None
    peak_mem_gb: Optional[float] = None
    final_loss: Optional[float] = None
    n_steps_logged: int = 0
    hw_key: Optional[str] = None
    notes: List[str] = field(default_factory=list)


# ------------------------------------------------------------- variant detect
_VARIANT_RE = re.compile(r"(?P<variant>" + "|".join(KNOWN_VARIANTS) + r")", re.IGNORECASE)


def detect_variant(name: str) -> Optional[str]:
    """Return the first known variant token in ``name`` (case-insensitive)."""
    m = _VARIANT_RE.search(name.lower())
    return m.group("variant") if m else None


# --------------------------------------------------------------- titan parse
def _tail_mean(rows: List[Dict[str, Any]], key: str, *, warmup_frac: float = 0.2) -> Optional[float]:
    if not rows:
        return None
    skip = max(int(len(rows) * warmup_frac), 1)
    vals = [r.get(key) for r in rows[skip:] if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_titan_run(run_dir: Path) -> Optional[RunMetrics]:
    """Parse a titan run directory into a ``RunMetrics`` (or None if unusable)."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None

    rows = _read_jsonl(metrics_path)
    if not rows:
        return None

    variant = detect_variant(run_dir.name)
    if variant is None:
        return None

    out = RunMetrics(framework="titan", variant=variant, run_dir=run_dir)
    out.tokens_per_sec = _tail_mean(rows, "tokens_per_sec")
    out.step_ms = _tail_mean(rows, "step_time_ms")
    out.mfu_pct = _tail_mean(rows, "mfu_pct")
    # Peak memory: take the max observed (memory monotonically saturates after
    # the first few steps, so max is more informative than mean).
    mems = [r.get("peak_mem_gb") for r in rows if isinstance(r.get("peak_mem_gb"), (int, float))]
    if mems:
        out.peak_mem_gb = float(max(mems))
    out.final_loss = next(
        (float(r["loss"]) for r in reversed(rows) if isinstance(r.get("loss"), (int, float))),
        None,
    )
    out.n_steps_logged = len(rows)

    # Pull config metadata from titan_config.json in any saved checkpoint dir
    # (rank-0 writes it alongside model.safetensors).
    cfg = _load_titan_config_sidecar(run_dir)
    if cfg is not None:
        out.world_size = cfg.get("parallelism", {}).get("dp_replicate_degree")
        training = cfg.get("training", {}) or {}
        data = cfg.get("data", {}) or {}
        out.seq_len = data.get("seq_len")
        per_dev = training.get("per_device_batch_size")
        accum = training.get("gradient_accumulation_steps")
        if per_dev and accum and out.world_size:
            out.effective_batch = int(per_dev * accum * out.world_size)
        out.bf16 = training.get("bf16")

    return out


def _load_titan_config_sidecar(run_dir: Path) -> Optional[Dict[str, Any]]:
    for ckpt in sorted(run_dir.glob("checkpoint-*")):
        sidecar = ckpt / "titan_config.json"
        if sidecar.exists():
            try:
                with open(sidecar) as fh:
                    return json.load(fh)
            except json.JSONDecodeError:
                continue
    return None


# ----------------------------------------------------------------- LF parse
def parse_lf_run(run_dir: Path) -> Optional[RunMetrics]:
    """Parse a LLaMA-Factory run directory into a ``RunMetrics`` (or None)."""
    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        return None

    with open(state_path) as fh:
        state = json.load(fh)

    log_history: List[Dict[str, Any]] = state.get("log_history", []) or []
    if not log_history:
        return None

    variant = detect_variant(run_dir.name)
    if variant is None:
        return None

    out = RunMetrics(framework="lf", variant=variant, run_dir=run_dir)

    # The HF Trainer's run-summary entry (last entry) is the one that carries
    # ``train_runtime`` / ``train_tokens_per_second`` / ``total_flos``.
    summary = next(
        (entry for entry in reversed(log_history) if "train_runtime" in entry),
        None,
    )
    if summary is not None:
        runtime = float(summary.get("train_runtime", 0.0)) or None
        out.tokens_per_sec = (
            float(summary["train_tokens_per_second"])
            if "train_tokens_per_second" in summary
            else None
        )
        # Step time: train_runtime / global_step. Falls back to None if global
        # step isn't recorded.
        steps = state.get("global_step") or summary.get("step")
        if runtime and steps:
            out.step_ms = (runtime / int(steps)) * 1000.0
        out.final_loss = (
            float(summary.get("train_loss"))
            if "train_loss" in summary
            else None
        )
        out.n_steps_logged = int(state.get("global_step", 0))

    # Per-step entries also help for in-progress runs that haven't closed out.
    if out.tokens_per_sec is None:
        out.notes.append(
            "no train_tokens_per_second in trainer_state.json — set "
            "include_tokens_per_second: true in the LF YAML"
        )
    if out.final_loss is None:
        # Use the last per-step ``loss`` entry as a coarse fallback.
        for entry in reversed(log_history):
            if isinstance(entry.get("loss"), (int, float)):
                out.final_loss = float(entry["loss"])
                break

    # Read config alongside the run dir if HF wrote training_args.bin or a YAML.
    # We do best-effort metadata extraction; missing fields just stay None.
    out.bf16, out.seq_len, out.effective_batch, out.world_size = _scan_lf_config(run_dir)

    return out


def _scan_lf_config(run_dir: Path) -> Tuple[Optional[bool], Optional[int], Optional[int], Optional[int]]:
    """Best-effort metadata sniffing from a LF run dir.

    Looks at ``args.json`` / ``adapter_config.json`` / sibling YAMLs. Returns
    ``(bf16, seq_len, effective_batch, world_size)``; any field that can't be
    determined stays ``None``.
    """
    bf16: Optional[bool] = None
    seq_len: Optional[int] = None
    effective_batch: Optional[int] = None
    world_size: Optional[int] = None

    args_json = run_dir / "args.json"
    if args_json.exists():
        try:
            with open(args_json) as fh:
                args = json.load(fh)
            bf16 = args.get("bf16")
            seq_len = args.get("cutoff_len") or args.get("max_seq_length")
            per_dev = args.get("per_device_train_batch_size")
            accum = args.get("gradient_accumulation_steps")
            world_size = args.get("world_size") or args.get("n_gpu") or 1
            if per_dev and accum and world_size:
                effective_batch = int(per_dev * accum * world_size)
        except (json.JSONDecodeError, OSError):
            pass

    # Pretraining YAML often sits next to or inside the run dir.
    if seq_len is None:
        for yaml_candidate in run_dir.glob("*.yaml"):
            seq_len, effective_batch, world_size, bf16 = _try_lf_yaml(
                yaml_candidate, seq_len, effective_batch, world_size, bf16
            )
            break

    return bf16, seq_len, effective_batch, world_size


def _try_lf_yaml(path: Path, seq_len, effective_batch, world_size, bf16):
    try:
        import yaml  # type: ignore
    except ImportError:
        return seq_len, effective_batch, world_size, bf16
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
    except (yaml.YAMLError, OSError):
        return seq_len, effective_batch, world_size, bf16
    seq_len = seq_len or data.get("cutoff_len") or data.get("max_seq_length")
    per_dev = data.get("per_device_train_batch_size")
    accum = data.get("gradient_accumulation_steps")
    # LF YAMLs don't record world_size (torchrun-time setting); default to 1
    # so we can compute an effective_batch when args.json is absent.
    if per_dev and accum:
        ws = world_size or 1
        effective_batch = effective_batch or int(per_dev * accum * ws)
        world_size = ws
    if bf16 is None:
        bf16 = data.get("bf16")
    return seq_len, effective_batch, world_size, bf16


# ------------------------------------------------------------------- pairing
@dataclass
class PairedRow:
    variant: str
    lf: Optional[RunMetrics]
    titan: Optional[RunMetrics]
    notes: List[str] = field(default_factory=list)

    @property
    def speedup(self) -> Optional[float]:
        if self.lf and self.titan and self.lf.tokens_per_sec and self.titan.tokens_per_sec:
            return self.titan.tokens_per_sec / self.lf.tokens_per_sec
        return None


def _index_by_variant(runs: Iterable[RunMetrics]) -> Dict[str, RunMetrics]:
    """If multiple runs share a variant, keep the most recent (mtime)."""
    out: Dict[str, RunMetrics] = {}
    for run in runs:
        existing = out.get(run.variant)
        if existing is None:
            out[run.variant] = run
            continue
        # Prefer the run with more steps logged; tiebreak by mtime.
        a_score = (run.n_steps_logged, run.run_dir.stat().st_mtime if run.run_dir.exists() else 0)
        b_score = (existing.n_steps_logged, existing.run_dir.stat().st_mtime if existing.run_dir.exists() else 0)
        if a_score > b_score:
            out[run.variant] = run
    return out


def pair_runs(lf_runs: List[RunMetrics], titan_runs: List[RunMetrics]) -> List[PairedRow]:
    lf_idx = _index_by_variant(lf_runs)
    titan_idx = _index_by_variant(titan_runs)
    variants = sorted(set(lf_idx) | set(titan_idx), key=lambda v: KNOWN_VARIANTS.index(v))

    rows: List[PairedRow] = []
    for v in variants:
        lf = lf_idx.get(v)
        titan = titan_idx.get(v)
        notes: List[str] = []
        if lf and titan:
            for field_name, label in [
                ("seq_len", "seq_len"),
                ("effective_batch", "effective_batch"),
                ("world_size", "world_size"),
                ("bf16", "bf16"),
            ]:
                lv, tv = getattr(lf, field_name), getattr(titan, field_name)
                if lv is not None and tv is not None and lv != tv:
                    notes.append(f"{label} mismatch (lf={lv}, titan={tv})")
        if lf is None:
            notes.append("no LF run")
        if titan is None:
            notes.append("no titan run")
        rows.append(PairedRow(variant=v, lf=lf, titan=titan, notes=notes))
    return rows


# ----------------------------------------------------------------- rendering
def _fmt(v: Optional[float], spec: str = ".0f") -> str:
    return format(v, spec) if isinstance(v, (int, float)) else "—"


def _fmt_speedup(s: Optional[float]) -> str:
    if s is None:
        return "—"
    return f"**{s:.2f}×**" if s >= 1.0 else f"{s:.2f}×"


def render_markdown(rows: List[PairedRow]) -> str:
    """Build the comparison table as markdown."""
    headers = [
        "variant", "world", "eff_batch", "seq_len",
        "LF tok/s", "titan tok/s", "speedup",
        "LF step ms", "titan step ms",
        "titan MFU %", "titan peak GB",
        "notes",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lf, t = row.lf, row.titan
        notes = "; ".join(row.notes + (lf.notes if lf else []) + (t.notes if t else []))
        cells = [
            row.variant,
            str((t.world_size if t else (lf.world_size if lf else None)) or "—"),
            str((t.effective_batch if t else (lf.effective_batch if lf else None)) or "—"),
            str((t.seq_len if t else (lf.seq_len if lf else None)) or "—"),
            _fmt(lf.tokens_per_sec if lf else None, ",.0f"),
            _fmt(t.tokens_per_sec if t else None, ",.0f"),
            _fmt_speedup(row.speedup),
            _fmt(lf.step_ms if lf else None, ".1f"),
            _fmt(t.step_ms if t else None, ".1f"),
            _fmt(t.mfu_pct if t else None, ".1f"),
            _fmt(t.peak_mem_gb if t else None, ".1f"),
            notes or "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_csv(rows: List[PairedRow], path: Path) -> None:
    fields = [
        "variant", "world_size", "effective_batch", "seq_len", "bf16",
        "lf_tokens_per_sec", "titan_tokens_per_sec", "speedup",
        "lf_step_ms", "titan_step_ms",
        "titan_mfu_pct", "titan_peak_mem_gb",
        "lf_final_loss", "titan_final_loss",
        "notes",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            lf, t = row.lf, row.titan
            writer.writerow({
                "variant": row.variant,
                "world_size": (t.world_size if t else (lf.world_size if lf else "")),
                "effective_batch": (t.effective_batch if t else (lf.effective_batch if lf else "")),
                "seq_len": (t.seq_len if t else (lf.seq_len if lf else "")),
                "bf16": (t.bf16 if t else (lf.bf16 if lf else "")),
                "lf_tokens_per_sec": lf.tokens_per_sec if lf else "",
                "titan_tokens_per_sec": t.tokens_per_sec if t else "",
                "speedup": row.speedup if row.speedup is not None else "",
                "lf_step_ms": lf.step_ms if lf else "",
                "titan_step_ms": t.step_ms if t else "",
                "titan_mfu_pct": t.mfu_pct if t else "",
                "titan_peak_mem_gb": t.peak_mem_gb if t else "",
                "lf_final_loss": lf.final_loss if lf else "",
                "titan_final_loss": t.final_loss if t else "",
                "notes": "; ".join(row.notes),
            })


def render_summary_plot(rows: List[PairedRow], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    paired = [r for r in rows if r.lf and r.titan and r.lf.tokens_per_sec and r.titan.tokens_per_sec]
    if not paired:
        return False

    variants = [r.variant for r in paired]
    lf_vals = [r.lf.tokens_per_sec for r in paired]
    titan_vals = [r.titan.tokens_per_sec for r in paired]

    x = list(range(len(variants)))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(6, len(variants) * 1.2), 4.5))
    ax.bar([xi - width / 2 for xi in x], lf_vals, width, label="LLaMA-Factory")
    ax.bar([xi + width / 2 for xi in x], titan_vals, width, label="torchtitan")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_ylabel("tokens / sec")
    ax.set_title("Pretraining throughput: LF vs torchtitan")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ------------------------------------------------------------------------- CLI
def collect_runs(root: Path, parser_fn) -> List[RunMetrics]:
    if not root.exists():
        return []
    runs: List[RunMetrics] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        result = parser_fn(sub)
        if result is not None:
            runs.append(result)
    return runs


def build_report_header(rows: List[PairedRow]) -> str:
    paired = [r for r in rows if r.lf and r.titan and r.speedup is not None]
    lines = [
        "# LF vs torchtitan throughput comparison",
        "",
        "_Generated by `scripts/throughput_compare.py`. Tokens/sec for titan is",
        "the tail-mean of `metrics.jsonl` (skipping the first 20% as warmup);",
        "for LF it's `train_tokens_per_second` from `trainer_state.json`'s final",
        "summary entry. MFU% is reported only for titan since LF doesn't log it._",
        "",
    ]
    if paired:
        speedups = [r.speedup for r in paired if r.speedup is not None]
        lines.append(
            f"**Coverage:** {len(paired)} paired variants; titan speedup ranges "
            f"{min(speedups):.2f}× – {max(speedups):.2f}× "
            f"(geomean {sum(speedups) / len(speedups):.2f}×)."
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pair and compare LF vs torchtitan throughput logs."
    )
    parser.add_argument("--lf_runs", type=Path, required=True,
                        help="Directory containing LF run subdirs (each with trainer_state.json)")
    parser.add_argument("--titan_runs", type=Path, required=True,
                        help="Directory containing titan run subdirs (each with metrics.jsonl)")
    parser.add_argument("--out", type=Path, default=Path("docs/throughput_comparison.md"),
                        help="Markdown output path; CSV is written alongside as <out>.csv")
    parser.add_argument("--summary_plot", type=Path, default=None,
                        help="Optional PNG path for a side-by-side bar chart")
    args = parser.parse_args(argv)

    lf_runs = collect_runs(args.lf_runs, parse_lf_run)
    titan_runs = collect_runs(args.titan_runs, parse_titan_run)
    if not lf_runs and not titan_runs:
        print(f"[throughput_compare] no runs found under {args.lf_runs} or {args.titan_runs}",
              file=sys.stderr)
        return 1

    rows = pair_runs(lf_runs, titan_runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    header = build_report_header(rows)
    table = render_markdown(rows)
    args.out.write_text(header + table)
    print(f"[throughput_compare] wrote {args.out}")

    csv_path = args.out.with_suffix(".csv")
    render_csv(rows, csv_path)
    print(f"[throughput_compare] wrote {csv_path}")

    if args.summary_plot:
        ok = render_summary_plot(rows, args.summary_plot)
        if ok:
            print(f"[throughput_compare] wrote {args.summary_plot}")
        else:
            print("[throughput_compare] summary plot skipped (matplotlib missing or no paired runs)",
                  file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
