"""Tests for the LF-vs-torchtitan throughput comparison harness.

We synthesize both formats (LF ``trainer_state.json`` + LF YAML, titan
``metrics.jsonl`` + ``titan_config.json``) so the parsers, variant pairing,
mismatch detection, and markdown/CSV rendering can be exercised without
running real training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Import from scripts/ — keep the import path resilient even when pytest's
# rootdir doesn't include scripts/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import throughput_compare as tc  # noqa: E402


# ----------------------------------------------------------------- variant detect
@pytest.mark.parametrize(
    "name,expected",
    [
        ("small_baseline_lr3e4", "baseline"),
        ("small_mla_lr3e4", "mla"),
        ("small_moe_lr3e4_ddp4", "moe"),
        ("titan_small_deltanet_gated_lr3e4", "deltanet_gated"),
        ("titan_small_deltanet_kda_lr3e4", "deltanet_kda"),
        ("Small_PoPE_lr3e4", "pope"),
        ("titan_small_swa512_lr3e4_ddp8", "swa512"),
        ("not_a_known_variant", None),
    ],
)
def test_detect_variant(name: str, expected):
    assert tc.detect_variant(name) == expected


# ----------------------------------------------------------------- titan parse
def _make_titan_run(root: Path, *, variant: str, world_size: int = 1,
                    seq_len: int = 2048, per_device: int = 8,
                    grad_accum: int = 16, n_steps: int = 50,
                    bf16: bool = True) -> Path:
    run_dir = root / f"titan_small_{variant}_lr3e4"
    run_dir.mkdir(parents=True)

    # Synthetic metrics.jsonl: warmup spike then a steady-state plateau.
    rows = []
    for step in range(1, n_steps + 1):
        # First 10 steps run slow (warmup); rest steady.
        if step <= 10:
            tokens_per_sec = 5_000.0
            step_ms = 200.0
            mfu = 25.0
        else:
            tokens_per_sec = 12_000.0
            step_ms = 80.0
            mfu = 55.0
        rows.append({
            "step": step,
            "loss": 6.0 - step * 0.01,
            "lr": 3e-4,
            "tokens_per_sec": tokens_per_sec,
            "step_time_ms": step_ms,
            "mfu_pct": mfu,
            "peak_mem_gb": 12.5,
            "grad_norm": 0.5,
        })
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # Sidecar config in a fake checkpoint dir (mimics what save_checkpoint writes).
    ckpt = run_dir / "checkpoint-50"
    ckpt.mkdir()
    (ckpt / "titan_config.json").write_text(json.dumps({
        "training": {
            "per_device_batch_size": per_device,
            "gradient_accumulation_steps": grad_accum,
            "bf16": bf16,
        },
        "data": {"seq_len": seq_len},
        "parallelism": {"dp_replicate_degree": world_size},
    }))
    return run_dir


def test_parse_titan_run_extracts_steady_state(tmp_path: Path):
    run = _make_titan_run(tmp_path, variant="baseline")
    parsed = tc.parse_titan_run(run)
    assert parsed is not None
    assert parsed.variant == "baseline"
    # Tail mean (skipping warmup 20%) should be 12,000 tok/s exactly,
    # not the 5,000 warmup value.
    assert parsed.tokens_per_sec == pytest.approx(12_000.0, rel=0.01)
    assert parsed.step_ms == pytest.approx(80.0, rel=0.01)
    assert parsed.mfu_pct == pytest.approx(55.0, rel=0.01)
    assert parsed.peak_mem_gb == pytest.approx(12.5)
    assert parsed.world_size == 1
    assert parsed.seq_len == 2048
    assert parsed.effective_batch == 128
    assert parsed.bf16 is True
    assert parsed.n_steps_logged == 50


def test_parse_titan_run_returns_none_for_empty_jsonl(tmp_path: Path):
    run_dir = tmp_path / "titan_small_baseline_lr3e4"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text("")
    assert tc.parse_titan_run(run_dir) is None


def test_parse_titan_run_returns_none_when_no_metrics(tmp_path: Path):
    run_dir = tmp_path / "titan_small_baseline_lr3e4"
    run_dir.mkdir()
    assert tc.parse_titan_run(run_dir) is None


# ----------------------------------------------------------------- LF parse
def _make_lf_run(root: Path, *, variant: str, tokens_per_sec: float = 9_500.0,
                 train_runtime: float = 1234.5, train_loss: float = 4.2,
                 global_step: int = 3814, with_tokens_per_sec: bool = True,
                 with_yaml: bool = True) -> Path:
    run_dir = root / f"small_{variant}_lr3e4"
    run_dir.mkdir(parents=True)

    log_history = [
        {"step": 50, "loss": 5.5, "learning_rate": 3e-4},
        {"step": 100, "loss": 4.7, "learning_rate": 3e-4},
        {
            "step": global_step,
            "train_runtime": train_runtime,
            "train_samples_per_second": 4.6,
            "total_flos": 1.2e16,
            "train_loss": train_loss,
        },
    ]
    if with_tokens_per_sec:
        log_history[-1]["train_tokens_per_second"] = tokens_per_sec
    state = {
        "global_step": global_step,
        "max_steps": global_step,
        "log_history": log_history,
    }
    (run_dir / "trainer_state.json").write_text(json.dumps(state))

    if with_yaml:
        (run_dir / "config.yaml").write_text(
            "cutoff_len: 2048\n"
            "per_device_train_batch_size: 8\n"
            "gradient_accumulation_steps: 16\n"
            "bf16: true\n"
        )

    return run_dir


def test_parse_lf_run_extracts_tokens_per_sec(tmp_path: Path):
    run = _make_lf_run(tmp_path, variant="baseline")
    parsed = tc.parse_lf_run(run)
    assert parsed is not None
    assert parsed.variant == "baseline"
    assert parsed.tokens_per_sec == 9_500.0
    assert parsed.final_loss == pytest.approx(4.2)
    assert parsed.step_ms == pytest.approx(1234.5 / 3814 * 1000.0, rel=0.01)
    assert parsed.seq_len == 2048
    assert parsed.effective_batch == 128
    assert parsed.bf16 is True


def test_parse_lf_run_warns_when_tokens_per_sec_missing(tmp_path: Path):
    run = _make_lf_run(tmp_path, variant="baseline", with_tokens_per_sec=False)
    parsed = tc.parse_lf_run(run)
    assert parsed is not None
    assert parsed.tokens_per_sec is None
    assert any("include_tokens_per_second" in n for n in parsed.notes)


# ----------------------------------------------------------------- pairing
def test_pair_runs_produces_matched_table(tmp_path: Path):
    lf_root = tmp_path / "lf_runs"
    titan_root = tmp_path / "titan_runs"
    lf_root.mkdir()
    titan_root.mkdir()

    for variant in ("baseline", "mla", "moe"):
        _make_lf_run(lf_root, variant=variant, tokens_per_sec=9_000.0)
        _make_titan_run(titan_root, variant=variant)

    lf_runs = tc.collect_runs(lf_root, tc.parse_lf_run)
    titan_runs = tc.collect_runs(titan_root, tc.parse_titan_run)
    rows = tc.pair_runs(lf_runs, titan_runs)

    by_variant = {r.variant: r for r in rows}
    assert set(by_variant) == {"baseline", "mla", "moe"}
    for r in by_variant.values():
        assert r.lf is not None
        assert r.titan is not None
        # Both have eff_batch = 128 in fixtures, no mismatch notes.
        mismatches = [n for n in r.notes if "mismatch" in n]
        assert mismatches == []
        # titan = 12,000 / lf = 9,000 → ~1.33×
        assert r.speedup == pytest.approx(12_000.0 / 9_000.0, rel=0.01)


def test_pair_runs_flags_mismatched_effective_batch(tmp_path: Path):
    lf_root = tmp_path / "lf"
    titan_root = tmp_path / "titan"
    lf_root.mkdir()
    titan_root.mkdir()

    _make_lf_run(lf_root, variant="baseline")
    # Titan run with grad_accum=8 → eff_batch=64, mismatched against LF=128.
    _make_titan_run(titan_root, variant="baseline", grad_accum=8)

    lf_runs = tc.collect_runs(lf_root, tc.parse_lf_run)
    titan_runs = tc.collect_runs(titan_root, tc.parse_titan_run)
    rows = tc.pair_runs(lf_runs, titan_runs)
    assert len(rows) == 1
    assert any("effective_batch mismatch" in n for n in rows[0].notes)


def test_pair_runs_handles_unpaired_variants(tmp_path: Path):
    lf_root = tmp_path / "lf"
    titan_root = tmp_path / "titan"
    lf_root.mkdir()
    titan_root.mkdir()

    _make_lf_run(lf_root, variant="baseline")
    _make_titan_run(titan_root, variant="mla")

    rows = tc.pair_runs(
        tc.collect_runs(lf_root, tc.parse_lf_run),
        tc.collect_runs(titan_root, tc.parse_titan_run),
    )
    by_variant = {r.variant: r for r in rows}
    assert by_variant["baseline"].titan is None
    assert "no titan run" in by_variant["baseline"].notes
    assert by_variant["mla"].lf is None
    assert "no LF run" in by_variant["mla"].notes


# ----------------------------------------------------------------- rendering
def test_render_markdown_contains_speedup_column(tmp_path: Path):
    lf_root = tmp_path / "lf"
    titan_root = tmp_path / "titan"
    lf_root.mkdir(); titan_root.mkdir()
    _make_lf_run(lf_root, variant="baseline")
    _make_titan_run(titan_root, variant="baseline")

    rows = tc.pair_runs(
        tc.collect_runs(lf_root, tc.parse_lf_run),
        tc.collect_runs(titan_root, tc.parse_titan_run),
    )
    md = tc.render_markdown(rows)
    assert "speedup" in md
    # 12,000 (titan tail-mean) / 9,500 (LF fixture default) ≈ 1.26
    assert "1.26×" in md
    assert "baseline" in md


def test_render_csv_writes_all_fields(tmp_path: Path):
    lf_root = tmp_path / "lf"
    titan_root = tmp_path / "titan"
    lf_root.mkdir(); titan_root.mkdir()
    _make_lf_run(lf_root, variant="baseline")
    _make_titan_run(titan_root, variant="baseline")
    rows = tc.pair_runs(
        tc.collect_runs(lf_root, tc.parse_lf_run),
        tc.collect_runs(titan_root, tc.parse_titan_run),
    )

    csv_path = tmp_path / "out.csv"
    tc.render_csv(rows, csv_path)
    text = csv_path.read_text()
    assert "lf_tokens_per_sec" in text
    assert "titan_tokens_per_sec" in text
    assert "speedup" in text
    assert "9500.0" in text
    assert "12000.0" in text


# ------------------------------------------------------------------- end-to-end
def test_main_writes_markdown_and_csv(tmp_path: Path, capsys):
    lf_root = tmp_path / "lf"
    titan_root = tmp_path / "titan"
    lf_root.mkdir(); titan_root.mkdir()
    for variant in ("baseline", "mtp"):
        _make_lf_run(lf_root, variant=variant)
        _make_titan_run(titan_root, variant=variant)

    out_md = tmp_path / "report.md"
    rc = tc.main([
        "--lf_runs", str(lf_root),
        "--titan_runs", str(titan_root),
        "--out", str(out_md),
    ])
    assert rc == 0
    assert out_md.exists()
    assert out_md.with_suffix(".csv").exists()
    md = out_md.read_text()
    assert "torchtitan" in md
    assert "baseline" in md and "mtp" in md
