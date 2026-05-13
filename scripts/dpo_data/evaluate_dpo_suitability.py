"""Evaluate DPO suitability for class-selected post-training datasets.

This script performs lightweight, streaming-first checks over named datasets and
emits a JSON + Markdown report answering:
- which datasets are directly DPO-ready
- which require transformation
- which are weaker fits

Usage:
  .venv/bin/python scripts/dpo_data/evaluate_dpo_suitability.py \
      --sample-size 32 \
    --streaming \
    --output-dir references/dpo_suitability
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_dataset_builder


DPO_DATASETS: list[dict[str, Any]] = [
    {
        "name": "openbmb/UltraFeedback",
        "splits": ["train", "validation", "test"],
        "priority": 1,
        "rag_relevance": "high",
        "repo_effort": "low",
        "notes": "UltraFeedback-first: existing repo prep path already supports UltraFeedback binarized conversion.",
    },
    {
        "name": "nvidia/HelpSteer3",
        "splits": ["train", "validation", "test"],
        "priority": 2,
        "rag_relevance": "medium",
        "repo_effort": "medium",
        "notes": "Contains paired responses and overall_preference; good DPO candidate after adapter conversion.",
    },
    {
        "name": "lmarena-ai/arena-human-preference-55k",
        "splits": ["train", "validation", "test"],
        "priority": 3,
        "rag_relevance": "medium",
        "repo_effort": "medium",
        "notes": "Pairwise winner format with response_a/response_b and winner flags.",
    },
    {
        "name": "openai/webgpt_comparisons",
        "splits": ["train", "validation", "test"],
        "priority": 4,
        "rag_relevance": "high",
        "repo_effort": "high",
        "notes": "Conceptually strong for RAG-style reasoning but may require loader/workaround in current datasets stack.",
    },
    {
        "name": "OpenAssistant/oasst1",
        "splits": ["train", "validation", "test"],
        "priority": 5,
        "rag_relevance": "medium",
        "repo_effort": "high",
        "notes": "Tree-structured chat data with rank metadata; requires pair construction.",
    },
    {
        "name": "nvidia/HelpSteer",
        "splits": ["train", "validation", "test"],
        "priority": 6,
        "rag_relevance": "low",
        "repo_effort": "high",
        "notes": "Single response with scalar scores; weak direct fit for DPO unless pairs are synthesized.",
    },
]

WARMUP_DATASETS: list[dict[str, str]] = [
    {
        "name": "nvidia/OpenMathInstruct-2",
        "use": "SFT warmup for math reasoning",
    },
    {
        "name": "sahil280114/codealpaca",
        "use": "SFT warmup for code instruction following",
    },
    {
        "name": "OpenBMB/ToolBench",
        "use": "Tool-use SFT/evaluation warmup",
    },
]

WEBGPT_PARQUET_PATHS: list[str] = [
    "hf://datasets/openai/webgpt_comparisons@refs/convert/parquet/default/train/0000.parquet",
]

WEBGPT_DPO_MAPPING = {
    "prompt": "question.full_text",
    "chosen": "answer_0 if score_0 > score_1 else answer_1",
    "rejected": "answer_1 if score_0 > score_1 else answer_0",
    "exclude": "score_0 == score_1",
}


@dataclass
class CriteriaResult:
    prompt_context: bool | str
    pair_constructable: bool | str
    chosen_rejected_signal: bool | str
    rag_relevance: str
    license_risk: str
    license_status: str
    implementation_effort: str
    llamafactory_compat: bool | str


def _first_rows(dataset_name: str, splits: list[str], sample_size: int, streaming: bool) -> tuple[list[dict[str, Any]], str | None, str | None]:
    last_error: str | None = None
    for split in splits:
        try:
            ds = load_dataset(dataset_name, split=split, streaming=streaming)
            rows = [r for r in islice(iter(ds), sample_size) if isinstance(r, dict)]
            if rows:
                return rows, split, None
            last_error = f"No rows yielded for split '{split}'"
        except Exception as exc:  # pragma: no cover - depends on external hub state
            last_error = f"{type(exc).__name__}: {exc}"
    return [], None, last_error


def _safe_license(dataset_name: str) -> tuple[str, str, str]:
    try:
        builder = load_dataset_builder(dataset_name)
        license_str = (builder.info.license or "").strip() if builder.info else ""
        if not license_str:
            return "unknown", "requires_manual_review", "unverified"
        lowered = license_str.lower()
        if any(x in lowered for x in ["cc-by", "apache", "mit", "bsd"]):
            return license_str, "low", "verified"
        if any(x in lowered for x in ["unknown", "custom"]):
            return license_str, "requires_manual_review", "unverified"
        return license_str, "medium", "verified"
    except Exception:  # pragma: no cover - depends on external hub state
        return "unavailable", "requires_manual_review", "unverified"


def _probe_webgpt_parquet(sample_size: int) -> tuple[dict[str, Any] | None, str | None]:
    last_error: str | None = None
    for path in WEBGPT_PARQUET_PATHS:
        try:
            ds = load_dataset(
                "parquet",
                data_files={"train": path},
                split="train",
                streaming=True,
            )
            rows = [r for r in islice(iter(ds), sample_size) if isinstance(r, dict)]
            if not rows:
                continue
            first = rows[0]
            q = first.get("question")
            question_has_full_text = isinstance(q, dict) and bool(q.get("full_text"))
            required = all(k in first for k in ["answer_0", "answer_1", "score_0", "score_1"])
            non_ties = sum(1 for row in rows if row.get("score_0") != row.get("score_1"))
            return {
                "parquet_probe_status": "success",
                "parquet_path": path,
                "sample_size": len(rows),
                "required_present": required and question_has_full_text,
                "non_tie_preferences": non_ties,
                "question_full_text_present": question_has_full_text,
                "dpo_mapping": WEBGPT_DPO_MAPPING,
            }, None
        except Exception as exc:  # pragma: no cover - depends on external hub state
            last_error = f"{type(exc).__name__}: {exc}"
    return None, last_error


def _looks_like_prompt_field(row: dict[str, Any]) -> bool:
    if isinstance(row.get("prompt"), str) and row["prompt"].strip():
        return True
    if isinstance(row.get("instruction"), str) and row["instruction"].strip():
        return True
    ctx = row.get("context")
    if isinstance(ctx, list):
        return any(
            isinstance(x, dict)
            and str(x.get("role", "")).lower() in {"user", "prompter", "human"}
            and isinstance(x.get("content"), str)
            and x.get("content", "").strip()
            for x in ctx
        )
    if isinstance(row.get("text"), str) and str(row.get("role", "")).lower() in {"prompter", "user", "human"}:
        return True
    return False


def _looks_like_pair_data(row: dict[str, Any]) -> bool:
    if "chosen" in row and "rejected" in row:
        return True
    if all(k in row for k in ["response_a", "response_b"]):
        return True
    if all(k in row for k in ["response1", "response2"]):
        return True
    completions = row.get("completions")
    if isinstance(completions, list) and len(completions) >= 2:
        return True
    return False


def _looks_like_signal(row: dict[str, Any]) -> bool:
    if "chosen" in row and "rejected" in row:
        return True
    if any(k in row for k in ["winner_model_a", "winner_model_b", "winner_tie"]):
        return True
    if "overall_preference" in row:
        return True
    completions = row.get("completions")
    if isinstance(completions, list):
        # Heuristic: ranked/comparative annotations often include rating/preference fields.
        for c in completions:
            if not isinstance(c, dict):
                continue
            ann = c.get("annotations")
            if isinstance(ann, dict) and ann:
                return True
    return False


def _llamafactory_compat(dataset_name: str, row: dict[str, Any], load_error: str | None) -> bool | str:
    if load_error:
        return "unknown"
    if dataset_name == "openbmb/UltraFeedback":
        # Conversion is feasible, but requires ranking extraction from completions.
        return _looks_like_prompt_field(row) and _looks_like_pair_data(row)
    if dataset_name == "nvidia/HelpSteer3":
        return _looks_like_prompt_field(row) and all(k in row for k in ["response1", "response2", "overall_preference"])
    if dataset_name == "lmarena-ai/arena-human-preference-55k":
        return _looks_like_prompt_field(row) and all(
            k in row for k in ["response_a", "response_b", "winner_model_a", "winner_model_b"]
        )
    if dataset_name == "openai/webgpt_comparisons":
        return _looks_like_prompt_field(row) and _looks_like_pair_data(row) and _looks_like_signal(row)
    if dataset_name == "OpenAssistant/oasst1":
        return False
    if dataset_name == "nvidia/HelpSteer":
        return False
    return False


def _classify(dataset_name: str, criteria: CriteriaResult, load_error: str | None) -> str:
    if load_error:
        return "blocked_by_loader"
    if dataset_name == "openbmb/UltraFeedback":
        if criteria.prompt_context and criteria.pair_constructable and criteria.chosen_rejected_signal:
            return "high_priority_convertible"
        return "requires_transformation"
    if criteria.prompt_context and criteria.pair_constructable and criteria.chosen_rejected_signal and criteria.llamafactory_compat:
        return "directly_dpo_ready"
    if criteria.prompt_context and criteria.pair_constructable:
        return "requires_transformation"
    return "weaker_fit"


def _evaluate_dataset(spec: dict[str, Any], sample_size: int, streaming: bool) -> dict[str, Any]:
    dataset_name = spec["name"]
    rows, split, load_error = _first_rows(dataset_name, spec["splits"], sample_size, streaming)
    row = rows[0] if rows else {}

    license_text, license_risk, license_status = _safe_license(dataset_name)

    prompt_context: bool | str = _looks_like_prompt_field(row)
    pair_constructable: bool | str = _looks_like_pair_data(row)
    chosen_rejected_signal: bool | str = _looks_like_signal(row)
    if load_error:
        prompt_context = "unknown"
        pair_constructable = "unknown"
        chosen_rejected_signal = "unknown"

    notes = spec["notes"]
    extra_probe: dict[str, Any] | None = None
    webgpt_probe: dict[str, Any] | None = None
    if dataset_name == "openai/webgpt_comparisons" and load_error:
        probe, probe_error = _probe_webgpt_parquet(sample_size)
        webgpt_probe = {
            "standard_loader_error": load_error,
            "parquet_probe_status": "failed",
            "parquet_path": WEBGPT_PARQUET_PATHS[0],
            "sample_size": 0,
            "required_present": False,
            "non_tie_preferences": 0,
            "question_full_text_present": False,
            "dpo_mapping": WEBGPT_DPO_MAPPING,
        }
        if probe:
            extra_probe = probe
            prompt_context = bool(probe["question_full_text_present"])
            pair_constructable = bool(probe["required_present"])
            chosen_rejected_signal = bool(probe["required_present"] and probe["non_tie_preferences"] > 0)
            webgpt_probe.update(probe)
            notes = (
                "Normal load_dataset('openai/webgpt_comparisons') is blocked by the legacy dataset-script restriction. "
                "Parquet sampling works using the recorded path. WebGPT is conceptually strong for Reasoning-First RAG. "
                "DPO mapping is feasible with prompt=question.full_text, chosen/rejected from score_0 vs score_1, "
                "excluding ties where score_0 == score_1."
            )
        elif probe_error:
            notes = f"{spec['notes']} Parquet probe error: {probe_error}"
            webgpt_probe["parquet_probe_status"] = "failed"

    criteria = CriteriaResult(
        prompt_context=prompt_context,
        pair_constructable=pair_constructable,
        chosen_rejected_signal=chosen_rejected_signal,
        rag_relevance=spec["rag_relevance"],
        license_risk=license_risk,
        license_status=license_status,
        implementation_effort=spec["repo_effort"],
        llamafactory_compat=_llamafactory_compat(dataset_name, row, load_error),
    )

    classification = _classify(dataset_name, criteria, load_error)
    if dataset_name == "openai/webgpt_comparisons" and extra_probe:
        classification = "requires_transformation"

    reported_split = split
    reported_sample_size = len(rows)
    if dataset_name == "openai/webgpt_comparisons" and extra_probe:
        reported_split = "parquet_fallback"
        reported_sample_size = int(extra_probe["sample_size"])

    return {
        "dataset": dataset_name,
        "priority": spec["priority"],
        "checked_split": reported_split,
        "sample_size": reported_sample_size,
        "classification": classification,
        "criteria": {
            "prompt_context": criteria.prompt_context,
            "pair_constructable": criteria.pair_constructable,
            "chosen_rejected_signal": criteria.chosen_rejected_signal,
            "rag_relevance": criteria.rag_relevance,
            "license_risk": criteria.license_risk,
            "license_status": criteria.license_status,
            "implementation_effort": criteria.implementation_effort,
            "llamafactory_compat": criteria.llamafactory_compat,
        },
        "license": license_text,
        "load_error": load_error,
        "notes": notes,
        "webgpt_probe": webgpt_probe,
    }


def _to_markdown(results: list[dict[str, Any]], output_json: Path) -> str:
    def _yn(value: bool | str) -> str:
        if value is True:
            return "Y"
        if value is False:
            return "N"
        return "?"

    header = [
        "# DPO Dataset Suitability Report",
        "",
        "This report evaluates six class-selected human-preference datasets for DPO readiness.",
        "",
        "## Criteria",
        "1. prompt/context present",
        "2. pair constructable",
        "3. chosen/rejected signal",
        "4. relevance to Reasoning-First RAG Hybrid",
        "5. license risk/status",
        "6. implementation effort in current repo",
        "7. LLaMA-Factory prompt/chosen/rejected compatibility",
        "",
        "## Summary",
        "| Priority | Dataset | Classification | Prompt | Pair | Signal | RAG | License Risk | License Status | Effort | LLaMA-Factory Compat |",
        "|---:|---|---|:---:|:---:|:---:|---|---|---|---|:---:|",
    ]

    lines = []
    for r in sorted(results, key=lambda x: x["priority"]):
        c = r["criteria"]
        lines.append(
            "| {p} | {d} | {cl} | {pr} | {pa} | {sg} | {rag} | {lr} | {ls} | {ef} | {lf} |".format(
                p=r["priority"],
                d=r["dataset"],
                cl=r["classification"],
                pr=_yn(c["prompt_context"]),
                pa=_yn(c["pair_constructable"]),
                sg=_yn(c["chosen_rejected_signal"]),
                rag=c["rag_relevance"],
                lr=c["license_risk"],
                ls=c["license_status"],
                ef=c["implementation_effort"],
                lf=_yn(c["llamafactory_compat"]),
            )
        )

    details = ["", "## Dataset Notes"]
    for r in sorted(results, key=lambda x: x["priority"]):
        details.append(f"### {r['dataset']}")
        details.append(f"- Classification: {r['classification']}")
        details.append(f"- Checked split: {r['checked_split']}")
        details.append(f"- Sample rows inspected: {r['sample_size']}")
        details.append(f"- License: {r['license']}")
        details.append(f"- License status: {r['criteria']['license_status']}")
        if r.get("load_error"):
            details.append(f"- Load issue: {r['load_error']}")
        details.append(f"- Notes: {r['notes']}")
        if r["dataset"] == "openai/webgpt_comparisons" and r.get("webgpt_probe"):
            wp = r["webgpt_probe"]
            details.append(
                "- normal load_dataset(\"openai/webgpt_comparisons\") is blocked by the legacy dataset-script restriction"
            )
            if wp.get("parquet_probe_status") == "success":
                details.append(f"- Parquet sampling works using path: {wp['parquet_path']}")
            else:
                details.append(f"- Parquet sampling failed for path: {wp['parquet_path']}")
            details.append("- WebGPT is conceptually strong for Reasoning-First RAG")
            details.append("- DPO mapping is feasible as:")
            details.append("  - prompt = question.full_text")
            details.append("  - chosen = answer_0 if score_0 > score_1 else answer_1")
            details.append("  - rejected = answer_1 if score_0 > score_1 else answer_0")
            details.append("  - exclude ties where score_0 == score_1")
        details.append("")

    warmup = [
        "## Non-DPO Warmup Datasets (Documented, Not Classified as DPO)",
    ]
    for d in WARMUP_DATASETS:
        warmup.append(f"- {d['name']}: {d['use']}")

    footer = [
        "",
        "## Output Artifacts",
        f"- JSON: {output_json.name}",
    ]

    return "\n".join(header + lines + details + warmup + footer) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DPO suitability for selected datasets")
    parser.add_argument("--sample-size", type=int, default=32, help="Rows to inspect per dataset")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode for lightweight checks")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("references/dpo_suitability"),
        help="Directory for generated report artifacts",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "dpo_suitability_report.json"
    output_md = args.output_dir / "dpo_suitability_report.md"

    results = [_evaluate_dataset(spec, args.sample_size, args.streaming) for spec in DPO_DATASETS]

    command_used = (
        f".venv/bin/python scripts/dpo_data/evaluate_dpo_suitability.py "
        f"--sample-size {args.sample_size} "
        f"{'--streaming ' if args.streaming else ''}"
        f"--output-dir {args.output_dir}"
    ).strip()

    run_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": args.sample_size,
        "streaming": args.streaming,
        "output_dir": str(args.output_dir),
        "command_from_repo_root": command_used,
    }

    payload = {
        "run_metadata": run_metadata,
        "criteria": [
            "prompt_context",
            "pair_constructable",
            "chosen_rejected_signal",
            "rag_relevance",
            "license_risk",
            "license_status",
            "implementation_effort",
            "llamafactory_compat",
        ],
        "datasets": sorted(results, key=lambda x: x["priority"]),
        "warmup_datasets": WARMUP_DATASETS,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = _to_markdown(results, output_json)
    metadata_lines = [
        "## Run Metadata",
        f"- Timestamp (UTC): {run_metadata['timestamp_utc']}",
        f"- Sample size: {run_metadata['sample_size']}",
        f"- Streaming mode: {run_metadata['streaming']}",
        f"- Output directory: {run_metadata['output_dir']}",
        f"- Command (repo root): {run_metadata['command_from_repo_root']}",
        "",
    ]
    output_md.write_text("\n".join(metadata_lines) + markdown, encoding="utf-8")

    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
