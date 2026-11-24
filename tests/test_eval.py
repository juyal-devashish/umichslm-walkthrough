"""Tests for scripts/run_eval.py — evaluate() and helpers.

Usage:
    pytest tests/test_eval.py -v
    pytest tests/test_eval.py::test_evaluate_task_structure -v
    pytest tests/test_eval.py -k "gsm8k" -v

"""

import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow importing from scripts/ without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_eval import evaluate, print_res_table, _primary_metric


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2_model_and_tokenizer():
    """Load gpt2 once for all tests in this module."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


def test_primary_metric():
    results = {"some_other_metric": 0.7}
    key, val = _primary_metric(results, ["accuracy,none"])
    assert key == "unknown"
    assert val == 0.0


def test_print_res_table_returns_dict():
    results = {
        "results": {
            "hellaswag": {"accuracy,none": 0.312},
        }
    }
    out = print_res_table(results, metric_priority=["accuracy,none"])
    assert isinstance(out, dict)
    assert "hellaswag" in out
    assert out["hellaswag"] == pytest.approx(0.312, abs=1e-4)


def test_print_res_table_empty():
    """Should return None (early return) when results dict is empty."""
    out = print_res_table({"results": {}}, metric_priority=["accuracy,none"])
    assert out is None


# ---------------------------------------------------------------------------
# Integration tests for evaluate() — one test per benchmark task
# ---------------------------------------------------------------------------

# (task_name, num_fewshot, substring that must appear in at least one metric key)
TASK_PARAMS = [
    ("hellaswag", 0, "acc"),
    ("arc_challenge", 0, "acc"),
    ("mmlu", 5, "acc"),
    ("gsm8k", 4, "exact_match"),
    ("truthfulqa_mc1", 0, "acc"),
    ("truthfulqa_mc2", 0, "acc"),
]


@pytest.mark.slow
@pytest.mark.parametrize("task,fewshot,metric_substr", TASK_PARAMS)
def test_evaluate_task_structure(gpt2_model_and_tokenizer, tmp_path, task, fewshot, metric_substr):
    """Each task should return a result dict with at least one expected metric
    and write a valid JSONL record."""
    model, tokenizer = gpt2_model_and_tokenizer
    output_file = tmp_path / f"eval_{task}.jsonl"

    results = evaluate(
        model=model,
        tokenizer=tokenizer,
        seed=42,
        batch_size=1,
        max_length=1024,
        tasks=[(task, fewshot)],
        output_path=str(output_file),
    )

    # --- result structure ---
    assert task in results, f"Task '{task}' missing from results"
    task_metrics = results[task]
    assert isinstance(task_metrics, dict), f"Expected dict, got {type(task_metrics)}"
    matching = [k for k in task_metrics if metric_substr in k.lower()]
    assert matching, f"No metric containing '{metric_substr}' in {list(task_metrics.keys())}"

    # --- JSONL output ---
    assert output_file.exists(), "JSONL file was not created"
    lines = output_file.read_text().strip().splitlines()
    assert len(lines) == 1, "Expected exactly one JSONL record"
    record = json.loads(lines[0])
    assert "timestamp" in record
    assert "elapsed_secs" in record
    assert task in record["results"]


@pytest.mark.slow
def test_evaluate_appends_jsonl(gpt2_model_and_tokenizer, tmp_path):
    """Calling evaluate() twice should append two lines to the JSONL file."""
    model, tokenizer = gpt2_model_and_tokenizer
    output_file = tmp_path / "eval_results.jsonl"

    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        seed=42,
        batch_size=1,
        max_length=1024,
        tasks=[("hellaswag", 0)],
        output_path=str(output_file),
    )

    evaluate(**kwargs)
    evaluate(**kwargs)

    lines = output_file.read_text().strip().splitlines()
    assert len(lines) == 2, "Expected two JSONL records after two evaluate() calls"
    for line in lines:
        record = json.loads(line)
        assert "results" in record
