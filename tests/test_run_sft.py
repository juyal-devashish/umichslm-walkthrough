"""Tests for the SFT launcher config resolution."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_sft import resolve_sft_config, run_sft


def test_resolve_sft_config_writes_max_samples():
    config = resolve_sft_config(
        dataset="umichslm_sft_train",
        max_samples=123,
        output_dir="/tmp/out",
        logging_dir="/tmp/log",
    )

    assert config["dataset"] == "umichslm_sft_train"
    assert config["max_samples"] == 123
    assert config["output_dir"] == "/tmp/out"
    assert config["logging_dir"] == "/tmp/log"


def test_run_sft_dry_run_does_not_launch_training(tmp_path, monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("run_sft.subprocess.run", fake_run)

    config_path = run_sft(
        dataset="umichslm_sft_train",
        max_samples=5,
        output_dir=str(tmp_path / "out"),
        logging_dir=str(tmp_path / "log"),
        generated_config_dir=str(tmp_path / "configs"),
        dry_run=True,
    )

    assert calls == []
    assert config_path.exists()
    assert "max_samples: 5" in config_path.read_text()


def test_run_sft_launches_wrapper_with_generated_config(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check, env):
        calls.append((cmd, check, env))

    monkeypatch.setattr("run_sft.subprocess.run", fake_run)

    config_path = run_sft(
        dataset="umichslm_sft_train",
        output_dir=str(tmp_path / "out"),
        logging_dir=str(tmp_path / "log"),
        generated_config_dir=str(tmp_path / "configs"),
        keep_config=True,
    )

    assert calls[0][0] == ["python", "scripts/llamafactory_train.py", "train", str(config_path)]
    assert calls[0][1] is True
    assert config_path.exists()


def test_unsupported_dataset_fails_clearly():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        resolve_sft_config(dataset="alpaca_gpt4_en")
