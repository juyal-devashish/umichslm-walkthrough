#!/usr/bin/env python3
"""Shared helpers for rendering UMichSLM SLURM batch scripts."""

from __future__ import annotations

import argparse
import ast
import itertools
import shlex
import subprocess
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal local envs
    yaml = None


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return an empty dict for empty files."""
    with open(path) as f:
        text = f.read()
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _load_simple_yaml(text)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            inner = value[1:-1].strip()
            return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by workflow configs when PyYAML is absent."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if "#" in line:
            line = line.split("#", 1)[0].rstrip()
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = _parse_scalar(raw_value)

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        parent[key] = value
        if isinstance(value, dict) and raw_value.strip() == "":
            stack.append((indent, value))

    return root


def ensure_list(value: Any) -> list[Any]:
    """Normalize a scalar/list config value to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def shell_join(parts: list[Any]) -> str:
    """Quote a command list for Bash."""
    return " ".join(shlex.quote(str(part)) for part in parts)


def make_run_slug(parts: list[Any]) -> str:
    """Build a filesystem-friendly run slug from non-empty values."""
    raw = "-".join(str(part) for part in parts if part not in (None, ""))
    safe = []
    for char in raw:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("-")
    return "".join(safe).strip("-")


def expand_sweep_grid(base: dict[str, Any], sweep: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Expand a simple Cartesian product sweep into run dictionaries."""
    if not sweep:
        return [dict(base)]

    keys = list(sweep.keys())
    values = [ensure_list(sweep[key]) for key in keys]
    runs = []
    for combo in itertools.product(*values):
        run = dict(base)
        for key, value in zip(keys, combo):
            run[key] = value
        runs.append(run)
    return runs


def render_sbatch_header(
    slurm: dict[str, Any],
    job_name: str,
    output_path: str,
    error_path: str | None = None,
    array: str | None = None,
) -> str:
    """Render the #SBATCH header shared by generated workflow scripts."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output_path}",
    ]
    if error_path:
        lines.append(f"#SBATCH --error={error_path}")

    fields = [
        ("partition", "partition"),
        ("time", "time"),
        ("nodes", "nodes"),
        ("ntasks_per_node", "ntasks-per-node"),
        ("cpus_per_task", "cpus-per-task"),
        ("mem", "mem"),
        ("mem_per_cpu", "mem-per-cpu"),
        ("account", "account"),
        ("qos", "qos"),
        ("gpus", "gpus"),
        ("gres", "gres"),
        ("mail_user", "mail-user"),
        ("mail_type", "mail-type"),
    ]
    for key, directive in fields:
        value = slurm.get(key)
        if value not in (None, ""):
            lines.append(f"#SBATCH --{directive}={value}")
    if array:
        lines.append(f"#SBATCH --array={array}")
    return "\n".join(lines)


def render_environment(config: dict[str, Any]) -> str:
    """Render module loads, conda activation, project cd, and env exports."""
    slurm = config.get("slurm", {})
    env_vars = config.get("env", {})
    required_env = ensure_list(config.get("required_env"))
    project_dir = config.get("project_dir", "$HOME/umich-slm")
    conda_env_name = slurm.get("conda_env_name", config.get("conda_env_name"))
    conda_env_path = slurm.get("conda_env_path", config.get("conda_env_path"))
    legacy_conda_env = slurm.get("conda_env", config.get("conda_env"))
    log_dir = slurm.get("log_dir", "slurm-logs")
    runtime_log_dir = "slurm-logs" if "%" in str(log_dir) else str(log_dir)

    lines = [
        "",
        "set -euo pipefail",
        "",
        'echo "Host:  $(hostname)"',
        'echo "Start: $(date)"',
        "",
    ]
    for module in ensure_list(slurm.get("modules")):
        lines.append(f"module load {module}")
    if slurm.get("modules"):
        lines.append("")

    if conda_env_path and conda_env_name:
        raise ValueError("Set only one of conda_env_path or conda_env_name")

    conda_target = conda_env_path or conda_env_name or legacy_conda_env or "umich-slm"
    if conda_target:
        lines.extend(
            [
                'source "$(conda info --base)/etc/profile.d/conda.sh"',
                f"conda activate {shlex.quote(str(conda_target))}",
                "",
            ]
        )

    lines.extend(
        [
            f'export PROJECT_DIR="{project_dir}"',
            'cd "$PROJECT_DIR"',
            f"mkdir -p {shlex.quote(runtime_log_dir)}",
        ]
    )

    for key, value in env_vars.items():
        lines.append(f"export {key}={shlex.quote(str(value))}")
    for key in required_env:
        lines.append(f': "${{{key}:?Set {key} before submitting this job}}"')
        lines.append(f"export {key}")
    lines.append("")
    return "\n".join(lines)


def submit_script(path: Path, dry_run: bool = False) -> str | None:
    """Submit a rendered script with sbatch."""
    if dry_run:
        print(f"[DRY RUN] Would submit: sbatch {path}")
        return "DRY_RUN"
    result = subprocess.run(["sbatch", str(path)], check=True, capture_output=True, text=True)
    print(result.stdout.strip())
    return result.stdout.strip().split()[-1]


def add_render_submit_args(parser: argparse.ArgumentParser) -> None:
    """Add common render/submit CLI arguments."""
    parser.add_argument("--config", required=True, help="Workflow YAML config")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for generated SLURM scripts (defaults to config output_dir)",
    )
    parser.add_argument("--submit", action="store_true", help="Submit rendered scripts with sbatch")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print rendered scripts and submission commands without submitting",
    )
