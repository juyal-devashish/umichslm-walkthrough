#!/usr/bin/env python3
"""Generate SLURM scripts for UMichSLM pretraining workflows."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

from slurm_common import (
    add_render_submit_args,
    expand_sweep_grid,
    load_yaml_config,
    make_run_slug,
    render_environment,
    render_sbatch_header,
    shell_join,
    submit_script,
)


def _flag(args: list[str], name: str, value: Any) -> None:
    if value not in (None, ""):
        args.extend([name, value])


def build_pretrain_command(run: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Build the python command for a single pretraining run."""
    training = config.get("training", {})
    data = config.get("data", {})
    wandb = config.get("wandb", {})

    cmd: list[Any] = ["python", "scripts/pretrain_fineweb.py"]
    for name, value in [
        ("--size", run.get("size")),
        ("--dataset", run.get("dataset", data.get("dataset"))),
        ("--dataset_dir", run.get("dataset_dir", data.get("dataset_dir"))),
        ("--eval_dataset", run.get("eval_dataset", data.get("eval_dataset"))),
        ("--target_tokens_b", run.get("target_tokens_b", training.get("target_tokens_b"))),
        ("--batch_size", run.get("batch_size", training.get("batch_size"))),
        (
            "--gradient_accumulation_steps",
            run.get(
                "gradient_accumulation_steps",
                training.get("gradient_accumulation_steps"),
            ),
        ),
        ("--learning_rate", run.get("learning_rate", training.get("learning_rate"))),
        ("--save_steps", run.get("save_steps", training.get("save_steps"))),
        ("--eval_steps", run.get("eval_steps", training.get("eval_steps"))),
        ("--logging_steps", run.get("logging_steps", training.get("logging_steps"))),
        ("--cutoff_len", run.get("cutoff_len", training.get("cutoff_len"))),
        ("--wandb_project", run.get("wandb_project", wandb.get("project"))),
        ("--wandb_entity", run.get("wandb_entity", wandb.get("entity"))),
        ("--run_name", run.get("run_name")),
        ("--output_dir", run["output_dir"]),
    ]:
        _flag(cmd, name, value)

    if run.get("do_eval", data.get("do_eval", True)):
        cmd.append("--do_eval")
    if training.get("config_only"):
        cmd.append("--config_only")
    if training.get("dry_run"):
        cmd.append("--dry_run")

    # Token-Superposition Training (TST). Surfaced as ``training.tst.*`` /
    # sweepable as ``tst_enabled`` / ``tst_bag_size`` / ``tst_recovery_ratio``.
    tst_cfg = training.get("tst", {}) if isinstance(training.get("tst"), dict) else {}
    tst_enabled = run.get("tst_enabled", tst_cfg.get("enabled", False))
    if tst_enabled:
        cmd.append("--tst_enabled")
        _flag(cmd, "--tst_bag_size", run.get("tst_bag_size", tst_cfg.get("bag_size")))
        _flag(
            cmd,
            "--tst_recovery_ratio",
            run.get("tst_recovery_ratio", tst_cfg.get("recovery_ratio")),
        )
        _flag(
            cmd,
            "--tst_loss_weighting",
            run.get("tst_loss_weighting", tst_cfg.get("loss_weighting")),
        )
    return [str(part) for part in cmd]


def make_pretrain_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand configured pretraining runs and fill derived names/paths."""
    training = config.get("training", {})
    base = {
        "size": training.get("size"),
        "target_tokens_b": training.get("target_tokens_b"),
        "learning_rate": training.get("learning_rate"),
    }
    runs = expand_sweep_grid(base, config.get("sweep"))
    output_root = config.get("outputs", {}).get("root", "./outputs/pretrain")
    prefix = config.get("outputs", {}).get("run_prefix", "pretrain")

    for idx, run in enumerate(runs):
        slug = make_run_slug(
            [
                prefix,
                run.get("size"),
                f"{run.get('target_tokens_b')}Btok",
                f"lr{run.get('learning_rate')}",
            ]
        )
        if len(runs) > 1:
            slug = f"{idx:03d}-{slug}"
        run["slug"] = slug
        run.setdefault("run_name", slug)
        run["output_dir"] = f"{output_root}/{slug}"
    return runs


def render_pretrain_script(config: dict[str, Any]) -> str:
    """Render a standalone or array pretraining SLURM script."""
    runs = make_pretrain_runs(config)
    slurm = config.get("slurm", {})
    job_name = slurm.get("job_name", "umichslm-pretrain")
    log_dir = slurm.get("log_dir", "/home/%u/umich-slm/slurm-logs")
    is_array = len(runs) > 1
    suffix = "%A_%a" if is_array else "%j"

    header = render_sbatch_header(
        slurm,
        job_name,
        f"{log_dir}/%x-{suffix}.out",
        f"{log_dir}/%x-{suffix}.err",
        array=f"0-{len(runs) - 1}" if is_array else None,
    )
    body = [header, render_environment(config)]

    body.append(f"mkdir -p {shlex.quote(config.get('outputs', {}).get('root', './outputs/pretrain'))}")
    body.append("")

    if is_array:
        body.append("case ${SLURM_ARRAY_TASK_ID} in")
        for idx, run in enumerate(runs):
            command = shell_join(build_pretrain_command(run, config))
            body.append(f"  {idx})")
            body.append(f"    RUN_SLUG={shlex.quote(run['slug'])}")
            body.append(f"    CMD={shlex.quote(command)}")
            body.append("    ;;")
        body.append("  *) echo \"Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}\" >&2; exit 2 ;;")
        body.append("esac")
        body.append("")
        body.append('echo "Run: ${RUN_SLUG}"')
        body.append('echo "Command: ${CMD}"')
        body.append("eval \"$CMD\"")
    else:
        command = shell_join(build_pretrain_command(runs[0], config))
        body.append(f"echo {shlex.quote('Run: ' + runs[0]['slug'])}")
        body.append(f"echo {shlex.quote('Command: ' + command)}")
        body.append(command)

    body.extend(
        [
            "",
            "EXIT_CODE=$?",
            'echo "Done:  $(date)"',
            'echo "Exit:  ${EXIT_CODE}"',
            "exit ${EXIT_CODE}",
            "",
        ]
    )
    return "\n".join(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_render_submit_args(parser)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = Path(args.output_dir or config.get("output_dir", "generated-slurm/pretrain"))
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / config.get("script_name", "pretrain_workflow.sh")
    script = render_pretrain_script(config)

    if args.dry_run:
        print(script)
    else:
        script_path.write_text(script)
        script_path.chmod(0o755)
        print(f"Wrote {script_path}")

    if args.submit:
        if args.dry_run:
            submit_script(script_path, dry_run=True)
        else:
            submit_script(script_path)


if __name__ == "__main__":
    main()
