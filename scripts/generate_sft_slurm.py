#!/usr/bin/env python3
"""Generate SLURM scripts for UMichSLM SFT workflows."""

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


def _flag(args: list[Any], name: str, value: Any) -> None:
    if value not in (None, ""):
        args.extend([name, value])


def build_sft_command(run: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Build the SFT launcher command for one run."""
    sft = config.get("sft", {})
    wandb = config.get("wandb", {})
    cmd: list[Any] = ["python", "scripts/run_sft.py"]

    for name, value in [
        ("--dataset", run.get("dataset", sft.get("dataset"))),
        ("--base_config", run.get("base_config", sft.get("base_config"))),
        ("--model_name_or_path", run.get("model_name_or_path", sft.get("model_name_or_path"))),
        ("--eval_dataset", run.get("eval_dataset", sft.get("eval_dataset"))),
        ("--max_samples", run.get("max_samples", sft.get("max_samples"))),
        ("--learning_rate", run.get("learning_rate", sft.get("learning_rate"))),
        ("--num_train_epochs", run.get("num_train_epochs", sft.get("num_train_epochs"))),
        (
            "--per_device_train_batch_size",
            run.get("per_device_train_batch_size", sft.get("per_device_train_batch_size")),
        ),
        (
            "--gradient_accumulation_steps",
            run.get("gradient_accumulation_steps", sft.get("gradient_accumulation_steps")),
        ),
        ("--save_steps", run.get("save_steps", sft.get("save_steps"))),
        ("--eval_steps", run.get("eval_steps", sft.get("eval_steps"))),
        ("--logging_steps", run.get("logging_steps", sft.get("logging_steps"))),
        ("--output_suffix", run["slug"]),
        ("--output_dir", run["output_dir"]),
        ("--logging_dir", f"{run['output_dir']}/tb"),
        ("--run_name", run["slug"]),
        ("--wandb_project", run.get("wandb_project", wandb.get("project"))),
        ("--wandb_entity", run.get("wandb_entity", wandb.get("entity"))),
        ("--generated_config_dir", config.get("generated_config_dir", "generated-slurm/sft-configs")),
    ]:
        _flag(cmd, name, value)

    return [str(part) for part in cmd]


def make_sft_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand SFT sweep config into concrete run dictionaries."""
    sft = config.get("sft", {})
    base = {
        "dataset": sft.get("dataset"),
        "learning_rate": sft.get("learning_rate"),
        "model_name_or_path": sft.get("model_name_or_path"),
    }
    runs = expand_sweep_grid(base, config.get("sweep"))
    output_root = config.get("outputs", {}).get("root", "./outputs/sft")
    prefix = config.get("outputs", {}).get("run_prefix", "sft")

    for idx, run in enumerate(runs):
        slug = make_run_slug(
            [
                prefix,
                run.get("dataset"),
                f"lr{run.get('learning_rate')}",
                Path(str(run.get("model_name_or_path", "model"))).name,
            ]
        )
        if len(runs) > 1:
            slug = f"{idx:03d}-{slug}"
        run["slug"] = slug
        run["output_dir"] = f"{output_root}/{slug}"
    return runs


def render_sft_script(config: dict[str, Any]) -> str:
    """Render a standalone or array SFT SLURM script."""
    runs = make_sft_runs(config)
    slurm = config.get("slurm", {})
    job_name = slurm.get("job_name", "umichslm-sft")
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
    body.append(f"mkdir -p {shlex.quote(config.get('outputs', {}).get('root', './outputs/sft'))}")
    body.append("")

    if is_array:
        body.append("case ${SLURM_ARRAY_TASK_ID} in")
        for idx, run in enumerate(runs):
            command = shell_join(build_sft_command(run, config))
            body.append(f"  {idx})")
            body.append(f"    RUN_SLUG={shlex.quote(run['slug'])}")
            body.append(f"    LEARNING_RATE={shlex.quote(str(run.get('learning_rate')))}")
            body.append(f"    CMD={shlex.quote(command)}")
            body.append("    ;;")
        body.append("  *) echo \"Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}\" >&2; exit 2 ;;")
        body.append("esac")
        body.append("")
        body.append('echo "Run: ${RUN_SLUG}"')
        body.append('echo "Learning rate: ${LEARNING_RATE}"')
        body.append('echo "Command: ${CMD}"')
        body.append("eval \"$CMD\"")
    else:
        command = shell_join(build_sft_command(runs[0], config))
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
    output_dir = Path(args.output_dir or config.get("output_dir", "generated-slurm/sft"))
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / config.get("script_name", "sft_workflow.sh")
    script = render_sft_script(config)

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
