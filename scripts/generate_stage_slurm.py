"""Emit a 5-job SLURM chain for a layer-stacked pretraining pilot.

The chain is:

    sbatch                                stage_1.sbatch     -> JOB1
    sbatch --dependency=afterok:JOB1      transition_1to2.sbatch  -> JOB2
    sbatch --dependency=afterok:JOB2      stage_2.sbatch     -> JOB3
    sbatch --dependency=afterok:JOB3      transition_2to3.sbatch  -> JOB4
    sbatch --dependency=afterok:JOB4      stage_3.sbatch     -> JOB5

Stage sbatch jobs invoke ``llamafactory-cli train <yaml>`` with the per-stage
YAML produced by :mod:`scripts.generate_stage_yaml`. Transition jobs invoke
``python -m umich_slm.staging.transition`` against the corresponding StageSpec
JSON.

Usage::

    python scripts/generate_stage_slurm.py \\
        --pilot stack_50m_b_dn \\
        --yaml-dir configs/llama_factory/stacking \\
        --output-dir ./outputs/stacking/stack_50m_b_dn \\
        --stage1-spec configs/torchtitan/stacking/specs/50m_b_dn_stage1.json \\
        --stage2-spec configs/torchtitan/stacking/specs/50m_b_dn_stage2.json \\
        --stage3-spec configs/torchtitan/stacking/specs/50m_b_dn_stage3.json \\
        --partition gpu --gres gpu:A6000:4 --time 24:00:00 \\
        --slurm-out generated-slurm/stacking

The script writes 5 sbatch files plus a top-level ``submit.sh`` that
``sbatch``-submits them with the right dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

STAGE_SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={pilot}_stage{stage_index}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --output={log_dir}/{pilot}_stage{stage_index}-%j.log

set -euo pipefail
cd {repo_root}

llamafactory-cli train {yaml_path}
"""

TRANSITION_SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={pilot}_trans{from_index}to{to_index}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --time={time_short}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --output={log_dir}/{pilot}_trans{from_index}to{to_index}-%j.log

set -euo pipefail
cd {repo_root}

bash {transition_script_path}
"""

SUBMIT_SH_TEMPLATE = """\
#!/bin/bash
# Auto-generated SLURM submission chain for {pilot}.
# Submits 5 dependent jobs: stage1 -> trans1->2 -> stage2 -> trans2->3 -> stage3.
set -euo pipefail

J1=$(sbatch --parsable {stage1})
echo "stage1 job id: $J1"

T12=$(sbatch --parsable --dependency=afterok:$J1 {trans12})
echo "trans 1->2 job id: $T12"

J2=$(sbatch --parsable --dependency=afterok:$T12 {stage2})
echo "stage2 job id: $J2"

T23=$(sbatch --parsable --dependency=afterok:$J2 {trans23})
echo "trans 2->3 job id: $T23"

J3=$(sbatch --parsable --dependency=afterok:$T23 {stage3})
echo "stage3 job id: $J3"

echo "All 5 jobs submitted. Tail logs in {log_dir}/."
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", required=True)
    p.add_argument("--yaml-dir", required=True,
                   help="Directory with the per-stage YAMLs (output of generate_stage_yaml.py).")
    p.add_argument("--output-dir", required=True,
                   help="Top-level pilot output dir.")
    p.add_argument("--partition", default="gpu")
    p.add_argument("--gres", default="gpu:A6000:4")
    p.add_argument("--time", default="24:00:00", help="Wall time per stage.")
    p.add_argument("--time-short", default="2:00:00",
                   help="Wall time per transition (probes + grow + save).")
    p.add_argument("--cpus", type=int, default=16)
    p.add_argument("--slurm-out", required=True,
                   help="Output directory for the generated sbatch files.")
    p.add_argument("--repo-root", default=str(Path.cwd()),
                   help="Repository root passed via cd in the sbatch jobs.")
    p.add_argument("--log-dir", default=None,
                   help="Where SLURM should write job logs; defaults to <output-dir>/logs.")
    args = p.parse_args(argv)

    yaml_dir = Path(args.yaml_dir)
    slurm_out = Path(args.slurm_out)
    slurm_out.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else (Path(args.output_dir) / "logs")

    # Stage sbatch files.
    stage_sbatches = []
    for i in (1, 2, 3):
        sbatch_path = slurm_out / f"{args.pilot}_stage{i}.sbatch"
        sbatch_path.write_text(STAGE_SBATCH_TEMPLATE.format(
            pilot=args.pilot,
            stage_index=i,
            partition=args.partition,
            gres=args.gres,
            time=args.time,
            cpus=args.cpus,
            log_dir=log_dir,
            repo_root=args.repo_root,
            yaml_path=yaml_dir / f"{args.pilot}_stage{i}.yaml",
        ))
        sbatch_path.chmod(0o755)
        stage_sbatches.append(sbatch_path)

    # Transition sbatch files (call into the per-pilot transition script).
    trans_sbatches = []
    for (from_i, to_i) in ((1, 2), (2, 3)):
        sbatch_path = slurm_out / f"{args.pilot}_trans{from_i}to{to_i}.sbatch"
        sbatch_path.write_text(TRANSITION_SBATCH_TEMPLATE.format(
            pilot=args.pilot,
            from_index=from_i,
            to_index=to_i,
            partition=args.partition,
            gres=args.gres,
            time_short=args.time_short,
            cpus=args.cpus,
            log_dir=log_dir,
            repo_root=args.repo_root,
            transition_script_path=yaml_dir / f"{args.pilot}_transition_{from_i}to{to_i}.sh",
        ))
        sbatch_path.chmod(0o755)
        trans_sbatches.append(sbatch_path)

    # Top-level submit.sh
    submit_path = slurm_out / f"{args.pilot}_submit.sh"
    submit_path.write_text(SUBMIT_SH_TEMPLATE.format(
        pilot=args.pilot,
        stage1=stage_sbatches[0],
        stage2=stage_sbatches[1],
        stage3=stage_sbatches[2],
        trans12=trans_sbatches[0],
        trans23=trans_sbatches[1],
        log_dir=log_dir,
    ))
    submit_path.chmod(0o755)
    print(f"Wrote {len(stage_sbatches)} stage sbatch files, "
          f"{len(trans_sbatches)} transition sbatch files, and {submit_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
