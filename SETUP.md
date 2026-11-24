This guide helps you set up the environment for running scaling law experiments on Great Lakes.

Create Conda Environment

```bash
# Load conda if not already available
module load python/anaconda

# Create environment from file
conda env create -f environment.yml

# Activate
conda activate umich-slm

# Install the umich-slm package
pip install -e .
```

> **Note:** If `conda env create` fails due to platform-specific packages, try:
> ```bash
> conda env create -f environment.yml
> ```

Results directory:

```bash
export SCRATCH="/path/to/data"
mkdir -p "$SCRATCH/results/$USER"
```

Step 1: Prepare Data

```bash
sbatch local-slurm/prepare_data.sh
```

Step 2: Run Scaling Experiment

Edit `run_scaling.sh` to set your desired model config, then:

```bash
sbatch local-slurm/run_scaling.sh
```

Pretraining and SFT Workflows

For repeatable pretraining and SFT jobs, use the config-driven SLURM generators
instead of hand-editing batch scripts. See `docs/slurm_workflows.md` for the
full collaborator guide.

The scripts already set `HF_HOME` to shared scratch. If you hit home directory quota issues elsewhere, ensure this is set:

```bash
export HF_HOME="/path/to/data"
```
