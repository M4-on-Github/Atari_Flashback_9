#!/bin/bash
# Usage: sbatch slurm/probe.sbatch [bc_probe args...]
#   e.g. sbatch slurm/probe.sbatch --out runs/bc_probe/v1.json
#SBATCH --partition=pleiades
#SBATCH --nodelist=pleiades-1-3
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=fb9-probe
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "job $SLURM_JOB_ID on $(hostname), cpus=$SLURM_CPUS_PER_TASK, gpu=${CUDA_VISIBLE_DEVICES:-none}"
container/run.sh python tools/bc_probe.py --workers $((SLURM_CPUS_PER_TASK - 2)) "$@"
