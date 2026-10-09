#!/bin/bash
# Usage: sbatch slurm/league.sbatch --init-ckpt <ckpt> --run-name <name> [extra league.py args...]
#   e.g. sbatch slurm/league.sbatch --init-ckpt checkpoints/surround_v4/latest.pt --run-name surround_league_v1
#SBATCH --partition=pleiades
#SBATCH --nodelist=pleiades-1-3
#SBATCH --gpus=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=fb9-league
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "job $SLURM_JOB_ID on $(hostname), cpus=$SLURM_CPUS_PER_TASK, gpu=${CUDA_VISIBLE_DEVICES:-none}"
container/run.sh python league.py --num-workers $((SLURM_CPUS_PER_TASK - 2)) "$@"
