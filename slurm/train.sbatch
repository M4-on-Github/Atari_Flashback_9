#!/bin/bash
# Usage: sbatch slurm/train.sbatch <game> [extra train.py args...]
#   e.g. sbatch slurm/train.sbatch surround --run-name surround_v1
#        sbatch --time=3-00:00:00 slurm/train.sbatch combat --total-samples 150000000 --run-name combat_v1
#SBATCH --partition=pleiades
#SBATCH --nodelist=pleiades-1-3
#SBATCH --gpus=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=fb9-train
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
GAME="$1"; shift
echo "job $SLURM_JOB_ID on $(hostname), cpus=$SLURM_CPUS_PER_TASK, gpu=${CUDA_VISIBLE_DEVICES:-none}, game=$GAME"
container/run.sh python train.py --game "$GAME" --num-workers $((SLURM_CPUS_PER_TASK - 2)) "$@"
