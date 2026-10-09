#!/bin/bash
# Usage: sbatch slurm/eval.sbatch <game> <run-name> [extra evaluate.py args...]
#   Watches checkpoints/<run>/ and evaluates each new checkpoint (CPU only). Cancel it when training ends.
#SBATCH --partition=pleiades
#SBATCH --nodelist=pleiades-1-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=fb9-eval
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
GAME="$1"; RUN="$2"; shift 2
echo "job $SLURM_JOB_ID on $(hostname), game=$GAME run=$RUN"
container/run.sh python fb9/evaluate.py --game "$GAME" --run "$RUN" --watch "$@"
