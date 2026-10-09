#!/bin/bash
# A4 speed benchmark: env throughput at several worker counts + short trainer runs.
# Usage: sbatch slurm/bench.sbatch [game]
#SBATCH --partition=pleiades
#SBATCH --nodelist=pleiades-1-3
#SBATCH --gpus=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=fb9-bench
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
GAME="${1:-surround}"
echo "job $SLURM_JOB_ID on $(hostname), cpus=$SLURM_CPUS_PER_TASK"
container/run.sh python scripts/bench_env.py --game "$GAME" --workers 8 16 32 46 --games-per-worker 2
container/run.sh python scripts/bench_env.py --game "$GAME" --workers 16 46 --games-per-worker 4
for W in 16 30 46; do
  container/run.sh python train.py --game "$GAME" --num-workers $W --num-games $(( W * 3 > 64 ? W * 3 : 64 )) \
    --total-samples 3000000 --run-name "bench_${GAME}_w${W}" 2>&1 | tail -5
done
