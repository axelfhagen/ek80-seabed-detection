#!/bin/bash
set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env
export MPLCONFIGDIR=/tmp/matplotlib-${USER}
mkdir -p "$MPLCONFIGDIR"

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
cd "$REPO_ROOT"
python -u code/experiments/hake_finetune/run_hake_finetune.py \
    --mode conformal \
    --resume-existing \
    --device cpu
