#!/bin/bash
set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env
export MPLCONFIGDIR=/tmp/matplotlib-${USER}
mkdir -p "$MPLCONFIGDIR"

cd /cluster/work/axelfh/Master
python -u code/experiments/hake_finetune/summarize_results.py
