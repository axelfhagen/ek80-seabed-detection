#!/bin/bash
#SBATCH --job-name=test-raw-echo
#SBATCH --account=share-ie-imf
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/test_raw_echo_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/test_raw_echo_%j.err

# Generate echogram PNGs for all raw files in test_raw (CPU only, no GPU).
# Submit with: sbatch run_test_raw_echograms.sh

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
DATA_ROOT="${DATA_ROOT:-/cluster/work/axelfh}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"

echo "============================================================"
echo "Test raw echograms | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "============================================================"

python -c "
import sys, os
sys.path.insert(0, 'code')
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

from infer_test_raw import generate_echogram

raw_dir = '${DATA_ROOT}/test_raw'
out_dir = 'code/plots_test_raw/echograms'
os.makedirs(out_dir, exist_ok=True)

raw_files = sorted(Path(raw_dir).rglob('*.raw'))
raw_files = [f for f in raw_files if not f.name.startswith('._')]
print(f'Found {len(raw_files)} raw files')

for i, rf in enumerate(raw_files, 1):
    print(f'[{i}/{len(raw_files)}] {rf.name}')
    try:
        generate_echogram(str(rf), out_dir)
    except Exception as e:
        print(f'  ERROR: {e}')

print('Done!')
"

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
