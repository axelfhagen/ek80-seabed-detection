#!/bin/bash
#SBATCH --job-name=r2z_single
#SBATCH --account=share-ie-imf
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/convert_r2z_%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/convert_r2z_%x_%j.err

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
ZARR_OUT="${PROJECT}/Datasets/echopype_zarr"
RAW_FILE="${RAW_FILE:?ERROR: RAW_FILE not set}"

echo "============================================================"
echo "Raw → Zarr (single file)"
echo "  RAW_FILE: ${RAW_FILE}"
echo "  ZARR_OUT: ${ZARR_OUT}"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "  Node:     $(hostname)"
echo "  Start:    $(date)"
echo "============================================================"

cd "$PROJECT"

python -c "
import sys
sys.path.insert(0, 'code')
from pathlib import Path
from raw_sensor_integration.data_loader import raw_to_zarr

raw = Path('${RAW_FILE}')
out = Path('${ZARR_OUT}') / raw.with_suffix('.zarr').name
if out.exists():
    print(f'SKIP: {out} already exists')
else:
    raw_to_zarr(raw, out)
"

echo "Finished: $(date)"
