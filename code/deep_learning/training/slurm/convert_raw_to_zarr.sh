#!/bin/bash
#SBATCH --job-name=raw2zarr
#SBATCH --account=share-ie-imf
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/convert_%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/convert_%x_%j.err

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
ZARR_OUT="${PROJECT}/Datasets/echopype_zarr"

RAW_DIR="${RAW_DIR:?ERROR: RAW_DIR not set}"

echo "============================================================"
echo "Raw → Zarr Conversion"
echo "  RAW_DIR:  ${RAW_DIR}"
echo "  ZARR_OUT: ${ZARR_OUT}"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "  Node:     $(hostname)"
echo "  Start:    $(date)"
echo "============================================================"

cd "$PROJECT"

python -c "
import sys
sys.path.insert(0, 'code')
from raw_sensor_integration.data_loader import convert_raw_dir
convert_raw_dir('${RAW_DIR}', '${ZARR_OUT}', skip_existing=True)
"

echo "Finished: $(date)"
