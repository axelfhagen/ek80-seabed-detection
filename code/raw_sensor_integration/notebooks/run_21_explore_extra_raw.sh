#!/bin/bash
#SBATCH --job-name=explore-extra-raw
#SBATCH --account=share-ie-imf
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/raw_sensor_integration/notebooks/logs/explore_extra_raw_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/raw_sensor_integration/notebooks/logs/explore_extra_raw_%j.err

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
NOTEBOOK_DIR="${PROJECT}/code/raw_sensor_integration/notebooks"
LOG_DIR="${NOTEBOOK_DIR}/logs"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
mv "${LOG_DIR}/explore_extra_raw_${SLURM_JOB_ID}.out" "${LOG_DIR}/explore_extra_raw_${TIMESTAMP}_${SLURM_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/explore_extra_raw_${SLURM_JOB_ID}.err" "${LOG_DIR}/explore_extra_raw_${TIMESTAMP}_${SLURM_JOB_ID}.err" 2>/dev/null || true

cd "$NOTEBOOK_DIR"

echo "============================================================"
echo "Running notebook: 21_explore_extra_raw.ipynb"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start:  $(date)"
echo "============================================================"

jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=14400 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output "21_explore_extra_raw_executed.ipynb" \
    "21_explore_extra_raw.ipynb"

echo "Exit code: $?"
echo "Finished: $(date)"
