#!/bin/bash
# In-situ split-conformal recalibration on NOAA Hake EK80 files.
#
# Submit:
#   sbatch code/validation/slurm/run_conformal_hake_recalibration.sh
#
# Extra arguments are forwarded to generate_conformal_hake_recalibration.py.

#SBATCH --job-name=conformal_hake_recalibration
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/validation/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/validation/slurm/logs/%x_%j.err

set -euo pipefail
module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"
mkdir -p code/validation/slurm/logs

echo "===================================================================="
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname)"
echo "CPUs/Mem: ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?}M"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "Start:  $(date)"
echo "===================================================================="

python -u code/validation/generate_conformal_hake_recalibration.py "$@"

echo "===================================================================="
echo "Finished: $(date)"
echo "===================================================================="
