#!/bin/bash
# Conformal-prediction exchangeability checks across survey campaigns.
# Runs U-Net inference per survey, caches per-survey bottom-depth nonconformity
# scores, then runs three exchangeability checks (two-sample score tests, a
# Vovk conformal test-martingale, and stratified coverage) and writes all
# figures to Figures/conformal/exchangeability/.
#
# Submit:    sbatch code/validation/slurm/run_conformal_exchangeability.sh
# Override:  sbatch code/validation/slurm/run_conformal_exchangeability.sh --max-files-per-survey 25
#
#SBATCH --job-name=conformal_exch
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/validation/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/validation/slurm/logs/%x_%j.err

set -euo pipefail
module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

MAX_FILES=15
EXTRA_ARGS=("$@")

echo "===================================================================="
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:       $(hostname)"
echo "CPUs/Mem:   ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?}M"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "Max files/survey: ${MAX_FILES}   extras=${EXTRA_ARGS[*]:-}"
echo "Start:      $(date)"
echo "===================================================================="

python -u code/validation/conformal_exchangeability.py \
    --max-files-per-survey "$MAX_FILES" \
    "${EXTRA_ARGS[@]}"

echo "===================================================================="
echo "Finished:   $(date)"
echo "===================================================================="
