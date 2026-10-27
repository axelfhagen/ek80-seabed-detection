#!/bin/bash
# Recalibrated 3-method conformal comparison on the exchangeable jan/feb/jun_2025
# split (pooled, seeded random calib/test) with micro-averaged coverage. This is
# the corrected counterpart to run_conformal_base.sh: it fixes the non-
# exchangeable split, the per-file averaging bias, and swaps the degenerate
# profile method for the locally-weighted band. Writes into
# Figures/conformal/v2_recalibrated/.
#
# Submit:  sbatch code/validation/slurm/run_conformal_v2.sh
#
#SBATCH --job-name=conformal_v2_recalibrated_exchangeable
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

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"

echo "===================================================================="
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname)"
echo "CPUs/Mem: ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?}M"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "Start:  $(date)"
echo "===================================================================="

python -u code/validation/generate_conformal_plot_v2.py --device auto "$@"

echo "===================================================================="
echo "Finished: $(date)"
echo "===================================================================="
