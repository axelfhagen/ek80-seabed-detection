#!/bin/bash
# Canonical 3-method conformal comparison on the survey-based split
# (calibration = jan_2025, test = feb/jun_2025). Constant-width vs profile-based
# vs locally-weighted bands, coverage calibration, score distributions and a
# sigma-vs-error diagnostic. This is the naive-split baseline whose under-
# coverage motivates the recalibrated v2 run; kept for side-by-side comparison.
# Writes into Figures/conformal/.
#
# Submit:  sbatch code/validation/slurm/run_conformal_base.sh
#
#SBATCH --job-name=conformal_base_survey_split
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/validation/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/validation/slurm/logs/%x_%j.err

set -euo pipefail
module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
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

python -u code/validation/generate_conformal_plot.py --device auto "$@"

echo "===================================================================="
echo "Finished: $(date)"
echo "===================================================================="
