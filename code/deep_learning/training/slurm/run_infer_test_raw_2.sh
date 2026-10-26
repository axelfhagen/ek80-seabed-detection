#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Inference + visualization on /cluster/work/axelfh/test_raw_2 (~72 raw files)
# Generates per-file echograms and 3x3 grids (hard + soft predictions)
# using the best U-Net checkpoint from exp_all_surveys_jun2025.
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=infer_test_raw_2
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

echo "════════════════════════════════════════════════════════════════════"
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK:-?}     Mem: ${SLURM_MEM_PER_NODE:-?}M"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
fi
echo "Start:  $(date)"
echo "════════════════════════════════════════════════════════════════════"

python code/infer_test_raw.py \
    --raw-dir /cluster/work/axelfh/test_raw_2 \
    --output-dir code/plots_test_raw_2 \
    --max-pings 2000

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
