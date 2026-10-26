#!/bin/bash
# Isolated Hake fine-tuning experiment from chunk_arch_sweep_2.
#
# Submit:
#   sbatch code/experiments/hake_finetune/run_hake_finetune.slurm
#
# Extra arguments are forwarded to run_hake_finetune.py.

#SBATCH --job-name=hake_ft_chunk_arch2
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/experiments/hake_finetune/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/experiments/hake_finetune/logs/%x_%j.err

set -euo pipefail
module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"
mkdir -p code/experiments/hake_finetune/logs

echo "===================================================================="
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname)"
echo "CPUs/Mem: ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?}M"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "Start:  $(date)"
echo "===================================================================="

python -u code/experiments/hake_finetune/run_hake_finetune.py "$@"

echo "===================================================================="
echo "Finished: $(date)"
echo "===================================================================="
