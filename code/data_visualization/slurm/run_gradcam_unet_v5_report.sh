#!/bin/bash
#SBATCH --job-name=gradcam_v5_report
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/%x_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"

export XDG_CACHE_HOME=/tmp/xdg-cache-gradcam-${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/mplconfig-gradcam-${SLURM_JOB_ID}
export PYTHONUNBUFFERED=1

mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

echo "Job:   ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

python code/data_visualization/gen_gradcam_unet_v5_report.py --n 5

echo "Done:  $(date)"
