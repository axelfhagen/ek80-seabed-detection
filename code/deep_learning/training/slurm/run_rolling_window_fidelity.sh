#!/bin/bash
#SBATCH --job-name=rwf-analysis
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/rwf_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/rwf_%j.err

module load Anaconda3/2024.02-1
conda activate master_env

cd /cluster/work/axelfh/Master

echo "============================================================"
echo "Rolling-window fidelity analysis | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "============================================================"

python -u code/data_visualization/analyze_rolling_window_fidelity.py --max-files 14

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
