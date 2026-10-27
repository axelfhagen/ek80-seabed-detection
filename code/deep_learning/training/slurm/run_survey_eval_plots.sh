#!/bin/bash
#SBATCH --job-name=survey-eval-plots
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/survey_eval_plots_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/survey_eval_plots_%j.err

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
cd "$REPO_ROOT"

echo "============================================================"
echo "Survey evaluation plots | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "============================================================"

python code/data_visualization/plot_survey_evaluations.py --output-dir code/plots_june2025/survey_eval

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
