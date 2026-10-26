#!/bin/bash
#SBATCH --job-name=xsurvey
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/xsurvey_%A_%a.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/xsurvey_%A_%a.err

# Cross-survey generalization: train on survey $SLURM_ARRAY_TASK_ID, eval on all others.
# Submit with: sbatch --array=1-8 launch_cross_survey.sh

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"
export PYTHONPATH="$PROJECT/code:$PYTHONPATH"

TASK_ID=$SLURM_ARRAY_TASK_ID

echo "============================================================"
echo "Cross-Survey Generalization | Task: ${TASK_ID} | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "============================================================"

python -m validation.dataset_shift.cross_survey train \
    --survey-index "$TASK_ID" \
    --zarr-dir Datasets/echopype_zarr \
    --cache-dir Datasets/patch_cache \
    --output-dir code/checkpoints/cross_survey \
    --max-train-files 80 \
    --max-eval-files 25 \
    --epochs 50 \
    --batch-size 4 \
    --num-workers 4 \
    --early-stopping-patience 10

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
