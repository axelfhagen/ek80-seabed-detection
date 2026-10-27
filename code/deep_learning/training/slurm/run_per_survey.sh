#!/bin/bash
#SBATCH --job-name=survey
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/survey_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/survey_%j.err

# Train on each survey independently using the best-run hyperparameters.
# Submit individually: bash submit_per_survey.sh
# Or as array:         sbatch --array=1-8 run_per_survey.sh

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"

SURVEYS=(dec_2023 jan_2024 apr_2024 jul_2024 sep_2024 oct_2024 jan_2025 feb_2025 jun_2025)
TASK_ID=${FORCE_TASK_ID:-$SLURM_ARRAY_TASK_ID}
IDX=$((TASK_ID - 1))
SURVEY=${SURVEYS[$IDX]}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

ZARR_PATHS=$(python -c "
import sys; sys.path.insert(0, 'code')
from deep_learning.training.survey_groups import get_survey_files
files = get_survey_files('${SURVEY}', 'Datasets/echopype_zarr')
print(' '.join(str(f) for f in files))
")

N_FILES=$(echo "$ZARR_PATHS" | wc -w)
CHECKPOINT_DIR="code/checkpoints/exp_${TIMESTAMP}_survey_${SURVEY}"

echo "============================================================"
echo "Survey: ${SURVEY} | Files: ${N_FILES} | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "============================================================"

python code/deep_learning/training/train.py \
    --split random \
    --zarr-paths $ZARR_PATHS \
    --cache-dir Datasets/patch_cache \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --encoder resnet34 \
    --freeze stages \
    --encoder-lr 1e-05 \
    --decoder-lr 0.001 \
    --scheduler-type plateau \
    --epochs 250 \
    --batch-size 4 \
    --num-workers 4 \
    --early-stopping-patience 20 \
    --patch-width 512 \
    --stride 512 \
    --target-size 512 512 \
    --in-channels 3 \
    --freq1 38000 \
    --freq2 200000 \
    --bce-weight 0.5 \
    --seed 42

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
