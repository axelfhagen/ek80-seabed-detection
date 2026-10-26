#!/bin/bash
#SBATCH --job-name=repro_24393977_dec2023_resnet34_stages_dicebce
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/repro_24393977_dec2023_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/repro_24393977_dec2023_%j.err

# Reproduce run 24393977 (2026-05-06, exp_20260506_163907_survey_dec_2023).
#
# That run reported eval_iou=0.9978 / bottom_err=0.495 m -- far better than the
# recent full-dataset sweeps (~0.984 / ~0.6 m). Investigation showed why: it
# trained on only the 21 Dec-2023 zarr files (Dec 20-21, one homogeneous survey)
# with a random file-level split that held out just 4 files. The recent runs use
# split_mode=survey over all 1728 files (378 held out across many surveys) -- a
# much harder, more honest benchmark. This job retrains on that SAME 21-file
# subset with the SAME hyperparameters to confirm the number is real and is an
# easy-dataset artifact, not a regression in the later runs.
#
# NOT a chunked-resume job: small dataset, runs to convergence in one shot.

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

# Exactly the 21 files run 24393977 trained on (Dec 20-21 2023 survey).
mapfile -t ZARR_FILES < <(ls -1 "${PROJECT}/Datasets/echopype_zarr/"D202312*.zarr)
echo "Training on ${#ZARR_FILES[@]} zarr files (expected 21):"
printf '  %s\n' "${ZARR_FILES[@]##*/}"
if [ "${#ZARR_FILES[@]}" -ne 21 ]; then
    echo "ERROR: expected 21 Dec-2023 files, found ${#ZARR_FILES[@]}" >&2
    exit 1
fi

CHECKPOINT_DIR="${PROJECT}/code/checkpoints/repro_24393977_dec2023"
CACHE_DIR="${PROJECT}/Datasets/patch_cache_repro_dec2023"
mkdir -p "$CHECKPOINT_DIR" "$CACHE_DIR"

echo "============================================================"
echo "Repro of 24393977 | Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Ckpt: ${CHECKPOINT_DIR}"
echo "Start: $(date)"
echo "============================================================"

python code/deep_learning/training/train.py \
    --zarr-paths "${ZARR_FILES[@]}" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --num-workers 4 \
    --split random \
    --val-fraction 0.2 \
    --seed 42 \
    --target-size 512 512 \
    --patch-width 512 \
    --stride 512 \
    --freq1 38000 \
    --freq2 200000 \
    --in-channels 3 \
    --architecture smp_unet \
    --encoder resnet34 \
    --encoder-weights imagenet \
    --freeze stages \
    --loss-type dice_bce \
    --bce-weight 0.5 \
    --mask-mode below \
    --line-thickness 3 \
    --crop-mode full \
    --crop-padding 64 \
    --batch-size 4 \
    --epochs 250 \
    --encoder-lr 1e-05 \
    --decoder-lr 1e-03 \
    --lr-patience 5 \
    --lr-factor 0.5 \
    --scheduler-type plateau \
    --early-stopping-patience 20

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
