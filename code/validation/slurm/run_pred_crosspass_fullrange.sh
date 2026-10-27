#!/bin/bash
# Full-range predicted-bottom crossing table for the failure-mining spotlight.
# Runs U-Net inference on every echopype zarr with the best generalist
# checkpoint (baseline_random_split_2, trained on all 1728 files), extracts a
# per-ping predicted bottom depth, detects one-way crossings, and projects each
# ping onto the 1-D route corridor. Unlike the conformal run, no depth cap is
# applied, so deep-channel predictions (100-200 m) survive for the cross-pass
# disagreement plot. Writes a fresh ping_table.parquet to a new directory so the
# existing capped conformal table is left intact.
#
# Submit:  sbatch code/validation/slurm/run_pred_crosspass_fullrange.sh
#
#SBATCH --job-name=pred_crosspass_fullrange_baseline_random_split2
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

CHECKPOINT="${PROJECT}/code/checkpoints/exp_20260525_220040_baseline_random_split_2/smp_unet_resnet34_best.pth"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
OUT_DIR="${PROJECT}/Figures/conformal/labelless_pred_fullrange"

echo "===================================================================="
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:       $(hostname)"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "Checkpoint: ${CHECKPOINT}"
echo "Zarr dir:   ${ZARR_DIR}"
echo "Out dir:    ${OUT_DIR}"
echo "Start:      $(date)"
echo "===================================================================="

# No --max-pred-depth: keep full-range predictions including the deep channel.
python -u code/validation/generate_labelless_conformal_plot.py \
    --checkpoint "$CHECKPOINT" \
    --zarr-dir "$ZARR_DIR" \
    --out-dir "$OUT_DIR" \
    --bin-size 50 \
    --alpha 0.10 \
    --score-mode pairwise

echo "===================================================================="
echo "Finished:   $(date)"
echo "===================================================================="
