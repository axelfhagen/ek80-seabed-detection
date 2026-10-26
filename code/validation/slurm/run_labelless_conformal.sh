#!/bin/bash
# Label-less conformal prediction from repeated Bastø ferry crossings.
# Runs U-Net inference on every echopype zarr, stitches consecutive files into
# continuous tracks, detects one-way crossings via the midpoint-line state
# machine, projects pings onto a 1-D arc-length corridor, calibrates a
# distribution-free band from |pred_A - pred_B| within 20 m bins, and writes
# coverage diagnostics to Figures/conformal/labelless/.
#
# Submit:    sbatch code/validation/slurm/run_labelless_conformal.sh
# Override:  sbatch --time=08:00:00 --mem=64G code/validation/slurm/run_labelless_conformal.sh \
#                  --bin-size 10 --score-mode median --per-direction
#
#SBATCH --job-name=labelless_conformal
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

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

# ── Defaults (can be overridden by extra sbatch arguments) ────────────────────
CHECKPOINT="${PROJECT}/code/checkpoints/exp_20260511_191547_all_surveys_jun2025/smp_unet_resnet34_best.pth"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
OUT_DIR="${PROJECT}/Figures/conformal/labelless"
BIN_SIZE=20
ALPHA=0.10
SCORE_MODE=pairwise
MAX_PRED_DEPTH=100
EXTRA_ARGS=("$@")

echo "===================================================================="
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:       $(hostname)"
echo "CPUs/Mem:   ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?}M"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "Checkpoint: ${CHECKPOINT}"
echo "Zarr dir:   ${ZARR_DIR}"
echo "Out dir:    ${OUT_DIR}"
echo "Args:       bin=${BIN_SIZE} alpha=${ALPHA} mode=${SCORE_MODE} "\
"max-pred-depth=${MAX_PRED_DEPTH} extras=${EXTRA_ARGS[*]:-}"
echo "Start:      $(date)"
echo "===================================================================="

python -u code/validation/generate_labelless_conformal_plot.py \
    --checkpoint "$CHECKPOINT" \
    --zarr-dir "$ZARR_DIR" \
    --out-dir "$OUT_DIR" \
    --bin-size "$BIN_SIZE" \
    --alpha "$ALPHA" \
    --score-mode "$SCORE_MODE" \
    --max-pred-depth "$MAX_PRED_DEPTH" \
    "${EXTRA_ARGS[@]}"

echo "===================================================================="
echo "Finished:   $(date)"
echo "===================================================================="
