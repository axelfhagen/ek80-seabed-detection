#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Ensemble evaluation — seed-diverse ResNet-34 U-Net deep ensemble (5 members).
# Runs AFTER all 5 trainers succeed (--dependency=afterok set by submit_all.sh).
#
# Three stages, all reusing existing project code:
#   1. evaluate_ensemble.py        → mean/std prob maps, 4-panel uncertainty
#                                    PNGs, ensemble_metrics.json, results CSV.
#   2. evaluate_ensemble_critical.py → ensemble-vs-best-single, member
#                                      diversity, size-scaling, conformal
#                                      with disagreement-weighted bands.
#   3. analyze_ensemble_uncertainty.py → disagreement-vs-error evidence figure
#                                        (Figures/ensemble/uncertainty_assessment/).
#
# NOTE: --checkpoints lists are explicit because the existing glob looks for
# unet_*_best.pth, which does NOT match smp_unet_resnet34_best.pth.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=eval_ensemble_resnet34_seeds
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

ENSEMBLE_TAG="resnet34_seeds5"
CKPT_PARENT="${PROJECT}/code/checkpoints/deep_ensemble_resnet34_seeds"
OUTPUT_DIR="${PROJECT}/code/ensemble_results/${ENSEMBLE_TAG}"

CHECKPOINTS=(
  "${CKPT_PARENT}/member_seed42/smp_unet_resnet34_best.pth"
  "${CKPT_PARENT}/member_seed1/smp_unet_resnet34_best.pth"
  "${CKPT_PARENT}/member_seed7/smp_unet_resnet34_best.pth"
  "${CKPT_PARENT}/member_seed123/smp_unet_resnet34_best.pth"
  "${CKPT_PARENT}/member_seed2025/smp_unet_resnet34_best.pth"
)

echo "════════════════════════════════════════════════════════════════════"
echo "Job:      ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Tag:      ${ENSEMBLE_TAG}"
echo "Out dir:  ${OUTPUT_DIR}"
echo "Start:    $(date)"
echo "════════════════════════════════════════════════════════════════════"
echo "Checkpoints:"
for c in "${CHECKPOINTS[@]}"; do echo "  $c"; done
echo

# Sanity-check: all 5 checkpoints must exist
MISSING=0
for c in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$c" ]]; then
    echo "ERROR: missing checkpoint $c" >&2
    MISSING=$((MISSING+1))
  fi
done
if [[ $MISSING -gt 0 ]]; then
  echo "ERROR: $MISSING checkpoints missing — aborting." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ─────────────────────────────────────────────────────────────────────────
# Stage 1 — Ensemble inference, metrics, 4-panel uncertainty figures.
# Uses survey split (validation files only); patches 512/256 mirror chunk_arch_sweep_2 eval.
# ─────────────────────────────────────────────────────────────────────────
echo ">>> Stage 1: evaluate_ensemble.py"
python code/deep_learning/training/evaluate_ensemble.py \
    --checkpoints "${CHECKPOINTS[@]}" \
    --split survey \
    --patch-width 512 \
    --stride 256 \
    --threshold 0.5 \
    --output-dir "$OUTPUT_DIR"

# ─────────────────────────────────────────────────────────────────────────
# Stage 2 — Critical evaluation: ensemble-vs-best-single, member diversity
# (pairwise disagreement, error correlation), size-scaling, disagreement
# vs error (Spearman), conformal disagreement-weighted bands.
# Writes to code/ensemble_results/${tag}/ensemble_metrics.json and figures
# to Figures/ensemble/<tag>/.
# ─────────────────────────────────────────────────────────────────────────
echo ">>> Stage 2: evaluate_ensemble_critical.py"
python code/validation/evaluate_ensemble_critical.py \
    --tag "${ENSEMBLE_TAG}" \
    --max-calib-files 60 \
    --checkpoints "${CHECKPOINTS[@]}"

# ─────────────────────────────────────────────────────────────────────────
# Stage 3 — Disagreement vs error evidence figure (constant vs disagreement-
# weighted conformal band). The script reads
# code/ensemble_results/${ENSEMBLE_TAG}/ensemble_metrics.json by default if
# pointed via env (DIV5_JSON is hardcoded; we override by symlinking).
# Falls back gracefully — Stage 3 failure does NOT mark the job failed.
# ─────────────────────────────────────────────────────────────────────────
echo ">>> Stage 3: analyze_ensemble_uncertainty.py"
DIV5_LINK="${PROJECT}/code/ensemble_results/div5"
if [[ ! -e "$DIV5_LINK" ]]; then
    ln -s "$OUTPUT_DIR" "$DIV5_LINK" || true
fi
python code/data_visualization/analyze_ensemble_uncertainty.py || \
    echo "WARN: Stage 3 (analyze_ensemble_uncertainty.py) failed; see log."

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "Results:  ${OUTPUT_DIR}"
echo "════════════════════════════════════════════════════════════════════"
