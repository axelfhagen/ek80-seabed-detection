#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Recompute all GPU-dependent TODO(recompute) thesis metrics with the
# chunk_arch_sweep_2 (Bastø Standard U-Net) checkpoint. Adapted from slurm_template.sh.
#   1. Regenerate the conformal-exchangeability score cache + figures (all surveys, 30/survey)
#   2. recompute_conformal.py   — canonical supervised-conformal summary (E)
#   3. diag_calibration.py      — probability calibration (D)
#   4. diag_mc_dropout.py       — MC-dropout sweep / correlation (F)
#   5. eval_aug_p95.py          — augmentation per-file P95 (C)
#   6. eval_hake.py             — hake transfer agreement / depth / aspect-ratio (G)
#   7. generate_conformal_hake.py — hake conformal coverage + reliability ECE (G)
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=recompute_thesis_metrics_conf_calib_mcdrop_hake_aug
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=07:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

set -uo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"
REGEN=code/thesis_metrics_regen
export PYTHONPATH="$PROJECT/code:${PYTHONPATH:-}"

echo "════════════════════════════════════════════════════════════════════"
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname)"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
fi
echo "Start:  $(date)"
echo "════════════════════════════════════════════════════════════════════"

# 0. Preserve the existing (mixed-date) conformal score cache before regenerating.
CACHE=code/validation/cache/exchangeability
if [[ -d "$CACHE" && ! -d "$REGEN/outputs/exchangeability_cache_backup" ]]; then
    cp -r "$CACHE" "$REGEN/outputs/exchangeability_cache_backup"
    echo "backed up old cache -> $REGEN/outputs/exchangeability_cache_backup"
fi

echo "[1/7] Regenerate conformal score cache + figures (all surveys, 30/survey) ..."
python -m validation.conformal_exchangeability --max-files-per-survey 30 \
    --checkpoint "$PROJECT/code/checkpoints/chunk_arch_sweep_2" || echo "  (step 1 nonzero exit)"
echo "[1b] focused calib-vs-test figures (from fresh cache) ..."
python -m validation.conformal_exchangeability --calib-vs-test || echo "  (step 1b nonzero exit)"

echo "[2/7] Canonical supervised-conformal summary ..."
python "$REGEN/recompute_conformal.py" || echo "  (step 2 nonzero exit)"

echo "[3/7] Probability calibration ..."
python "$REGEN/diag_calibration.py" --max-files 24 --seed 42 || echo "  (step 3 nonzero exit)"

echo "[4/7] MC-dropout ..."
python "$REGEN/diag_mc_dropout.py" --n-files 4 --n-samples 20 --seed 42 || echo "  (step 4 nonzero exit)"

echo "[5/7] Augmentation per-file P95 ..."
python "$REGEN/eval_aug_p95.py" || echo "  (step 5 nonzero exit)"

echo "[6/8] Hake transfer ..."
python "$REGEN/eval_hake.py" --open-water-scan 95 || echo "  (step 6 nonzero exit)"

echo "[7/8] Hake conformal + reliability ..."
python code/validation/generate_conformal_hake.py --n-calib 40 --n-groups 6 \
    --checkpoint "$PROJECT/code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth" \
    || echo "  (step 7 nonzero exit)"

echo "[8/8] Bottom-error distribution (chunk_arch_sweep_2) ..."
python "$REGEN/diag_bottom_error.py" --n-files 80 --seed 7 || echo "  (step 8 nonzero exit)"

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "Summaries in $REGEN/outputs/"
echo "════════════════════════════════════════════════════════════════════"
