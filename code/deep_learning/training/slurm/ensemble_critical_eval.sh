#!/bin/bash
#SBATCH --job-name=ensemble_critical_eval_div5_div10_feb_jun2025
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/ens_crit_%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/ens_crit_%x_%j.err

# Critical evaluation of heterogeneous deep ensembles (5 and 10 members).
# Diversity comes from architecture (all members share seed 42 + same data),
# so members are chosen to span U-Net / FPN / DeepLabV3+ / MAnet / SegFormer /
# vanilla-UNet / lightweight-CNN families. Calibrate on jan_2025, evaluate on
# the held-out feb_2025 + jun_2025 test surveys.

set -euo pipefail

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"
CK="$PROJECT/code/checkpoints"

# ── 5-member ensemble: maximally diverse architectures, all top-tier ──
DIV5=(
  "$CK/exp_20260513_100101_arch_sweep_2/smp_unet_resnet34_best.pth"
  "$CK/exp_20260513_100101_arch_sweep_3/smp_unet_resnet50_best.pth"
  "$CK/exp_20260513_180631_arch_sweep_7/smp_fpn_resnet34_best.pth"
  "$CK/exp_20260513_180708_arch_sweep_8/smp_deeplabv3plus_resnet34_best.pth"
  "$CK/exp_20260523_062831_attention_sweep_3/smp_segformer_mit_b1_best.pth"
)

# ── 10-member ensemble: DIV5 + 5 more architecture families ──
DIV10=(
  "${DIV5[@]}"
  "$CK/exp_20260523_062753_attention_sweep_2/smp_manet_resnet34_best.pth"
  "$CK/exp_20260523_060859_attention_sweep_1/smp_unet_resnet34_best.pth"
  "$CK/chunk_encoder_sweep_4/smp_unet_efficientnet-b3_best.pth"
  "$CK/exp_20260513_174359_arch_sweep_5/vanilla_unet_d4_f32_best.pth"
  "$CK/exp_20260513_222509_arch_sweep_9/lightweight_cnn_16_32_64_best.pth"
)

echo "============================================================"
echo "Ensemble critical eval  |  job ${SLURM_JOB_ID}  |  $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================================"

SCRIPT="code/validation/evaluate_ensemble_critical.py"
# jan_2025 has 391 files; conformal needs only a few thousand pings, so
# subsample evenly to keep wall time bounded while spanning the survey.
MAX_CALIB=60

echo ">>> 5-member ensemble (div5)"
python "$SCRIPT" --tag div5  --max-calib-files "$MAX_CALIB" --checkpoints "${DIV5[@]}"

echo ">>> 10-member ensemble (div10)"
python "$SCRIPT" --tag div10 --max-calib-files "$MAX_CALIB" --checkpoints "${DIV10[@]}"

echo ">>> Pair comparison (5 vs 10)"
python "$SCRIPT" --pair-compare \
    code/ensemble_results/div5/ensemble_metrics.json \
    code/ensemble_results/div10/ensemble_metrics.json

echo "Done: $(date)"
