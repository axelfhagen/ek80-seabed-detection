#!/bin/bash
#SBATCH --job-name=dlbench
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/dlbench_%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/dlbench_%x_%j.err

# Dataloader A/B benchmark: how many epochs fit in a fixed 10-min window.
#   arg 1: "old"  -> .npz path (decompress whole file + resize per sample)
#          "fast" -> --fast-cache (pre-resized, uncompressed, mmap'd .npy)
# Everything else is identical between the two: same 30 files, same random
# split (seed 42), same model (smp_unet/resnet34, frozen encoder), same
# batch size, same workers, NO early stopping, eval/diagnostics off.
#
# train.py runs with a high --epochs and is hard-killed by `timeout` at 600s;
# the SIGTERM handler exits gracefully. Epochs completed = count of "Epoch "
# log lines (history is written per-epoch, so the count is exact).

set -uo pipefail
MODE="${1:?Usage: sbatch run_dl_bench.sh <old|fast>}"

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"
cd "$PROJECT"

TS=$(date +%Y%m%d_%H%M%S)
BENCH_ROOT="${PROJECT}/code/checkpoints/dlbench_${MODE}_${TS}"
mkdir -p "$BENCH_ROOT"

# Fixed 30-file subset (deterministic order), identical for both modes.
mapfile -t ZARR_PATHS < <(ls -d "${ZARR_DIR}"/*.zarr | sort | head -n 30)

FAST_ARG=""
[ "$MODE" = "fast" ] && FAST_ARG="--fast-cache"

COMMON=(
    --architecture smp_unet
    --encoder resnet34
    --encoder-weights imagenet
    --freeze encoder
    --in-channels 3
    --freq1 38000 --freq2 200000
    --target-size 512 512
    --patch-width 512 --stride 512
    --batch-size 16
    --num-workers 4
    --split random --val-fraction 0.2 --seed 42
    --epochs 500
    --scheduler-type plateau
    --no-eval --no-diagnostics
    --results-log ""
)

echo "============================================================"
echo "DL benchmark | MODE=${MODE} ${FAST_ARG:-(npz)} | Job ${SLURM_JOB_ID}"
echo "Node: $(hostname)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Files: ${#ZARR_PATHS[@]}   Start: $(date)"
echo "============================================================"

# 600s hard wall on the training process; graceful SIGTERM (train.py handles it).
START=$(date +%s)
timeout --signal=TERM 600 \
    python code/deep_learning/training/train.py \
        "${COMMON[@]}" \
        $FAST_ARG \
        --zarr-paths "${ZARR_PATHS[@]}" \
        --cache-dir "$CACHE_DIR" \
        --checkpoint-dir "$BENCH_ROOT"
rc=$?
END=$(date +%s)

echo "============================================================"
echo "MODE=${MODE}  timeout_rc=${rc}  wall=$((END-START))s"
echo "Finished: $(date)"
echo "============================================================"
