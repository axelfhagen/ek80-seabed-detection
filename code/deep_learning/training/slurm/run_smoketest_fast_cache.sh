#!/bin/bash
#SBATCH --job-name=smoke-fast-cache
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/smoke_fast_cache_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/smoke_fast_cache_%j.err

# Smoketest for the fast patch cache (--fast-cache: pre-resized mmap .npy) on a
# real GPU. Validates the full train -> eval path end-to-end and confirms a fast
# .npy pair is actually built and consumed.
#
# Isolation: seeds a private temp cache dir with copies of the matching .npz so
# the shared Datasets/patch_cache is never written. Outputs go to a fresh
# timestamped dir; --results-log "" leaves results.csv untouched.

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
SHARED_CACHE="${PROJECT}/Datasets/patch_cache"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SMOKE_ROOT="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_smoke_fast_cache"
TRAIN_DIR="${SMOKE_ROOT}/train"
TMP_CACHE="${SMOKE_ROOT}/tmp_cache"
cd "$PROJECT"
mkdir -p "$TRAIN_DIR" "$TMP_CACHE"

# 4 zarr files; seed the temp cache with their matching .npz (any param hash).
mapfile -t ZARR_PATHS < <(ls -d "${ZARR_DIR}"/*.zarr | head -n 4)
for zp in "${ZARR_PATHS[@]}"; do
    stem=$(basename "$zp" .zarr)
    cp "${SHARED_CACHE}/${stem}"_*.npz "$TMP_CACHE/" 2>/dev/null || true
done
echo "Seeded temp cache with $(ls "$TMP_CACHE"/*.npz 2>/dev/null | wc -l) npz files"

COMMON=(
    --architecture lightweight_cnn
    --encoder-weights none
    --freeze none
    --in-channels 3
    --freq1 38000 --freq2 200000
    --target-size 512 512
    --patch-width 512 --stride 512
    --batch-size 8
    --split random --val-fraction 0.25
    --num-workers 4
    --fast-cache
    --cache-dir "$TMP_CACHE"
    --results-log ""
)

echo "============================================================"
echo "Smoketest: --fast-cache GPU train+eval | Job ${SLURM_JOB_ID}"
echo "Node: $(hostname)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================================"

set -x
python code/deep_learning/training/train.py \
    "${COMMON[@]}" \
    --zarr-paths "${ZARR_PATHS[@]}" \
    --epochs 1 \
    --checkpoint-dir "$TRAIN_DIR"
rc=$?
set +x

echo "------------------------------------------------------------"
NPY_COUNT=$(ls "$TMP_CACHE"/*.patches.npy 2>/dev/null | wc -l)
echo "Fast .npy pairs built in temp cache: ${NPY_COUNT}"
echo "eval_metrics.json present: $([ -f "${TRAIN_DIR}/eval_metrics.json" ] && echo yes || echo no)"

if [ "$rc" -eq 0 ] && [ "$NPY_COUNT" -gt 0 ] && [ -f "${TRAIN_DIR}/eval_metrics.json" ]; then
    echo "SMOKETEST: PASS"
    echo "Finished: $(date)"
    exit 0
else
    echo "SMOKETEST: FAIL (rc=$rc, npy=$NPY_COUNT)"
    echo "Finished: $(date)"
    exit 1
fi
