#!/bin/bash
#SBATCH --job-name=convert_patchcache_fast_512
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

# One-time pre-build of the fast patch cache: for every .npz in the cache dir,
# write a pre-resized, uncompressed, mmap-able .npy pair (512x512). Reads the
# .npz files; never modifies them, so .npz-based jobs keep working. Idempotent
# and re-runnable. Peak RAM ~ workers x largest-decompressed-file, so 8 workers
# on 48G has comfortable headroom for the largest (~4GB) files.

module purge
module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
CACHE_DIR="${CACHE_DIR:-${PROJECT}/Datasets/patch_cache}"
cd "$PROJECT"

echo "════════════════════════════════════════════════════════════════════"
echo "Convert patch cache -> fast (.npy)  | Job ${SLURM_JOB_ID}"
echo "Cache dir: ${CACHE_DIR}"
echo "Node: $(hostname)   CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "Start: $(date)"
echo "════════════════════════════════════════════════════════════════════"

python code/deep_learning/training/convert_cache_to_fast.py \
    --cache-dir "$CACHE_DIR" \
    --target-size 512 512 \
    --workers "${SLURM_CPUS_PER_TASK:-8}"

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
