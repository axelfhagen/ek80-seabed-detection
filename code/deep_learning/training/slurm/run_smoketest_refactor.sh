#!/bin/bash
#SBATCH --job-name=smoke-refactor
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/smoke_refactor_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/smoke_refactor_%j.err

# Smoketest for the train.py module split (cli.py / model_setup.py /
# evaluation.py / checkpointing.py / evaluate_checkpoint.py). Validates that all
# refactored entry points still run end-to-end. NOT a training experiment.
#
# Kept fast on purpose: smallest architecture (lightweight_cnn, no pretrained
# encoder), 4 zarr files, 1 epoch, and pw512/s512 patch params that hit the
# existing patch_cache so no extraction runs. Typically finishes in ~1-2 min;
# 10 min wall is just queue headroom.
#
# Isolation guarantees:
#   * all outputs go under a fresh timestamped exp_smoke_refactor_* dir
#   * --results-log "" so the shared results.csv is never touched
#   * never reads or writes any existing checkpoint dir

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SMOKE_ROOT="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_smoke_refactor"
TRAIN_DIR="${SMOKE_ROOT}/train"
EVALONLY_DIR="${SMOKE_ROOT}/eval_only_via_train"
STANDALONE_DIR="${SMOKE_ROOT}/eval_standalone"
RESUME_DIR="${SMOKE_ROOT}/resume_train"

cd "$PROJECT"
mkdir -p "$TRAIN_DIR" "$EVALONLY_DIR" "$STANDALONE_DIR" "$RESUME_DIR"

# 4 zarr files is enough for a 1-val-file random split and keeps extraction tiny.
mapfile -t ZARR_PATHS < <(ls -d "${ZARR_DIR}"/*.zarr | head -n 4)

# Fast config shared by every phase. Patch params match the canonical survey
# extraction (pw512/s512, 38k/200k, default range+norm), so every file hits the
# existing patch_cache and NO extraction runs. Model is the smallest arch.
COMMON=(
    --architecture lightweight_cnn
    --encoder-weights none
    --freeze none
    --in-channels 3
    --freq1 38000
    --freq2 200000
    --target-size 512 512
    --patch-width 512
    --stride 512
    --batch-size 4
    --split random
    --val-fraction 0.25
    --num-workers 4
    --cache-dir "$CACHE_DIR"
    --results-log ""
)

echo "============================================================"
echo "Smoketest: train.py refactor"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Files:  ${#ZARR_PATHS[@]} zarr files"
echo "Output root: $SMOKE_ROOT"
echo "============================================================"

# Run each phase without aborting the whole job on first failure, so we get a
# full pass/fail picture in one run.
declare -A RESULT

run_phase () {
    local name="$1"; shift
    echo ""
    echo ">>> PHASE: ${name}"
    echo ">>> CMD: $*"
    if "$@"; then
        RESULT[$name]="PASS"
        echo ">>> ${name}: PASS"
    else
        RESULT[$name]="FAIL (exit $?)"
        echo ">>> ${name}: FAIL"
    fi
}

# Phase 1 — full train path: cli -> model_setup -> train loop -> evaluation ->
# diagnostics. Produces the checkpoint the later phases consume.
run_phase "1_train" \
    python code/deep_learning/training/train.py \
        "${COMMON[@]}" \
        --zarr-paths "${ZARR_PATHS[@]}" \
        --epochs 1 \
        --checkpoint-dir "$TRAIN_DIR"

# Phase 2 — preserved train.py --eval-only path (what the eval slurm scripts
# use): exercises checkpointing.load_resume_state eval branch + evaluation.
run_phase "2_eval_only_via_train" \
    python code/deep_learning/training/train.py \
        "${COMMON[@]}" \
        --zarr-paths "${ZARR_PATHS[@]}" \
        --eval-only \
        --no-diagnostics \
        --resume "$TRAIN_DIR" \
        --checkpoint-dir "$EVALONLY_DIR"

# Phase 3 — new standalone evaluate_checkpoint.py entry point.
run_phase "3_eval_standalone" \
    python code/deep_learning/training/evaluate_checkpoint.py \
        "${COMMON[@]}" \
        --zarr-paths "${ZARR_PATHS[@]}" \
        --no-diagnostics \
        --resume "$TRAIN_DIR" \
        --checkpoint-dir "$STANDALONE_DIR"

# Phase 4 — resume-into-training path: exercises load_resume_state non-eval
# branch (optimizer state restore) and continues for one more epoch. Writes to
# its own dir so phase-1 outputs stay intact.
run_phase "4_resume_train" \
    python code/deep_learning/training/train.py \
        "${COMMON[@]}" \
        --zarr-paths "${ZARR_PATHS[@]}" \
        --epochs 2 \
        --no-diagnostics \
        --resume "$TRAIN_DIR" \
        --checkpoint-dir "$RESUME_DIR"

echo ""
echo "============================================================"
echo "Smoketest summary"
for k in 1_train 2_eval_only_via_train 3_eval_standalone 4_resume_train; do
    printf "  %-24s %s\n" "$k" "${RESULT[$k]:-SKIPPED}"
done
echo "Finished: $(date)"
echo "============================================================"

# Non-zero exit if any phase failed, so sacct/CI shows the job as failed.
for k in "${!RESULT[@]}"; do
    [[ "${RESULT[$k]}" == PASS ]] || exit 1
done
