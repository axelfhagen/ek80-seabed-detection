#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Extract patches for the TEST + CALIBRATION surveys (512-wide, stride 512).
# Adapted from slurm_template.sh (MODE=cpu).
#
# The training/eval pipeline only ever warms patch_cache for the train+val
# surveys; calibration (Jan 2025) and test (Feb/Jun 2025) have no patch cache
# because conformal runs use full-echogram inference instead. This job warms
# patch_cache for those held-out surveys so a 512/512 patch-based evaluation
# can run without on-the-fly extraction.
#
# Resumable: ensure_cached() skips files whose npz already exists, so re-running
# (or raising --time) continues where a 30-minute slot left off.
#
# Submit:  sbatch code/raw_sensor_integration/slurm/run_extract_test_calib_patches.sh
# Status:  squeue -u $USER   |   sacct -j <jobid>
# ═══════════════════════════════════════════════════════════════════════════════
# ── SBATCH directives ────────────────────────────────────────────────────────
#SBATCH --job-name=extract_test_calib
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/work/axelfh/Master/code/raw_sensor_integration/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/raw_sensor_integration/slurm/logs/%x_%j.err

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"

# ── Diagnostics header ────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════"
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK:-?}     Mem: ${SLURM_MEM_PER_NODE:-?}M"
echo "Start:  $(date)"
echo "════════════════════════════════════════════════════════════════════"

# Soft wall-clock budget (seconds) — exit cleanly before Slurm SIGTERM at 30 min.
export EXTRACT_BUDGET_SEC=1680   # 28 min
export EXTRACT_WORKERS="${SLURM_CPUS_PER_TASK:-4}"

python -u - <<'PY'
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "code")

from deep_learning.training.config import PipelineConfig
from deep_learning.training.survey_groups import get_split_files
from deep_learning.echo_dataset import ensure_cached

CACHE_DIR = Path("Datasets/patch_cache")
BUDGET = float(os.environ.get("EXTRACT_BUDGET_SEC", "1680"))
WORKERS = int(os.environ.get("EXTRACT_WORKERS", "4"))

# 512-wide patches, stride 512 (no overlap). Everything else = pipeline defaults
# (38/200 kHz, minmax norm, mask_mode 'below', require_bottom) so the resulting
# cache matches what a 512/512 training/eval run would request.
cfg = PipelineConfig(patch_width=512, stride=512)

splits = get_split_files()
files = list(splits["calibration"]) + list(splits["test"])
print(f"Calibration files: {len(splits['calibration'])}", flush=True)
print(f"Test files:        {len(splits['test'])}", flush=True)
print(f"Total to process:  {len(files)}  (workers={WORKERS}, budget={BUDGET:.0f}s)",
      flush=True)
print(f"patch_width={cfg.patch_width} stride={cfg.stride} "
      f"freq1={cfg.freq1_hz} freq2={cfg.freq2_hz} mask_mode={cfg.mask_mode}",
      flush=True)

def work(zarr_path):
    npz, count = ensure_cached(zarr_path, CACHE_DIR, cfg, verbose=False)
    return str(zarr_path), count

start = time.time()
done = 0
total_patches = 0
it = iter(files)

with ProcessPoolExecutor(max_workers=WORKERS) as ex:
    futures = {}

    def submit_next():
        """Queue one more file if any remain and the budget allows."""
        if time.time() - start >= BUDGET:
            return
        try:
            p = next(it)
        except StopIteration:
            return
        futures[ex.submit(work, p)] = p

    for _ in range(WORKERS):
        submit_next()

    while futures:
        fut = next(as_completed(futures))
        p = futures.pop(fut)
        try:
            name, count = fut.result()
            total_patches += count
            done += 1
            print(f"[{done}/{len(files)}] {Path(name).name}: {count} patches",
                  flush=True)
        except Exception as e:
            done += 1
            print(f"[{done}/{len(files)}] {Path(p).name}: ERROR {e}", flush=True)
        submit_next()

elapsed = time.time() - start
print("────────────────────────────────────────────────────────────────────",
      flush=True)
print(f"Processed {done}/{len(files)} files, {total_patches} patches, "
      f"{elapsed:.0f}s elapsed.", flush=True)
if done < len(files):
    print("Budget reached before finishing — re-submit to resume "
          "(cache hits are skipped).", flush=True)
PY

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
