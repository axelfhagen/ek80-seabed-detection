#!/bin/bash
#SBATCH --job-name=gen-jun2025-caches
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/raw_sensor_integration/slurm/logs/gen_jun2025_caches_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/raw_sensor_integration/slurm/logs/gen_jun2025_caches_%j.err

# Build the Jun 2025 test patch caches from zarr (idempotent: ensure_cached
# skips files already present). Bottom detection over full echograms is
# memory-heavy, so this runs with 64G off the shared login node.

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"

python - <<'PY'
import sys; sys.path.insert(0, 'code')
from pathlib import Path
from deep_learning.training.config import PipelineConfig
from deep_learning.echo_dataset import ensure_cached
from project_paths import REPO_ROOT as ROOT
CACHE_DIR = ROOT / 'Datasets/patch_cache'
ZARR_DIR = ROOT / 'Datasets/echopype_zarr'
STEMS = ['D20250624-T181715', 'D20250625-T015027', 'D20250625-T070947',
         'D20250625-T115550', 'D20250625-T155644', 'D20250625-T202259',
         'D20250626-T042617', 'D20250626-T084615', 'D20250626-T125158',
         'D20250626-T163945', 'D20250626-T211527']

cfg = PipelineConfig()
cfg.freq1_hz = 38000
cfg.freq2_hz = 200000
cfg.patch_width = 512
cfg.stride = 512

for i, stem in enumerate(STEMS, 1):
    zp = ZARR_DIR / f'{stem}.zarr'
    print(f'[{i}/{len(STEMS)}] {stem} ...', flush=True)
    npz, count = ensure_cached(zp, CACHE_DIR, cfg, verbose=True)
    print(f'    -> {npz.name} ({count} patches)', flush=True)
print('DONE')
PY
