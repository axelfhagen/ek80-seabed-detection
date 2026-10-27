#!/bin/bash
# Regenerate the per-survey Sv distribution figures (KDE overlays + KS/JSD
# heatmaps) into Figures/distribution_comparison/v2/ with the thesis style.
# CPU-only; reads zarr directly, no model needed. Runs on a dedicated node so
# the long per-file zarr loop is not killed by login-node memory pressure.
#SBATCH --job-name=gen_sv_distribution_comparison_v2_thesisstyle
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT/code"

echo "════════════════════════════════════════════════════════════════════"
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname)   Mem: ${SLURM_MEM_PER_NODE:-?}M"
echo "Start:  $(date)"
echo "════════════════════════════════════════════════════════════════════"

python -m validation.dataset_shift.distribution_comparison \
    --zarr-dir "$PROJECT/Datasets/echopype_zarr" \
    --output-dir "$PROJECT/Figures/distribution_comparison/v2"

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
