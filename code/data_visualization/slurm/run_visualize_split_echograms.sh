#!/bin/bash
#SBATCH --job-name=viz_split_echograms
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/%x_%j.err

# Render one 38 kHz echogram PNG (ensemble bottom line overlaid) per file in a
# dataset survey split. Select the split with the SPLIT env var, e.g.
#   sbatch --job-name=viz_split_echograms_val  --export=ALL,SPLIT=val  run_visualize_split_echograms.sh
#   sbatch --job-name=viz_split_echograms_test --export=ALL,SPLIT=test run_visualize_split_echograms.sh

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"
mkdir -p code/data_visualization/slurm/logs

SPLIT="${SPLIT:-val}"

echo "════════════════════════════════════════════════════════════════════"
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Split:  ${SPLIT}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "════════════════════════════════════════════════════════════════════"

python code/data_visualization/visualize_split_echograms.py --split "$SPLIT"

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
