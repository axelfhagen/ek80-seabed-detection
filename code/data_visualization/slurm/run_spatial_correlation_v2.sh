#!/bin/bash
#SBATCH --job-name=fig_spatial_correlation_v2_legend
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/%x_%j.err

# Regenerate the echogram autocorrelation/variogram figure with the legend moved
# outside the panels and a labelled +/-1 SD ribbon entry. Writes the revised
# figure to Figures/spatial_correlation/v2/. The login node OOM-kills this job,
# so it must run on a compute node.

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"
mkdir -p code/data_visualization/slurm/logs

echo "════════════════════════════════════════════════════════════════════"
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "════════════════════════════════════════════════════════════════════"

python code/data_visualization/analyze_spatial_correlation.py \
    --output-dir Figures/spatial_correlation/v2

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
