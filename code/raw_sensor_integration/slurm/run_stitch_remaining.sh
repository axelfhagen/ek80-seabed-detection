#!/bin/bash
#SBATCH --job-name=stitch-remaining
#SBATCH --account=share-ie-imf
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/raw_sensor_integration/slurm/logs/stitch_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/raw_sensor_integration/slurm/logs/stitch_%j.err

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
LOG_DIR="${PROJECT}/code/raw_sensor_integration/slurm/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Stitching remaining low-ping datasets"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "============================================================"

cd "$PROJECT"
python code/raw_sensor_integration/stitch_remaining.py

echo "Exit code: $?"
echo "Finished: $(date)"
