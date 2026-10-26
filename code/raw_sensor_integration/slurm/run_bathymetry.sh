#!/bin/bash
#SBATCH --job-name=bathymetry-map
#SBATCH --account=share-ie-imf
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/raw_sensor_integration/slurm/logs/bathymetry_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/raw_sensor_integration/slurm/logs/bathymetry_%j.err

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

mkdir -p code/raw_sensor_integration/slurm/logs
mkdir -p Figures/bathymetry

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

python code/raw_sensor_integration/bathymetry_map.py

echo "Finished: $(date)"
