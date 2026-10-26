#!/bin/bash
#SBATCH --job-name=gen-patches
#SBATCH --account=share-ie-imf
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=1-10%10
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/gen_patches_%A_%a.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/gen_patches_%A_%a.err

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
NUM_CHUNKS=10
CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

cd "$PROJECT"

echo "============================================================"
echo "Patch Generation | Chunk ${CHUNK_INDEX}/${NUM_CHUNKS}"
echo "Job ID: ${SLURM_JOB_ID} | Array: ${SLURM_ARRAY_JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "============================================================"

python code/generate_patches_chunk.py \
    --chunk-index "$CHUNK_INDEX" \
    --num-chunks "$NUM_CHUNKS"

echo "Finished: $(date)"
