#!/bin/bash
# Render only the chunk_arch_sweep_2 prediction-grid single plots with
# locally weighted conformal overlays, reusing the q_hat from job 24705920.

#SBATCH --job-name=pred_conf_singles_arch2
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/cluster/work/axelfh/Master/code/thesis_metrics_regen/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/thesis_metrics_regen/slurm/logs/%x_%j.err

set -euo pipefail
module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"
export PYTHONPATH="$PROJECT/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python -u code/data_visualization/gen_prediction_grids_regen.py \
    --checkpoint-dir "$PROJECT/code/checkpoints/chunk_arch_sweep_2" \
    --out-dir "$PROJECT/Latex/Figures/prediction_grids_regen" \
    --singles-only \
    --q-hat 1.667
