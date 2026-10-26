#!/bin/bash
# Recompute final thesis figures/results that must use the Bastø Standard U-Net
# (`chunk_arch_sweep_2`) rather than older all-surveys or BCE-weight checkpoints.
#
# Submit:
#   sbatch code/thesis_metrics_regen/slurm/run_final_figure_recompute.sh

#SBATCH --job-name=final_fig_recompute_chunk_arch2
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/thesis_metrics_regen/slurm/logs/%x_%A_%a.out
#SBATCH --error=/cluster/work/axelfh/Master/code/thesis_metrics_regen/slurm/logs/%x_%A_%a.err

set -uo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"
mkdir -p code/thesis_metrics_regen/slurm/logs

export PYTHONPATH="$PROJECT/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=/tmp/xdg-cache-final-figs-${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/mplconfig-final-figs-${SLURM_JOB_ID}
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

CKPT="$PROJECT/code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth"
CKPT_DIR="$PROJECT/code/checkpoints/chunk_arch_sweep_2"

status=0
run_step() {
    label="$1"
    shift
    echo
    echo "[$(date)] ${label}"
    echo "Command: $*"
    "$@"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "FAILED (${rc}): ${label}"
        status=$rc
    fi
}

echo "===================================================================="
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:   $(hostname)"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "Start:  $(date)"
echo "Checkpoint: $CKPT"
echo "===================================================================="

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        run_step "bottom-error distribution" \
            python -u code/thesis_metrics_regen/diag_bottom_error.py --n-files 80 --seed 7

        run_step "prediction grids, restyled" \
            python -u code/data_visualization/gen_prediction_grids_regen.py \
                --checkpoint-dir "$CKPT_DIR" \
                --out-dir "$PROJECT/Latex/Figures/prediction_grids_regen"
        ;;
    1)
        run_step "channel importance, chunk_arch_sweep_2" \
            python -u code/data_visualization/analyze_channel_importance_v2.py \
                --checkpoint "$CKPT" \
                --output-dir "$PROJECT/Latex/Figures/channel_importance/chunk_arch_sweep_2" \
                --max-per-file 12 \
                --max-attribution-patches 60 \
                --ig-steps 24 \
                --shuffle-trials 3 \
                --batch-size 8 \
                --seed 0

        run_step "Grad-CAM report, chunk_arch_sweep_2" \
            python -u code/data_visualization/gen_gradcam_unet_v5_report.py \
                --checkpoint "$CKPT" \
                --output-dir "$PROJECT/Latex/Figures/gradcam_unet/chunk_arch_sweep_2_v5_report" \
                --n 5

        run_step "patch probability hard cases, chunk_arch_sweep_2" \
            python -u code/data_visualization/gen_patch_prob_eval.py \
                --checkpoint "$CKPT" \
                --output-dir "$PROJECT/Latex/Figures/patch_prob_eval/chunk_arch_sweep_2_rgb_hard" \
                --split test \
                --display rgb \
                --selection hardest \
                --n 12 \
                --candidates-per-file 8 \
                --max-candidates 128 \
                --batch-size 8 \
                --seed 0

        run_step "patch conformal examples, chunk_arch_sweep_2" \
            python -u code/data_visualization/gen_patch_conformal_failures.py \
                --checkpoint "$CKPT" \
                --output-dir "$PROJECT/Latex/Figures/patch_conformal_failures/chunk_arch_sweep_2" \
                --calibration-split calibration \
                --test-split test \
                --n-each 2 \
                --calib-max-patches 160 \
                --test-max-patches 240 \
                --candidates-per-file 24 \
                --seed 0
        ;;
    2)
        run_step "Hake conformal recalibration, chunk_arch_sweep_2" \
            python -u code/validation/generate_conformal_hake_recalibration.py \
                --checkpoint "$CKPT" \
                --recompute-cache \
                --n-calib 20 \
                --seed 0 \
                --sweep-sizes 10 20 30 \
                --sweep-repeats 30 \
                --bootstrap 1000
        ;;
    *)
        echo "Unknown array task: ${SLURM_ARRAY_TASK_ID}"
        exit 2
        ;;
esac

echo
echo "===================================================================="
echo "Finished: $(date)"
echo "Exit status: $status"
echo "===================================================================="
exit "$status"
