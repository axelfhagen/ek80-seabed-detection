#!/bin/bash
#SBATCH --job-name=test-diagnostics
#SBATCH --account=share-ie-imf
#SBATCH --partition=short
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/test_diag_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/test_diag_%j.err

module load Anaconda3/2024.02-1
conda activate master_env

export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

# Test all diagnostics plot functions with synthetic data.
# This bypasses the slow evaluate_file/load_file_data and tests
# purely the plotting code we wrote.
python -c "
import sys, json, numpy as np
from pathlib import Path

sys.path.insert(0, 'code')
from deep_learning.training.diagnostics import (
    FileResult, plot_training_curves, plot_prediction_grid,
    plot_error_maps, plot_reliability, plot_bottom_error_distribution,
    plot_threshold_sensitivity_curve, plot_roc_curve,
    plot_precision_recall_curve, plot_entropy_maps,
    plot_metric_vs_size, plot_per_file_metrics,
)
from deep_learning.evaluate import FileMetrics

out_dir = Path('code/checkpoints/exp_20260506_163907_survey_dec_2023/diagnostics')
out_dir.mkdir(parents=True, exist_ok=True)

# --- Synthetic history ---
n_epochs = 20
history = {
    'train_loss': list(np.linspace(0.5, 0.1, n_epochs) + np.random.randn(n_epochs) * 0.02),
    'val_loss':   list(np.linspace(0.55, 0.12, n_epochs) + np.random.randn(n_epochs) * 0.03),
    'train_iou':  list(np.linspace(0.6, 0.95, n_epochs) + np.random.randn(n_epochs) * 0.02),
    'val_iou':    list(np.linspace(0.55, 0.92, n_epochs) + np.random.randn(n_epochs) * 0.03),
    'train_dice': list(np.linspace(0.7, 0.97, n_epochs) + np.random.randn(n_epochs) * 0.01),
    'val_dice':   list(np.linspace(0.65, 0.95, n_epochs) + np.random.randn(n_epochs) * 0.02),
}

print('1. Training curves...')
plot_training_curves(history, out_dir / 'training_curves.png')
print('   OK')

# --- Synthetic file results ---
def make_fake_result(name, n_range=200, n_pings=300, quality=0.9):
    rng = np.random.RandomState(hash(name) % 2**31)
    range_arr = np.linspace(0, 100, n_range)
    # Simulate a bottom at ~60% depth
    bottom_row = int(n_range * 0.6)
    gt_mask = np.zeros((n_range, n_pings), dtype=np.uint8)
    gt_mask[bottom_row:, :] = 1
    # Prediction: mostly correct with some noise
    prediction = np.zeros((n_range, n_pings), dtype=np.float32)
    for p in range(n_pings):
        noise = int(rng.randn() * (1 - quality) * 20)
        pred_row = max(0, min(n_range - 1, bottom_row + noise))
        prediction[pred_row:, p] = rng.uniform(0.7, 1.0)
        prediction[:pred_row, p] = rng.uniform(0.0, 0.3, pred_row)
    # Echogram-like channels
    sv = rng.rand(3, n_range, n_pings).astype(np.float32)
    sv[1, :bottom_row, :] *= 0.3  # water column darker on 200kHz

    pred_bin = (prediction >= 0.5).astype(bool)
    gt_bool = gt_mask.astype(bool)
    tp = int(np.sum(gt_bool & pred_bin))
    fp = int(np.sum(~gt_bool & pred_bin))
    fn = int(np.sum(gt_bool & ~pred_bin))
    tn = int(np.sum(~gt_bool & ~pred_bin))
    eps = 1e-7
    iou = (tp+eps)/(tp+fp+fn+eps)
    dice = (2*tp+eps)/(2*tp+fp+fn+eps)

    metrics = FileMetrics(
        file_name=name, iou=iou, dice=dice,
        precision=(tp+eps)/(tp+fp+eps), recall=(tp+eps)/(tp+fn+eps),
        tp=tp, fp=fp, fn=fn, tn=tn,
        bottom_error_mean_m=rng.uniform(0.5, 5.0),
        bottom_error_median_m=rng.uniform(0.3, 3.0),
        bottom_error_p95_m=rng.uniform(2.0, 10.0),
        n_pings_with_bottom=n_pings,
    )
    return FileResult(
        zarr_path=Path(f'fake/{name}'), metrics=metrics,
        prediction=prediction, gt_mask=gt_mask,
        sv_channels=sv, range_arr=range_arr,
    )

results = [
    make_fake_result('survey_A.zarr', n_pings=200, quality=0.95),
    make_fake_result('survey_B.zarr', n_pings=350, quality=0.85),
    make_fake_result('survey_C.zarr', n_pings=500, quality=0.90),
    make_fake_result('survey_D.zarr', n_pings=150, quality=0.70),
    make_fake_result('survey_E.zarr', n_pings=400, quality=0.92),
]
print(f'Created {len(results)} synthetic file results')

print('2. Prediction grid (random)...')
plot_prediction_grid(results, 'random', out_dir / 'prediction_grid_random.png')
print('   OK')

print('3. Prediction grid (worst_mean)...')
plot_prediction_grid(results, 'worst_mean', out_dir / 'prediction_grid_worst_mean.png')
print('   OK')

print('4. Prediction grid (worst_max)...')
plot_prediction_grid(results, 'worst_max', out_dir / 'prediction_grid_worst_max.png')
print('   OK')

print('5. Error maps...')
plot_error_maps(results, out_dir)
print('   OK')

print('6. Reliability diagram...')
plot_reliability(results, out_dir / 'reliability_diagram.png')
print('   OK')

print('7. Bottom error distribution...')
plot_bottom_error_distribution(results, out_dir / 'bottom_error_distribution.png')
print('   OK')

print('8. Threshold sensitivity...')
plot_threshold_sensitivity_curve(results, out_dir / 'threshold_sensitivity.png')
print('   OK')

print('9. ROC curve...')
plot_roc_curve(results, out_dir / 'roc_curve.png')
print('   OK')

print('10. Precision-Recall curve...')
plot_precision_recall_curve(results, out_dir / 'precision_recall_curve.png')
print('   OK')

print('11. Entropy maps...')
plot_entropy_maps(results, out_dir / 'entropy_map_worst.png')
print('   OK')

print('12. Metric vs size scatter...')
plot_metric_vs_size(results, out_dir / 'metric_vs_size_scatter.png')
print('   OK')

print('13. Per-file metrics chart...')
plot_per_file_metrics(results, out_dir / 'per_file_metrics.png')
print('   OK')

import os
pngs = sorted(out_dir.glob('*.png'))
print(f'\nAll done! {len(pngs)} plots generated:')
for p in pngs:
    size_kb = os.path.getsize(p) / 1024
    print(f'  {p.name:40s} {size_kb:6.1f} KB')
"

echo ""
echo "Finished: $(date)"
