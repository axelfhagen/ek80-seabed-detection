"""Cross-survey generalization matrix: train on one survey, evaluate on all.

Trains a U-Net on data from a single survey period and measures segmentation
performance (IoU, Dice, bottom-depth error) on all other surveys, producing
an NxN generalization matrix.

Usage
-----
    # Single run (for testing):
    python -m validation.dataset_shift.cross_survey \
        --train-survey jan_2024 \
        --output-dir code/checkpoints/cross_survey

    # SLURM array jobs (submit all 7 surveys):
    sbatch --array=1-7 code/deep_learning/training/slurm/launch_cross_survey.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from deep_learning.training.config import PipelineConfig
from deep_learning.training.survey_groups import SURVEYS, get_survey_files
from deep_learning.echo_dataset import (
    EchogramPatchDataset,
    get_augmentation_pipeline,
    save_patches_cached,
)
from deep_learning.architectures import create_model
from deep_learning.unet_model import (
    get_trainable_param_count,
    build_loss,
    train_one_epoch,
    validate,
)
from deep_learning.evaluate import evaluate_files


def subsample_files(files: list[Path], max_files: int,
                    seed: int) -> list[Path]:
    """Deterministic random subsample if len(files) > max_files."""
    if len(files) <= max_files:
        return files
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(files), size=max_files, replace=False)
    return [files[i] for i in sorted(idx)]


def extract_patches(zarr_paths: list[Path], cache_dir: Path,
                    cfg: PipelineConfig):
    """Extract patches from zarr files, returning (patches, masks) lists."""
    all_patches, all_masks = [], []
    for zp in zarr_paths:
        patches, masks = save_patches_cached(
            zp, cache_dir,
            freq1_hz=cfg.freq1_hz, freq2_hz=cfg.freq2_hz,
            patch_width=cfg.patch_width, stride=cfg.stride,
            max_range_m=cfg.max_range_m,
            max_range_samples=cfg.max_range_samples,
            vmin=cfg.vmin, vmax=cfg.vmax,
            require_bottom=cfg.require_bottom,
            mask_mode=cfg.mask_mode,
            line_thickness=cfg.line_thickness,
            norm_method=cfg.norm_method,
            clip_percentile=cfg.clip_percentile,
            single_freq=cfg.single_freq,
            crop_mode=cfg.crop_mode,
            crop_padding=cfg.crop_padding,
        )
        all_patches.extend(patches)
        all_masks.extend(masks)
    return all_patches, all_masks


def run_cross_survey(
    train_survey: str,
    zarr_dir: Path = Path("Datasets/echopype_zarr"),
    cache_dir: Path = Path("Datasets/patch_cache"),
    output_dir: Path = Path("code/checkpoints/cross_survey"),
    max_train_files: int = 80,
    max_eval_files: int = 25,
    epochs: int = 50,
    batch_size: int = 4,
    num_workers: int = 4,
    early_stopping_patience: int = 10,
    seed: int = 42,
) -> dict:
    """Train on one survey and evaluate on all others.

    Returns the full results dict (also saved as JSON).
    """
    eval_surveys = [s for s in SURVEYS if s != train_survey]

    print(f"Train survey: {train_survey} ({SURVEYS[train_survey]['label']})")
    print(f"Eval surveys: {', '.join(eval_surveys)}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = PipelineConfig(
        patch_width=256,
        stride=64,
        val_fraction=0.15,
        seed=seed,
        batch_size=batch_size,
        num_epochs=epochs,
        num_workers=num_workers,
        early_stopping_patience=early_stopping_patience,
        encoder_name="resnet34",
        freeze_strategy="encoder",
        in_channels=3,
        target_size=(512, 512),
    )

    device = cfg.device
    print(f"Device: {device}")

    # ── 1. Prepare training data ─────────────────────────────────────
    print(f"\n=== Preparing training data ({train_survey}) ===")
    train_files = get_survey_files(train_survey, zarr_dir)
    train_files = subsample_files(train_files, max_train_files, seed)
    print(f"Using {len(train_files)} files for training")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(train_files))
    n_val = max(1, int(len(train_files) * cfg.val_fraction))
    val_idx = set(indices[:n_val])

    train_zarr = [train_files[i] for i in range(len(train_files))
                  if i not in val_idx]
    val_zarr = [train_files[i] for i in val_idx]

    train_patches, train_masks = extract_patches(
        train_zarr, cache_dir, cfg)
    val_patches, val_masks = extract_patches(
        val_zarr, cache_dir, cfg)

    print(f"Training patches: {len(train_patches)}")
    print(f"Validation patches: {len(val_patches)}")

    if len(train_patches) == 0:
        print("ERROR: No training patches extracted.")
        sys.exit(1)

    transform = get_augmentation_pipeline()
    train_ds = EchogramPatchDataset(
        train_patches, train_masks,
        target_size=cfg.target_size, transform=transform)
    val_ds = EchogramPatchDataset(
        val_patches, val_masks,
        target_size=cfg.target_size, transform=None)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device not in ('cpu', 'mps')))
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device not in ('cpu', 'mps')))

    # ── 2. Model ─────────────────────────────────────────────────────
    print("\n=== Model Setup ===")
    model = create_model(
        architecture="smp_unet",
        encoder_name=cfg.encoder_name,
        encoder_weights="imagenet",
        in_channels=cfg.in_channels,
        classes=1,
    )
    model.freeze_encoder()
    counts = get_trainable_param_count(model)
    print(f"Trainable: {counts['trainable']:,} / {counts['total']:,}")
    model = model.to(device)

    # ── 3. Optimizer + Loss ──────────────────────────────────────────
    param_groups = [
        {'params': [p for p in model.get_encoder_params() if p.requires_grad],
         'lr': cfg.encoder_lr},
        {'params': [p for p in model.get_decoder_params() if p.requires_grad],
         'lr': cfg.decoder_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.lr_factor,
        patience=cfg.lr_patience)
    loss_fn = build_loss(cfg.loss_type, bce_weight=cfg.bce_weight)
    scaler = torch.amp.GradScaler() if cfg.use_amp else None

    # ── 4. Training loop ─────────────────────────────────────────────
    print("\n=== Training ===")
    run_dir = output_dir / f"train_{train_survey}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pth"

    best_val_iou = 0.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(cfg.num_epochs):
        train_m = train_one_epoch(model, train_loader, optimizer,
                                  loss_fn, device, scaler)
        val_m = validate(model, val_loader, loss_fn, device)
        scheduler.step(val_m['loss'])

        improved = val_m['iou'] > best_val_iou
        if improved:
            best_val_iou = val_m['iou']
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_iou': best_val_iou,
                'train_survey': train_survey,
            }, checkpoint_path)
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1:3d}/{cfg.num_epochs}  "
              f"train_loss={train_m['loss']:.4f}  "
              f"val_loss={val_m['loss']:.4f}  "
              f"val_IoU={val_m['iou']:.4f}"
              f"{'  *' if improved else ''}")

        if (cfg.early_stopping_patience is not None
                and epochs_no_improve >= cfg.early_stopping_patience):
            print(f"\nEarly stopping after {epoch+1} epochs.")
            break

    print(f"\nBest val IoU: {best_val_iou:.4f} at epoch {best_epoch + 1}")

    # ── 5. Reload best and evaluate on all other surveys ─────────────
    print("\n=== Cross-Survey Evaluation ===")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    results = {
        'train_survey': train_survey,
        'train_label': SURVEYS[train_survey]['label'],
        'train_n_files': len(train_files),
        'train_n_patches': len(train_patches),
        'best_val_iou': best_val_iou,
        'best_epoch': best_epoch + 1,
        'evaluations': {},
    }

    all_eval_surveys = [train_survey] + eval_surveys

    for eval_name in all_eval_surveys:
        print(f"\n--- Evaluating on {eval_name} "
              f"({SURVEYS[eval_name]['label']}) ---")
        eval_files = get_survey_files(eval_name, zarr_dir)
        eval_files = subsample_files(
            eval_files, max_eval_files, seed + 1)
        print(f"  Using {len(eval_files)} files")

        if len(eval_files) == 0:
            print(f"  Skipping {eval_name}: no files")
            continue

        all_metrics, summary = evaluate_files(
            model, eval_files,
            freq1_hz=cfg.freq1_hz, freq2_hz=cfg.freq2_hz,
            target_size=cfg.target_size, device=device,
            patch_width=cfg.patch_width, stride=cfg.stride,
            vmin=cfg.vmin, vmax=cfg.vmax,
        )

        results['evaluations'][eval_name] = {
            'label': SURVEYS[eval_name]['label'],
            'n_files': len(eval_files),
            'mean_iou': summary['mean_iou'],
            'mean_dice': summary['mean_dice'],
            'mean_bottom_error_m': summary['mean_bottom_error_m'],
            'std_iou': summary['std_iou'],
            'per_file': [
                {
                    'file': m.file_name,
                    'iou': m.iou,
                    'dice': m.dice,
                    'bottom_error_mean_m': m.bottom_error_mean_m,
                }
                for m in all_metrics
            ],
        }

        print(f"  Mean IoU: {summary['mean_iou']:.4f}  "
              f"Mean Dice: {summary['mean_dice']:.4f}  "
              f"Bottom err: {summary['mean_bottom_error_m']:.2f}m")

    # ── 6. Save results ──────────────────────────────────────────────
    results_path = run_dir / "cross_survey_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    return results


# ---------------------------------------------------------------------------
# Analysis: aggregate results into generalization matrix
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> dict[str, dict]:
    """Load all cross_survey_results.json files from subdirectories."""
    results = {}
    for sub in sorted(results_dir.iterdir()):
        json_path = sub / "cross_survey_results.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            results[data['train_survey']] = data
    return results


def build_matrix(results: dict[str, dict],
                 metric: str = 'mean_iou') -> tuple[np.ndarray, list[str]]:
    """Build the NxN generalization matrix.

    matrix[i, j] = metric when trained on survey i, evaluated on survey j.
    """
    survey_names = sorted(results.keys())
    n = len(survey_names)
    matrix = np.full((n, n), np.nan)

    for i, train_name in enumerate(survey_names):
        evals = results[train_name].get('evaluations', {})
        for j, eval_name in enumerate(survey_names):
            if eval_name in evals:
                matrix[i, j] = evals[eval_name].get(metric, np.nan)

    return matrix, survey_names


def plot_matrix(matrix: np.ndarray, survey_names: list[str],
                metric: str = 'mean_iou',
                save_path: str | Path | None = None):
    """Plot the generalization matrix as an annotated heatmap."""
    import matplotlib.pyplot as plt

    labels = [SURVEYS[s]['label'] for s in survey_names]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    if metric in ('mean_iou', 'mean_dice'):
        cmap = 'RdYlGn'
        vmin = max(0.5, vmin - 0.05)
        vmax = min(1.0, vmax + 0.01)
    else:
        cmap = 'RdYlGn_r'
        vmin = 0
        vmax = min(50, vmax * 1.2)

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            if metric in ('mean_iou', 'mean_dice'):
                text = f"{val:.3f}"
            else:
                text = f"{val:.1f}"
            color = 'white' if val < (vmin + vmax) / 2 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=9, color=color, fontweight=weight)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Evaluation survey', fontsize=12)
    ax.set_ylabel('Training survey', fontsize=12)

    metric_label = metric.replace('_', ' ').replace('mean ', '').title()
    ax.set_title(f'Cross-Survey Generalization Matrix ({metric_label})',
                 fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric_label, fontsize=11)

    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor='black',
                                   linewidth=2))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")

    return fig


def print_summary(matrix: np.ndarray, survey_names: list[str],
                  metric: str):
    """Print diagonal vs off-diagonal summary."""
    n = len(survey_names)
    diag = np.array([matrix[i, i] for i in range(n)
                     if not np.isnan(matrix[i, i])])
    off_diag = np.array([matrix[i, j] for i in range(n) for j in range(n)
                         if i != j and not np.isnan(matrix[i, j])])

    print("\n" + "=" * 60)
    print(f"CROSS-SURVEY GENERALIZATION SUMMARY ({metric})")
    print("=" * 60)

    print("\nDiagonal (in-survey performance):")
    print(f"  Mean: {diag.mean():.4f}  Std: {diag.std():.4f}  "
          f"Range: [{diag.min():.4f}, {diag.max():.4f}]")

    print("\nOff-diagonal (cross-survey generalization):")
    print(f"  Mean: {off_diag.mean():.4f}  Std: {off_diag.std():.4f}  "
          f"Range: [{off_diag.min():.4f}, {off_diag.max():.4f}]")

    gap = diag.mean() - off_diag.mean()
    print(f"\nGeneralization gap (diag - off-diag): {gap:.4f}")

    if metric in ('mean_iou', 'mean_dice'):
        if gap < 0.02:
            print("  → Minimal gap: surveys are highly compatible.")
        elif gap < 0.05:
            print("  → Moderate gap: some domain shift present.")
        else:
            print("  → Large gap: significant distributional differences.")
    print()

    print(f"{'Survey':<16} {'In-survey':>10} {'Cross-survey':>13} {'Gap':>8}")
    print("-" * 50)
    for i, name in enumerate(survey_names):
        label = SURVEYS[name]['label']
        in_val = matrix[i, i]
        cross_vals = [matrix[i, j] for j in range(n)
                      if j != i and not np.isnan(matrix[i, j])]
        cross_mean = np.mean(cross_vals) if cross_vals else np.nan
        g = in_val - cross_mean if not np.isnan(in_val) else np.nan
        print(f"{label:<16} {in_val:>10.4f} {cross_mean:>13.4f} {g:>8.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Cross-survey generalization training and analysis")
    sub = p.add_subparsers(dest="command")

    # ── train subcommand ─────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train on one survey, eval on all")
    g = train_p.add_mutually_exclusive_group(required=True)
    g.add_argument("--train-survey", type=str, choices=list(SURVEYS.keys()))
    g.add_argument("--survey-index", type=int,
                   help="1-based index into SURVEYS (for SLURM arrays)")
    train_p.add_argument("--zarr-dir", type=Path,
                         default=Path("Datasets/echopype_zarr"))
    train_p.add_argument("--cache-dir", type=Path,
                         default=Path("Datasets/patch_cache"))
    train_p.add_argument("--output-dir", type=Path,
                         default=Path("code/checkpoints/cross_survey"))
    train_p.add_argument("--max-train-files", type=int, default=80)
    train_p.add_argument("--max-eval-files", type=int, default=25)
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--batch-size", type=int, default=4)
    train_p.add_argument("--num-workers", type=int, default=4)
    train_p.add_argument("--early-stopping-patience", type=int, default=10)
    train_p.add_argument("--seed", type=int, default=42)

    # ── analyze subcommand ───────────────────────────────────────────
    analyze_p = sub.add_parser("analyze", help="Aggregate results into matrix")
    analyze_p.add_argument("--results-dir", type=Path,
                           default=Path("code/checkpoints/cross_survey"))
    analyze_p.add_argument("--metric", type=str, default="mean_iou",
                           choices=["mean_iou", "mean_dice",
                                    "mean_bottom_error_m"])
    analyze_p.add_argument("--save-fig", type=str, default=None)
    analyze_p.add_argument("--no-plot", action="store_true")

    return p.parse_args()


def main():
    args = _parse_args()

    if args.command == "train":
        survey_names = list(SURVEYS.keys())
        if args.survey_index is not None:
            if args.survey_index < 1 or args.survey_index > len(survey_names):
                print(f"ERROR: --survey-index {args.survey_index} out of "
                      f"range [1, {len(survey_names)}]", file=sys.stderr)
                sys.exit(1)
            train_survey = survey_names[args.survey_index - 1]
        else:
            train_survey = args.train_survey

        run_cross_survey(
            train_survey=train_survey,
            zarr_dir=args.zarr_dir,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            max_train_files=args.max_train_files,
            max_eval_files=args.max_eval_files,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            early_stopping_patience=args.early_stopping_patience,
            seed=args.seed,
        )

    elif args.command == "analyze":
        results = load_results(args.results_dir)
        if not results:
            print(f"No results found in {args.results_dir}")
            print("Run 'cross_survey train' first (or submit via SLURM).")
            return

        print(f"Loaded results for {len(results)} surveys: "
              f"{', '.join(sorted(results.keys()))}")

        matrix, survey_names = build_matrix(results, metric=args.metric)
        print_summary(matrix, survey_names, args.metric)

        if not args.no_plot:
            save_path = (args.save_fig or
                         f"Figures/cross_survey_{args.metric}.pdf")
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plot_matrix(matrix, survey_names, metric=args.metric,
                        save_path=save_path)

    else:
        print("Usage: python -m validation.dataset_shift.cross_survey "
              "{train,analyze}")
        sys.exit(1)


if __name__ == "__main__":
    main()
