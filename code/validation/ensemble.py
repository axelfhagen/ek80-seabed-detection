"""Deep ensemble inference and evaluation for seabed segmentation.

Loads N independently trained U-Net checkpoints, runs inference with
each, and aggregates predictions via probability averaging.  The
inter-model disagreement (standard deviation) provides a calibrated
uncertainty estimate without requiring dropout at inference time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from deep_learning.architectures import create_model
from deep_learning.unet_model import predict_full_echogram
from deep_learning.evaluate import (
    FileMetrics,
    load_file_data,
    extract_bottom_line,
    bottom_depth_error,
    plot_evaluation,
    plot_reliability_diagram,
)
from validation.uncertainty import prediction_entropy, bottom_uncertainty_band


# ---------------------------------------------------------------------------
# Load ensemble
# ---------------------------------------------------------------------------


def load_ensemble(
    checkpoint_paths: list[Path],
    device: str = "cuda",
    in_channels: int = 3,
) -> tuple[list[torch.nn.Module], dict]:
    """Load N models from checkpoint ``.pth`` files.

    Members may use *different* architectures and encoders — a heterogeneous
    ensemble draws its diversity from exactly that. Only the data geometry
    (``target_size``, ``freq1_hz``, ``freq2_hz``) must match so every member
    consumes the same input and yields an aligned probability map.

    Parameters
    ----------
    checkpoint_paths : paths to ``.pth`` checkpoint files
    device           : torch device string
    in_channels      : number of input channels (must match training)

    Returns
    -------
    models : list of segmentation models in eval mode on *device*
    config : dict with shared data config from checkpoint 0; additionally
             carries ``_member_labels`` (per-member "arch/encoder" strings).
    """
    if len(checkpoint_paths) < 2:
        raise ValueError(f"Need at least 2 checkpoints, got {len(checkpoint_paths)}")

    models = []
    ref_cfg = None
    member_labels: list[str] = []

    for i, ckpt_path in enumerate(checkpoint_paths):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]

        if ref_cfg is None:
            ref_cfg = cfg
        else:
            for key in ("target_size", "freq1_hz", "freq2_hz"):
                if cfg.get(key) != ref_cfg.get(key):
                    raise ValueError(
                        f"Checkpoint {ckpt_path} has {key}={cfg.get(key)}, "
                        f"expected {ref_cfg.get(key)} — data geometry must match."
                    )

        architecture = cfg.get("architecture", "smp_unet")
        encoder = cfg.get("encoder_name", "resnet34")
        model = create_model(
            architecture=architecture,
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
            base_filters=cfg.get("base_filters", 32),
            depth=cfg.get("model_depth", 4),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval().to(device)
        models.append(model)
        member_labels.append(f"{architecture}/{encoder}")
        print(
            f"  Loaded member {i + 1}/{len(checkpoint_paths)}: {ckpt_path.name} "
            f"[{architecture}/{encoder}] (val_iou={ckpt.get('val_iou', '?')})"
        )

    ref_cfg = dict(ref_cfg)
    ref_cfg["_member_labels"] = member_labels
    return models, ref_cfg


# ---------------------------------------------------------------------------
# Ensemble inference
# ---------------------------------------------------------------------------


def ensemble_predict_file(
    models: list[torch.nn.Module],
    zarr_path: Path,
    freq1_hz: float,
    freq2_hz: float,
    target_size: tuple[int, int],
    device: str,
    patch_width: int = 256,
    stride: int = 128,
    max_range_m: Optional[float] = None,
    max_range_samples: Optional[int] = None,
    vmin: float = -80.0,
    vmax: float = -34.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run ensemble inference on a single zarr file.

    Loads the data once and passes it through every model.

    Returns
    -------
    mean_prob     : (n_range, n_pings) float32
    std_prob      : (n_range, n_pings) float32 — inter-model disagreement
    all_preds     : (N, n_range, n_pings) float32
    gt_mask       : (n_range, n_pings) uint8
    gt_bottom_idx : (n_pings,) int
    sv_channels   : (3, n_range, n_pings) float32
    range_arr     : (n_range,) float64
    """
    sv_channels, gt_mask, gt_bottom_idx, range_arr = load_file_data(
        zarr_path,
        freq1_hz,
        freq2_hz,
        max_range_m=max_range_m,
        max_range_samples=max_range_samples,
        vmin=vmin,
        vmax=vmax,
    )

    n = len(models)
    _, n_range, n_pings = sv_channels.shape
    all_preds = np.zeros((n, n_range, n_pings), dtype=np.float32)

    with torch.no_grad():
        for i, model in enumerate(models):
            all_preds[i] = predict_full_echogram(
                model, sv_channels, target_size, device, patch_width=patch_width, stride=stride
            )

    mean_prob = all_preds.mean(axis=0)
    std_prob = all_preds.std(axis=0)

    return mean_prob, std_prob, all_preds, gt_mask, gt_bottom_idx, sv_channels, range_arr


# ---------------------------------------------------------------------------
# Ensemble evaluation (single file)
# ---------------------------------------------------------------------------


def _compute_metrics(mean_prob, gt_mask, gt_bottom_idx, range_arr, file_name, threshold=0.5):
    """Pixel metrics + bottom-depth error from an ensemble mean map."""
    pred_binary = (mean_prob >= threshold).astype(np.uint8)
    gt_flat = gt_mask.astype(bool).ravel()
    pred_flat = pred_binary.astype(bool).ravel()

    tp = int(np.sum(gt_flat & pred_flat))
    fp = int(np.sum(~gt_flat & pred_flat))
    fn = int(np.sum(gt_flat & ~pred_flat))
    tn = int(np.sum(~gt_flat & ~pred_flat))

    eps = 1e-7
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    pred_bottom_idx = extract_bottom_line(mean_prob, threshold=threshold)
    errors_m = bottom_depth_error(pred_bottom_idx, gt_bottom_idx, range_arr)

    if len(errors_m) > 0:
        err_mean = float(np.mean(errors_m))
        err_median = float(np.median(errors_m))
        err_p95 = float(np.percentile(errors_m, 95))
        n_valid = len(errors_m)
    else:
        err_mean = err_median = err_p95 = float("nan")
        n_valid = 0

    return FileMetrics(
        file_name=file_name,
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        bottom_error_mean_m=err_mean,
        bottom_error_median_m=err_median,
        bottom_error_p95_m=err_p95,
        n_pings_with_bottom=n_valid,
    )


def ensemble_evaluate_file(
    models: list[torch.nn.Module],
    zarr_path: Path,
    freq1_hz: float,
    freq2_hz: float,
    target_size: tuple[int, int],
    device: str,
    threshold: float = 0.5,
    **kwargs,
) -> tuple[FileMetrics, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ensemble inference + metrics for one zarr file.

    Returns
    -------
    metrics     : FileMetrics (computed from ensemble mean)
    mean_prob   : (n_range, n_pings) float32
    std_prob    : (n_range, n_pings) float32
    all_preds   : (N, n_range, n_pings) float32
    gt_mask     : (n_range, n_pings) uint8
    sv_channels : (3, n_range, n_pings) float32
    range_arr   : (n_range,) float64
    """
    (mean_prob, std_prob, all_preds, gt_mask, gt_bottom_idx, sv_channels, range_arr) = (
        ensemble_predict_file(models, zarr_path, freq1_hz, freq2_hz, target_size, device, **kwargs)
    )

    metrics = _compute_metrics(
        mean_prob,
        gt_mask,
        gt_bottom_idx,
        range_arr,
        file_name=Path(zarr_path).name,
        threshold=threshold,
    )

    return metrics, mean_prob, std_prob, all_preds, gt_mask, sv_channels, range_arr


# ---------------------------------------------------------------------------
# Ensemble evaluation (multiple files)
# ---------------------------------------------------------------------------


def ensemble_evaluate_files(
    models: list[torch.nn.Module],
    zarr_paths: list[Path],
    freq1_hz: float,
    freq2_hz: float,
    target_size: tuple[int, int],
    device: str,
    output_dir: Optional[Path] = None,
    threshold: float = 0.5,
    **kwargs,
) -> tuple[list[FileMetrics], dict, list[dict]]:
    """Evaluate the ensemble on multiple zarr files.

    Parameters
    ----------
    output_dir : if given, save per-file visualisation PNGs here

    Returns
    -------
    all_metrics      : list of FileMetrics
    summary          : aggregate dict
    uncertainty_info : per-file uncertainty statistics
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[FileMetrics] = []
    uncertainty_info: list[dict] = []

    for zp in zarr_paths:
        fname = Path(zp).stem
        print(f"Evaluating {Path(zp).name} ...")

        (metrics, mean_prob, std_prob, all_preds, gt_mask, sv_channels, range_arr) = (
            ensemble_evaluate_file(
                models, zp, freq1_hz, freq2_hz, target_size, device, threshold=threshold, **kwargs
            )
        )

        all_metrics.append(metrics)

        entropy = prediction_entropy(mean_prob)
        upper_d, lower_d, band_w = bottom_uncertainty_band(mean_prob, range_arr)

        u_info = {
            "file": Path(zp).name,
            "disagreement_mean": float(np.mean(std_prob)),
            "disagreement_max": float(np.max(std_prob)),
            "entropy_mean": float(np.mean(entropy)),
            "band_width_mean_m": float(np.nanmean(band_w)),
            "band_width_median_m": float(np.nanmedian(band_w)),
        }
        uncertainty_info.append(u_info)

        print(
            f"  IoU={metrics.iou:.4f}  Dice={metrics.dice:.4f}  "
            f"Bottom err: mean={metrics.bottom_error_mean_m:.2f}m  "
            f"Disagreement: {u_info['disagreement_mean']:.4f}"
        )

        if output_dir is not None:
            plot_evaluation(
                sv_channels,
                gt_mask,
                mean_prob,
                range_arr,
                metrics,
                threshold=threshold,
                save_path=output_dir / f"{fname}_eval.png",
            )

            plot_reliability_diagram(
                mean_prob,
                gt_mask,
                save_path=output_dir / f"{fname}_reliability.png",
                title=f"Reliability — {fname} (ensemble, N={len(models)})",
            )

            plot_ensemble_uncertainty(
                sv_channels,
                mean_prob,
                std_prob,
                range_arr,
                file_name=Path(zp).name,
                save_path=output_dir / f"{fname}_uncertainty.png",
            )

            import matplotlib.pyplot as plt

            plt.close("all")

    # Aggregate
    ious = [m.iou for m in all_metrics]
    dices = [m.dice for m in all_metrics]
    errs = [m.bottom_error_mean_m for m in all_metrics if not np.isnan(m.bottom_error_mean_m)]

    summary = {
        "n_files": len(all_metrics),
        "n_members": len(models),
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "mean_bottom_error_m": float(np.mean(errs)) if errs else float("nan"),
        "std_iou": float(np.std(ious)),
        "mean_disagreement": float(np.mean([u["disagreement_mean"] for u in uncertainty_info])),
        "mean_entropy": float(np.mean([u["entropy_mean"] for u in uncertainty_info])),
    }

    return all_metrics, summary, uncertainty_info


# ---------------------------------------------------------------------------
# Uncertainty visualisation
# ---------------------------------------------------------------------------


def plot_ensemble_uncertainty(
    sv_channels: np.ndarray,
    mean_prob: np.ndarray,
    std_prob: np.ndarray,
    range_array: np.ndarray,
    file_name: str,
    save_path: Optional[Path] = None,
):
    """Four-panel uncertainty figure: echogram, prediction, disagreement, entropy."""
    import matplotlib.pyplot as plt

    norm1 = sv_channels[0]
    n_range, n_pings = norm1.shape
    extent = [0, n_pings, range_array[-1], range_array[0]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)

    axes[0, 0].imshow(
        norm1, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper", extent=extent
    )
    axes[0, 0].set_title("Echogram (freq 1)")
    axes[0, 0].set_ylabel("Range (m)")

    axes[0, 1].imshow(
        mean_prob, aspect="auto", cmap="gray_r", vmin=0, vmax=1, origin="upper", extent=extent
    )
    axes[0, 1].set_title("Ensemble mean prediction")

    im_std = axes[1, 0].imshow(
        std_prob, aspect="auto", cmap="hot", vmin=0, vmax=0.25, origin="upper", extent=extent
    )
    axes[1, 0].set_title("Inter-model disagreement (std)")
    axes[1, 0].set_ylabel("Range (m)")
    axes[1, 0].set_xlabel("Ping")
    fig.colorbar(im_std, ax=axes[1, 0], fraction=0.046, pad=0.04)

    entropy = prediction_entropy(mean_prob)
    im_ent = axes[1, 1].imshow(
        entropy, aspect="auto", cmap="inferno", vmin=0, vmax=1.0, origin="upper", extent=extent
    )
    axes[1, 1].set_title("Prediction entropy")
    axes[1, 1].set_xlabel("Ping")
    fig.colorbar(im_ent, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{file_name}  —  disagreement mean={np.mean(std_prob):.4f}  "
        f"entropy mean={np.mean(entropy):.4f}",
        fontsize=12,
    )
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig
