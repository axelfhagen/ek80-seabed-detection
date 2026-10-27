"""Requested ensemble uncertainty maps for one echopype zarr file.

This script avoids ``xarray.open_zarr`` because the local zarr-v3 stores can
hang in the current environment. It reads the needed v3 chunks directly, runs
the ResNet34 seed ensemble, and writes:

- conformal bottom band overlay
- ensemble/member probability maps
- ensemble probability, entropy, and disagreement maps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from numcodecs import Zstd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, savefig
from deep_learning.architectures import create_model
from deep_learning.evaluate import extract_bottom_line
from deep_learning.training.survey_groups import get_split_files
from raw_sensor_integration.bottom_labelling import (
    bottom_to_sample_index,
    detect_bottom_ensemble_pipeline,
)
from raw_sensor_integration.preprocessing import normalize_echogram
from validation.conformal import conformal_quantile
from validation.uncertainty import prediction_entropy

DEFAULT_ZARR_NAME = "D20241006-T173200.zarr"
DEFAULT_CKPT_DIR = PROJECT_ROOT / "code/checkpoints/deep_ensemble_resnet34_seeds"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Figures/ensemble/requested_D20241006_T173200"


def _dtype(meta: dict) -> np.dtype:
    name = meta["data_type"]
    if name == "float64":
        return np.dtype("<f8")
    if name == "float32":
        return np.dtype("<f4")
    if name == "int64":
        return np.dtype("<i8")
    if name == "int32":
        return np.dtype("<i4")
    raise ValueError(f"Unsupported zarr dtype: {name}")


def read_zarr_v3_array(array_path: Path) -> np.ndarray:
    """Read a zarr-v3 array whose chunks use bytes+zstd codecs."""
    meta = json.loads((array_path / "zarr.json").read_text())
    shape = tuple(meta["shape"])
    chunks = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    dtype = _dtype(meta)
    fill_value = meta.get("fill_value", 0)
    if fill_value == "NaN":
        fill_value = np.nan

    out = np.full(shape, fill_value, dtype=dtype)
    codec = Zstd(level=0)
    grid_shape = tuple((s + c - 1) // c for s, c in zip(shape, chunks))
    for chunk_idx in np.ndindex(grid_shape):
        chunk_file = array_path / "c" / Path(*map(str, chunk_idx))
        if not chunk_file.exists():
            continue
        raw = codec.decode(chunk_file.read_bytes())
        block = np.frombuffer(raw, dtype=dtype).reshape(chunks)
        out_slices = []
        block_slices = []
        for axis, ci in enumerate(chunk_idx):
            start = ci * chunks[axis]
            stop = min(start + chunks[axis], shape[axis])
            out_slices.append(slice(start, stop))
            block_slices.append(slice(0, stop - start))
        out[tuple(out_slices)] = block[tuple(block_slices)]
    return out


def read_zarr_v3_channel(array_path: Path, channel_idx: int) -> np.ndarray:
    """Read one channel from a (channel, ping_time, range_sample) v3 array."""
    meta = json.loads((array_path / "zarr.json").read_text())
    shape = tuple(meta["shape"])
    chunks = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    dtype = _dtype(meta)
    out = np.full((shape[1], shape[2]), np.nan, dtype=dtype)
    codec = Zstd(level=0)
    n_ping_chunks = (shape[1] + chunks[1] - 1) // chunks[1]
    n_range_chunks = (shape[2] + chunks[2] - 1) // chunks[2]
    for ping_chunk in range(n_ping_chunks):
        for range_chunk in range(n_range_chunks):
            chunk_file = array_path / "c" / str(channel_idx) / str(ping_chunk) / str(range_chunk)
            if not chunk_file.exists():
                continue
            raw = codec.decode(chunk_file.read_bytes())
            block = np.frombuffer(raw, dtype=dtype).reshape(chunks)
            ps = slice(ping_chunk * chunks[1], min((ping_chunk + 1) * chunks[1], shape[1]))
            rs = slice(range_chunk * chunks[2], min((range_chunk + 1) * chunks[2], shape[2]))
            out[ps, rs] = block[0, : ps.stop - ps.start, : rs.stop - rs.start]
    return out.T


def load_echopype_inputs(
    zarr_path: Path, freq1_hz: float, freq2_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq = read_zarr_v3_array(zarr_path / "frequency_nominal").astype(float)
    i1 = int(np.argmin(np.abs(freq - freq1_hz)))
    i2 = int(np.argmin(np.abs(freq - freq2_hz)))

    sv1 = read_zarr_v3_channel(zarr_path / "Sv", i1).astype(np.float32)
    sv2 = read_zarr_v3_channel(zarr_path / "Sv", i2).astype(np.float32)
    echo_range = read_zarr_v3_channel(zarr_path / "echo_range", i1).astype(np.float64)
    with np.errstate(all="ignore"):
        range_arr = np.nanmean(echo_range, axis=1)
    valid_range = np.isfinite(range_arr)
    keep = np.where(valid_range)[0]
    sv1 = sv1[keep]
    sv2 = sv2[keep]
    range_arr = range_arr[keep]

    finite_rows = np.where(~np.isnan(sv1).all(axis=1) | ~np.isnan(sv2).all(axis=1))[0]
    if finite_rows.size:
        sample_dist = float(range_arr[1] - range_arr[0]) if len(range_arr) > 1 else 1.0
        margin = max(50, int(10.0 / sample_dist))
        trim = min(len(range_arr), int(finite_rows[-1]) + margin)
        sv1, sv2, range_arr = sv1[:trim], sv2[:trim], range_arr[:trim]

    norm1 = normalize_echogram(sv1, method="minmax", vmin=-80.0, vmax=-34.0, fill_nan=0.0).astype(
        np.float32
    )
    norm2 = normalize_echogram(sv2, method="minmax", vmin=-80.0, vmax=-34.0, fill_nan=0.0).astype(
        np.float32
    )
    return np.stack([norm1, norm2, norm1 - norm2], axis=0), sv1, range_arr


def load_reference_bottom(zarr_path: Path, sv38: np.ndarray, range_arr: np.ndarray) -> np.ndarray:
    heave_path = zarr_path / "heave"
    heave = (
        read_zarr_v3_array(heave_path).astype(np.float64)
        if heave_path.exists()
        else np.zeros(sv38.shape[1])
    )
    bottom_range, _, _ = detect_bottom_ensemble_pipeline(
        sv38,
        range_arr,
        heave=heave,
        pulse_duration=0.0,
        frequency_hz=38000.0,
        transducer_draft=0.0,
        threshold_log_sv=-31.0,
        detectors=("simple", "edge"),
    )
    return bottom_to_sample_index(bottom_range, range_arr)


def load_models(checkpoint_dir: Path, device: str) -> tuple[list[torch.nn.Module], dict]:
    ckpts = sorted(checkpoint_dir.glob("member_seed*/smp_unet_resnet34_best.pth"))
    if len(ckpts) < 2:
        raise FileNotFoundError(f"Expected at least two ensemble checkpoints in {checkpoint_dir}")
    models: list[torch.nn.Module] = []
    ref_cfg = None
    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        if ref_cfg is None:
            ref_cfg = dict(cfg)
        model = create_model(
            architecture=cfg.get("architecture", "smp_unet"),
            encoder_name=cfg.get("encoder_name", "resnet34"),
            encoder_weights=None,
            in_channels=3,
            classes=1,
            base_filters=cfg.get("base_filters", 32),
            depth=cfg.get("model_depth", 4),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval().to(device)
        models.append(model)
        print(f"Loaded {ckpt_path}", flush=True)
    assert ref_cfg is not None
    return models, ref_cfg


@torch.no_grad()
def predict_batched(
    model: torch.nn.Module,
    sv_channels: np.ndarray,
    target_size: tuple[int, int],
    device: str,
    patch_width: int = 512,
    stride: int = 256,
    batch_size: int = 4,
) -> np.ndarray:
    _, n_range, n_pings = sv_channels.shape
    h, w = target_size
    starts = list(range(0, max(n_pings - patch_width + 1, 1), stride))
    if n_pings > patch_width and starts[-1] != n_pings - patch_width:
        starts.append(n_pings - patch_width)

    pred_sum = np.zeros((n_range, n_pings), dtype=np.float32)
    pred_count = np.zeros((n_range, n_pings), dtype=np.float32)
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        patches = []
        widths = []
        for start in batch_starts:
            end = min(start + patch_width, n_pings)
            patch = sv_channels[:, :, start:end]
            widths.append(end - start)
            if end - start < patch_width:
                pad = np.zeros(
                    (sv_channels.shape[0], n_range, patch_width - (end - start)),
                    dtype=sv_channels.dtype,
                )
                patch = np.concatenate([patch, pad], axis=2)
            patch_hwc = np.transpose(patch, (1, 2, 0))
            patch_resized = cv2.resize(patch_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
            patches.append(np.transpose(patch_resized, (2, 0, 1)))
        x = torch.from_numpy(np.stack(patches)).float().to(device)
        probs = torch.sigmoid(model(x)).squeeze(1).detach().cpu().numpy()
        for prob, start, width in zip(probs, batch_starts, widths):
            prob_orig = cv2.resize(prob, (patch_width, n_range), interpolation=cv2.INTER_LINEAR)[
                :, :width
            ]
            pred_sum[:, start : start + width] += prob_orig
            pred_count[:, start : start + width] += 1.0
    return np.divide(pred_sum, pred_count, out=np.zeros_like(pred_sum), where=pred_count > 0)


def depths_from_idx(idx: np.ndarray, range_arr: np.ndarray) -> np.ndarray:
    out = np.full(idx.shape, np.nan, dtype=float)
    valid = idx >= 0
    out[valid] = range_arr[idx[valid]]
    return out


def predict_ensemble(
    models: list[torch.nn.Module],
    sv: np.ndarray,
    target_size: tuple[int, int],
    device: str,
    patch_width: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = []
    for i, model in enumerate(models, 1):
        print(f"Predicting member {i}/{len(models)} ...", flush=True)
        preds.append(
            predict_batched(model, sv, target_size, device, patch_width=patch_width, stride=stride)
        )
    all_preds = np.stack(preds)
    return all_preds.mean(axis=0), all_preds.std(axis=0), all_preds


def calibrate_constant_q(
    models: list[torch.nn.Module],
    cfg: dict,
    paths: list[Path],
    device: str,
    n_calib: int,
    threshold: float,
    patch_width: int,
    stride: int,
    alpha: float,
) -> float:
    errs = []
    for zp in paths[:n_calib]:
        print(f"Calibrating on {zp.name} ...", flush=True)
        sv, sv38, ra = load_echopype_inputs(zp, cfg["freq1_hz"], cfg["freq2_hz"])
        mean_prob, _, _ = predict_ensemble(
            models, sv, tuple(cfg["target_size"]), device, patch_width, stride
        )
        pred_d = depths_from_idx(extract_bottom_line(mean_prob, threshold=threshold), ra)
        gt_d = depths_from_idx(load_reference_bottom(zp, sv38, ra), ra)
        m = np.isfinite(pred_d) & np.isfinite(gt_d) & (pred_d > 0) & (gt_d > 0)
        errs.append(np.abs(pred_d[m] - gt_d[m]))
    return float(conformal_quantile(np.concatenate(errs), alpha=alpha))


def plot_conformal(
    zarr_name: str,
    sv: np.ndarray,
    mean_prob: np.ndarray,
    std_prob: np.ndarray,
    all_preds: np.ndarray,
    gt_idx: np.ndarray,
    range_arr: np.ndarray,
    q_const: float,
    threshold: float,
    out_path: Path,
) -> None:
    pred_idx = extract_bottom_line(mean_prob, threshold=threshold)
    pred_d = depths_from_idx(pred_idx, range_arr)
    gt_d = depths_from_idx(gt_idx, range_arr)
    member_depths = np.stack(
        [depths_from_idx(extract_bottom_line(p, threshold), range_arr) for p in all_preds]
    )
    lo = np.nanmin(member_depths, axis=0)
    hi = np.nanmax(member_depths, axis=0)
    sig = std_prob[np.clip(pred_idx, 0, std_prob.shape[0] - 1), np.arange(std_prob.shape[1])]

    x = np.arange(mean_prob.shape[1])
    extent = [0, mean_prob.shape[1], range_arr[-1], range_arr[0]]
    dvals = np.concatenate([pred_d, gt_d, lo, hi])
    dvals = dvals[np.isfinite(dvals)]
    top = max(0.0, np.percentile(dvals, 1) - 8.0)
    bottom = min(float(range_arr[-1]), np.percentile(dvals, 99) + 18.0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)
    axes[0].imshow(
        sv[0], aspect="auto", origin="upper", cmap="viridis", vmin=0, vmax=1, extent=extent
    )
    axes[0].fill_between(x, lo, hi, color="#9467bd", alpha=0.45, lw=0, label="Member spread")
    axes[0].plot(x, gt_d, color="#d62728", lw=1.2, label="Reference")
    axes[0].plot(x, pred_d, color="#ffbf00", lw=1.0, label="Ensemble mean")
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_ylim(bottom, top)
    axes[0].legend(loc="upper right", fontsize=10)

    axes[1].imshow(
        sv[0], aspect="auto", origin="upper", cmap="viridis", vmin=0, vmax=1, extent=extent
    )
    axes[1].fill_between(
        x,
        pred_d - q_const,
        pred_d + q_const,
        color="#ff7f0e",
        alpha=0.45,
        lw=0,
        label=f"90% constant conformal band (+/-{q_const:.2f} m)",
    )
    axes[1].plot(x, pred_d, color="#ffbf00", lw=1.0, label="Ensemble mean")
    axes[1].plot(x, gt_d, color="#d62728", lw=1.0, label="Reference")
    axes[1].set_ylabel("Depth (m)")
    axes[1].set_xlabel("Ping")
    axes[1].set_ylim(bottom, top)
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].set_title(
        f"Median probability disagreement at predicted bottom: {np.nanmedian(sig):.3f}", fontsize=12
    )
    fig.suptitle(f"{zarr_name}: conformal bottom band", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, out_path)
    plt.close(fig)


def _map_panel(ax, data, title, extent, cmap, vmin=None, vmax=None):
    im = ax.imshow(
        data, aspect="auto", origin="upper", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Depth (m)")
    return im


def plot_maps(
    zarr_name: str,
    sv: np.ndarray,
    mean_prob: np.ndarray,
    std_prob: np.ndarray,
    all_preds: np.ndarray,
    range_arr: np.ndarray,
    out_dir: Path,
) -> None:
    extent = [0, mean_prob.shape[1], range_arr[-1], range_arr[0]]
    entropy = prediction_entropy(mean_prob)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
    ims = [
        _map_panel(axes[0, 0], sv[0], "38 kHz input (normalised)", extent, "viridis", 0, 1),
        _map_panel(axes[0, 1], mean_prob, "Ensemble mean probability", extent, "magma", 0, 1),
        _map_panel(axes[1, 0], entropy, "Probability entropy", extent, "inferno", 0, 1),
        _map_panel(
            axes[1, 1],
            std_prob,
            "Ensemble probability std.",
            extent,
            "cividis",
            0,
            float(np.nanpercentile(std_prob, 99)),
        ),
    ]
    for ax, im in zip(axes.ravel(), ims):
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    axes[1, 0].set_xlabel("Ping")
    axes[1, 1].set_xlabel("Ping")
    fig.suptitle(f"{zarr_name}: probability and uncertainty maps", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, out_dir / f"{Path(zarr_name).stem}_probability_entropy_maps.png")
    plt.close(fig)

    n = all_preds.shape[0]
    fig, axes = plt.subplots(1, n + 1, figsize=(4.8 * (n + 1), 5.2), sharex=True, sharey=True)
    for i in range(n):
        im = _map_panel(axes[i], all_preds[i], f"Member {i + 1} probability", extent, "magma", 0, 1)
        axes[i].set_xlabel("Ping")
    im = _map_panel(axes[-1], mean_prob, "Ensemble mean", extent, "magma", 0, 1)
    axes[-1].set_xlabel("Ping")
    fig.subplots_adjust(left=0.045, right=0.89, bottom=0.12, top=0.84, wspace=0.16)
    cax = fig.add_axes([0.91, 0.16, 0.012, 0.68])
    fig.colorbar(im, cax=cax)
    fig.suptitle(f"{zarr_name}: member probability maps", fontsize=16)
    savefig(fig, out_dir / f"{Path(zarr_name).stem}_member_probability_maps.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zarr", default=str(PROJECT_ROOT / "Datasets/echopype_zarr" / DEFAULT_ZARR_NAME)
    )
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-calib", type=int, default=5)
    parser.add_argument(
        "--q-const-m",
        type=float,
        default=None,
        help="reuse an existing constant conformal half-width and skip calibration",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patch-width", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    args = parser.parse_args()

    apply_thesis_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = Path(args.zarr)

    models, cfg = load_models(Path(args.checkpoint_dir), args.device)
    if args.q_const_m is None:
        split_files = get_split_files(PROJECT_ROOT / "Datasets/echopype_zarr")
        q_const = calibrate_constant_q(
            models,
            cfg,
            sorted(split_files["calibration"]),
            args.device,
            args.n_calib,
            args.threshold,
            args.patch_width,
            args.stride,
            args.alpha,
        )
    else:
        q_const = float(args.q_const_m)
    print(f"q_const={q_const:.3f} m", flush=True)

    sv, sv38, range_arr = load_echopype_inputs(zarr_path, cfg["freq1_hz"], cfg["freq2_hz"])
    gt_idx = load_reference_bottom(zarr_path, sv38, range_arr)
    mean_prob, std_prob, all_preds = predict_ensemble(
        models, sv, tuple(cfg["target_size"]), args.device, args.patch_width, args.stride
    )

    stem = zarr_path.stem
    plot_conformal(
        zarr_path.name,
        sv,
        mean_prob,
        std_prob,
        all_preds,
        gt_idx,
        range_arr,
        q_const,
        args.threshold,
        out_dir / f"{stem}_conformal_band.png",
    )
    plot_maps(zarr_path.name, sv, mean_prob, std_prob, all_preds, range_arr, out_dir)
    np.savez_compressed(
        out_dir / f"{stem}_maps.npz",
        mean_prob=mean_prob.astype(np.float32),
        std_prob=std_prob.astype(np.float32),
        entropy=prediction_entropy(mean_prob).astype(np.float32),
        range_arr=range_arr.astype(np.float32),
        q_const_m=np.array(q_const, dtype=np.float32),
    )
    summary = {
        "zarr": str(zarr_path),
        "checkpoint_dir": str(Path(args.checkpoint_dir)),
        "n_members": len(models),
        "n_calib": args.n_calib if args.q_const_m is None else None,
        "q_const_source": "calibrated" if args.q_const_m is None else "provided",
        "alpha": args.alpha,
        "q_const_m": q_const,
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
