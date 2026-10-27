"""Rolling-window inference as a free ensemble for bottom-depth uncertainty.

The sliding-window inference sees every ping inside many overlapping patches, at
a different position within each patch. Collecting the predicted bottom depth
across all those windows gives a distribution per ping, from which we draw a
non-symmetric 95% band. Also produces a positional-bias bar plot (predicted
depth deviation as a function of position within the patch).

Output: Figures/rolling_window_uncertainty/
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import (
    apply_thesis_style,
    FIGSIZE_FULL,
    FIGSIZE_STD,
    style_colorbar,
    savefig,
)
from infer_test_raw import (
    load_model,
    prepare_raw_channels,
    DEFAULT_CHECKPOINT,
    TARGET_SIZE,
    PATCH_WIDTH,
)
from deep_learning.unet_model import predict_full_echogram

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "rolling_window_uncertainty"
DEFAULT_RAW_DIR = str(PROJECT_ROOT / "Datasets" / "raw_files")


@torch.no_grad()
def window_bottom_distribution(
    model,
    sv_channels,
    target_size,
    device,
    patch_width=PATCH_WIDTH,
    stride=4,
    batch_size=16,
    threshold=0.5,
):
    """Run a dense sliding window and record the bottom index from every window.

    Returns
    -------
    pos_ping  : (M,) global ping index of each (window, column) record
    pos_idx   : (M,) column position within the patch [0, patch_width)
    pos_depth : (M,) bottom sample index, -1 where no bottom crossed threshold
    n_pings   : int
    n_windows : int
    """
    model.eval()
    _, n_range, n_pings = sv_channels.shape
    h, w = target_size

    starts = list(range(0, n_pings - patch_width + 1, stride))
    if n_pings >= patch_width and starts[-1] != n_pings - patch_width:
        starts.append(n_pings - patch_width)

    pos_ping, pos_idx, pos_depth = [], [], []
    cols = np.arange(patch_width)

    for b in range(0, len(starts), batch_size):
        chunk = starts[b : b + batch_size]
        batch_inp = np.empty((len(chunk), 3, h, w), dtype=np.float32)
        for k, s in enumerate(chunk):
            patch = sv_channels[:, :, s : s + patch_width]
            patch_hwc = np.transpose(patch, (1, 2, 0))
            patch_resized = cv2.resize(patch_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
            batch_inp[k] = np.transpose(patch_resized, (2, 0, 1))

        inp = torch.from_numpy(batch_inp).to(device)
        logits = model(inp)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (B, h, w)

        for k, s in enumerate(chunk):
            probs_orig = cv2.resize(
                probs[k], (patch_width, n_range), interpolation=cv2.INTER_LINEAR
            )
            binary = probs_orig >= threshold
            has_b = binary.any(axis=0)
            first = binary.argmax(axis=0).astype(np.int64)
            first[~has_b] = -1
            pos_ping.append(s + cols)
            pos_idx.append(cols.copy())
            pos_depth.append(first)

    return (
        np.concatenate(pos_ping),
        np.concatenate(pos_idx),
        np.concatenate(pos_depth),
        n_pings,
        len(starts),
    )


def per_ping_stats(pos_ping, pos_depth, n_pings):
    """Per-ping median and non-symmetric 95% band over valid window predictions."""
    valid = pos_depth >= 0
    df = pd.DataFrame({"ping": pos_ping[valid], "depth": pos_depth[valid]})
    g = df.groupby("ping")["depth"]
    stats = pd.DataFrame(
        {
            "median": g.median(),
            "q025": g.quantile(0.025),
            "q975": g.quantile(0.975),
            "count": g.size(),
        }
    ).reindex(range(n_pings))
    return stats


def positional_bias(pos_ping, pos_idx, pos_depth, consensus, dz):
    """Deviation (m) from the per-ping consensus, grouped by position-in-patch."""
    valid = pos_depth >= 0
    cons = consensus[pos_ping]
    cons_valid = valid & np.isfinite(cons)
    dev_m = (pos_depth[cons_valid] - cons[cons_valid]) * dz
    df = pd.DataFrame({"pos": pos_idx[cons_valid], "dev": dev_m})
    g = df.groupby("pos")["dev"]
    bias = pd.DataFrame(
        {
            "mean": g.mean(),
            "std": g.std(),
            "q025": g.quantile(0.025),
            "q975": g.quantile(0.975),
        }
    )
    det = pd.DataFrame({"pos": pos_idx, "valid": valid.astype(float)})
    bias["detect_rate"] = det.groupby("pos")["valid"].mean()
    return bias


def plot_echogram_band(name, sv_channels, range_arr, stats, save_path):
    echogram = sv_channels[1]  # 200 kHz, normalised [0, 1]
    n_range, n_pings = echogram.shape
    extent = [0, n_pings, range_arr[-1], range_arr[0]]
    pings = np.arange(n_pings)

    def to_m(idx):
        return range_arr[np.clip(idx.astype(int), 0, n_range - 1)]

    valid = stats["count"].fillna(0).values > 0
    band_m = np.full(n_pings, np.nan)
    band_m[valid] = (stats["q975"].values[valid] - stats["q025"].values[valid]) * (
        range_arr[1] - range_arr[0]
    )

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(FIGSIZE_FULL[0], FIGSIZE_FULL[1] + 2), gridspec_kw={"height_ratios": [3, 1]}
    )

    im = ax0.imshow(
        echogram, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper", extent=extent
    )
    if valid.any():
        pv = pings[valid]
        ax0.fill_between(
            pv,
            to_m(stats["q025"].values[valid]),
            to_m(stats["q975"].values[valid]),
            color="orange",
            alpha=0.45,
            linewidth=0,
            label="95% window band",
        )
        ax0.plot(
            pv,
            to_m(stats["median"].values[valid]),
            color="red",
            linewidth=1.5,
            label="Median bottom",
        )
    cbar = fig.colorbar(im, ax=ax0, pad=0.01)
    style_colorbar(cbar, label="Normalised Sv")
    ax0.set_ylabel("Range (m)")
    ax0.set_xlabel("Ping")
    ax0.set_title("Rolling-window bottom: median and 95% band")
    ax0.legend(loc="upper right", framealpha=0.9)

    ax1.plot(pings[valid], band_m[valid], color="darkorange", linewidth=1.5)
    ax1.set_ylabel("Band width (m)")
    ax1.set_xlabel("Ping")
    ax1.set_xlim(0, n_pings)
    ax1.set_title("95% band width across pings")

    fig.suptitle(name)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)


def plot_example_pings(
    name, pos_ping, pos_depth, range_arr, stats, n_pings, save_path, n_examples=6
):
    valid = stats["count"].fillna(0).values >= 5
    cand = np.where(valid)[0]
    if len(cand) == 0:
        return
    sel = cand[np.linspace(0, len(cand) - 1, n_examples).astype(int)]

    data, labels = [], []
    vmask = pos_depth >= 0
    for p in sel:
        d = pos_depth[(pos_ping == p) & vmask]
        if d.size == 0:
            continue
        data.append(range_arr[np.clip(d, 0, len(range_arr) - 1)])
        labels.append(str(int(p)))
    if not data:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.invert_yaxis()  # depth increases downward
    ax.set_xlabel("Ping index")
    ax.set_ylabel("Predicted bottom depth (m)")
    ax.set_title("Per-ping depth distribution across windows")
    fig.suptitle(name)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)


def plot_positional_bias(bias, n_files, stride, save_path):
    pos = bias.index.values
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(FIGSIZE_FULL[0], FIGSIZE_FULL[1] + 1), sharex=True
    )

    ax0.bar(pos, bias["mean"].values, width=1.0, color="steelblue", label="Mean deviation")
    ax0.fill_between(
        pos,
        bias["q025"].values,
        bias["q975"].values,
        color="steelblue",
        alpha=0.25,
        linewidth=0,
        label="95% spread",
    )
    ax0.axhline(0.0, color="black", linewidth=1.0)
    ax0.set_ylabel("Depth deviation (m)")
    ax0.set_title("Bottom-depth bias by position within patch")
    ax0.legend(loc="upper center", ncol=2, framealpha=0.9)

    ax1.bar(pos, bias["detect_rate"].values, width=1.0, color="seagreen")
    ax1.set_ylabel("Detection rate")
    ax1.set_xlabel("Position within patch (ping column)")
    ax1.set_xlim(0, len(pos))
    ax1.set_ylim(0, 1.02)
    ax1.set_title("Bottom-detection rate by position")

    fig.suptitle(f"Positional diagnostics ({n_files} files, stride {stride})")
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)


def write_readme(out_dir, script_rel, n_files, stride, below_frac):
    txt = (
        "# Rolling-Window Bottom Uncertainty\n\n"
        "rolling-window  ensemble  uncertainty-band  positional-bias  echosounder\n\n"
        "Sliding-window inference reused as a free ensemble: each ping is predicted "
        "in many overlapping patches, giving a per-ping bottom-depth distribution "
        "(non-symmetric 95% band) plus a positional-bias diagnostic. Files where the "
        f"U-Net collapses (mean below-bottom fraction > {below_frac}) are skipped.\n\n"
        f"Source: {script_rel}\n"
        f"Files: {n_files}, stride: {stride}\n"
        f"Created: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
    )
    (out_dir / "_README.md").write_text(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / DEFAULT_CHECKPOINT))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--n-files", type=int, default=3)
    ap.add_argument("--max-pings", type=int, default=1500)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="explicit file stems to use (overrides auto-selection)",
    )
    ap.add_argument(
        "--max-below-frac",
        type=float,
        default=0.88,
        help="skip files whose mean predicted below-bottom fraction "
        "exceeds this (model-collapse guard)",
    )
    args = ap.parse_args()

    apply_thesis_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    raw_files = sorted(p for p in Path(args.raw_dir).rglob("*.raw") if not p.name.startswith("._"))
    if args.files:
        wanted = set(args.files)
        raw_files = [p for p in raw_files if p.stem in wanted]
    print(
        f"Scanning {len(raw_files)} raw files in {args.raw_dir} "
        f"for {args.n_files} usable dual-frequency files"
    )
    if not raw_files:
        print("No raw files found.")
        return

    model = load_model(args.checkpoint, device)

    bias_frames = []
    n_done = 0

    for rf in raw_files:
        if n_done >= args.n_files:
            break
        name = rf.stem
        try:
            sv_channels, sv1, sv2, range_arr = prepare_raw_channels(
                str(rf), max_pings=args.max_pings
            )
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue

        n_pings = sv_channels.shape[2]
        if n_pings < PATCH_WIDTH:
            print(f"  skip {name}: only {n_pings} pings (< {PATCH_WIDTH})")
            continue

        below_frac = float(
            predict_full_echogram(
                model, sv_channels, TARGET_SIZE, device, patch_width=PATCH_WIDTH, stride=256
            ).mean()
        )
        if below_frac > args.max_below_frac:
            print(
                f"  skip {name}: model collapsed "
                f"(below-bottom frac {below_frac:.2f} > {args.max_below_frac})"
            )
            continue

        n_done += 1
        print(f"[{n_done}/{args.n_files}] {name} (below-bottom frac {below_frac:.2f})")

        pos_ping, pos_idx, pos_depth, n_pings, n_win = window_bottom_distribution(
            model, sv_channels, TARGET_SIZE, device, stride=args.stride, batch_size=args.batch_size
        )
        print(f"  {n_win} windows, {len(pos_ping)} records")

        stats = per_ping_stats(pos_ping, pos_depth, n_pings)
        dz = float(range_arr[1] - range_arr[0])
        consensus = stats["median"].values

        plot_echogram_band(name, sv_channels, range_arr, stats, out_dir / f"{name}_band.png")
        plot_example_pings(
            name,
            pos_ping,
            pos_depth,
            range_arr,
            stats,
            n_pings,
            out_dir / f"{name}_ping_distributions.pdf",
        )
        stats.to_csv(out_dir / f"{name}_per_ping_stats.csv", index_label="ping")

        bias_frames.append(positional_bias(pos_ping, pos_idx, pos_depth, consensus, dz))

        med_band = float(np.nanmedian((stats["q975"] - stats["q025"]).values) * dz)
        print(f"  median 95% band width: {med_band:.3f} m")

    if bias_frames:
        bias = pd.concat(bias_frames).groupby(level=0).mean()
        bias.to_csv(out_dir / "positional_bias.csv", index_label="pos")
        plot_positional_bias(bias, len(bias_frames), args.stride, out_dir / "positional_bias.pdf")

    write_readme(
        out_dir,
        "code/data_visualization/analyze_rolling_window_uncertainty.py",
        n_done,
        args.stride,
        args.max_below_frac,
    )
    print(f"Done. Output in {out_dir}")


if __name__ == "__main__":
    main()
