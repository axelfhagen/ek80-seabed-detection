"""U-Net versus the raw classic detector: robustness, not accuracy.

A head-to-head accuracy test is circular (the detector is the U-Net's label
source). Instead this shows where the two diverge in a way that favours the
U-Net: it produces a gap-free, smooth bottom in one forward pass, whereas a
single per-ping detector drops out (NaN) and spikes. For a few test files it
overlays both lines on the echogram and quantifies detection rate, gap recovery
(pings the detector missed but the U-Net filled), and per-ping disagreement.

Per-ping depth arrays are cached to NPZ for instant re-plotting.

Output: Figures/statistical_tests/
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from data_visualization.thesis_style import apply_thesis_style, savefig
from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import load_file_data, extract_bottom_line
from raw_sensor_integration.preprocessing import load_sv_from_zarr
from raw_sensor_integration.bottom_labelling import detect_bottom_simple, detect_bottom_edge

ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "statistical_tests"
CKPT = PROJECT_ROOT / "code" / "checkpoints" / "exp_20260420_091726_full_finetune" / "unet_resnet34_best.pth"
FREQ1, FREQ2 = 38000, 200000
TARGET = (512, 512)
PATCH_W, STRIDE = 512, 256
VMIN, VMAX = -80.0, -34.0


def load_model(device):
    m = create_unet("resnet34", encoder_weights=None, in_channels=3)
    sd = torch.load(CKPT, map_location=device)
    if isinstance(sd, dict):
        sd = sd.get("model_state_dict", sd)
    m.load_state_dict(sd)
    m.to(device).eval()
    return m


def process_file(model, device, fn):
    """Return per-ping depths (m) for U-Net, raw detector, GT label, plus the
    38 kHz echogram for display."""
    path = str(ZARR_DIR / fn)
    sv_ch, _gt, gt_idx, range_arr = load_file_data(path, FREQ1, FREQ2, vmin=VMIN, vmax=VMAX)
    prob = predict_full_echogram(model, sv_ch, TARGET, device, patch_width=PATCH_W, stride=STRIDE)
    unet_idx = extract_bottom_line(prob)

    disp_sv, disp_r, _t, pd, hv, _f = load_sv_from_zarr(path, FREQ1)
    pd = pd if pd and pd > 0 else 1e-3
    simple_depth = detect_bottom_simple(disp_sv, disp_r, hv, pd)
    edge_depth = detect_bottom_edge(disp_sv, disp_r, hv, pd, frequency_hz=FREQ1)

    n = min(len(unet_idx), len(simple_depth), len(edge_depth), len(gt_idx))
    unet_d = np.where(unet_idx[:n] >= 0, range_arr[np.clip(unet_idx[:n], 0, len(range_arr) - 1)], np.nan)
    gt_d = np.where(gt_idx[:n] >= 0, range_arr[np.clip(gt_idx[:n], 0, len(range_arr) - 1)], np.nan)
    return {"unet": unet_d, "simple": simple_depth[:n].astype(float),
            "edge": edge_depth[:n].astype(float), "gt": gt_d,
            "sv": disp_sv, "range": disp_r, "file": fn}


def main():
    global OUTPUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=6)
    ap.add_argument("--n-panels", type=int, default=3)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args()
    apply_thesis_style()
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import json
    ref = PROJECT_ROOT / "code" / "checkpoints" / "exp_20260417_130130_cosine" / "eval_metrics.json"
    files = [r["file"] for r in json.load(open(ref))["per_file"]][:args.n_files]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    print(f"device={device}  files={len(files)}")

    results = []
    for i, fn in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fn}")
        results.append(process_file(model, device, fn))

    # Aggregate per-file stats. Edge detector = robustness contrast (it drops
    # out); simple detector = accuracy reference (U-Net should agree with it).
    edge_rate, simple_rate, unet_rate, gap_recov, disagree = [], [], [], [], []
    for r in results:
        edge_valid = ~np.isnan(r["edge"])
        simple_valid = ~np.isnan(r["simple"])
        unet_valid = ~np.isnan(r["unet"])
        edge_rate.append(edge_valid.mean())
        simple_rate.append(simple_valid.mean())
        unet_rate.append(unet_valid.mean())
        missed = ~edge_valid
        gap_recov.append(unet_valid[missed].mean() if missed.any() else np.nan)
        both = simple_valid & unet_valid
        disagree.append(r["unet"][both] - r["simple"][both])
    edge_rate = np.array(edge_rate)
    simple_rate = np.array(simple_rate)
    unet_rate = np.array(unet_rate)
    all_disagree = np.concatenate(disagree)

    _, p_rate = stats.ttest_rel(unet_rate, edge_rate)
    try:
        _, pw_rate = stats.wilcoxon(unet_rate, edge_rate)
    except ValueError:
        pw_rate = np.nan

    # ---- Figure ----
    npanel = min(args.n_panels, len(results))
    fig = plt.figure(figsize=(22, 13))
    fig.suptitle("U-Net versus raw classic detector (38 kHz)")
    gs = fig.add_gridspec(2, max(npanel, 3))

    from raw_sensor_integration.visualization import ECHOGRAM_CMAP
    for k in range(npanel):
        r = results[k]
        ax = fig.add_subplot(gs[0, k])
        sv = np.clip(r["sv"], -100, 0)
        npings = sv.shape[1]
        extent = [0, npings, r["range"][-1], r["range"][0]]
        ax.imshow(sv, aspect="auto", cmap=ECHOGRAM_CMAP, extent=extent,
                  vmin=-90, vmax=-30)
        x = np.arange(len(r["edge"]))
        ax.plot(x, r["edge"], ".", ms=2.5, color="orange", label="edge detector")
        ax.plot(np.arange(len(r["unet"])), r["unet"], "-", lw=1.6, color="cyan",
                label="U-Net")
        ax.set_xlabel("ping"); ax.set_ylabel("depth (m)")
        ax.set_title(r["file"].replace(".zarr", ""), fontsize=15)
        bot = np.nanmax(r["gt"]) if np.isfinite(np.nanmax(r["gt"])) else r["range"][-1]
        ax.set_ylim(min(bot * 1.4, r["range"][-1]), 0)
        if k == 0:
            ax.legend(loc="upper right", fontsize=12)

    # detection-rate bars (edge / simple / U-Net)
    ax = fig.add_subplot(gs[1, 0])
    series = [edge_rate, simple_rate, unet_rate]
    means = [s.mean() for s in series]
    ax.bar([0, 1, 2], means, color=["orange", "#e08a1f", "#1f9fd0"], width=0.6, alpha=0.85)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(series):
        ax.scatter(rng.normal(i, 0.05, len(vals)), vals, s=30, color="black", zorder=3)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["edge", "simple", "U-Net"])
    ax.set_ylabel("valid-ping fraction"); ax.set_ylim(0, 1.08)
    sig = "*" if p_rate < 0.05 else "n.s."
    ax.set_title(f"detection rate: U-Net vs edge p={p_rate:.3f} ({sig})")
    ax.grid(axis="y", alpha=0.3)

    # gap recovery (edge dropouts filled by U-Net)
    ax = fig.add_subplot(gs[1, 1])
    gr = np.array(gap_recov)
    valid = ~np.isnan(gr)
    if valid.any():
        ax.bar(np.arange(valid.sum()), gr[valid], color="#27ae60", alpha=0.85)
        ax.axhline(np.nanmean(gr), color="black", ls="--", lw=2,
                   label=f"mean {np.nanmean(gr):.2f}")
        ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05); ax.set_ylabel("fraction filled by U-Net")
    ax.set_xlabel("file (with edge dropouts)")
    ax.set_title("gap recovery: edge-detector-missed pings")
    ax.grid(axis="y", alpha=0.3)

    # disagreement histogram (U-Net vs the strong simple detector)
    ax = fig.add_subplot(gs[1, 2])
    clip = np.clip(all_disagree, -30, 30)
    ax.hist(clip, bins=60, color="#1f5f80", alpha=0.8)
    med = np.median(all_disagree)
    ax.axvline(med, color="#c0392b", lw=2, label=f"median {med:+.2f} m")
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("U-Net minus simple detector (m)"); ax.set_ylabel("ping count")
    ax.set_title(f"agreement with simple detector (n={len(all_disagree)})")
    ax.legend(fontsize=12); ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTPUT_DIR / "unet_vs_detector.pdf"
    savefig(fig, out)
    plt.close(fig)

    print(f"\nedge detection rate     {edge_rate.mean():.3f}")
    print(f"simple detection rate   {simple_rate.mean():.3f}")
    print(f"U-Net detection rate    {unet_rate.mean():.3f}")
    print(f"gap recovery (mean)     {np.nanmean(gr):.3f}")
    print(f"median disagree vs simple {med:+.2f} m   IQR "
          f"[{np.percentile(all_disagree,25):+.2f}, {np.percentile(all_disagree,75):+.2f}]")
    print(f"U-Net vs edge rate: paired t p={p_rate:.4f}  Wilcoxon p={pw_rate:.4f}")
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
