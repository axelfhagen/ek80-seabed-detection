#!/usr/bin/env python
"""Bottom-line extraction strategies on the shallow_C2 dock probability maps.

Output: Figures/shallow_C2_bottomline_strategies/

The default extractor takes the first pixel from the top that crosses a
probability threshold. On the shallow dock files that first crossing can latch
onto near-surface reverberation speckle above the seabed. This script compares
that baseline against three strategies that exploit the spatial structure of the
(clean) probability map from chunk_arch_sweep_2:

  - first-crossing (baseline): first row with P >= tau, per ping;
  - longest-run top: top of the longest contiguous above-threshold run per ping;
  - largest-component top: threshold, keep the single largest connected blob,
    take its top edge per ping (enforces along-ping continuity);
  - max-gradient: row of steepest probability rise per ping (water/seabed edge).

It also sweeps the threshold for the baseline. Probability maps are recomputed
from the cached broadband Sv (no echopype rerun).

Usage:
    python code/data_visualization/analyze_shallow_C2_bottomline_strategies.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "data_visualization"))
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import label
import torch

from raw_sensor_integration.preprocessing import normalize_echogram
from deep_learning.unet_model import create_unet, predict_full_echogram
from data_visualization.thesis_style import apply_thesis_style, savefig, style_colorbar

# ── Constants ──────────────────────────────────────────────────────────────
STEMS = [
    "shallow_C2-D20260526-T134234",
    "shallow_C2-D20260526-T134826",
    "shallow_C2-D20260526-T135419",
]
FREQ1, FREQ2 = 38_000, 200_000
VMIN, VMAX = -80.0, -34.0
TARGET_SIZE = (512, 512)
PATCH_WIDTH, STRIDE = 512, 256
TAU = 0.5
SWEEP_TAUS = [0.3, 0.5, 0.7, 0.9]
DISP_MAX = 9.0  # all strategy lines live in the top few m; zoom there for clarity

DEFAULT_CKPT_NAME = "chunk_arch_sweep_2"
SV_CACHE_DIR = PROJECT_ROOT / "Figures" / "shallow_C2_failure" / "cache"
OUT_DIR = PROJECT_ROOT / "Figures" / "shallow_C2_bottomline_strategies"
CHECKPOINT = PROJECT_ROOT / "code/checkpoints" / DEFAULT_CKPT_NAME / "smp_unet_resnet34_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# strategy name -> colour
STRAT_COLORS = {
    "First-crossing (baseline)": "#ff3b30",
    "Longest-run top": "#00e5ff",
    "Largest-component top": "#ffb000",
    "Max-gradient edge": "#c042ff",
}


# ── Extraction strategies (all return per-ping row index, -1 = none) ─────────
def strat_first_crossing(prob, tau=TAU):
    binary = prob >= tau
    idx = np.full(prob.shape[1], -1, dtype=np.int64)
    for p in range(prob.shape[1]):
        hits = np.where(binary[:, p])[0]
        if hits.size:
            idx[p] = hits[0]
    return idx


def _longest_run_top(col_bool):
    best_len, best_start, cur, start = 0, -1, 0, 0
    for i, v in enumerate(col_bool):
        if v:
            if cur == 0:
                start = i
            cur += 1
            if cur > best_len:
                best_len, best_start = cur, start
        else:
            cur = 0
    return best_start if best_len > 0 else -1


def strat_longest_run_top(prob, tau=TAU):
    binary = prob >= tau
    idx = np.full(prob.shape[1], -1, dtype=np.int64)
    for p in range(prob.shape[1]):
        idx[p] = _longest_run_top(binary[:, p])
    return idx


def strat_largest_component_top(prob, tau=TAU):
    binary = prob >= tau
    lbl, n = label(binary)
    idx = np.full(prob.shape[1], -1, dtype=np.int64)
    if n == 0:
        return idx
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    keep = int(sizes.argmax())
    comp = lbl == keep
    for p in range(prob.shape[1]):
        hits = np.where(comp[:, p])[0]
        if hits.size:
            idx[p] = hits[0]
    return idx


def strat_max_gradient(prob, tau=TAU, min_grad=0.05):
    """Row of steepest downward probability rise; gated so flat columns are skipped."""
    grad = np.diff(prob, axis=0)  # (n_range-1, n_pings)
    idx = np.full(prob.shape[1], -1, dtype=np.int64)
    for p in range(prob.shape[1]):
        g = grad[:, p]
        r = int(np.argmax(g))
        # only accept if there is a real rise AND the column ever crosses tau
        if g[r] >= min_grad and (prob[:, p] >= tau).any():
            idx[p] = r + 1
    return idx


STRATEGIES = {
    "First-crossing (baseline)": strat_first_crossing,
    "Longest-run top": strat_longest_run_top,
    "Largest-component top": strat_largest_component_top,
    "Max-gradient edge": strat_max_gradient,
}


def idx_to_depth(idx, range_arr):
    d = np.full(idx.shape, np.nan, dtype=np.float64)
    ok = idx >= 0
    d[ok] = range_arr[np.clip(idx[ok], 0, len(range_arr) - 1)]
    return d


# ── Data / model ─────────────────────────────────────────────────────────
def load_cached_sv(stem):
    d = np.load(SV_CACHE_DIR / f"{stem}_sv.npz")
    return d["sv1"], d["range1"], d["sv2"], d["range2"]


def resample_to_grid(sv2, range2, range1):
    out = np.full((len(range1), sv2.shape[1]), np.nan, dtype=np.float32)
    inr = range1 <= range2[-1]
    for p in range(sv2.shape[1]):
        out[inr, p] = np.interp(range1[inr], range2, sv2[:, p])
    return out


def load_model():
    model = create_unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    sd = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state = sd.get("model_state_dict", sd)
    model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=False)
    return model.to(DEVICE).eval()


def process(stem, model):
    sv1, range1, sv2, range2 = load_cached_sv(stem)
    npings = min(sv1.shape[1], sv2.shape[1])
    sv1 = sv1[:, :npings]
    sv2r = resample_to_grid(sv2, range2, range1)[:, :npings]
    norm1 = normalize_echogram(sv1, method="minmax", vmin=VMIN, vmax=VMAX, fill_nan=0.0).astype(
        np.float32
    )
    norm2 = normalize_echogram(sv2r, method="minmax", vmin=VMIN, vmax=VMAX, fill_nan=0.0).astype(
        np.float32
    )
    diff = (norm1 - norm2).astype(np.float32)
    chans = np.stack([norm1, norm2, diff], axis=0)
    with torch.no_grad():
        prob = predict_full_echogram(
            model, chans, TARGET_SIZE, DEVICE, patch_width=PATCH_WIDTH, stride=STRIDE
        )
    lines = {name: idx_to_depth(fn(prob), range1) for name, fn in STRATEGIES.items()}
    sweep = {t: idx_to_depth(strat_first_crossing(prob, t), range1) for t in SWEEP_TAUS}
    return dict(
        stem=stem, sv1=sv1, prob=prob, range1=range1, npings=npings, lines=lines, sweep=sweep
    )


# ── Figures ─────────────────────────────────────────────────────────────
def _overlay_lines(ax, g, lines, color_map, lw=1.6):
    pings = np.arange(g["npings"])
    for name, depth in lines.items():
        v = np.isfinite(depth)
        if v.any():
            cov = 100 * v.mean()
            ax.plot(
                pings[v],
                depth[v],
                color=color_map[name],
                lw=lw,
                alpha=0.9,
                label=f"{name} ({cov:.0f}% cov)",
            )


def fig_strategies(g, path):
    ra = g["range1"]
    extent = [0, g["npings"], ra[-1], ra[0]]
    fig, axes = plt.subplots(1, 3, figsize=(28, 8))

    im0 = axes[0].imshow(
        g["sv1"], aspect="auto", cmap="viridis", vmin=VMIN, vmax=VMAX, extent=extent
    )
    _overlay_lines(axes[0], g, g["lines"], STRAT_COLORS)
    axes[0].set_title("38 kHz Sv + extraction strategies")
    style_colorbar(plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04), label="Sv (dB)")
    axes[0].legend(loc="lower left", fontsize=11, framealpha=0.9)

    im1 = axes[1].imshow(g["prob"], aspect="auto", cmap="magma", vmin=0, vmax=1, extent=extent)
    _overlay_lines(axes[1], g, g["lines"], STRAT_COLORS)
    axes[1].set_title("P(bottom) + extraction strategies")
    style_colorbar(plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04), label="P(bottom)")

    # threshold sweep of the baseline first-crossing
    axes[2].imshow(g["sv1"], aspect="auto", cmap="viridis", vmin=VMIN, vmax=VMAX, extent=extent)
    pings = np.arange(g["npings"])
    cmap = plt.cm.cool(np.linspace(0, 1, len(SWEEP_TAUS)))
    for c, t in zip(cmap, SWEEP_TAUS):
        d = g["sweep"][t]
        v = np.isfinite(d)
        if v.any():
            axes[2].plot(
                pings[v],
                d[v],
                color=c,
                lw=1.6,
                alpha=0.9,
                label=f"tau={t} ({100 * v.mean():.0f}% cov)",
            )
    axes[2].set_title("First-crossing: threshold sweep")
    axes[2].legend(loc="lower left", fontsize=11, framealpha=0.9)

    for ax in axes:
        ax.set_xlabel("Ping")
        ax.set_ylabel("Depth (m)")
        ax.set_ylim(DISP_MAX, 0)
    fig.suptitle(
        f"{g['stem']}: bottom-line extraction strategies "
        f"({DEFAULT_CKPT_NAME}), depth zoomed to top {DISP_MAX:.0f} m",
        y=1.02,
    )
    fig.tight_layout()
    savefig(fig, path)
    plt.close(fig)


def fig_summary(items, path):
    n = len(items)
    fig, axes = plt.subplots(n, 1, figsize=(16, 5 * n))
    if n == 1:
        axes = [axes]
    for ax, g in zip(axes, items):
        ra = g["range1"]
        extent = [0, g["npings"], ra[-1], ra[0]]
        ax.imshow(g["sv1"], aspect="auto", cmap="viridis", vmin=VMIN, vmax=VMAX, extent=extent)
        _overlay_lines(ax, g, g["lines"], STRAT_COLORS, lw=1.8)
        ax.set_title(g["stem"].split("-")[-1])
        ax.set_xlabel("Ping")
        ax.set_ylabel("Depth (m)")
        ax.set_ylim(DISP_MAX, 0)
    handles = [Line2D([0], [0], color=c, lw=2.5, label=n) for n, c in STRAT_COLORS.items()]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(STRAT_COLORS),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=13,
    )
    fig.suptitle("shallow_C2 dock files: bottom-line extraction strategies on 38 kHz Sv", y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    savefig(fig, path)
    plt.close(fig)


def write_readme(items):
    import datetime

    lines = []
    for g in items:
        parts = []
        for name, d in g["lines"].items():
            v = np.isfinite(d)
            med = np.nanmedian(d[v]) if v.any() else np.nan
            parts.append(f"{name.split()[0]}={med:.1f}m/{100 * v.mean():.0f}%")
        lines.append(f"  {g['stem']}: " + "  ".join(parts))
    (OUT_DIR / "_README.md").write_text(
        "# shallow_C2 bottom-line extraction strategies\n\n"
        "bottom-line  extraction  threshold  connected-component  gradient\n\n"
        "Comparison of bottom-line extraction strategies (first-crossing, "
        "longest-run top, largest-connected-component top, max-gradient edge) "
        "applied to the chunk_arch_sweep_2 probability maps on the three shallow "
        "dock files, plus a first-crossing threshold sweep.\n\n"
        "Source: code/data_visualization/analyze_shallow_C2_bottomline_strategies.py\n"
        f"Created: {datetime.datetime.now():%Y-%m-%d %H:%M}\n"
        f"Checkpoint: {DEFAULT_CKPT_NAME}\n\n"
        "Per-file median depth / coverage by strategy:\n" + "\n".join(lines) + "\n"
    )


def main():
    global CHECKPOINT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    args = ap.parse_args()
    CHECKPOINT = Path(args.checkpoint)

    apply_thesis_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}  Checkpoint: {CHECKPOINT}")

    model = load_model()
    items = []
    for stem in STEMS:
        print(f"[process] {stem}", flush=True)
        g = process(stem, model)
        items.append(g)
        fig_strategies(g, OUT_DIR / f"strategies_{stem}.png")
    fig_summary(items, OUT_DIR / "summary_strategies.png")
    write_readme(items)
    print(f"Done. {len(items)} files -> {OUT_DIR}")


if __name__ == "__main__":
    main()
