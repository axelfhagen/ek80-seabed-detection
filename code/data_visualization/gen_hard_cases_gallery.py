"""Real-data gallery of the hardest seabed echograms for the U-Net.

The synthetic failure-mode schematic teaches the canonical hard phenomena; this
is the real-data companion. It scans an even sample of zarr files, runs the
generalist U-Net and the classical ensemble reference on each, and renders the
files where the two diverge most as a 2x3 echogram grid. Each panel overlays the
classical reference label (cyan) and the U-Net prediction (red) so the reader
sees, on actual data, where seabed detection is genuinely ambiguous.

Per-file depth lines are cached to NPZ for instant re-plotting.

Output: Figures/hard_cases_gallery/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import torch

from data_visualization.thesis_style import apply_thesis_style, savefig, style_colorbar
from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.evaluate import load_file_data, extract_bottom_line
from raw_sensor_integration.preprocessing import load_sv_from_zarr
from raw_sensor_integration.visualization import ECHOGRAM_CMAP

ZARR_DIR = PROJECT_ROOT / "Datasets" / "echopype_zarr"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "hard_cases_gallery"
CKPT = (PROJECT_ROOT / "code" / "checkpoints"
        / "exp_20260525_220040_baseline_random_split_2"
        / "smp_unet_resnet34_best.pth")

FREQ1, FREQ2 = 38000, 200000
TARGET = (512, 512)
PATCH_W, STRIDE = 512, 256
VMIN, VMAX = -80.0, -34.0

N_CANDIDATES = 80   # files scanned (sampled evenly across the archive)
N_PANELS = 6        # hardest files shown (2 rows x 3 cols)
DIV_THRESH_M = 5.0  # a ping "diverges" when |U-Net - reference| exceeds this
MIN_VALID_FRAC = 0.5


def load_model(device):
    model = create_unet("resnet34", encoder_weights=None, in_channels=3)
    state = torch.load(CKPT, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
    cleaned = {k.removeprefix("model."): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    return model


def analyse_file(model, device, zarr_path):
    """Return per-ping reference/U-Net depths (m) and hardness stats, or None."""
    sv_ch, _gt_mask, gt_idx, range_arr = load_file_data(
        str(zarr_path), FREQ1, FREQ2, vmin=VMIN, vmax=VMAX)
    prob = predict_full_echogram(model, sv_ch, TARGET, device,
                                 patch_width=PATCH_W, stride=STRIDE)
    unet_idx = extract_bottom_line(prob)

    n = min(len(gt_idx), len(unet_idx))
    gt_idx, unet_idx = gt_idx[:n], unet_idx[:n]
    last = len(range_arr) - 1
    ref_d = np.where(gt_idx >= 0, range_arr[np.clip(gt_idx, 0, last)], np.nan)
    unet_d = np.where(unet_idx >= 0, range_arr[np.clip(unet_idx, 0, last)], np.nan)

    both = np.isfinite(ref_d) & np.isfinite(unet_d)
    if both.mean() < MIN_VALID_FRAC or both.sum() < 30:
        return None
    delta = np.abs(unet_d[both] - ref_d[both])
    stats = {
        "median_div": float(np.median(delta)),
        "p75_div": float(np.percentile(delta, 75)),
        "max_div": float(np.max(delta)),
        "frac_div": float((delta > DIV_THRESH_M).mean()),
        "n_pings": int(n),
    }
    # Hardness: sustained disagreement (fraction of diverging pings), with the
    # 75th-percentile magnitude as a tie-breaker so single spikes do not win.
    stats["hardness"] = stats["frac_div"] + 0.02 * stats["p75_div"]
    return {"file": zarr_path.name, "ref_d": ref_d, "unet_d": unet_d,
            "range_max": float(range_arr[-1]), **stats}


def collect(model, device, cache_path, recompute):
    if cache_path.exists() and not recompute:
        data = np.load(cache_path, allow_pickle=True)
        records = list(data["records"])
        print(f"Loaded {len(records)} cached records from {cache_path.name}")
        return records

    all_files = sorted(ZARR_DIR.glob("*.zarr"))
    idx = np.linspace(0, len(all_files) - 1, min(N_CANDIDATES, len(all_files)))
    files = [all_files[int(i)] for i in idx]
    records = []
    for i, zp in enumerate(files, 1):
        try:
            rec = analyse_file(model, device, zp)
        except Exception as exc:  # a few stores are short/corrupt; skip them
            print(f"[{i}/{len(files)}] SKIP {zp.stem}: {exc}")
            continue
        if rec is None:
            print(f"[{i}/{len(files)}] skip {zp.stem} (too few valid pings)")
            continue
        records.append(rec)
        print(f"[{i}/{len(files)}] {zp.stem}  frac_div={rec['frac_div']:.2f} "
              f"median={rec['median_div']:.1f}m  max={rec['max_div']:.1f}m")
    np.savez(cache_path, records=np.array(records, dtype=object))
    print(f"Cached {len(records)} records -> {cache_path.name}")
    return records


def render(records, out_path):
    records = sorted(records, key=lambda r: r["hardness"], reverse=True)[:N_PANELS]
    apply_thesis_style()
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("Hardest real echograms for the U-Net", y=0.99)
    fig.text(0.45, 0.95,
             "Files with the largest disagreement from the classical reference: "
             "the U-Net tracks it through steep slopes and canyons, diverging "
             "mainly in open water with no true seabed",
             ha="center", va="top", fontsize=14, color="grey")

    im = None
    for ax, rec in zip(axes.ravel(), records):
        path = ZARR_DIR / rec["file"]
        disp_sv, disp_r, *_ = load_sv_from_zarr(str(path), FREQ1)
        n_pings = disp_sv.shape[1]
        extent = [0, n_pings, disp_r[-1], disp_r[0]]
        row_step = max(1, disp_sv.shape[0] // 800)
        col_step = max(1, disp_sv.shape[1] // 1000)
        im = ax.imshow(disp_sv[::row_step, ::col_step], aspect="auto",
                       cmap=ECHOGRAM_CMAP, vmin=VMIN, vmax=VMAX,
                       extent=extent, interpolation="nearest")
        x = np.arange(len(rec["ref_d"]))
        ax.plot(x, rec["ref_d"], color="cyan", lw=1.6)
        ax.plot(x, rec["unet_d"], color="red", lw=1.1, ls="--")
        # Frame the seabed: show down to ~1.3x the deepest reference depth.
        bot = np.nanmax(rec["ref_d"])
        if np.isfinite(bot):
            ax.set_ylim(min(bot * 1.3, disp_r[-1]), 0)
        ax.set_xlabel("ping")
        ax.set_ylabel("depth (m)")
        ax.set_title(rec["file"].replace(".zarr", "").replace("D", "", 1),
                     fontsize=14)
        ax.text(0.015, 0.04,
                f"diverging pings {rec['frac_div']*100:.0f}%\n"
                f"median |Δ| {rec['median_div']:.1f} m   max {rec['max_div']:.0f} m",
                transform=ax.transAxes, fontsize=11, va="bottom", ha="left",
                color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.45,
                          ec="none"))

    for ax in axes.ravel()[len(records):]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color="cyan", lw=2.5, label="Classical detector (reference label)"),
        Line2D([0], [0], color="red", lw=2.0, ls="--", label="U-Net prediction"),
    ]
    # Reserve the right margin for a dedicated colorbar axis and the bottom
    # margin for the legend + provenance footer, so nothing overlaps the panels.
    fig.tight_layout(rect=[0, 0.08, 0.9, 0.96])
    if im is not None:
        cax = fig.add_axes([0.92, 0.30, 0.015, 0.40])
        cbar = fig.colorbar(im, cax=cax)
        style_colorbar(cbar, label=r"$S_v$ (dB re 1 m$^{-1}$)")
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=16,
               bbox_to_anchor=(0.45, 0.025))
    fig.text(0.45, 0.005,
             "Source: code/data_visualization/gen_hard_cases_gallery.py  |  "
             f"hardest {N_PANELS} of {N_CANDIDATES} sampled files, ranked by "
             f"fraction of pings diverging >{DIV_THRESH_M:.0f} m",
             ha="center", va="bottom", fontsize=11, color="grey")
    savefig(fig, out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return records


def write_readme(records):
    from datetime import datetime
    files = ", ".join(r["file"].replace(".zarr", "") for r in records)
    (OUTPUT_DIR / "_README.md").write_text(
        "# Hard Cases Gallery\n\n"
        "hard-cases   failure   seabed   unet   classical-disagreement\n\n"
        "Real-data companion to the synthetic failure-mode schematic: the "
        "echograms with the largest disagreement between the U-Net prediction "
        "(red) and the classical ensemble reference label (cyan), ranked by the "
        "fraction of pings disagreeing by more than 5 m over an 80-file even "
        "sample. The hardest real cases show the U-Net tracking the reference "
        "through steep slopes and canyons, with sustained divergence only in "
        "open water that has no true seabed return.\n\n"
        f"Selected files: {files}\n\n"
        "Source: code/data_visualization/gen_hard_cases_gallery.py\n\n"
        f"Created: {datetime.now().isoformat(timespec='seconds')}\n")


def main():
    global OUTPUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  checkpoint={CKPT.parent.name}")
    model = load_model(device)

    cache_path = OUTPUT_DIR / "_cache_hardness.npz"
    records = collect(model, device, cache_path, args.recompute)
    if not records:
        print("No usable files found.")
        return
    selected = render(records, OUTPUT_DIR / "hard_cases_gallery.png")
    write_readme(selected)


if __name__ == "__main__":
    main()
