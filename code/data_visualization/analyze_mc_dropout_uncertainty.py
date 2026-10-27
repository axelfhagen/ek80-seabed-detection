"""MC-Dropout uncertainty for the U-Net seabed segmentation.

Dropout layers are injected into the (dropout-free) trained decoder and left
active at inference, so N stochastic forward passes give a per-pixel mean and
standard-deviation probability map. From the per-sample bottom lines we also
get a per-ping bottom-depth distribution (mean line + non-symmetric 95% band).

The script answers two questions: where does MC-Dropout place its uncertainty,
and does that uncertainty carry information the single deterministic pass does
not already have (compared here against single-pass prediction entropy). A
dropout-rate sweep shows how the band scales with p.

Output: Figures/mc_dropout_uncertainty/
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
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
from infer_test_raw import load_model, prepare_raw_channels, TARGET_SIZE, PATCH_WIDTH
from deep_learning.unet_model import predict_full_echogram
from deep_learning.evaluate import extract_bottom_line
from validation.uncertainty import mc_dropout_predict, prediction_entropy

OUTPUT_DIR = PROJECT_ROOT / "Figures" / "mc_dropout_uncertainty"
DEFAULT_RAW_DIR = str(PROJECT_ROOT / "Datasets" / "raw_files")

# The canonical best model (survey split, referenced across recent analyses).
DEFAULT_CHECKPOINT = "code/checkpoints/exp_20260513_100101_arch_sweep_2/smp_unet_resnet34_best.pth"


def bottom_line_distribution(all_preds, det_prob, range_arr, threshold=0.5, search_m=25.0):
    """Per-ping bottom-depth stats over the MC samples.

    The naive first-crossing rule is unstable here: dropout occasionally lifts
    a near-surface scattering layer above threshold, so the *first* crossing
    jumps tens of metres up the water column. We separate two quantities:

    - Seabed-boundary uncertainty (the useful band): each sample's bottom is
      the first crossing *within ``search_m`` of the deterministic seabed
      line*, so the band measures how much the boundary itself wobbles.
    - Flip rate: the fraction of sample-pings whose unconstrained first
      crossing lands more than ``search_m`` shallower than the deterministic
      seabed, i.e. how often MC-Dropout flips onto the near-surface layer.

    Returns mean / 2.5% / 97.5% / std boundary depth in metres (NaN where < 3
    samples have a boundary), per-ping flip rate, and per-ping valid count.
    """
    n_samples, n_range, n_pings = all_preds.shape
    det_line = extract_bottom_line(det_prob, threshold=threshold)
    dz = float(np.median(np.diff(range_arr)))
    win = max(1, int(round(search_m / dz)))

    depths = np.full((n_samples, n_pings), np.nan)
    flips = np.zeros((n_samples, n_pings), dtype=bool)
    has_det = det_line >= 0

    for s in range(n_samples):
        binary = all_preds[s] >= threshold
        for ping in np.where(has_det)[0]:
            d0 = det_line[ping]
            lo = max(0, d0 - win)
            hi = min(n_range, d0 + win + 1)
            seg = np.where(binary[lo:hi, ping])[0]
            if seg.size:
                depths[s, ping] = range_arr[lo + seg[0]]
            # unconstrained first crossing, for the flip diagnostic
            full = np.where(binary[:, ping])[0]
            if full.size and full[0] < d0 - win:
                flips[s, ping] = True

    n_valid = np.sum(np.isfinite(depths), axis=0)
    with np.errstate(invalid="ignore"):
        mean_d = np.nanmean(depths, axis=0)
        q025 = np.nanpercentile(depths, 2.5, axis=0)
        q975 = np.nanpercentile(depths, 97.5, axis=0)
        std_d = np.nanstd(depths, axis=0)
    bad = n_valid < 3
    for arr in (mean_d, q025, q975, std_d):
        arr[bad] = np.nan
    flip_rate = flips.mean(axis=0)
    flip_rate[~has_det] = np.nan
    return mean_d, q025, q975, std_d, flip_rate, n_valid


def plot_per_file(
    name, sv_channels, range_arr, std_prob, mean_d, q025, q975, std_d, flip_rate, save_path
):
    echogram = sv_channels[1]  # 200 kHz, normalised [0, 1]
    n_range, n_pings = echogram.shape
    extent = [0, n_pings, range_arr[-1], range_arr[0]]
    pings = np.arange(n_pings)
    valid = np.isfinite(mean_d)

    fig, (ax0, ax1, ax2) = plt.subplots(
        3,
        1,
        figsize=(FIGSIZE_FULL[0], FIGSIZE_FULL[1] + 4),
        gridspec_kw={"height_ratios": [3, 3, 1.4]},
    )

    # 1) echogram + mean bottom line + 95% boundary band
    im0 = ax0.imshow(
        echogram, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper", extent=extent
    )
    if valid.any():
        ax0.fill_between(
            pings[valid],
            q025[valid],
            q975[valid],
            color="orange",
            alpha=0.6,
            linewidth=0,
            label="95% MC boundary band",
        )
        ax0.plot(pings[valid], mean_d[valid], color="red", linewidth=1.4, label="Mean bottom")
    cbar0 = fig.colorbar(im0, ax=ax0, pad=0.01)
    style_colorbar(cbar0, label="Normalised Sv")
    ax0.set_ylabel("Range (m)")
    ax0.set_title("MC-Dropout seabed boundary: mean line and 95% band")
    ax0.legend(loc="upper right", framealpha=0.9)

    # 2) per-pixel MC standard deviation
    im1 = ax1.imshow(
        std_prob,
        aspect="auto",
        cmap="magma",
        origin="upper",
        extent=extent,
        vmin=0,
        vmax=max(1e-3, float(np.percentile(std_prob, 99.5))),
    )
    if valid.any():
        ax1.plot(pings[valid], mean_d[valid], color="cyan", linewidth=0.8, alpha=0.8)
    cbar1 = fig.colorbar(im1, ax=ax1, pad=0.01)
    style_colorbar(cbar1, label="MC std of p")
    ax1.set_ylabel("Range (m)")
    ax1.set_title("Per-pixel predictive standard deviation")

    # 3) per-ping boundary std and near-surface flip rate
    ax2.plot(
        pings[valid], std_d[valid], color="darkorange", linewidth=1.3, label="Boundary std (m)"
    )
    ax2.set_ylabel("Boundary std (m)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_xlabel("Ping")
    ax2.set_xlim(0, n_pings)
    ax2b = ax2.twinx()
    fvalid = np.isfinite(flip_rate)
    ax2b.plot(pings[fvalid], flip_rate[fvalid], color="steelblue", linewidth=1.0, alpha=0.7)
    ax2b.set_ylabel("Flip rate", color="steelblue")
    ax2b.set_ylim(0, 1.02)
    ax2b.tick_params(axis="y", labelcolor="steelblue")
    ax2b.spines["top"].set_visible(False)
    ax2.set_title("Boundary uncertainty (orange) and near-surface flip rate (blue)")

    fig.suptitle(name)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)


def plot_spaghetti(name, all_preds, sv_channels, range_arr, save_path, threshold=0.5):
    echogram = sv_channels[1]
    n_samples, n_range, n_pings = all_preds.shape
    extent = [0, n_pings, range_arr[-1], range_arr[0]]
    pings = np.arange(n_pings)

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    ax.imshow(
        echogram, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper", extent=extent
    )
    for s in range(n_samples):
        line = extract_bottom_line(all_preds[s], threshold=threshold)
        valid = line >= 0
        if valid.any():
            ax.plot(
                pings[valid],
                range_arr[np.clip(line[valid], 0, n_range - 1)],
                color="red",
                linewidth=0.5,
                alpha=0.25,
            )
    ax.set_ylabel("Range (m)")
    ax.set_xlabel("Ping")
    ax.set_title(f"{n_samples} MC-Dropout bottom-line samples")
    fig.suptitle(name)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)


def plot_aggregate(std_all, ent_at_std, band_all, n_files, save_path):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(FIGSIZE_FULL[0], FIGSIZE_FULL[1] - 1))

    # per-ping bottom-depth std distribution
    band = band_all[np.isfinite(band_all)]
    ax0.hist(band, bins=60, color="darkorange", alpha=0.85)
    med = float(np.median(band)) if band.size else float("nan")
    ax0.axvline(med, color="black", linestyle="--", linewidth=1.2, label=f"median {med:.2f} m")
    ax0.set_xlabel("Per-ping seabed-boundary std (m)")
    ax0.set_ylabel("Ping count")
    ax0.set_title("Seabed-boundary uncertainty")
    ax0.legend(framealpha=0.9)

    # does MC std track single-pass entropy?
    s = std_all.ravel()
    e = ent_at_std.ravel()
    keep = (s > 1e-4) | (e > 1e-3)
    s, e = s[keep], e[keep]
    if s.size > 200_000:
        idx = np.random.default_rng(0).choice(s.size, 200_000, replace=False)
        s, e = s[idx], e[idx]
    hb = ax1.hexbin(e, s, gridsize=60, cmap="viridis", bins="log", mincnt=1)
    if s.size > 2:
        r = float(np.corrcoef(e, s)[0, 1])
        ax1.set_title(f"MC std vs single-pass entropy (r={r:.2f})")
    else:
        ax1.set_title("MC std vs single-pass entropy")
    cbar = fig.colorbar(hb, ax=ax1, pad=0.01)
    style_colorbar(cbar, label="log count")
    ax1.set_xlabel("Single-pass entropy (bits)")
    ax1.set_ylabel("MC std of p")

    fig.suptitle(f"MC-Dropout uncertainty diagnostics ({n_files} files)")
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)


def plot_dropout_sweep(
    name, model, sv_channels, det_prob, range_arr, device, ps, n_samples, save_path
):
    """Sweep the dropout rate (model must be dropout-free on entry/exit)."""
    med_band, med_pix = [], []
    for p in ps:
        _, std_prob, all_preds = mc_dropout_predict(
            model,
            sv_channels,
            TARGET_SIZE,
            device,
            n_samples=n_samples,
            patch_width=PATCH_WIDTH,
            stride=256,
            dropout_p=p,
        )
        _strip_dropout(model)  # next p starts from a clean model
        _, _, _, std_d, _, _ = bottom_line_distribution(all_preds, det_prob, range_arr)
        med_band.append(float(np.nanmedian(std_d)))
        med_pix.append(float(np.median(std_prob)))

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.plot(ps, med_band, marker="o", color="darkorange", label="median seabed-boundary std (m)")
    ax.set_xlabel("Dropout rate p")
    ax.set_ylabel("Median boundary std (m)")
    ax.set_title(f"Uncertainty vs dropout rate: {name}")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    savefig(fig, save_path)
    plt.close(fig)
    return ps, med_band, med_pix


def _strip_dropout(model):
    """Undo add_mc_dropout so a fresh dropout rate can be injected."""
    import torch.nn as nn

    for block in model.decoder.blocks:
        for conv_relu in [block.conv1, block.conv2]:
            keep = [m for m in conv_relu if not isinstance(m, nn.Dropout2d)]
            for name in list(conv_relu._modules):
                del conv_relu._modules[name]
            for i, m in enumerate(keep):
                conv_relu.add_module(str(i), m)


def write_readme(out_dir, n_files, n_samples, dropout_p, ckpt):
    txt = (
        "# MC-Dropout Bottom Uncertainty\n\n"
        "mc-dropout  predictive-uncertainty  bottom-line-band  entropy  echosounder\n\n"
        "Dropout injected into the trained (dropout-free) decoder and kept active "
        f"for {n_samples} stochastic passes (p={dropout_p}). Per-pixel mean/std "
        "probability maps plus a per-ping bottom-depth distribution; an aggregate "
        "figure compares MC std against single-pass prediction entropy, and a sweep "
        "shows how the band scales with the dropout rate.\n\n"
        f"Checkpoint: {ckpt}\n"
        f"Files: {n_files}, samples: {n_samples}, dropout_p: {dropout_p}\n"
        f"Created: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
    )
    (out_dir / "_README.md").write_text(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / DEFAULT_CHECKPOINT))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--n-files", type=int, default=4)
    ap.add_argument("--max-pings", type=int, default=1500)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--dropout-p", type=float, default=0.2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument(
        "--max-below-frac",
        type=float,
        default=0.88,
        help="skip files where the deterministic model collapses",
    )
    ap.add_argument("--no-sweep", action="store_true", help="skip the dropout-rate sweep figure")
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
    print(f"Scanning {len(raw_files)} raw files for {args.n_files} usable files")
    if not raw_files:
        print("No raw files found.")
        return

    model = load_model(args.checkpoint, device)

    std_chunks, ent_chunks, band_chunks = [], [], []
    n_done = 0
    swept = False

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

        # deterministic single pass: collapse guard + entropy reference
        det_prob = predict_full_echogram(
            model, sv_channels, TARGET_SIZE, device, patch_width=PATCH_WIDTH, stride=256
        )
        below_frac = float(det_prob.mean())
        if below_frac > args.max_below_frac:
            print(f"  skip {name}: model collapsed (below frac {below_frac:.2f})")
            continue

        n_done += 1
        print(f"[{n_done}/{args.n_files}] {name} (below frac {below_frac:.2f})")

        _, std_prob, all_preds = mc_dropout_predict(
            model,
            sv_channels,
            TARGET_SIZE,
            device,
            n_samples=args.n_samples,
            patch_width=PATCH_WIDTH,
            stride=256,
            dropout_p=args.dropout_p,
        )
        _strip_dropout(model)  # keep model dropout-free for det passes / sweep

        det_entropy = prediction_entropy(det_prob)
        mean_d, q025, q975, std_d, flip_rate, n_valid = bottom_line_distribution(
            all_preds, det_prob, range_arr
        )

        plot_per_file(
            name,
            sv_channels,
            range_arr,
            std_prob,
            mean_d,
            q025,
            q975,
            std_d,
            flip_rate,
            out_dir / f"{name}_mc_uncertainty.png",
        )
        plot_spaghetti(
            name, all_preds, sv_channels, range_arr, out_dir / f"{name}_mc_spaghetti.png"
        )

        std_chunks.append(std_prob.ravel())
        ent_chunks.append(det_entropy.ravel())
        band_chunks.append(std_d)

        med_band = float(np.nanmedian(std_d))
        mean_flip = float(np.nanmean(flip_rate))
        print(
            f"  median boundary std: {med_band:.3f} m, "
            f"mean flip rate: {mean_flip:.2f}, "
            f"max per-pixel std: {std_prob.max():.3f}"
        )

        if not args.no_sweep and not swept:
            plot_dropout_sweep(
                name,
                model,
                sv_channels,
                det_prob,
                range_arr,
                device,
                ps=[0.05, 0.1, 0.2, 0.3, 0.4],
                n_samples=args.n_samples,
                save_path=out_dir / "dropout_rate_sweep.pdf",
            )
            swept = True

    if std_chunks:
        std_all = np.concatenate(std_chunks)
        ent_all = np.concatenate(ent_chunks)
        band_all = np.concatenate(band_chunks)
        plot_aggregate(std_all, ent_all, band_all, n_done, out_dir / "mc_dropout_diagnostics.pdf")

    write_readme(out_dir, n_done, args.n_samples, args.dropout_p, Path(args.checkpoint).parent.name)
    print(f"Done. Output in {out_dir}")


if __name__ == "__main__":
    main()
