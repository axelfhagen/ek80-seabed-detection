"""MC-Dropout diagnostics (F) with chunk_arch_sweep_2.

On held-out Jan-2024 files, injects dropout into the trained dropout-free decoder and runs 20
stochastic passes. Produces the dropout-rate sweep of the median seabed-boundary standard
deviation, the p=0.2 median, and the Spearman correlation between the MC boundary std and the
single-pass prediction entropy. Writes outputs/mc_dropout_summary.json and regenerates the
sweep and diagnostics figures into Latex/Figures/regen_2026/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from deep_learning.unet_model import create_unet, predict_full_echogram  # noqa: E402
from deep_learning.evaluate import load_file_data, extract_bottom_line  # noqa: E402
from deep_learning.training.survey_groups import get_survey_files  # noqa: E402
from validation.uncertainty import mc_dropout_predict, prediction_entropy  # noqa: E402
from data_visualization.analyze_mc_dropout_uncertainty import (  # noqa: E402
    bottom_line_distribution, _strip_dropout)

CKPT = ROOT / "code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth"
FIG_DIR = ROOT / "Latex/Figures/regen_2026"
OUT = Path(__file__).resolve().parent / "outputs/mc_dropout_summary.json"
FREQ1, FREQ2 = 38000, 200000
VMIN, VMAX = -80.0, -34.0
TARGET = (512, 512)
RATES = [0.05, 0.1, 0.2, 0.3, 0.4]


def load_model(device):
    model = create_unet(encoder_name="resnet34", in_channels=3)
    state = torch.load(CKPT, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    model.load_state_dict({k.removeprefix("model."): v for k, v in sd.items()}, strict=True)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(device)
    files = get_survey_files("jan_2024", str(ROOT / "Datasets/echopype_zarr"))
    rng = np.random.default_rng(args.seed)
    sel = rng.permutation(len(files))[: args.n_files]
    files = [files[i] for i in sorted(sel)]
    print(f"MC-dropout over {len(files)} jan_2024 files on {device}", flush=True)

    data = []
    for zp in files:
        sv, gt, gt_idx, rng_arr = load_file_data(
            zp, FREQ1, FREQ2, max_range_m=None, max_range_samples=None,
            vmin=VMIN, vmax=VMAX, mask_mode="below", line_thickness=1)
        data.append((zp.stem, sv, rng_arr))

    # Sweep: median seabed-boundary std vs dropout rate.
    sweep = {}
    for p in RATES:
        stds = []
        for name, sv, rng_arr in data:
            det = predict_full_echogram(model, sv, TARGET, device,
                                        patch_width=512, stride=256)
            _, _, all_preds = mc_dropout_predict(
                model, sv, TARGET, device, n_samples=args.n_samples,
                patch_width=512, stride=256, dropout_p=p)
            _strip_dropout(model)
            _, _, _, std_d, _, _ = bottom_line_distribution(all_preds, det, rng_arr)
            stds.append(std_d[np.isfinite(std_d)])
        allstd = np.concatenate(stds) if stds else np.array([])
        sweep[str(p)] = float(np.median(allstd)) if allstd.size else None
        print(f"  p={p}: median boundary std = {sweep[str(p)]}", flush=True)

    # At p=0.2: median + Spearman(MC std, single-pass entropy) at the boundary.
    boundary_std_all, entropy_all = [], []
    for name, sv, rng_arr in data:
        det = predict_full_echogram(model, sv, TARGET, device,
                                    patch_width=512, stride=256)
        _, std_prob, all_preds = mc_dropout_predict(
            model, sv, TARGET, device, n_samples=args.n_samples,
            patch_width=512, stride=256, dropout_p=0.2)
        _strip_dropout(model)
        ent = prediction_entropy(det)
        det_line = extract_bottom_line(det, 0.5)
        for ping in np.where(det_line >= 0)[0]:
            r = det_line[ping]
            boundary_std_all.append(std_prob[r, ping])
            entropy_all.append(ent[r, ping])
    bstd = np.array(boundary_std_all)
    ent = np.array(entropy_all)
    rho = float(stats.spearmanr(bstd, ent).correlation)

    res = {
        "checkpoint": "chunk_arch_sweep_2",
        "survey": "jan_2024",
        "n_files": len(files),
        "files": [d[0] for d in data],
        "n_samples": args.n_samples,
        "dropout_rate_sweep_median_boundary_std_m": sweep,
        "median_boundary_std_m_at_p0p2": sweep["0.2"],
        "spearman_mcstd_vs_entropy": rho,
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # Figure: dropout-rate sweep.
    xs = [float(k) for k in sweep]
    ys = [sweep[str(x)] for x in xs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-", color="darkorange")
    ax.set_xlabel("Dropout rate p")
    ax.set_ylabel("Median boundary std (m)")
    ax.set_title("MC-Dropout band width vs dropout rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dropout_rate_sweep.pdf")
    plt.close(fig)

    # Figure: diagnostics (boundary-std hist + scatter vs entropy).
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4))
    a0.hist(bstd[np.isfinite(bstd)], bins=60, color="darkorange")
    a0.set_xlabel("MC std of p at boundary")
    a0.set_ylabel("Count")
    a0.set_title(f"Boundary MC std (p=0.2), median {sweep['0.2']:.2f} m")
    a1.scatter(ent, bstd, s=2, alpha=0.2, color="steelblue")
    a1.set_xlabel("Single-pass entropy")
    a1.set_ylabel("MC std of p")
    a1.set_title(f"MC std vs entropy (Spearman r = {rho:.2f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mc_dropout_diagnostics.pdf")
    plt.close(fig)
    print(f"wrote {OUT}\nfigures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
