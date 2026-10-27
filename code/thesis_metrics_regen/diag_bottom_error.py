"""Regenerate the bottom-error distribution (B) with chunk_arch_sweep_2.

The existing Figures/bottom_error_distribution/signed_bottom_errors.npy was produced with the
arch_sweep_6 vanilla U-Net, but the thesis figure attributes it to the Basto Standard U-Net
(chunk_arch_sweep_2). This recomputes the per-ping signed bottom-depth errors with
chunk_arch_sweep_2 over its held-out Oct-2024 evaluation files, writes the array and the
corrected median / P95 into the regen outputs, and regenerates the distribution figure into
Latex/Figures/regen_2026/.
"""

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from deep_learning.architectures import create_model  # noqa: E402
from deep_learning.evaluate import evaluate_file, extract_bottom_line  # noqa: E402

CKPT = ROOT / "code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth"
EVAL = ROOT / "code/checkpoints/chunk_arch_sweep_2/eval_metrics.json"
ZARR_DIR = ROOT / "Datasets/echopype_zarr"
FIG_DIR = ROOT / "Latex/Figures/regen_2026"
OUT_NPY = Path(__file__).resolve().parent / "outputs/signed_bottom_errors_chunk_arch_sweep_2.npy"
OUT_JSON = Path(__file__).resolve().parent / "outputs/bottom_error_summary.json"
FREQ1, FREQ2 = 38000, 200000
TARGET = (512, 512)
THRESH = 0.5


def load_model(device):
    state = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = state.get("config", {})
    model = create_model(
        architecture=cfg.get("architecture", "smp_unet"),
        encoder_name=cfg.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=cfg.get("in_channels", 3),
        classes=1,
    )
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to(device).eval()


def signed_errors(prediction, gt_mask, range_array):
    pi = extract_bottom_line(prediction, threshold=THRESH)
    gi = extract_bottom_line(gt_mask, threshold=0.5)
    valid = (pi >= 0) & (gi >= 0)
    if not valid.any():
        return np.array([])
    n = len(range_array)
    return range_array[np.clip(pi[valid], 0, n - 1)] - range_array[np.clip(gi[valid], 0, n - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    names = [r["file"] for r in json.loads(EVAL.read_text())["per_file"]]
    random.Random(args.seed).shuffle(names)
    paths = [ZARR_DIR / n for n in names if (ZARR_DIR / n).exists()][: args.n_files]
    model = load_model(device)
    print(f"bottom-error over {len(paths)} files on {device}", flush=True)

    allerr = []
    for i, zp in enumerate(paths, 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, pred, gt, _, ra = evaluate_file(
                    model,
                    zp,
                    FREQ1,
                    FREQ2,
                    TARGET,
                    device,
                    vmin=-80.0,
                    vmax=-34.0,
                    threshold=THRESH,
                )
            allerr.append(signed_errors(pred, gt, ra))
        except Exception as e:  # noqa: BLE001
            print(f"  {zp.name}: FAIL {e}", flush=True)
        if i % 10 == 0:
            print(f"  {i}/{len(paths)}", flush=True)

    err = np.concatenate([e for e in allerr if len(e)])
    np.save(OUT_NPY, err)
    ae = np.abs(err)
    res = {
        "checkpoint": "chunk_arch_sweep_2",
        "n_files": len(paths),
        "n_pings": int(err.size),
        "median_abs_err_m": float(np.median(ae)),
        "p95_abs_err_m": float(np.percentile(ae, 95)),
        "mean_abs_err_m": float(ae.mean()),
        "signed_median_m": float(np.median(err)),
        "signed_mean_m": float(err.mean()),
    }
    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    med, p95 = res["median_abs_err_m"], res["p95_abs_err_m"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(ae, 0, 10), bins=80, color="#4575b4", edgecolor="white", linewidth=0.2)
    ax.axvline(med, color="#1a1a1a", ls="--", lw=2, label=f"median = {med:.2f} m")
    ax.axvline(p95, color="#d73027", ls=":", lw=2, label=f"P95 = {p95:.2f} m")
    ax.set_xlim(0, 10)
    ax.set_xlabel("Absolute bottom-depth difference (m)")
    ax.set_ylabel("Number of pings")
    ax.set_title("Bottom-depth difference: Basto Standard U-Net vs reference (Oct 2024)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chunk_arch_sweep_2_bottom_error_distribution.png", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT_NPY}\nwrote {OUT_JSON}\nfigure -> {FIG_DIR}")


if __name__ == "__main__":
    main()
