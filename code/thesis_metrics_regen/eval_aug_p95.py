"""Verify the augmentation-ablation per-file P95 claim (C).

The thesis states file D20250118-T095329 has a 95th-percentile depth error of ~133 m against
a sub-metre median, and that it sits in every augmentation-ablation configuration's validation
set. The aggregate aug_ablation_summary.csv has no per-file rows, so this re-evaluates that one
file with the augmentation-ablation checkpoints and records per-file IoU / median / P95.
Appends the result to outputs/headline_summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from deep_learning.unet_model import create_unet  # noqa: E402
from deep_learning.evaluate import evaluate_file  # noqa: E402

ZARR = ROOT / "Datasets/echopype_zarr/D20250118-T095329.zarr"
OUT = Path(__file__).resolve().parent / "outputs/headline_summary.json"
FREQ1, FREQ2 = 38000, 200000
VMIN, VMAX = -80.0, -34.0
TARGET = (512, 512)


def load_model(ckpt, device):
    model = create_unet(encoder_name="resnet34", in_channels=3)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    model.load_state_dict({k.removeprefix("model."): v for k, v in sd.items()}, strict=True)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = sorted((ROOT / "code/checkpoints").glob("exp_*_aug_ablation_*/smp_unet_resnet34_best.pth"))
    print(f"found {len(ckpts)} aug-ablation checkpoints", flush=True)
    rows = []
    for ck in ckpts:
        cfg = ck.parent.name
        m, *_ = evaluate_file(
            load_model(ck, device), ZARR, freq1_hz=FREQ1, freq2_hz=FREQ2,
            target_size=TARGET, device=device, patch_width=512, stride=512,
            max_range_m=None, max_range_samples=None, vmin=VMIN, vmax=VMAX,
            threshold=0.5, mask_mode="below")
        row = {"config": cfg, "iou": float(m.iou),
               "bottom_error_median_m": float(m.bottom_error_median_m),
               "bottom_error_p95_m": float(m.bottom_error_p95_m),
               "bottom_error_mean_m": float(m.bottom_error_mean_m)}
        rows.append(row)
        print(f"  {cfg}: IoU {row['iou']:.3f}  median {row['bottom_error_median_m']:.2f} m  "
              f"P95 {row['bottom_error_p95_m']:.1f} m", flush=True)

    p95s = [r["bottom_error_p95_m"] for r in rows]
    summary = {
        "file": "D20250118-T095329",
        "n_checkpoints": len(rows),
        "p95_min_m": min(p95s) if p95s else None,
        "p95_max_m": max(p95s) if p95s else None,
        "per_checkpoint": rows,
    }
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data["aug_p95_check"] = summary
    OUT.write_text(json.dumps(data, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"appended aug_p95_check -> {OUT}")


if __name__ == "__main__":
    main()
