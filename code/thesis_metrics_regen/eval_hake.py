"""Foreign-survey (NOAA hake) transfer recompute (G) with chunk_arch_sweep_2.

Reuses the heavy lifting in code/data_visualization/eval_hake_test_raw.py (echopype decode,
stitching, algorithmic 38 kHz label, below-region IoU/Dice/depth error) but with the Basto
Standard U-Net checkpoint, and adds a machine-readable summary of every hake number the thesis
cites: per-stitched-echogram IoU and depth deviation, seabed depth range, the 642-ping vs
214-ping (stitched vs native) IoU comparison on the shelf, and the open-water fraction.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import data_visualization.eval_hake_test_raw as H  # noqa: E402
from deep_learning.unet_model import predict_full_echogram  # noqa: E402

CKPT = ROOT / "code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth"
OUT = Path(__file__).resolve().parent / "outputs/hake_summary.json"


def label_depths(res):
    rc = res["range_crop"]
    idx = res["label_idx_crop"]
    v = idx >= 0
    if not v.any():
        return np.array([])
    return rc[np.clip(idx[v], 0, len(rc) - 1)]


def eval_groups(model, device, groups, label):
    out = []
    for stems in groups:
        try:
            res = H.prepare_group(stems)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {stems[0]}: {e}", flush=True)
            continue
        pred = predict_full_echogram(model, res["sv_channels"], H.TARGET_SIZE, device,
                                     patch_width=H.PATCH_WIDTH, stride=H.STRIDE)
        m = H.compute_metrics(res["label_mask"], pred, res["range_crop"])
        ld = label_depths(res)
        rec = {
            "group": f"{stems[0]}+{len(stems) - 1}",
            "n_pings": int(res["sv_channels"].shape[2]),
            "iou": float(m["iou"]), "dice": float(m["dice"]),
            "depth_mae_m": float(m["mae_m"]), "depth_median_m": float(m["median_m"]),
            "coverage": float(m["detect_frac"]),
            "seabed_min_m": float(ld.min()) if ld.size else None,
            "seabed_max_m": float(ld.max()) if ld.size else None,
        }
        out.append((rec, label))
        print(f"  [{label}] {rec['group']} pings={rec['n_pings']} IoU={rec['iou']:.3f} "
              f"MAE={rec['depth_mae_m']:.1f}m seabed=[{rec['seabed_min_m']},{rec['seabed_max_m']}]",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--open-water-scan", type=int, default=0,
                    help="if >0, scan this many raw files for the open-water (empty-label) fraction")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = H.load_model(str(CKPT), device)
    print(f"hake transfer with chunk_arch_sweep_2 on {device}", flush=True)

    stems = [p.stem for p in sorted(H.RAW_DIR.glob("*.raw")) if not p.name.startswith("._")]
    shelf, slope = stems[0:9], stems[56:65]

    # Stitched (group-size 3, ~642 pings) on shelf + slope.
    shelf3 = [shelf[i:i + 3] for i in range(0, len(shelf), 3) if len(shelf[i:i + 3]) == 3]
    slope3 = [slope[i:i + 3] for i in range(0, len(slope), 3) if len(slope[i:i + 3]) == 3]
    stitched = eval_groups(model, device, shelf3, "shelf-642") + \
        eval_groups(model, device, slope3, "slope-642")

    # Native (group-size 1, ~214 pings) on shelf only, for the aspect-ratio comparison.
    native_shelf = eval_groups(model, device, [[s] for s in shelf], "shelf-214")

    shelf_stitched_iou = [r["iou"] for r, lab in stitched if lab == "shelf-642"]
    slope_stitched = [r for r, lab in stitched if lab == "slope-642"]
    shelf_native_iou = [r["iou"] for r, lab in native_shelf]

    all_recs = [r for r, _ in stitched]
    ious = [r["iou"] for r in all_recs]
    seabeds = [v for r in all_recs for v in (r["seabed_min_m"], r["seabed_max_m"]) if v]

    res = {
        "checkpoint": "chunk_arch_sweep_2",
        "n_stitched_echograms": len(all_recs),
        "iou_min": float(min(ious)) if ious else None,
        "iou_max": float(max(ious)) if ious else None,
        "seabed_depth_min_m": float(min(seabeds)) if seabeds else None,
        "seabed_depth_max_m": float(max(seabeds)) if seabeds else None,
        "shelf_depth_mae_range_m": [
            float(min(r["depth_mae_m"] for r, lab in stitched if lab == "shelf-642")),
            float(max(r["depth_mae_m"] for r, lab in stitched if lab == "shelf-642")),
        ] if shelf_stitched_iou else None,
        "deepest_slope_depth_mae_m": float(max(r["depth_mae_m"] for r in slope_stitched))
        if slope_stitched else None,
        "shelf_iou_642_mean": float(np.mean(shelf_stitched_iou)) if shelf_stitched_iou else None,
        "shelf_iou_214_mean": float(np.mean(shelf_native_iou)) if shelf_native_iou else None,
        "aspect_ratio_iou_gain": (float(np.mean(shelf_stitched_iou) - np.mean(shelf_native_iou))
                                  if shelf_stitched_iou and shelf_native_iou else None),
        "pings_per_raw_file": 214,
        "stitched_window_pings": 642,
        "n_raw_files_total": len(stems),
        "stitched_echograms": all_recs,
        "native_shelf_echograms": [r for r, _ in native_shelf],
    }

    if args.open_water_scan:
        empty = 0
        scanned = 0
        for s in stems[: args.open_water_scan]:
            try:
                r = H.prepare_group([s])
            except Exception:  # noqa: BLE001
                continue
            scanned += 1
            if not (r["label_idx_crop"] >= 0).any():
                empty += 1
        res["open_water_fraction"] = empty / scanned if scanned else None
        res["open_water_scanned"] = scanned
        print(f"open-water fraction: {empty}/{scanned}", flush=True)

    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("stitched_echograms", "native_shelf_echograms")}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
