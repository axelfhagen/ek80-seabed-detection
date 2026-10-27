#!/usr/bin/env python
"""Isolated Hake fine-tuning experiment for chunk_arch_sweep_2."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data_visualization.eval_hake_test_raw import (
    PATCH_WIDTH,
    RAW_DIR,
    STRIDE,
    TARGET_SIZE,
    ZARR_DIR,
    load_model as load_hake_model,
    prepare_group,
)
from deep_learning.echo_dataset import (
    LazyPatchDataset,
    _resize_patch,
    get_augmentation_pipeline,
)
from deep_learning.evaluate import evaluate_file, load_file_data
from deep_learning.training.survey_groups import get_split_files
from deep_learning.unet_model import (
    build_loss,
    create_unet,
    freeze_encoder_stages,
    get_param_groups,
    predict_full_echogram,
    train_one_epoch,
    validate,
)
from validation.conformal import (
    calibrate_depth,
    calibrate_locally_weighted,
    calibrate_profile,
    conformal_quantile,
    compute_transition_width,
    extract_bottom_depths,
)

BASE_CKPT = ROOT / "code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth"
CKPT_DIR = ROOT / "code/checkpoints/hake_finetune_chunk_arch_sweep_2_seed0"
FIG_DIR = ROOT / "Figures/hake_finetune_chunk_arch_sweep_2"
CACHE_DIR = ROOT / "code/experiments/hake_finetune/cache"
ALPHA = 0.10
METHODS = ("constant", "local", "profile")


def auto_device(value: str) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value


def valid_hake_stems(limit: int = 0) -> list[str]:
    raw = [p.stem for p in sorted(RAW_DIR.glob("*.raw")) if not p.name.startswith("._")]
    stems = [s for s in raw if (ZARR_DIR / f"{s}.zarr").is_dir()]
    skipped = sorted(set(raw) - set(stems))
    if skipped:
        print(
            f"Skipping {len(skipped)} Hake raw files without valid zarr: {', '.join(skipped)}",
            flush=True,
        )
    return stems[:limit] if limit else stems


def groups(stems: list[str], size: int, keep_tail: bool = False) -> list[list[str]]:
    out = [stems[i : i + size] for i in range(0, len(stems), size)]
    return out if keep_tail else [g for g in out if len(g) == size]


def make_split(stems: list[str], size: int, seed: int) -> dict[str, list[str]]:
    full = groups(stems, size)
    used = len(full) * size
    shuffled = [full[i] for i in np.random.default_rng(seed).permutation(len(full))]
    n_cal = max(1, round(0.10 * len(stems) / size))
    n_test = max(1, round(0.10 * len(stems) / size))
    n_val = max(1, round(0.15 * len(stems) / size))
    split_groups = {
        "calibration": shuffled[:n_cal],
        "test": shuffled[n_cal : n_cal + n_test],
        "val": shuffled[n_cal + n_test : n_cal + n_test + n_val],
        "train": shuffled[n_cal + n_test + n_val :],
    }
    tail = stems[used:]
    if tail:
        split_groups["train"].append(tail)
    return {name: [stem for group in split_groups[name] for stem in group] for name in split_groups}


def write_manifest(args, split: dict[str, list[str]]) -> None:
    data = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "raw_dir": str(RAW_DIR),
        "zarr_dir": str(ZARR_DIR),
        "seed": args.seed,
        "group_size": args.group_size,
        "fractions": {"calibration": 0.10, "test": 0.10, "val": 0.15},
        "counts": {k: len(v) for k, v in split.items()},
        "splits": split,
    }
    (args.checkpoint_dir / "hake_split_manifest.json").write_text(json.dumps(data, indent=2))


def patch_path(args, split_name: str, group: list[str]) -> Path:
    return (
        args.cache_dir
        / "patches"
        / split_name
        / f"{group[0]}_n{len(group)}_pw{args.patch_width}_s{args.stride}.npz"
    )


def cache_group(args, split_name: str, group: list[str]) -> tuple[Path, int]:
    path = patch_path(args, split_name, group)
    if path.exists() and not args.recompute_cache:
        with np.load(path, allow_pickle=False) as data:
            return path, int(data["patches"].shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    result = prepare_group(group)
    image = result["sv_channels"]
    mask = result["label_mask"].astype(np.uint8)
    n = image.shape[2]
    if n <= args.patch_width:
        windows = [(0, n)]
    else:
        starts = list(range(0, n - args.patch_width + 1, args.stride))
        if starts[-1] != n - args.patch_width:
            starts.append(n - args.patch_width)
        windows = [(start, start + args.patch_width) for start in starts]
    patches, masks = [], []
    for start, end in windows:
        p, m = _resize_patch(image[:, :, start:end], mask[:, start:end], tuple(args.target_size))
        patches.append(p)
        masks.append(m)
    np.savez_compressed(
        path,
        patches=np.stack(patches).astype(np.float32),
        masks=np.stack(masks).astype(np.uint8),
    )
    return path, len(patches)


def build_dataset(args, stems: list[str], split_name: str, augment: bool) -> LazyPatchDataset:
    grouped = groups(stems, args.group_size, keep_tail=True)
    if args.smoke:
        grouped = grouped[:1]
    index = []
    for i, group in enumerate(grouped, 1):
        print(
            f"  [{split_name} {i}/{len(grouped)}] {'+'.join(group)}",
            flush=True,
        )
        path, count = cache_group(args, split_name, group)
        index.extend((path, j) for j in range(count))
    transform = None
    if augment:
        transform = get_augmentation_pipeline(
            p_hflip=0.5,
            p_noise=0.15,
            p_brightness=0.15,
            p_channel_dropout=0.02,
            p_ping_dropout_zero=0.03,
            p_ping_dropout_shuffle=0.02,
            p_ping_dropout_noise=0.02,
        )
    dataset = LazyPatchDataset(index, target_size=tuple(args.target_size), transform=transform)
    print(
        f"  {split_name}: {len(dataset)} patches from {len(grouped)} groups",
        flush=True,
    )
    return dataset


def model_from_checkpoint(checkpoint: Path, device: str):
    model = create_unet(encoder_name="resnet34", encoder_weights=None, in_channels=3)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    weights = state.get("model_state_dict", state)
    weights = {k.removeprefix("model."): v for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return model.to(device)


def make_training_model(args, device: str):
    if args.init_mode == "checkpoint":
        model = model_from_checkpoint(args.base_checkpoint, device)
        print(f"Initialised from checkpoint: {args.base_checkpoint}", flush=True)
        return model
    model = create_unet(encoder_name="resnet34", encoder_weights=None, in_channels=3)
    print("Initialised from scratch: ResNet-34 U-Net, no pretrained weights", flush=True)
    return model.to(device)


def atomic_model_save(model, path: Path, epoch: int, metrics: dict, args):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_iou": float(metrics["iou"]),
            "val_bottom_mae": float(metrics["bottom_mae"]),
            "init_mode": args.init_mode,
            "base_checkpoint": (
                str(args.base_checkpoint.resolve()) if args.init_mode == "checkpoint" else None
            ),
        },
        tmp,
    )
    tmp.replace(path)


def train(args, split: dict[str, list[str]]) -> Path:
    device = auto_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_ds = build_dataset(args, split["train"], "train", True)
    val_ds = build_dataset(args, split["val"], "val", False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )
    model = make_training_model(args, device)
    freeze_encoder_stages(model, args.freeze_stages)
    optimizer = torch.optim.AdamW(
        get_param_groups(model, args.encoder_lr, args.decoder_lr),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    loss = build_loss("dice_bce", bce_weight=0.5)
    scaler = torch.amp.GradScaler() if device == "cuda" else None
    history = {
        f"{side}_{metric}": []
        for side in ("train", "val")
        for metric in ("loss", "iou", "dice", "bottom_mae")
    }
    best = math.inf
    stale = 0
    best_path = args.checkpoint_dir / "smp_unet_resnet34_best.pth"
    epochs = 1 if args.smoke else args.epochs
    for epoch in range(epochs):
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss,
            device,
            scaler=scaler,
            mask_mode="below",
            grad_clip_norm=1.0,
        )
        va = validate(model, val_loader, loss, device, mask_mode="below")
        scheduler.step(va["bottom_mae"])
        for metric in ("loss", "iou", "dice", "bottom_mae"):
            history[f"train_{metric}"].append(float(tr[metric]))
            history[f"val_{metric}"].append(float(va[metric]))
        print(
            f"Epoch {epoch + 1:02d}/{epochs} train_loss={tr['loss']:.5f} "
            f"val_loss={va['loss']:.5f} val_iou={va['iou']:.4f} "
            f"val_bottom_mae={va['bottom_mae']:.3f}",
            flush=True,
        )
        if va["bottom_mae"] < best:
            best = va["bottom_mae"]
            stale = 0
            atomic_model_save(model, best_path, epoch, va, args)
        else:
            stale += 1
        (args.checkpoint_dir / "training_history.json").write_text(json.dumps(history, indent=2))
        if stale >= args.early_stopping_patience:
            print(f"Early stopping after epoch {epoch + 1}", flush=True)
            break
    print(f"Best checkpoint: {best_path} (val_bottom_mae={best:.3f})", flush=True)
    return best_path


def depth_from_indices(indices: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    out = np.full(indices.shape, np.nan, dtype=np.float64)
    valid = indices >= 0
    out[valid] = ranges[np.clip(indices[valid], 0, len(ranges) - 1)]
    return out


@dataclass
class HakeMetrics:
    group: str
    iou: float
    dice: float
    precision: float
    recall: float
    depth_mae_m: float
    depth_median_m: float
    depth_p95_m: float
    detect_frac: float


def hake_metrics(name: str, result, prob: np.ndarray) -> HakeMetrics:
    pred = prob >= 0.5
    ref = result["label_mask"].astype(bool)
    tp = int((pred & ref).sum())
    fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum())
    eps = 1e-7
    pred_d = extract_bottom_depths(prob, result["range_crop"], threshold=0.5)
    ref_d = depth_from_indices(result["label_idx_crop"], result["range_crop"])
    valid = np.isfinite(pred_d) & np.isfinite(ref_d) & (pred_d > 0) & (ref_d > 0)
    errors = np.abs(pred_d[valid] - ref_d[valid])
    return HakeMetrics(
        name,
        (tp + eps) / (tp + fp + fn + eps),
        (2 * tp + eps) / (2 * tp + fp + fn + eps),
        (tp + eps) / (tp + fp + eps),
        (tp + eps) / (tp + fn + eps),
        float(np.mean(errors)) if errors.size else float("nan"),
        float(np.median(errors)) if errors.size else float("nan"),
        float(np.percentile(errors, 95)) if errors.size else float("nan"),
        float(pred.any(axis=0).mean()),
    )


def metric_summary(rows: list[HakeMetrics]) -> dict:
    return {
        "n_groups": len(rows),
        "mean_iou": float(np.nanmean([r.iou for r in rows])),
        "mean_dice": float(np.nanmean([r.dice for r in rows])),
        "mean_precision": float(np.nanmean([r.precision for r in rows])),
        "mean_recall": float(np.nanmean([r.recall for r in rows])),
        "mean_depth_mae_m": float(np.nanmean([r.depth_mae_m for r in rows])),
        "median_depth_mae_m": float(np.nanmedian([r.depth_mae_m for r in rows])),
        "mean_detect_frac": float(np.nanmean([r.detect_frac for r in rows])),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_overlay(name, result, before, after, path):
    ranges = result["range_crop"]
    extent = [0, result["sv1_crop"].shape[1], ranges[-1], ranges[0]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    ref_d = depth_from_indices(result["label_idx_crop"], ranges)
    for ax, prob, title in zip(axes, (before, after), ("Original", "Fine-tuned")):
        ax.imshow(
            result["sv1_crop"],
            aspect="auto",
            cmap="viridis",
            vmin=-80,
            vmax=-34,
            origin="upper",
            extent=extent,
        )
        pred_d = extract_bottom_depths(prob, ranges, threshold=0.5)
        pings = np.arange(len(ref_d))
        rv = np.isfinite(ref_d) & (ref_d > 0)
        pv = np.isfinite(pred_d) & (pred_d > 0)
        ax.plot(pings[rv], ref_d[rv], color="cyan", lw=2, label="Hake label")
        ax.plot(pings[pv], pred_d[pv], color="red", lw=1.5, label="Prediction")
        ax.set_title(title)
        ax.set_xlabel("Ping")
        ax.set_ylabel("Range (m)")
    axes[0].legend(loc="lower left")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(summary: dict, path: Path, title: str):
    depth_key = (
        "mean_depth_mae_m" if "mean_depth_mae_m" in summary["original"] else "mean_bottom_error_m"
    )
    keys = ("mean_iou", "mean_dice", depth_key)
    labels = ("IoU", "Dice", "Depth MAE (m)")
    x = np.arange(3, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    names = list(summary)
    width = min(0.8 / max(len(names), 1), 0.34)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width
        ax.bar(
            x + offset,
            [summary[name][k] for k in keys],
            width,
            label=name.replace("_", " "),
            color=colors[i % len(colors)],
        )
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def evaluate_hake(args, split, base, tuned) -> dict:
    device = auto_device(args.device)
    grouped = groups(split["test"], args.group_size)
    if args.smoke:
        grouped = grouped[:1]
    models = {
        "original": load_hake_model(base, device),
        "finetuned": load_hake_model(tuned, device),
    }
    metrics = {key: [] for key in models}
    csv_rows = []
    overlay_dir = args.figure_dir / "hake_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for i, group in enumerate(grouped, 1):
        name = f"{group[0]}+{len(group) - 1}more"
        print(f"  Hake eval [{i}/{len(grouped)}] {name}", flush=True)
        result = prepare_group(group)
        probs = {}
        for tag, model in models.items():
            probs[tag] = predict_full_echogram(
                model,
                result["sv_channels"],
                tuple(args.target_size),
                device,
                patch_width=args.patch_width,
                stride=args.stride,
            )
            row = hake_metrics(name, result, probs[tag])
            metrics[tag].append(row)
            csv_rows.append({"model": tag, **asdict(row)})
        if i <= args.max_overlay_groups:
            plot_overlay(
                name,
                result,
                probs["original"],
                probs["finetuned"],
                overlay_dir / f"{name}_before_after.png",
            )
        del result, probs
        gc.collect()
    summary = {tag: metric_summary(rows) for tag, rows in metrics.items()}
    write_csv(args.figure_dir / "hake_test_metrics.csv", csv_rows)
    (args.figure_dir / "hake_test_summary.json").write_text(json.dumps(summary, indent=2))
    plot_comparison(summary, args.figure_dir / "hake_metric_bars.png", "Hake test metrics")
    return summary


def score_cache_path(args, model_tag: str, domain: str, stem: str) -> Path:
    return args.cache_dir / "scores" / model_tag / domain / f"{stem}.npz"


def save_scores(path, prob, mask, pred_d, ref_d, sigma):
    valid = (
        np.isfinite(pred_d) & np.isfinite(ref_d) & np.isfinite(sigma) & (pred_d > 0) & (ref_d > 0)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        constant=calibrate_depth(pred_d, ref_d).astype(np.float32),
        local=calibrate_locally_weighted(pred_d, ref_d, sigma).astype(np.float32),
        local_sigmas=np.maximum(sigma[valid], 0.5).astype(np.float32),
        profile=calibrate_profile([prob], [mask]).astype(np.float32),
    )


def evaluate_basto(args, base, tuned) -> dict:
    device = auto_device(args.device)
    paths = get_split_files()["test"]
    if args.max_basto_files:
        paths = paths[: args.max_basto_files]
    if args.smoke:
        paths = paths[:2]
    summaries = {}
    for tag, checkpoint in (("original", base), ("finetuned", tuned)):
        model = model_from_checkpoint(checkpoint, device).eval()
        rows = []
        for i, path in enumerate(paths, 1):
            print(f"  Basto {tag} [{i}/{len(paths)}] {path.name}", flush=True)
            metric, prob, mask, sv, ranges = evaluate_file(
                model,
                path,
                38_000,
                200_000,
                tuple(args.target_size),
                device,
                patch_width=args.patch_width,
                stride=args.stride,
                vmin=-80,
                vmax=-34,
                mask_mode="below",
                line_thickness=3,
            )
            rows.append(
                {
                    "file": metric.file_name,
                    "iou": metric.iou,
                    "dice": metric.dice,
                    "precision": metric.precision,
                    "recall": metric.recall,
                    "bottom_error_mean_m": metric.bottom_error_mean_m,
                    "bottom_error_median_m": metric.bottom_error_median_m,
                    "bottom_error_p95_m": metric.bottom_error_p95_m,
                }
            )
            score_path = score_cache_path(args, tag, "basto", path.stem)
            if args.recompute_scores or not score_path.exists():
                pred_d = extract_bottom_depths(prob, ranges, threshold=0.5)
                ref_d = extract_bottom_depths(mask.astype(np.float32), ranges, threshold=0.5)
                save_scores(
                    score_path,
                    prob,
                    mask,
                    pred_d,
                    ref_d,
                    compute_transition_width(prob, ranges),
                )
            del prob, mask, sv
            gc.collect()
        summaries[tag] = {
            "n_files": len(rows),
            "mean_iou": float(np.mean([r["iou"] for r in rows])),
            "mean_dice": float(np.mean([r["dice"] for r in rows])),
            "mean_bottom_error_m": float(
                np.mean(
                    [
                        r["bottom_error_mean_m"]
                        for r in rows
                        if np.isfinite(r["bottom_error_mean_m"])
                    ]
                )
            ),
            "std_iou": float(np.std([r["iou"] for r in rows])),
        }
        write_csv(args.figure_dir / f"basto_test_metrics_{tag}.csv", rows)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    (args.figure_dir / "basto_test_summary.json").write_text(json.dumps(summaries, indent=2))
    plot_comparison(summaries, args.figure_dir / "basto_metric_bars.png", "Basto test metrics")
    return summaries


def score_hake_file(args, model, device, tag, stem) -> Path | None:
    path = score_cache_path(args, tag, "hake", stem)
    if path.exists() and not args.recompute_scores:
        return path
    try:
        result = prepare_group([stem])
        prob = predict_full_echogram(
            model,
            result["sv_channels"],
            tuple(args.target_size),
            device,
            patch_width=args.patch_width,
            stride=args.stride,
        )
        ranges = result["range_crop"]
        pred_d = extract_bottom_depths(prob, ranges, threshold=0.5)
        ref_d = depth_from_indices(result["label_idx_crop"], ranges)
        save_scores(
            path,
            prob,
            result["label_mask"],
            pred_d,
            ref_d,
            compute_transition_width(prob, ranges),
        )
        return path
    except Exception as exc:
        print(f"    skip Hake {stem}: {exc}", flush=True)
        return None


def score_basto_file(args, model, device, tag, zarr) -> Path | None:
    path = score_cache_path(args, tag, "basto", zarr.stem)
    if path.exists() and not args.recompute_scores:
        return path
    try:
        sv, mask, _, ranges = load_file_data(
            zarr,
            38_000,
            200_000,
            vmin=-80,
            vmax=-34,
            mask_mode="below",
            line_thickness=3,
        )
        prob = predict_full_echogram(
            model,
            sv,
            tuple(args.target_size),
            device,
            patch_width=args.patch_width,
            stride=args.stride,
        )
        pred_d = extract_bottom_depths(prob, ranges, threshold=0.5)
        ref_d = extract_bottom_depths(mask.astype(np.float32), ranges, threshold=0.5)
        save_scores(
            path,
            prob,
            mask,
            pred_d,
            ref_d,
            compute_transition_width(prob, ranges),
        )
        return path
    except Exception as exc:
        print(f"    skip Basto {zarr.name}: {exc}", flush=True)
        return None


def load_scores(paths: list[Path | None], names: list[str]) -> dict:
    out = {}
    for path, name in zip(paths, names):
        if path is None or not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            out[name] = {
                key: data[key].astype(np.float64)
                for key in ("constant", "local", "local_sigmas", "profile")
            }
    return out


def score_sets(args, split, model, device, tag) -> dict:
    hake_names = sorted(set(split["calibration"]) | set(split["test"]))
    if args.smoke:
        hake_names = hake_names[:4]
    print(f"  scoring Hake {tag}: {len(hake_names)} files", flush=True)
    hake = load_scores(
        [score_hake_file(args, model, device, tag, s) for s in hake_names],
        hake_names,
    )
    basto_split = get_split_files()
    cal = basto_split["calibration"]
    test = basto_split["test"]
    if args.max_basto_score_files:
        cal = cal[: args.max_basto_score_files]
        test = test[: args.max_basto_score_files]
    if args.smoke:
        cal, test = cal[:2], test[:2]
    all_paths = list(cal) + list(test)
    all_names = [p.stem for p in all_paths]
    print(f"  scoring Basto {tag}: {len(all_paths)} files", flush=True)
    basto = load_scores(
        [score_basto_file(args, model, device, tag, p) for p in all_paths],
        all_names,
    )
    return {
        "hake": {
            "scores": hake,
            "calibration": [s for s in split["calibration"] if s in hake],
            "test": [s for s in split["test"] if s in hake],
        },
        "basto": {
            "scores": basto,
            "calibration": [p.stem for p in cal if p.stem in basto],
            "test": [p.stem for p in test if p.stem in basto],
        },
    }


def concatenated(domain, files, key):
    arrays = [
        domain["scores"][f][key]
        for f in files
        if f in domain["scores"] and domain["scores"][f][key].size
    ]
    return np.concatenate(arrays) if arrays else np.array([])


def coverage(domain, files, method, q_hat):
    covered = total = 0
    file_cov = []
    widths = []
    for name in files:
        if name not in domain["scores"]:
            continue
        scores = domain["scores"][name][method]
        if not scores.size:
            continue
        hit = scores <= q_hat
        covered += int(hit.sum())
        total += len(scores)
        file_cov.append(float(hit.mean()))
        if method == "constant":
            widths.append(np.full(len(scores), 2 * q_hat))
        elif method == "local":
            widths.append(2 * q_hat * domain["scores"][name]["local_sigmas"])
    width = np.concatenate(widths) if widths else np.array([])
    return {
        "coverage": covered / total if total else float("nan"),
        "n": total,
        "n_files": len(file_cov),
        "file_mean": float(np.mean(file_cov)) if file_cov else float("nan"),
        "file_sd": float(np.std(file_cov, ddof=1)) if len(file_cov) > 1 else float("nan"),
        "median_width_m": float(np.median(width)) if width.size else float("nan"),
        "mean_width_m": float(np.mean(width)) if width.size else float("nan"),
    }


def coverage_rows(model_tag, source_name, target_name, source, target):
    rows = []
    for method in METHODS:
        cal = concatenated(source, source["calibration"], method)
        if not cal.size:
            continue
        q_hat = conformal_quantile(cal, ALPHA)
        result = coverage(target, target["test"], method, q_hat)
        rows.append(
            {
                "model": model_tag,
                "calibration_domain": source_name,
                "test_domain": target_name,
                "method": method,
                "q_hat": q_hat,
                "target_coverage": 1 - ALPHA,
                **result,
            }
        )
    return rows


def weighted_quantile(arrays: list[np.ndarray], q: float) -> float:
    sorted_arrays = [
        np.sort(arr[np.isfinite(arr)]) for arr in arrays if arr.size and np.isfinite(arr).any()
    ]
    if not sorted_arrays:
        return float("nan")
    candidates = np.sort(np.concatenate(sorted_arrays))
    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        value = candidates[mid]
        cdf = np.mean(
            [np.searchsorted(arr, value, side="right") / len(arr) for arr in sorted_arrays]
        )
        if cdf >= q:
            hi = mid
        else:
            lo = mid + 1
    return float(candidates[lo])


def balanced_pooled_rows(model_tag, target_name, domains, target):
    rows = []
    for method in METHODS:
        domain_scores = [
            concatenated(domains[name], domains[name]["calibration"], method)
            for name in ("basto", "hake")
        ]
        q_hat = weighted_quantile(domain_scores, 1 - ALPHA)
        if not np.isfinite(q_hat):
            continue
        result = coverage(target, target["test"], method, q_hat)
        rows.append(
            {
                "model": model_tag,
                "calibration_domain": "balanced_pooled",
                "test_domain": target_name,
                "method": method,
                "q_hat": q_hat,
                "target_coverage": 1 - ALPHA,
                **result,
            }
        )
    return rows


def pooled_calibration_domain(domains: dict) -> dict:
    scores = {}
    calibration = []
    for domain_name in ("basto", "hake"):
        prefix = f"{domain_name}:"
        for name in domains[domain_name]["calibration"]:
            if name in domains[domain_name]["scores"]:
                pooled_name = prefix + name
                scores[pooled_name] = domains[domain_name]["scores"][name]
                calibration.append(pooled_name)
    return {"scores": scores, "calibration": calibration, "test": []}


def plot_coverage(rows, path):
    labels = [
        f"{r['model']}\n{r['calibration_domain']}->{r['test_domain']}\n{r['method']}" for r in rows
    ]
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(rows)), 5.2))
    colors = ["#4c78a8" if r["model"] == "original" else "#f58518" for r in rows]
    ax.bar(np.arange(len(rows)), [r["coverage"] for r in rows], color=colors)
    ax.axhline(0.9, color="black", linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Empirical coverage")
    ax.set_xticks(np.arange(len(rows)), labels, rotation=60, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def conformal(args, split, base, tuned) -> list[dict]:
    device = auto_device(args.device)
    sets = {}
    for tag, checkpoint in (("original", base), ("finetuned", tuned)):
        model = model_from_checkpoint(checkpoint, device).eval()
        sets[tag] = score_sets(args, split, model, device, tag)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    rows = []
    for tag, domains in sets.items():
        pooled = pooled_calibration_domain(domains)
        for source, target in (
            ("basto", "hake"),
            ("hake", "hake"),
            ("basto", "basto"),
            ("hake", "basto"),
        ):
            rows += coverage_rows(tag, source, target, domains[source], domains[target])
        for target in ("hake", "basto"):
            rows += coverage_rows(tag, "pooled", target, pooled, domains[target])
            rows += balanced_pooled_rows(tag, target, domains, domains[target])
    write_csv(args.figure_dir / "conformal_results.csv", rows)
    plot_coverage(rows, args.figure_dir / "conformal_coverage_bars.png")
    return rows


def write_readme(args, hake, basto, rows):
    constant = [r for r in rows if r["method"] == "constant"]
    lines = [
        "# Hake fine-tune experiment",
        "",
        f"Created: {dt.datetime.now():%Y-%m-%d %H:%M}",
        f"Base checkpoint: `{args.base_checkpoint}`",
        f"Fine-tuned checkpoint: `{args.checkpoint_dir / 'smp_unet_resnet34_best.pth'}`",
        "",
        "## Hake test",
        f"- Original: IoU {hake['original']['mean_iou']:.3f}, "
        f"depth MAE {hake['original']['mean_depth_mae_m']:.3f} m.",
        f"- Fine-tuned: IoU {hake['finetuned']['mean_iou']:.3f}, "
        f"depth MAE {hake['finetuned']['mean_depth_mae_m']:.3f} m.",
        "",
        "## Basto test",
        f"- Original: IoU {basto['original']['mean_iou']:.3f}, "
        f"depth MAE {basto['original']['mean_bottom_error_m']:.3f} m.",
        f"- Fine-tuned: IoU {basto['finetuned']['mean_iou']:.3f}, "
        f"depth MAE {basto['finetuned']['mean_bottom_error_m']:.3f} m.",
        "",
        "## Constant-width conformal coverage",
    ]
    for row in constant:
        lines.append(
            f"- {row['model']} {row['calibration_domain']}->{row['test_domain']}: "
            f"q={row['q_hat']:.3f}, coverage={row['coverage']:.3f}, n={row['n']}."
        )
    (args.figure_dir / "_README.md").write_text("\n".join(lines) + "\n")


def prepare(args):
    args.base_checkpoint = Path(args.base_checkpoint)
    args.checkpoint_dir = Path(args.checkpoint_dir)
    args.figure_dir = Path(args.figure_dir)
    args.cache_dir = Path(args.cache_dir)
    if args.smoke:
        args.checkpoint_dir = Path(str(args.checkpoint_dir) + "_smoke")
        args.figure_dir = Path(str(args.figure_dir) + "_smoke")
        args.cache_dir = Path(str(args.cache_dir) + "_smoke")
        args.max_hake_files = args.max_hake_files or 24
        args.overwrite = True
    if not args.base_checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint: {args.base_checkpoint}")
    if args.checkpoint_dir.resolve() == args.base_checkpoint.parent.resolve():
        raise SystemExit("Refusing to write into source checkpoint directory")
    if args.overwrite:
        shutil.rmtree(args.checkpoint_dir, ignore_errors=True)
        shutil.rmtree(args.figure_dir, ignore_errors=True)
    elif args.checkpoint_dir.exists() and not args.resume_existing:
        raise SystemExit(
            f"Output exists: {args.checkpoint_dir}; use --resume-existing or --overwrite"
        )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.checkpoint_dir / "hake_split_manifest.json"
    if args.resume_existing and manifest.exists():
        split = json.loads(manifest.read_text())["splits"]
    else:
        split = make_split(valid_hake_stems(args.max_hake_files), args.group_size, args.seed)
        write_manifest(args, split)
    print("Hake split counts:", {k: len(v) for k, v in split.items()}, flush=True)
    return split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "train", "eval", "conformal"), default="full")
    parser.add_argument("--init-mode", choices=("checkpoint", "scratch"), default="checkpoint")
    parser.add_argument("--base-checkpoint", type=Path, default=BASE_CKPT)
    parser.add_argument("--checkpoint-dir", type=Path, default=CKPT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--target-size", type=int, nargs=2, default=list(TARGET_SIZE))
    parser.add_argument("--patch-width", type=int, default=PATCH_WIDTH)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-stages", type=int, default=5)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-hake-files", type=int, default=0)
    parser.add_argument("--max-basto-files", type=int, default=0)
    parser.add_argument("--max-basto-score-files", type=int, default=0)
    parser.add_argument("--max-overlay-groups", type=int, default=6)
    parser.add_argument("--recompute-cache", action="store_true")
    parser.add_argument("--recompute-scores", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Device: {auto_device(args.device)}", flush=True)
    split = prepare(args)
    tuned = args.checkpoint_dir / "smp_unet_resnet34_best.pth"
    if args.mode in ("full", "train"):
        tuned = train(args, split)
    elif not tuned.is_file():
        raise SystemExit(f"Fine-tuned checkpoint missing: {tuned}")
    hake = basto = {}
    rows = []
    if args.mode in ("full", "eval"):
        hake = evaluate_hake(args, split, args.base_checkpoint, tuned)
        basto = evaluate_basto(args, args.base_checkpoint, tuned)
    if args.mode in ("full", "conformal"):
        rows = conformal(args, split, args.base_checkpoint, tuned)
    if args.mode == "full":
        write_readme(args, hake, basto, rows)
    print(f"Done -> {args.figure_dir}", flush=True)


if __name__ == "__main__":
    main()
