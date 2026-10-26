"""End-to-end U-Net training for seabed segmentation.

Usage
-----
    # From a directory of zarr files:
    python code/deep_learning/train.py \
        --zarr-dir /path/to/zarr_directory \
        --epochs 30

    # From raw files (auto-converts to zarr first):
    python code/deep_learning/train.py \
        --raw-dirs /path/to/raw_folder1 /path/to/raw_folder2 \
        --zarr-cache-dir /path/to/zarr_cache \
        --epochs 30

    # Mixed: some zarr files + raw dirs:
    python code/deep_learning/train.py \
        --zarr-paths /path/to/existing.zarr \
        --raw-dirs /path/to/raw_folder \
        --epochs 30

Pipeline
--------
1. Convert .raw → zarr (if raw-dirs provided, skip existing)
2. Extract 3-channel patches (freq1, freq2, diff) with caching
3. Split into train / val at file level
4. Train U-Net with transfer learning
5. Evaluate on validation set
6. Save checkpoint + metrics
"""

from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure the project root is importable
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from deep_learning.training.config import PipelineConfig
from deep_learning.echo_dataset import build_datasets_from_config, _clean_owned_temp
from deep_learning.unet_model import (
    get_trainable_param_count,
    build_loss,
    train_one_epoch,
    validate,
)
from deep_learning.training.cli import parse_args, build_config
from deep_learning.training.model_setup import build_model
from deep_learning.training.evaluation import run_evaluation
from deep_learning.training.checkpointing import load_resume_state
from deep_learning.training.diagnostics import generate_diagnostics
from deep_learning.training.results_log import append_run, build_row


# Whether each monitored metric should be maximised or minimised.
_MONITOR_MODE = {'iou': 'max', 'dice': 'max', 'loss': 'min', 'bottom_mae': 'min'}


def _atomic_save_torch(obj, path: Path) -> None:
    """torch.save via a temp file + rename so a kill mid-write never corrupts
    the target — Slurm SIGTERM can land during the multi-second checkpoint
    write, and these files are now rewritten every epoch."""
    tmp = path.parent / (path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _atomic_save_json(obj, path: Path) -> None:
    """json.dump via a temp file + rename (see _atomic_save_torch)."""
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def _build_scheduler(cfg, optimizer):
    """Construct the LR scheduler for ``optimizer`` per ``cfg.scheduler_type``."""
    if cfg.scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.cosine_t0, T_mult=cfg.cosine_t_mult,
            eta_min=cfg.cosine_eta_min)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.lr_factor, patience=cfg.lr_patience)


def train(cfg: PipelineConfig):
    """Run the full training pipeline."""

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device
    print(f"Device: {device}")

    # Throughput knobs. Model input is a fixed target_size every forward pass
    # (training patches and the resize in predict_full_echogram), so cuDNN
    # autotune pays off and never re-tunes. TF32 speeds fp32 matmuls; under AMP
    # most compute is already fp16, so this only helps the fp32 remainder.
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    # ── 1. Data ───────────────────────────────────────────────────────
    print("\n=== Data Preparation ===")
    train_ds, val_ds, zarr_paths = build_datasets_from_config(cfg)
    print(f"Training patches:   {len(train_ds)}")
    print(f"Validation patches: {len(val_ds)}")

    if len(train_ds) == 0:
        print("ERROR: No training patches extracted. Check your data paths.")
        sys.exit(1)

    # Dedicated generator for the shuffle so the per-epoch ordering is a pure
    # function of (seed, epoch), independent of how many epochs already ran.
    # Without this, a resumed process re-seeds the global RNG and replays the
    # *epoch-0* permutation instead of continuing — feeding the optimizer a
    # batch sequence it never trained on, which is what kicked off the arch_6
    # resume divergence.
    train_gen = torch.Generator()
    train_gen.manual_seed(cfg.seed)
    # persistent_workers keeps the worker pool (and its open mmap handles) alive
    # across epochs instead of re-spawning; prefetch_factor lets each worker
    # stage batches ahead of the GPU. Both require num_workers > 0.
    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=(device not in ('cpu', 'mps')),
    )
    if cfg.num_workers > 0:
        loader_kwargs['persistent_workers'] = getattr(
            cfg, 'persistent_workers', True)
        loader_kwargs['prefetch_factor'] = getattr(cfg, 'prefetch_factor', 4)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        generator=train_gen, **loader_kwargs)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)

    # ── 2. Model ──────────────────────────────────────────────────────
    print("\n=== Model Setup ===")
    model = build_model(cfg)
    print(f"Architecture: {model.architecture_id}")

    if cfg.freeze_strategy == 'encoder':
        model.freeze_encoder()
        print("Froze entire encoder")
    elif cfg.freeze_strategy == 'stages':
        model.freeze_encoder_stages(cfg.freeze_stages)
        print(f"Froze first {cfg.freeze_stages} encoder stages")
    else:
        print("No freezing — full fine-tuning")

    counts = get_trainable_param_count(model)
    print(f"Trainable: {counts['trainable']:,} / {counts['total']:,} "
          f"({counts['trainable_pct']:.1f}%)")

    model = model.to(device)

    # ── 3. Optimizer + Loss ───────────────────────────────────────────
    param_groups = [
        {'params': [p for p in model.get_encoder_params() if p.requires_grad],
         'lr': cfg.encoder_lr},
        {'params': [p for p in model.get_decoder_params() if p.requires_grad],
         'lr': cfg.decoder_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    # Scheduler is built later (after the resume block) so it wraps the final
    # optimizer — the resume path may rebuild the optimizer for an unfrozen
    # encoder, and a scheduler bound to the discarded optimizer would silently
    # stop adjusting the live LR.
    loss_fn = build_loss(cfg.loss_type, bce_weight=cfg.bce_weight,
                         focal_alpha=cfg.focal_alpha, focal_gamma=cfg.focal_gamma,
                         tversky_alpha=cfg.tversky_alpha, tversky_beta=cfg.tversky_beta,
                         geo_reg_weight=cfg.geo_reg_weight,
                         geo_smooth_weight=cfg.geo_smooth_weight,
                         geo_depth_scale=cfg.geo_depth_scale,
                         geo_huber_delta=cfg.geo_huber_delta,
                         mask_mode=cfg.mask_mode)
    print(f"Loss: {cfg.loss_type} ({loss_fn})")
    scaler = torch.amp.GradScaler() if cfg.use_amp else None

    # ── 4. Training loop ──────────────────────────────────────────────
    print("\n=== Training ===")
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.checkpoint_dir / f"{model.architecture_id}_best.pth"
    last_path = cfg.checkpoint_dir / f"{model.architecture_id}_last.pth"
    history_path = cfg.checkpoint_dir / "training_history.json"

    history = {'train_loss': [], 'val_loss': [],
               'train_iou': [], 'val_iou': [],
               'train_dice': [], 'val_dice': [],
               'train_bottom_mae': [], 'val_bottom_mae': []}

    monitor = cfg.monitor_metric
    monitor_mode = _MONITOR_MODE[monitor]
    best_score = -float('inf') if monitor_mode == 'max' else float('inf')
    best_val_iou = 0.0   # tracked alongside for the results log
    best_epoch = -1
    epochs_without_improvement = 0
    start_epoch = 0
    print(f"Monitor: {monitor} ({'maximise' if monitor_mode == 'max' else 'minimise'})")

    # ── Resume from checkpoint ────────────────────────────────────────
    resume_scheduler_state = None
    if cfg.resume is not None:
        (optimizer, best_val_iou, best_epoch, start_epoch,
         resume_scheduler_state) = load_resume_state(
            cfg, model, optimizer, scaler, device, history)
        # Recover the best monitored score (and the epoch/IoU it came from) from
        # the restored history. This matters now that resume restarts from
        # _last.pth: the loaded checkpoint reflects the *last* epoch, not the
        # best one, so best_epoch/best_val_iou must be derived from the curve.
        # Older checkpoints predate val_bottom_mae, so fall back gracefully.
        hist_vals = history.get(f"val_{monitor}", [])
        if hist_vals:
            best_score = max(hist_vals) if monitor_mode == 'max' else min(hist_vals)
            best_epoch = hist_vals.index(best_score)
            val_iou_hist = history.get('val_iou', [])
            if val_iou_hist:
                best_val_iou = val_iou_hist[best_epoch]
            # Restore the early-stopping counter from the curve. Without this it
            # resets to 0 on every resume, so with frequent 3.5h chunk restarts
            # the epochs-since-best count never accumulates and early stopping
            # can never fire — a converged run wastes compute up to num_epochs.
            epochs_without_improvement = len(hist_vals) - 1 - best_epoch
        elif monitor == 'iou':
            best_score = best_val_iou

    if cfg.eval_only:
        if cfg.resume is None:
            raise SystemExit("ERROR: --eval-only requires --resume.")
        print("\n=== Eval-only: skipping training loop ===")
        return (model, zarr_paths, history, checkpoint_path,
                best_val_iou, best_epoch, 0)

    # Build the scheduler now that the optimizer is final. On resume, restore the
    # scheduler's dynamic state (cosine cycle position / plateau patience counter)
    # so chunked 3.5h resumes behave like one continuous run instead of restarting
    # the schedule each chunk — most visible for cosine, whose warm-restart cycle
    # otherwise jumps the LR back toward base at every chunk boundary. The CLI
    # hyperparameters are re-applied after loading so a changed --lr-patience /
    # --lr-factor / cosine period still takes effect (only the counters are
    # restored). --reset-lr-on-resume signals an intentional LR-regime change, so
    # the schedule is left fresh in that case.
    scheduler = _build_scheduler(cfg, optimizer)
    if (resume_scheduler_state is not None and not cfg.reset_lr_on_resume):
        scheduler.load_state_dict(resume_scheduler_state)
        if cfg.scheduler_type == "cosine":
            scheduler.T_0 = cfg.cosine_t0
            scheduler.T_mult = cfg.cosine_t_mult
            scheduler.eta_min = cfg.cosine_eta_min
        else:
            scheduler.patience = cfg.lr_patience
            scheduler.factor = cfg.lr_factor
        print("  restored scheduler state from checkpoint "
              "(schedule continues across resume)")
    if cfg.scheduler_type == "cosine":
        print(f"Scheduler: CosineAnnealingWarmRestarts "
              f"(T_0={cfg.cosine_t0}, T_mult={cfg.cosine_t_mult}, "
              f"eta_min={cfg.cosine_eta_min})")
    else:
        print(f"Scheduler: ReduceLROnPlateau "
              f"(factor={cfg.lr_factor}, patience={cfg.lr_patience})")

    for epoch in range(start_epoch, cfg.num_epochs):
        if cfg.unfreeze_after is not None and epoch == cfg.unfreeze_after:
            model.unfreeze_encoder()
            counts = get_trainable_param_count(model)
            print(f"\n--- Unfreezing encoder at epoch {epoch+1} ---")
            print(f"Trainable: {counts['trainable']:,} / {counts['total']:,} "
                  f"({counts['trainable_pct']:.1f}%)")

            # Rebuild optimizer so encoder params are included
            param_groups = [
                {'params': [p for p in model.get_encoder_params() if p.requires_grad],
                 'lr': cfg.encoder_lr},
                {'params': [p for p in model.get_decoder_params() if p.requires_grad],
                 'lr': cfg.decoder_lr},
            ]
            optimizer = torch.optim.AdamW(
                param_groups, weight_decay=cfg.weight_decay)

            # Fresh schedule for the new (unfrozen) optimizer — intentional, the
            # encoder LR regime restarts at unfreeze.
            scheduler = _build_scheduler(cfg, optimizer)

            if cfg.use_amp:
                scaler = torch.amp.GradScaler()

            # Reset early stopping counter after unfreeze
            epochs_without_improvement = 0

        # Deterministic, resume-stable shuffle for this epoch.
        train_gen.manual_seed(cfg.seed + epoch)
        train_m = train_one_epoch(model, train_loader, optimizer,
                                  loss_fn, device, scaler,
                                  mask_mode=cfg.mask_mode,
                                  grad_clip_norm=cfg.grad_clip_norm,
                                  task=cfg.task)
        if train_m.get('n_nonfinite', 0):
            print(f"  WARN: skipped {train_m['n_nonfinite']} batch(es) with "
                  f"non-finite loss this epoch (clipping/guard engaged)")
        val_m = validate(model, val_loader, loss_fn, device,
                         mask_mode=cfg.mask_mode, task=cfg.task)
        if cfg.scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_m['loss'])

        history['train_loss'].append(train_m['loss'])
        history['val_loss'].append(val_m['loss'])
        history['train_iou'].append(train_m['iou'])
        history['val_iou'].append(val_m['iou'])
        history['train_dice'].append(train_m['dice'])
        history['val_dice'].append(val_m['dice'])
        history['train_bottom_mae'].append(train_m['bottom_mae'])
        history['val_bottom_mae'].append(val_m['bottom_mae'])

        score = val_m[monitor]
        improved = (score > best_score) if monitor_mode == 'max' else (score < best_score)
        if improved:
            best_score = score
            best_val_iou = val_m['iou']
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'val_iou': val_m['iou'],
            'monitor_metric': monitor,
            'monitor_score': best_score,
            'config': {
                'architecture': cfg.architecture,
                'encoder_name': cfg.encoder_name,
                'encoder_weights': cfg.encoder_weights,
                'base_filters': cfg.base_filters,
                'model_depth': cfg.model_depth,
                'in_channels': cfg.in_channels,
                'target_size': cfg.target_size,
                'freeze_strategy': cfg.freeze_strategy,
                'freq1_hz': cfg.freq1_hz,
                'freq2_hz': cfg.freq2_hz,
                'vmin': cfg.vmin,
                'vmax': cfg.vmax,
            }
        }
        # Persist the curve and a per-epoch _last.pth so a chunk killed by the
        # wall clock resumes from exactly where it stopped. Resuming from
        # _best.pth instead would rewind to the best-val epoch and re-train the
        # gap — the train-metric dip seen at every chunk boundary.
        #
        # Write history *before* _last.pth: if a SIGTERM lands between the two,
        # history may be one epoch ahead of _last (harmless — the resume
        # truncates the curve to _last's epoch) but never behind, which would
        # leave an unrecoverable gap. Both writes are atomic.
        _atomic_save_json(history, history_path)
        _atomic_save_torch(ckpt, last_path)
        if improved:
            _atomic_save_torch(ckpt, checkpoint_path)

        print(f"Epoch {epoch+1:3d}/{cfg.num_epochs}  "
              f"train_loss={train_m['loss']:.4f}  "
              f"val_loss={val_m['loss']:.4f}  "
              f"val_IoU={val_m['iou']:.4f}  "
              f"val_Dice={val_m['dice']:.4f}  "
              f"val_botMAE={val_m['bottom_mae']:.3f}"
              f"{'  *' if improved else ''}")

        if (cfg.early_stopping_patience is not None
                and epochs_without_improvement >= cfg.early_stopping_patience):
            print(f"\nEarly stopping: no improvement for "
                  f"{cfg.early_stopping_patience} epochs.")
            break

    print(f"\nBest {monitor}: {best_score:.4f} (val IoU {best_val_iou:.4f}) "
          f"at epoch {best_epoch + 1}")
    print(f"Checkpoint: {checkpoint_path}")

    # Final history write (also written every epoch in the loop above).
    _atomic_save_json(history, history_path)
    print(f"History: {history_path}")

    n_epochs_completed = len(history['val_loss'])
    return (model, zarr_paths, history, checkpoint_path,
            best_val_iou, best_epoch, n_epochs_completed)


def main():
    args = parse_args()
    cfg = build_config(args)

    # Slurm sends SIGTERM with a grace period before SIGKILL. Clean up only the
    # temp dirs / lock files this process created; never touch arbitrary paths
    # in the shared cache dir (would clobber other live jobs' in-progress work).
    def _on_sigterm(signum, frame):
        print(f"\nReceived signal {signum} — cleaning owned cache temp files...",
              flush=True)
        try:
            _clean_owned_temp()
        finally:
            sys.exit(143)
    signal.signal(signal.SIGTERM, _on_sigterm)

    t0 = time.perf_counter()
    (model, zarr_paths, history, ckpt,
     best_val_iou, best_epoch, n_epochs_completed) = train(cfg)

    eval_summary = None
    if not args.no_eval:
        # In eval-only mode the model is already loaded from --resume; reloading
        # would look for a best.pth inside --checkpoint-dir (which may be a fresh
        # output dir holding no checkpoint). Only reload after a real training run.
        if not cfg.eval_only:
            ckpt_data = torch.load(ckpt, map_location=cfg.device, weights_only=False)
            model.load_state_dict(ckpt_data['model_state_dict'])
        eval_summary = run_evaluation(model, zarr_paths, cfg)

        if not args.no_diagnostics:
            try:
                generate_diagnostics(model, zarr_paths, cfg, history)
            except Exception as e:
                print(f"WARN: diagnostics failed: {e}")

    wall_time_sec = time.perf_counter() - t0

    # Append to shared results log (never fails training)
    log_path = args.results_log.strip() if args.results_log else ""
    if log_path and not args.eval_only:
        try:
            row = build_row(
                cfg=cfg,
                best_val_iou=best_val_iou,
                best_epoch=best_epoch + 1,  # 1-indexed for readability
                n_epochs_completed=n_epochs_completed,
                eval_summary=eval_summary,
                wall_time_sec=wall_time_sec,
                checkpoint_dir=cfg.checkpoint_dir,
            )
            append_run(Path(log_path), row)
            print(f"Logged run to {log_path}")
        except Exception as e:
            print(f"WARN: failed to log run: {e}")


if __name__ == "__main__":
    main()
