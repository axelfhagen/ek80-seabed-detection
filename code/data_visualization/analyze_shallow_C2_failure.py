#!/usr/bin/env python
"""Failure analysis of the shallow_C2 dock EK80 files.

Output: Figures/shallow_C2_failure/

The three files in "/cluster/work/axelfh/EK80 38 200 kHz ved kai/" are broadband
(LFM) EK80 recordings taken at the quay. Water depth is only ~10-19 m, far
shallower than the Basto crossings the U-Net was trained on, and the recording
is dominated by near-field transducer ringing, dock structure and a very short
water column. The model fails heavily here. This script documents *why* with:

  - per-channel input panels (38 kHz, 200 kHz, frequency-difference) in greyscale,
    exactly what the network sees, plus the raw Sv for context;
  - the predicted P(bottom) probability map and the binary prediction vs the
    algorithmic 38 kHz reference label;
  - conformal depth bands (constant, profile, locally-weighted) calibrated on the
    in-distribution Basto set, to show whether the uncertainty machinery flags
    the failure;
  - Grad-CAM over the worst-failing window, with an input-channel gradient
    ablation, to show where the decoder looks.

These files are not standard EK80: they mix ADCP, EC150 and ES transceivers, and
echopype 0.11.0 crashes building the Vendor_specific group. A small patch to
SetGroupsEK80.set_vendor (pad ragged narrowband tables, coerce numeric fields)
lets broadband Sv be computed for the ES38/ES200 channels.

Usage:
    python code/data_visualization/analyze_shallow_C2_failure.py
"""
from __future__ import annotations

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
import torch
import cv2

import argparse
import xarray as xr

from raw_sensor_integration.preprocessing import (
    normalize_echogram, load_sv_from_zarr)
from raw_sensor_integration.bottom_labelling import (
    detect_bottom_ensemble_pipeline, bottom_to_sample_index, create_bottom_mask)
from raw_sensor_integration.data_loader import load_zarr
from deep_learning.unet_model import create_unet, predict_full_echogram
from deep_learning.training.survey_groups import get_split_files
from validation.conformal import (
    conformal_quantile, predict_depth_band,
    predict_locally_weighted_band, compute_transition_width, extract_bottom_depths,
    calibrate_depth, calibrate_locally_weighted)
from data_visualization.thesis_style import apply_thesis_style, savefig, style_colorbar

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

# ── Constants ──────────────────────────────────────────────────────────────
RAW_DIR = Path("/cluster/work/axelfh/EK80 38 200 kHz ved kai")
STEMS = ["shallow_C2-D20260526-T134234",
         "shallow_C2-D20260526-T134826",
         "shallow_C2-D20260526-T135419"]
FREQ1, FREQ2 = 38_000, 200_000
VMIN, VMAX = -80.0, -34.0
BT_THRESHOLD = -31.0          # calibrated bottom-detection threshold (dB)
MIN_RANGE = 2.0               # shallow water: allow detections from 2 m down
TARGET_SIZE = (512, 512)
PATCH_WIDTH, STRIDE = 512, 256
ALPHA = 0.10

# Default checkpoint for the project (set 2026-05-31). Overridable via --checkpoint.
DEFAULT_CKPT_NAME = "chunk_arch_sweep_2"
MAX_CALIB_PINGS = 4000        # cap ping span per calibration file (bound memory)
N_CALIB = 20                  # in-distribution files used to calibrate q_hat

# These are (re)assigned in main() from CLI args so every output is keyed to the
# checkpoint and nothing from a previous run is overwritten.
OUT_DIR = PROJECT_ROOT / "Figures" / f"shallow_C2_failure_{DEFAULT_CKPT_NAME}"
CACHE_DIR = OUT_DIR / "cache"
# Broadband-Sv cache is checkpoint-independent (echopype compute_Sv), so it is
# shared across runs rather than living under a per-checkpoint output dir.
SV_CACHE_DIR = PROJECT_ROOT / "Figures" / "shallow_C2_failure" / "cache"
CHECKPOINT = (PROJECT_ROOT / "code/checkpoints" / DEFAULT_CKPT_NAME /
              "smp_unet_resnet34_best.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REF_COLOR = "#ff3b30"
PRED_COLOR = "#00e5ff"


# ── echopype vendor-group patch (mixed-transceiver EK80) ───────────────────
def patch_echopype_vendor():
    """Make echopype tolerate the mixed ADCP/EC150/ES channel set in these files.

    The stock set_vendor() builds the narrowband calibration table with a single
    np.array() over all channels; here the per-channel pulse_length tables have
    different lengths (and one channel lacks them entirely), which raises. We pad
    ragged rows with NaN, NaN-fill missing channels, and coerce numeric fields,
    then rebuild the broadband calibration and filter-coefficient sections as
    upstream does. Only the ES38/ES200 rows are used downstream, so NaN padding
    on the ADCP rows is harmless.
    """
    import xarray as xr
    from echopype.convert.set_groups_ek80 import (
        SetGroupsEK80, WIDE_BAND_TRANS, PULSE_COMPRESS,
        FILTER_IMAG, FILTER_REAL, DECIMATION)

    def _f(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    def set_vendor(self):
        config = self.parser_obj.config_datagram["configuration"]
        chans = self.sorted_channel["all"]
        freq = np.array([_f(config[ch]["transducer_frequency"]) for ch in chans])

        def table(p):
            rows = [np.atleast_1d(np.asarray(
                [_f(x) for x in np.atleast_1d(config[ch].get(p, [np.nan]))],
                dtype=float)) for ch in chans]
            m = max(r.shape[0] for r in rows)
            return np.array([np.pad(r, (0, m - r.shape[0]),
                                    constant_values=np.nan) for r in rows])

        sa, gn, pl = table("sa_correction"), table("gain"), table("pulse_duration")
        nb = min(sa.shape[1], gn.shape[1], pl.shape[1])
        sa, gn, pl = sa[:, :nb], gn[:, :nb], pl[:, :nb]
        ds_table = xr.Dataset(
            {"frequency_nominal": (["channel"], freq),
             "sa_correction": (["channel", "pulse_length_bin"], sa),
             "gain_correction": (["channel", "pulse_length_bin"], gn),
             "pulse_length": (["channel", "pulse_length_bin"], pl)},
            coords={"channel": (["channel"], chans),
                    "pulse_length_bin": (["pulse_length_bin"], np.arange(nb))})
        if all("impedance" in config[ch] for ch in chans):
            ds_table["impedance_transceiver"] = xr.DataArray(
                np.array([_f(config[ch]["impedance"]) for ch in chans]),
                dims=["channel"], coords={"channel": ds_table["channel"]})
        if all("rx_sample_frequency" in config[ch] for ch in chans):
            ds_table["receiver_sampling_frequency"] = xr.DataArray(
                np.array([_f(config[ch]["rx_sample_frequency"]) for ch in chans]),
                dims=["channel"], coords={"channel": ds_table["channel"]})
        if all("transceiver_type" in config[ch] for ch in chans):
            ds_table["transceiver_type"] = xr.DataArray(
                np.array([str(config[ch]["transceiver_type"]) for ch in chans]),
                dims=["channel"], coords={"channel": ds_table["channel"]})

        cal_ch_ids = [ch for ch in chans if "calibration" in config[ch]]
        ds_cal = []
        for ch_id in cal_ch_ids:
            cp = ["gain", "impedance", "phase", "beamwidth_alongship",
                  "beamwidth_athwartship", "angle_offset_alongship",
                  "angle_offset_athwartship"]
            pd2 = {p: (["cal_frequency"], config[ch_id]["calibration"][p])
                   for p in cp if p in config[ch_id]["calibration"]}
            ds_cal.append(xr.Dataset(
                data_vars=pd2,
                coords={"cal_frequency": (["cal_frequency"],
                                          config[ch_id]["calibration"]["frequency"])}
            ).expand_dims({"cal_channel_id": [ch_id]}))
        ds_cal = xr.merge(ds_cal, join="outer")
        if "impedance" in ds_cal:
            ds_cal = ds_cal.rename_vars({"impedance": "impedance_transducer"})

        pmap = {1: WIDE_BAND_TRANS, 2: PULSE_COMPRESS}
        cad = {t: {FILTER_IMAG: [], FILTER_REAL: [], DECIMATION: []}
               for t in pmap.values()}
        for ch in chans:
            fc = self.parser_obj.fil_coeffs.get(ch, None)
            fd = self.parser_obj.fil_df.get(ch, None)
            for tn in pmap:
                param = pmap[tn]
                cad[param][FILTER_IMAG].append(np.imag(fc[tn]) if fc else [np.nan])
                cad[param][FILTER_REAL].append(np.real(fc[tn]) if fc else [np.nan])
                cad[param][DECIMATION].append(fd[tn] if fd else np.nan)
        ds = xr.merge([ds_table, ds_cal]).pipe(self._add_filter_params, cad)
        ds["config_xml"] = self.parser_obj.config_datagram["xml"]
        return ds

    SetGroupsEK80.set_vendor = set_vendor


# ── Sv loading (broadband) ─────────────────────────────────────────────────
def load_sv_bb(raw_path: Path):
    """Compute broadband Sv for the ES38 and ES200 channels.

    Returns sv1, range1, sv2, range2, pulse_duration, heave — each echogram
    oriented (range, ping). Cached to npz on first run.
    """
    cache = SV_CACHE_DIR / f"{raw_path.stem}_sv.npz"
    if cache.exists():
        d = np.load(cache)
        return (d["sv1"], d["range1"], d["sv2"], d["range2"],
                float(d["pulse"]), d["heave"])

    import echopype as ep
    ed = ep.open_raw(str(raw_path), sonar_model="EK80")
    ds = ep.calibrate.compute_Sv(ed, waveform_mode="BB", encode_mode="complex")

    def pick(substr):
        for i, c in enumerate(ds.channel.values):
            if substr in str(c):
                return i
        raise RuntimeError(f"channel {substr} not found")

    i1, i2 = pick("ES38"), pick("ES200")

    def extract(i):
        sv = ds.Sv.isel(channel=i).values            # (ping, range_sample)
        er = ds["echo_range"].isel(channel=i).values
        rng = np.nanmax(er, axis=0)                   # range per sample (m)
        valid = np.isfinite(rng)
        sv = sv[:, valid].T                           # (range, ping)
        rng = rng[valid]
        return sv.astype(np.float32), rng.astype(np.float32)

    sv1, range1 = extract(i1)
    sv2, range2 = extract(i2)

    # heave (platform vertical motion), aligned to ping axis of ch1
    heave = np.zeros(sv1.shape[1], dtype=np.float32)
    try:
        plat = ed["Platform"]
        if "vertical_offset" in plat:
            h = np.asarray(plat["vertical_offset"].values).ravel()
            if h.size >= sv1.shape[1]:
                heave = h[:sv1.shape[1]].astype(np.float32)
    except Exception:
        pass

    # nominal pulse duration of the ES38 channel
    pulse = 0.000512
    try:
        bp = ed["Sonar/Beam_group1"]
        td = bp["transmit_duration_nominal"].values
        ch_ids = [str(c) for c in bp["channel"].values]
        k = next(j for j, c in enumerate(ch_ids) if "ES38" in c)
        pulse = float(np.nanmedian(td[k]))
    except Exception:
        pass

    SV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, sv1=sv1, range1=range1, sv2=sv2, range2=range2,
             pulse=pulse, heave=heave)
    del ed, ds
    return sv1, range1, sv2, range2, pulse, heave


def resample_to_grid(sv2, range2, range1):
    out = np.full((len(range1), sv2.shape[1]), np.nan, dtype=np.float32)
    inr = range1 <= range2[-1]
    for p in range(sv2.shape[1]):
        out[inr, p] = np.interp(range1[inr], range2, sv2[:, p])
    return out


# ── Model ──────────────────────────────────────────────────────────────────
def load_model():
    model = create_unet(encoder_name="resnet34", encoder_weights=None,
                        in_channels=3, classes=1)
    sd = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state = sd.get("model_state_dict", sd)
    model.load_state_dict({k.removeprefix("model."): v
                           for k, v in state.items()}, strict=False)
    return model.to(DEVICE).eval()


# ── Per-file processing ─────────────────────────────────────────────────────
def process(stem, model):
    raw = RAW_DIR / (stem + ".raw")
    sv1, range1, sv2, range2, pulse, heave = load_sv_bb(raw)

    n = min(sv1.shape[0], sv2.shape[0] if sv2.shape[0] else sv1.shape[0])
    # align ping counts
    npings = min(sv1.shape[1], sv2.shape[1])
    sv1 = sv1[:, :npings]
    sv2r = resample_to_grid(sv2, range2, range1)[:, :npings]
    heave = heave[:npings]

    norm1 = normalize_echogram(sv1, method="minmax", vmin=VMIN, vmax=VMAX,
                               fill_nan=0.0).astype(np.float32)
    norm2 = normalize_echogram(sv2r, method="minmax", vmin=VMIN, vmax=VMAX,
                               fill_nan=0.0).astype(np.float32)
    diff = (norm1 - norm2).astype(np.float32)
    sv_channels = np.stack([norm1, norm2, diff], axis=0)

    with torch.no_grad():
        prob = predict_full_echogram(model, sv_channels, TARGET_SIZE, DEVICE,
                                     patch_width=PATCH_WIDTH, stride=STRIDE)

    # algorithmic 38 kHz reference label (no human GT for these files)
    bottom_range, _, _ = detect_bottom_ensemble_pipeline(
        sv1, range1, heave=heave, pulse_duration=pulse, frequency_hz=FREQ1,
        transducer_draft=0.0, minimum_range=MIN_RANGE,
        threshold_log_sv=BT_THRESHOLD, detectors=("simple", "edge"))
    ref_idx = bottom_to_sample_index(bottom_range, range1)
    ref_mask = create_bottom_mask(sv1.shape[0], npings, ref_idx, mode="below")

    pred_d = extract_bottom_depths(prob, range1, threshold=0.5)
    ref_d = extract_bottom_depths(ref_mask.astype(np.float32), range1, threshold=0.5)
    sigma = compute_transition_width(prob, range1)

    return dict(stem=stem, sv1=sv1, sv2r=sv2r, norm1=norm1, norm2=norm2,
                diff=diff, sv_channels=sv_channels, prob=prob, range1=range1,
                ref_mask=ref_mask, pred_d=pred_d, ref_d=ref_d, sigma=sigma,
                npings=npings)


def metrics(g):
    prob, ref = g["prob"], g["ref_mask"].astype(bool)
    pred = prob >= 0.5
    inter = (pred & ref).sum()
    union = (pred | ref).sum()
    iou = inter / union if union else np.nan
    dice = 2 * inter / (pred.sum() + ref.sum()) if (pred.sum() + ref.sum()) else np.nan
    v = np.isfinite(g["pred_d"]) & np.isfinite(g["ref_d"]) & (g["ref_d"] > 0)
    err = (g["pred_d"][v] - g["ref_d"][v]) if v.any() else np.array([np.nan])
    return dict(iou=float(iou), dice=float(dice),
                mae=float(np.nanmean(np.abs(err))),
                bias=float(np.nanmean(err)),
                pred_cov=float(np.mean(np.isfinite(g["pred_d"]) & (g["pred_d"] > 0))),
                ref_cov=float(np.mean(np.isfinite(g["ref_d"]) & (g["ref_d"] > 0))))


# ── Figures ─────────────────────────────────────────────────────────────────
def fig_input_channels(g, path):
    ra = g["range1"]
    extent = [0, g["npings"], ra[-1], ra[0]]
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    im0 = axes[0].imshow(g["sv1"], aspect="auto", cmap="viridis",
                         vmin=VMIN, vmax=VMAX, extent=extent)
    axes[0].set_title("38 kHz Sv (context)")
    style_colorbar(plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04),
                   label="Sv (dB)")
    grey = [("38 kHz (ch 1)", g["norm1"], 0, 1),
            ("200 kHz (ch 2)", g["norm2"], 0, 1),
            ("38-200 difference (ch 3)", g["diff"], -1, 1)]
    for ax, (title, dat, lo, hi) in zip(axes[1:], grey):
        im = ax.imshow(dat, aspect="auto", cmap="gray", vmin=lo, vmax=hi,
                       extent=extent)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xlabel("Ping")
        ax.set_ylabel("Depth (m)")
    fig.suptitle(f"{g['stem']}: network input channels (greyscale)", y=1.02)
    fig.tight_layout()
    savefig(fig, path)
    plt.close(fig)


def fig_probability(g, m, path):
    ra = g["range1"]
    extent = [0, g["npings"], ra[-1], ra[0]]
    pings = np.arange(g["npings"])
    vpred = np.isfinite(g["pred_d"]) & (g["pred_d"] > 0)
    vref = np.isfinite(g["ref_d"]) & (g["ref_d"] > 0)
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    im0 = axes[0].imshow(g["sv1"], aspect="auto", cmap="viridis",
                         vmin=VMIN, vmax=VMAX, extent=extent)
    axes[0].plot(pings[vref], g["ref_d"][vref], color=REF_COLOR, lw=1.4,
                 ls="--", label="Reference (38 kHz ensemble)")
    axes[0].plot(pings[vpred], g["pred_d"][vpred], color=PRED_COLOR, lw=1.4,
                 label="U-Net prediction")
    axes[0].set_title("Sv + bottom lines")
    axes[0].legend(loc="lower left", fontsize=10, framealpha=0.9)
    style_colorbar(plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04),
                   label="Sv (dB)")
    im1 = axes[1].imshow(g["prob"], aspect="auto", cmap="magma", vmin=0, vmax=1,
                         extent=extent)
    axes[1].set_title("P(bottom)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    im2 = axes[2].imshow(g["prob"] >= 0.5, aspect="auto", cmap="gray",
                         extent=extent)
    axes[2].imshow(np.ma.masked_where(~g["ref_mask"].astype(bool),
                                      g["ref_mask"]), aspect="auto",
                   cmap="autumn", alpha=0.35, extent=extent)
    axes[2].set_title("Prediction (white) vs reference (red)")
    for ax in axes[:3]:
        ax.set_xlabel("Ping")
        ax.set_ylabel("Depth (m)")

    # depth profile: where do the detectors sit vs the actual returns?
    prof = np.nanmedian(g["sv1"], axis=1)
    axp = axes[3]
    axp.plot(prof, ra, color="#1f77b4", lw=2.0)
    axp.invert_yaxis()
    det = float(np.nanmedian(g["pred_d"][vpred])) if vpred.any() else np.nan
    axp.axhline(det, color=PRED_COLOR, lw=2.0,
                label=f"detected bottom ~{det:.1f} m")
    deep = ra[(ra > 12) & (ra < ra[-1] - 1)]
    if deep.size:
        seg = prof[(ra > 12) & (ra < ra[-1] - 1)]
        sb = float(deep[int(np.nanargmax(seg))])
        axp.axhline(sb, color="#2ca02c", lw=2.0, ls="--",
                    label=f"deep return ~{sb:.1f} m")
    axp.set_xlabel("Median Sv (dB)")
    axp.set_ylabel("Depth (m)")
    axp.set_title("Median Sv vs depth")
    axp.legend(loc="lower right", fontsize=10, framealpha=0.9)
    axp.grid(True, alpha=0.3)

    fig.suptitle(f"{g['stem']}: IoU={m['iou']:.2f}  Dice={m['dice']:.2f}  "
                 f"MAE={m['mae']:.1f} m  bias={m['bias']:+.1f} m "
                 f"(reference is co-fooled, see profile)", y=1.02)
    fig.tight_layout()
    savefig(fig, path)
    plt.close(fig)


def fig_conformal(g, q, path):
    ra = g["range1"]
    extent = [0, g["npings"], ra[-1], ra[0]]
    pings = np.arange(g["npings"])
    vpred = np.isfinite(g["pred_d"]) & (g["pred_d"] > 0)
    vref = np.isfinite(g["ref_d"]) & (g["ref_d"] > 0)

    cu, cl = predict_depth_band(g["pred_d"], q["const"])[:2]
    lu, ll, _ = predict_locally_weighted_band(g["pred_d"], g["sigma"], q["local"])
    bands = {"Constant ±%.1f m" % q["const"]: (cu, cl, "#ff7f0e"),
             "Locally-weighted": (lu, ll, "#9467bd")}

    def cover(u, l):
        ref = g["ref_d"]
        valid = np.isfinite(u) & np.isfinite(l) & np.isfinite(ref) & (ref > 0)
        if not valid.any():
            return np.nan
        return float(((ref[valid] >= u[valid]) & (ref[valid] <= l[valid])).mean())

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.imshow(g["sv1"], aspect="auto", cmap="viridis", vmin=VMIN, vmax=VMAX,
              extent=extent)
    for label, (u, l, c) in bands.items():
        v = np.isfinite(u) & np.isfinite(l)
        ax.fill_between(pings[v], u[v], l[v], color=c, alpha=0.30, zorder=2,
                        label=f"{label}: {cover(u, l)*100:.0f}% cover")
    ax.plot(pings[vpred], g["pred_d"][vpred], color=PRED_COLOR, lw=1.5, zorder=5,
            label="U-Net prediction")
    ax.plot(pings[vref], g["ref_d"][vref], color=REF_COLOR, lw=1.3, ls="--",
            zorder=6, label="Reference label")
    ax.set_xlabel("Ping")
    ax.set_ylabel("Depth (m)")
    ax.set_ylim(ra[-1], ra[0])
    ax.legend(loc="lower right", ncol=2, framealpha=0.92, fontsize=11)
    ax.set_title(f"{g['stem']}: conformal bands ({(1-ALPHA)*100:.0f}% target, "
                 f"calibrated in-distribution)")
    savefig(fig, path)
    plt.close(fig)


def worst_window(g, width=512):
    """Find the 512-ping window with the largest pred/ref depth disagreement."""
    err = np.abs(g["pred_d"] - g["ref_d"])
    err[~np.isfinite(err)] = 0.0  # missing prediction also counts where ref exists
    miss = np.isfinite(g["ref_d"]) & (g["ref_d"] > 0) & ~(
        np.isfinite(g["pred_d"]) & (g["pred_d"] > 0))
    score = err + miss * (g["range1"][-1])  # missed bottom = full-range penalty
    n = g["npings"]
    if n <= width:
        return 0, n
    best, bi = -1, 0
    for s in range(0, n - width + 1, 64):
        v = score[s:s + width].sum()
        if v > best:
            best, bi = v, s
    return bi, bi + width


def fig_gradcam(g, model, path):
    s, e = worst_window(g)
    patch = g["sv_channels"][:, :, s:e]                       # (3,H,W)
    ra = g["range1"]
    h, w = patch.shape[1], patch.shape[2]
    hwc = np.transpose(patch, (1, 2, 0))
    res = cv2.resize(hwc, (TARGET_SIZE[1], TARGET_SIZE[0]),
                     interpolation=cv2.INTER_LINEAR)
    inp = np.transpose(res, (2, 0, 1)).astype(np.float32)
    tens = torch.from_numpy(inp[None]).float().to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(tens)).squeeze().cpu().numpy()
    pred_mask = (prob >= 0.5).astype(np.float32)

    target = [SemanticSegmentationTarget(0, pred_mask if pred_mask.any()
                                         else np.ones_like(pred_mask))]
    layers = {"decoder (final block)": model.decoder.blocks[-1],
              "encoder (layer3)": model.encoder.layer3[-1]}
    cams = {}
    for name, layer in layers.items():
        with GradCAM(model=model, target_layers=[layer]) as cam:
            cams[name] = cam(input_tensor=tens, targets=target)[0]

    # input-channel gradient magnitude (which channel drives the output)
    tg = torch.from_numpy(inp[None]).float().to(DEVICE).requires_grad_(True)
    out = model(tg)
    sel = out.squeeze()[pred_mask.astype(bool)] if pred_mask.any() else out
    sel.sum().backward()
    chan_grad = tg.grad.detach().abs().mean(dim=(2, 3)).cpu().numpy().ravel()

    comp = np.clip(np.transpose(res, (2, 0, 1)), 0, 1)
    rgb = np.stack([np.clip(comp[0], 0, 1), np.clip(comp[1], 0, 1),
                    np.clip((comp[2] + 1) / 2, 0, 1)], axis=-1)
    extent = [s, e, ra[-1], ra[0]]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes[0, 0].imshow(rgb, aspect="auto", extent=extent)
    axes[0, 0].set_title("Input composite (R=38, G=200, B=diff)")
    im = axes[0, 1].imshow(prob, aspect="auto", cmap="magma", vmin=0, vmax=1,
                           extent=extent)
    axes[0, 1].set_title("P(bottom) on window")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[0, 2].bar(["38 kHz", "200 kHz", "diff"], chan_grad,
                   color=["#1f77b4", "#d62728", "#2ca02c"])
    axes[0, 2].set_title("Mean |∂logit/∂input| per channel")
    axes[0, 2].set_ylabel("gradient magnitude")
    for ax, (name, cam) in zip(axes[1, :2], cams.items()):
        ax.imshow(rgb, aspect="auto", extent=extent)
        ax.imshow(cam, aspect="auto", cmap="jet", alpha=0.5, extent=extent)
        ax.set_title(f"Grad-CAM: {name}")
    axes[1, 2].imshow(g["sv1"][:, s:e], aspect="auto", cmap="viridis",
                      vmin=VMIN, vmax=VMAX, extent=extent)
    axes[1, 2].set_title("38 kHz Sv (same window)")
    for ax in axes.ravel():
        ax.set_xlabel("Ping")
        ax.set_ylabel("Depth (m)")
    fig.suptitle(f"{g['stem']}: Grad-CAM on worst window (pings {s}-{e})", y=1.01)
    fig.tight_layout()
    savefig(fig, path)
    plt.close(fig)


def fig_summary(items, mets, path):
    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 11))
    if n == 1:
        axes = axes[:, None]
    for j, (g, m) in enumerate(zip(items, mets)):
        ra = g["range1"]
        extent = [0, g["npings"], ra[-1], ra[0]]
        pings = np.arange(g["npings"])
        vpred = np.isfinite(g["pred_d"]) & (g["pred_d"] > 0)
        vref = np.isfinite(g["ref_d"]) & (g["ref_d"] > 0)
        axes[0, j].imshow(g["sv1"], aspect="auto", cmap="viridis",
                          vmin=VMIN, vmax=VMAX, extent=extent)
        axes[0, j].plot(pings[vref], g["ref_d"][vref], color=REF_COLOR, lw=1.2,
                        ls="--")
        axes[0, j].plot(pings[vpred], g["pred_d"][vpred], color=PRED_COLOR, lw=1.2)
        axes[0, j].set_title(f"{g['stem'].split('-')[-1]}\nIoU={m['iou']:.2f} "
                             f"MAE={m['mae']:.1f} m")
        axes[1, j].imshow(g["prob"], aspect="auto", cmap="magma", vmin=0, vmax=1,
                          extent=extent)
        axes[1, j].set_title("P(bottom)")
        for r in (0, 1):
            axes[r, j].set_xlabel("Ping")
            axes[r, j].set_ylabel("Depth (m)")
    fig.legend(handles=[
        Line2D([0], [0], color=REF_COLOR, ls="--", lw=2, label="Reference label"),
        Line2D([0], [0], color=PRED_COLOR, lw=2, label="U-Net prediction")],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("shallow_C2 dock files: Sv + prediction (top), "
                 "P(bottom) (bottom)", y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    savefig(fig, path)
    plt.close(fig)


def write_readme(items, mets):
    import datetime
    lines = []
    for g, m in zip(items, mets):
        lines.append(f"  {g['stem']}: IoU={m['iou']:.3f} Dice={m['dice']:.3f} "
                     f"MAE={m['mae']:.2f} m bias={m['bias']:+.2f} m "
                     f"pred_cov={m['pred_cov']*100:.0f}% ref_cov={m['ref_cov']*100:.0f}%")
    (OUT_DIR / "_README.md").write_text(f"""# shallow_C2 dock failure analysis

shallow  dock  EK80  broadband  out-of-distribution  failure  gradcam  conformal

Failure analysis of the three broadband EK80 dock recordings in
"EK80 38 200 kHz ved kai" (shallow_C2-D20260526). Water depth ~10-19 m, far
shallower than the training crossings; near-field ringing and dock structure
dominate. Per file: greyscale input channels, P(bottom) map, prediction vs
algorithmic 38 kHz reference, in-distribution-calibrated conformal bands, and
Grad-CAM over the worst-failing window with input-channel gradient ablation.

Source: code/data_visualization/analyze_shallow_C2_failure.py
Created: {datetime.datetime.now():%Y-%m-%d %H:%M}
Checkpoint: {CHECKPOINT.parent.name}

Per-file metrics (reference = algorithmic 38 kHz ensemble label):
""" + "\n".join(lines) + "\n")


def _calib_one(zarr_path, model):
    """Inference + reference label on one in-distribution zarr file."""
    sv1, range1, _, _, _, _ = load_sv_from_zarr(str(zarr_path), FREQ1)
    sv2, range2, _, _, _, _ = load_sv_from_zarr(str(zarr_path), FREQ2)
    min_r = min(sv1.shape[0], sv2.shape[0])
    npings = min(sv1.shape[1], sv2.shape[1], MAX_CALIB_PINGS)
    sv1, sv2 = sv1[:min_r, :npings], sv2[:min_r, :npings]
    ra = range1[:min_r]

    norm1 = normalize_echogram(sv1, method="minmax", vmin=VMIN, vmax=VMAX,
                               fill_nan=0.0).astype(np.float32)
    norm2 = normalize_echogram(sv2, method="minmax", vmin=VMIN, vmax=VMAX,
                               fill_nan=0.0).astype(np.float32)
    diff = (norm1 - norm2).astype(np.float32)
    chans = np.stack([norm1, norm2, diff], axis=0)

    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    if "bottom_mask" in ds.data_vars:
        gt_mask = ds["bottom_mask"].values.astype(np.uint8)[:min_r, :npings]
        ds.close()
    else:
        ds.close()
        d1 = load_zarr(str(zarr_path), frequency=FREQ1)
        heave = np.asarray(d1.heave if d1.heave is not None
                           else np.zeros(npings))[:npings]
        bt = -31.0 if d1.calibrated else "auto"
        br, _, _ = detect_bottom_ensemble_pipeline(
            sv1, ra, heave=heave, pulse_duration=d1.pulse_duration,
            frequency_hz=d1.frequency_hz,
            transducer_draft=d1.transducer_draft or 0.0,
            threshold_log_sv=bt, detectors=("simple", "edge"))
        gt_idx = bottom_to_sample_index(br, ra)
        gt_mask = create_bottom_mask(sv1.shape[0], npings, gt_idx, mode="below")

    with torch.no_grad():
        prob = predict_full_echogram(model, chans, TARGET_SIZE, DEVICE,
                                     patch_width=PATCH_WIDTH, stride=STRIDE)
    pred_d = extract_bottom_depths(prob, ra, threshold=0.5)
    gt_d = extract_bottom_depths(gt_mask.astype(np.float32), ra, threshold=0.5)
    sigma = compute_transition_width(prob, ra)
    return pred_d, gt_d, sigma


def build_calibration(model, n_calib, seed):
    """Conformal q_hat for the *current* checkpoint (constant + locally-weighted).

    Calibrates on a random subset of the in-distribution calibration/test pool
    so the bands are valid for the checkpoint under test, not a different model.
    Cached per checkpoint under CACHE_DIR.
    """
    cache = CACHE_DIR / f"calib_{CHECKPOINT.parent.name}_n{n_calib}_s{seed}.npz"
    if cache.exists():
        d = np.load(cache)
        depth_s, local_s = d["depth_scores"], d["local_scores"]
    else:
        sf = get_split_files()
        pool = sorted(set(str(p) for p in sf["calibration"]) |
                      set(str(p) for p in sf["test"]))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(pool))
        files = [PROJECT_ROOT / pool[i] for i in perm[:min(n_calib, len(pool))]]
        print(f"In-distribution calibration: {len(files)} files", flush=True)
        depth_l, local_l = [], []
        for i, zp in enumerate(files, 1):
            try:
                pred_d, gt_d, sigma = _calib_one(zp, model)
            except Exception as e:
                print(f"  [calib {i}/{len(files)}] {zp.stem} skipped: {e}")
                continue
            depth_l.append(calibrate_depth(pred_d, gt_d))
            local_l.append(calibrate_locally_weighted(pred_d, gt_d, sigma))
            print(f"  [calib {i}/{len(files)}] {zp.stem}", flush=True)
        depth_s = np.concatenate(depth_l)
        local_s = np.concatenate(local_l)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, depth_scores=depth_s, local_scores=local_s)

    return {"const": float(conformal_quantile(depth_s, ALPHA)),
            "local": float(conformal_quantile(local_s, ALPHA))}


def main():
    global CHECKPOINT, OUT_DIR, CACHE_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT),
                    help="Path to the .pth checkpoint (default: chunk_arch_sweep_2).")
    ap.add_argument("--n-calib", type=int, default=N_CALIB)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    CHECKPOINT = Path(args.checkpoint)
    OUT_DIR = PROJECT_ROOT / "Figures" / f"shallow_C2_failure_{CHECKPOINT.parent.name}"
    CACHE_DIR = OUT_DIR / "cache"

    apply_thesis_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patch_echopype_vendor()
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")

    model = load_model()
    q = build_calibration(model, args.n_calib, args.seed)
    print(f"Conformal q (alpha={ALPHA}): {q}")
    items, mets = [], []
    for stem in STEMS:
        print(f"[process] {stem}", flush=True)
        g = process(stem, model)
        m = metrics(g)
        mets.append(m)
        items.append(g)
        print(f"   IoU={m['iou']:.3f} Dice={m['dice']:.3f} MAE={m['mae']:.2f} m "
              f"bias={m['bias']:+.2f} m pred_cov={m['pred_cov']*100:.0f}%")
        fig_input_channels(g, OUT_DIR / f"input_channels_{stem}.png")
        fig_probability(g, m, OUT_DIR / f"prediction_{stem}.png")
        fig_conformal(g, q, OUT_DIR / f"conformal_{stem}.png")
        fig_gradcam(g, model, OUT_DIR / f"gradcam_{stem}.png")

    fig_summary(items, mets, OUT_DIR / "summary_grid.png")
    write_readme(items, mets)
    print(f"\nDone. {len(items)} files -> {OUT_DIR}")


if __name__ == "__main__":
    main()
