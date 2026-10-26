"""Adaptive vs constant label-less conformal analysis and figures.

Reads the full ping table from a previous label-less conformal run
(Figures/conformal/labelless/ping_table.parquet) and, without re-running
inference, derives the adaptive (per-ping, dispersion-normalised) conformal
band, compares it against the constant band, computes coverage/efficiency
metrics, and renders a polished figure set.

Output: Figures/conformal/labelless_adaptive/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from data_visualization.thesis_style import (
    apply_thesis_style, FIGSIZE_STD, FIGSIZE_FULL, FIGSIZE_HALF,
    style_colorbar, savefig, keep_spines,
)
from validation.conformal_labelless import (
    split_crossings, build_calibration_scores, calibrate_labelless,
    pairwise_coverage, compute_bin_dispersion, assign_bin_sigma,
    bin_arclength, EPSILON_SIGMA_M, route_length_m,
)

DEFAULT_PARQUET = PROJECT_ROOT / 'Figures' / 'conformal' / 'labelless' / 'ping_table.parquet'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'conformal' / 'labelless_adaptive'

# Colour conventions (kept consistent across the set).
C_CONST = '#ff7f0e'    # constant band
C_ADAPT = '#1f77b4'    # adaptive band
C_PRED = '#d62728'     # predicted bottom line
C_OTHER = '#cccccc'    # other crossings

# Route segmentation for the harbour-vs-channel breakdown (metres of arc).
HARBOUR_HORTEN_MAX = 2000.0
HARBOUR_MOSS_MIN = 7000.0


def robust_reject(df: pd.DataFrame, bin_size_m: float, k: float) -> pd.DataFrame:
    """Drop pings > k robust-MADs from their bin median (uses redundancy)."""
    w = df.copy()
    w['_bin'] = bin_arclength(w['arc_length_m'].to_numpy(), bin_size_m)
    med = w.groupby('_bin')['pred_depth_m'].transform('median')
    mad = w.groupby('_bin')['pred_depth_m'].transform(
        lambda x: np.median(np.abs(x - np.median(x))))
    rs = 1.4826 * mad
    dev = (w['pred_depth_m'] - med).abs()
    keep = (rs <= 0) | (dev <= k * rs)
    return w[keep].drop(columns='_bin').reset_index(drop=True)


def region_of(arc_m: np.ndarray) -> np.ndarray:
    """Label each arc-length as harbour or channel."""
    return np.where((arc_m < HARBOUR_HORTEN_MAX) | (arc_m > HARBOUR_MOSS_MIN),
                    'harbour', 'channel')


# ─── Figures ──────────────────────────────────────────────────────────────────

def fig_example_crossing(df_test, q_const, q_adapt, bin_size_m, cid, out_path,
                         subtitle=None):
    """One crossing: constant band vs adaptive band, two stacked panels."""
    ex = df_test[df_test['crossing_id'] == cid].sort_values('arc_length_m')
    if ex.empty:
        ex = (df_test[df_test['crossing_id'] ==
              df_test.groupby('crossing_id').size().idxmax()]
              .sort_values('arc_length_m'))
        cid = ex['crossing_id'].iloc[0]
    s = ex['arc_length_m'].to_numpy() / 1000.0
    d = ex['pred_depth_m'].to_numpy()
    sig = np.maximum(ex['sigma_m'].to_numpy(), EPSILON_SIGMA_M)
    direction = ex['direction'].iloc[0]
    neigh = df_test[(df_test['direction'] == direction)
                    & (df_test['crossing_id'] != cid)]

    fig, axes = plt.subplots(2, 1, figsize=(FIGSIZE_FULL[0], 9), sharex=True,
                             sharey=True)
    half_const = np.full_like(d, q_const)
    half_adapt = q_adapt * sig
    for ax, half, colour, label in (
        (axes[0], half_const, C_CONST, rf'Constant band  ($\pm${q_const:.1f} m)'),
        (axes[1], half_adapt, C_ADAPT,
         rf'Adaptive band  (median $\pm${np.median(half_adapt):.1f} m, '
         f'{np.min(half_adapt):.1f} to {np.max(half_adapt):.0f} m)')):
        ax.set_facecolor('#202020')
        for _, g in neigh.groupby('crossing_id'):
            g = g.sort_values('arc_length_m')
            ax.plot(g['arc_length_m'].to_numpy() / 1000.0,
                    g['pred_depth_m'].to_numpy(),
                    color=C_OTHER, alpha=0.55, lw=0.8, zorder=1)
        ax.fill_between(s, d - half, d + half, color=colour, alpha=0.40,
                        zorder=2, label=label)
        ax.plot(s, d, color=C_PRED, lw=2.0, zorder=3)
        ax.set_ylim(100, 0)
        ax.set_ylabel('Bottom depth (m)')
        ax.legend(loc='upper center', framealpha=0.9)
    axes[0].plot([], [], color=C_OTHER, lw=1.2, label='other crossings')
    axes[0].plot([], [], color=C_PRED, lw=2.0, label='this crossing')
    axes[0].legend(loc='upper center', ncol=2, framealpha=0.9)
    axes[1].set_xlabel('Arc-length along route (km, Horten to Moss)')
    fig.suptitle('Label-less conformal band: constant vs adaptive')
    axes[0].set_title(subtitle or 'Same crossing, 90% target coverage',
                      color='grey')
    savefig(fig, out_path)
    plt.close(fig)
    return cid


def crossing_noisiness(df_test, bin_size_m, per_direction):
    """Per-crossing disagreement with the consensus of other crossings.

    For each ping, |pred - bin_median| (the bin median is taken over all
    crossings at that spot); summarised per crossing. A high median deviation
    means this crossing's bottom track sits far from where the other passes
    agree the seabed is — i.e. a genuinely noisy / off crossing.
    """
    w = df_test.copy()
    w['bin'] = bin_arclength(w['arc_length_m'].to_numpy(), bin_size_m)
    keys = ['bin', 'direction'] if per_direction else ['bin']
    w['binmed'] = w.groupby(keys)['pred_depth_m'].transform('median')
    w['dev'] = (w['pred_depth_m'] - w['binmed']).abs()
    return (w.groupby('crossing_id')
              .agg(direction=('direction', 'first'),
                   n_pings=('pred_depth_m', 'size'),
                   arc_span_m=('arc_length_m', lambda x: x.max() - x.min()),
                   median_dev_m=('dev', 'median'),
                   p90_dev_m=('dev', lambda x: x.quantile(0.9)))
              .reset_index()
              .sort_values('median_dev_m'))


def render_example_gallery(df_test, q_const, q_adapt, bin_size_m, per_direction,
                           out_dir, n=6):
    """Render constant-vs-adaptive figures for a diverse set of crossings
    spanning the calm-to-noisy spectrum, so no single example dominates.
    """
    gal = out_dir / 'example_crossings'
    gal.mkdir(parents=True, exist_ok=True)

    noise = crossing_noisiness(df_test, bin_size_m, per_direction)
    noise.to_csv(out_dir / 'crossing_noisiness.csv', index=False)

    # Restrict to "full" crossings so the figures show the whole route.
    full = noise[(noise['arc_span_m'] > 6000) & (noise['n_pings'] > 1500)]
    if len(full) < n:
        full = noise[noise['n_pings'] > 800]
    full = full.sort_values('median_dev_m').reset_index(drop=True)

    # Pick n evenly spaced along the noisiness ranking (calmest ... noisiest).
    idx = np.linspace(0, len(full) - 1, min(n, len(full))).round().astype(int)
    picks = full.iloc[idx]

    pct = (full['median_dev_m'].rank(pct=True) * 100)
    rank_lookup = dict(zip(full['crossing_id'], pct))

    rendered = []
    for _, row in picks.iterrows():
        cid = row['crossing_id']
        subtitle = (f"{cid}  |  median deviation from consensus "
                    f"{row['median_dev_m']:.2f} m "
                    f"({rank_lookup[cid]:.0f}th pct of crossings)")
        safe = cid.replace('/', '_')
        fig_example_crossing(df_test, q_const, q_adapt, bin_size_m, cid,
                             gal / f'{safe}.pdf', subtitle=subtitle)
        rendered.append((cid, float(row['median_dev_m'])))
    return rendered, noise


def fig_coverage_curves(curve_const, curve_adapt, out_path):
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.plot([0, 1], [0, 1], ls=':', color='grey', lw=1.5, label='ideal (y = x)')
    for curve, colour, label in (
        (curve_const, C_CONST, 'constant'),
        (curve_adapt, C_ADAPT, r'adaptive (dispersion $\sigma$)')):
        nominal = 1.0 - curve['alpha'].to_numpy()
        ax.plot(nominal, curve['empirical_coverage'].to_numpy(),
                marker='o', ms=5, lw=2.0, color=colour, label=label)
    ax.axvline(0.9, color='k', ls='--', lw=1.0, alpha=0.5)
    ax.set_xlabel(r'Nominal coverage ($1 - \alpha$)')
    ax.set_ylabel('Empirical pairwise coverage')
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_title('Coverage calibration: both track the diagonal')
    ax.legend(loc='lower right')
    savefig(fig, out_path)
    plt.close(fig)


def fig_width_vs_arclength(df, q_const, q_adapt, bin_size_m, out_path):
    """Spatial adaptivity: band half-width along the route."""
    w = df.copy()
    w['bin'] = bin_arclength(w['arc_length_m'].to_numpy(), bin_size_m)
    w['half_adapt'] = q_adapt * np.maximum(w['sigma_m'].to_numpy(), EPSILON_SIGMA_M)
    g = w.groupby('bin').agg(arc=('arc_length_m', 'mean'),
                             half_adapt=('half_adapt', 'median')).reset_index()
    g = g.sort_values('arc')
    arc_km = g['arc'].to_numpy() / 1000.0

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    # Harbour shading.
    ax.axvspan(0, HARBOUR_HORTEN_MAX / 1000.0, color='grey', alpha=0.12)
    ax.axvspan(HARBOUR_MOSS_MIN / 1000.0, route_length_m() / 1000.0,
               color='grey', alpha=0.12, label='harbour approaches')
    ax.axhline(q_const, color=C_CONST, lw=2.5, ls='--',
               label=rf'constant ($\pm${q_const:.1f} m everywhere)')
    ax.plot(arc_km, g['half_adapt'].to_numpy(), color=C_ADAPT, lw=2.2,
            label='adaptive (median per bin)')
    ax.fill_between(arc_km, 0, g['half_adapt'].to_numpy(), color=C_ADAPT,
                    alpha=0.18)
    ax.set_xlabel('Arc-length along route (km, Horten to Moss)')
    ax.set_ylabel('Band half-width (m)')
    ax.set_xlim(0, route_length_m() / 1000.0)
    ax.set_ylim(0, None)
    ax.set_title('Adaptive band widens over the deep channel, narrows at the harbours')
    ax.legend(loc='upper center', ncol=2)
    savefig(fig, out_path)
    plt.close(fig)


def fig_width_distribution(df, q_const, q_adapt, out_path):
    half_adapt = q_adapt * np.maximum(df['sigma_m'].to_numpy(), EPSILON_SIGMA_M)
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    hi = float(np.quantile(half_adapt, 0.99))
    ax.hist(np.clip(half_adapt, None, hi), bins=60, color=C_ADAPT, alpha=0.8,
            label='adaptive half-width')
    ax.axvline(q_const, color=C_CONST, lw=2.5, ls='--',
               label=rf'constant ($\pm${q_const:.1f} m)')
    ax.axvline(np.median(half_adapt), color=C_ADAPT, lw=2.5, ls='-',
               label=rf'adaptive median ($\pm${np.median(half_adapt):.1f} m)')
    ax.set_xlabel('Band half-width (m)')
    ax.set_ylabel('Ping count')
    ax.set_title('Per-ping band half-width: constant vs adaptive')
    ax.legend(loc='upper right')
    savefig(fig, out_path)
    plt.close(fig)


def fig_route_map(df, q_adapt, bin_size_m, out_path):
    """Geographic route coloured by adaptive band half-width."""
    w = df.copy()
    w['half'] = q_adapt * np.maximum(w['sigma_m'].to_numpy(), EPSILON_SIGMA_M)
    # Sub-sample for a clean scatter.
    if len(w) > 120_000:
        w = w.sample(120_000, random_state=0)
    vmax = float(np.quantile(w['half'], 0.97))
    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    sc = ax.scatter(w['lon'], w['lat'], c=np.clip(w['half'], None, vmax),
                    s=4, cmap='magma', vmin=0, vmax=vmax, edgecolors='none')
    cbar = fig.colorbar(sc, ax=ax)
    style_colorbar(cbar, label='Adaptive band half-width (m)')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(w['lat'].mean()))))
    ax.set_title('Where the band widens: predictive uncertainty along the route')
    keep_spines(ax)
    savefig(fig, out_path)
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parquet', default=str(DEFAULT_PARQUET))
    p.add_argument('--output-dir', default=str(OUTPUT_DIR))
    p.add_argument('--bin-size', type=float, default=20.0)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--per-direction', action='store_true', default=True)
    p.add_argument('--robust-reject', type=float, default=5.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--example-cid', default='group01_0093907_M2H')
    p.add_argument('--n-examples', type=int, default=6,
                   help='render a gallery of N crossings spanning the '
                        'calm-to-noisy spectrum (0 to skip)')
    args = p.parse_args()

    apply_thesis_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    n0 = len(df)
    if args.robust_reject is not None:
        df = robust_reject(df, args.bin_size, args.robust_reject)
    print(f"Loaded {n0:,} pings; {len(df):,} after robust-reject "
          f"(k={args.robust_reject}); {df['crossing_id'].nunique()} crossings")

    df_calib, df_test = split_crossings(df, calib_frac=0.5, seed=args.seed)

    # Dispersion sigma estimated on calibration crossings only.
    disp = compute_bin_dispersion(df_calib, bin_size_m=args.bin_size,
                                  per_direction=args.per_direction)
    df_calib = assign_bin_sigma(df_calib, disp, bin_size_m=args.bin_size,
                                per_direction=args.per_direction)
    df_test = assign_bin_sigma(df_test, disp, bin_size_m=args.bin_size,
                               per_direction=args.per_direction)

    # Calibrate both bands at the same alpha.
    sc_c, _ = build_calibration_scores(
        df_calib, bin_size_m=args.bin_size, per_direction=args.per_direction,
        score_mode='pairwise')
    q_const = calibrate_labelless(sc_c, args.alpha)
    cov_c = pairwise_coverage(df_test, q_const, bin_size_m=args.bin_size,
                              per_direction=args.per_direction,
                              score_mode='pairwise')

    sc_a, _ = build_calibration_scores(
        df_calib, bin_size_m=args.bin_size, per_direction=args.per_direction,
        score_mode='pairwise_norm', sigma_col='sigma_m')
    q_adapt = calibrate_labelless(sc_a, args.alpha)
    cov_a = pairwise_coverage(df_test, q_adapt, bin_size_m=args.bin_size,
                              per_direction=args.per_direction,
                              score_mode='pairwise_norm', sigma_col='sigma_m')

    # Per-ping adaptive half-widths on the full (post-reject) set.
    df_all = assign_bin_sigma(df, disp, bin_size_m=args.bin_size,
                              per_direction=args.per_direction)
    half_adapt_all = q_adapt * np.maximum(df_all['sigma_m'].to_numpy(),
                                          EPSILON_SIGMA_M)

    # Region breakdown (test set, using validation pairs' realised widths).
    reg = region_of(df_all['arc_length_m'].to_numpy())
    region_rows = []
    for name in ('channel', 'harbour'):
        m = reg == name
        region_rows.append({
            'region': name,
            'n_pings': int(m.sum()),
            'constant_halfwidth_m': float(q_const),
            'adaptive_halfwidth_median_m': float(np.median(half_adapt_all[m])),
            'adaptive_halfwidth_mean_m': float(np.mean(half_adapt_all[m])),
        })
    region_df = pd.DataFrame(region_rows)
    region_df.to_csv(out_dir / 'region_stats.csv', index=False)

    metrics = {
        'alpha': args.alpha,
        'bin_size_m': args.bin_size,
        'per_direction': bool(args.per_direction),
        'robust_reject_k': args.robust_reject,
        'n_pings_used': int(len(df)),
        'n_crossings': int(df['crossing_id'].nunique()),
        'constant': {
            'q_hat_m': float(q_const),
            'coverage': cov_c['coverage'],
            'mean_halfwidth_m': float(q_const),
            'median_halfwidth_m': float(q_const),
            'n_pairs': cov_c['n_pairs'],
        },
        'adaptive': {
            'q_hat_sigma_multiplier': float(q_adapt),
            'coverage': cov_a['coverage'],
            'mean_halfwidth_m': float(np.mean(half_adapt_all)),
            'median_halfwidth_m': float(np.median(half_adapt_all)),
            'p10_halfwidth_m': float(np.quantile(half_adapt_all, 0.10)),
            'p90_halfwidth_m': float(np.quantile(half_adapt_all, 0.90)),
            'n_pairs': cov_a['n_pairs'],
        },
        'median_width_reduction_pct': float(
            100 * (1 - np.median(half_adapt_all) / q_const)),
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    print(f"\nConstant:  q_hat = {q_const:.2f} m   coverage = {cov_c['coverage']:.1%}")
    print(f"Adaptive:  q_hat = {q_adapt:.2f}×σ   coverage = {cov_a['coverage']:.1%}")
    print(f"  median half-width {q_const:.2f} m -> {np.median(half_adapt_all):.2f} m "
          f"({metrics['median_width_reduction_pct']:.0f}% tighter for the typical ping)")
    print("\nRegion breakdown (adaptive half-width):")
    print(region_df.to_string(index=False))

    # Coverage curves (sweep alpha) for both.
    from validation.conformal_labelless import pairwise_coverage_curve
    alphas = np.arange(0.02, 0.51, 0.02)
    curve_c = pairwise_coverage_curve(
        df_calib, df_test, alphas=alphas, bin_size_m=args.bin_size,
        per_direction=args.per_direction, score_mode='pairwise')
    curve_a = pairwise_coverage_curve(
        df_calib, df_test, alphas=alphas, bin_size_m=args.bin_size,
        per_direction=args.per_direction, score_mode='pairwise_norm',
        sigma_col='sigma_m')
    curve_c.to_csv(out_dir / 'coverage_curve_constant.csv', index=False)
    curve_a.to_csv(out_dir / 'coverage_curve_adaptive.csv', index=False)

    # ── Figures ───────────────────────────────────────────────────────
    used_cid = fig_example_crossing(df_test, q_const, q_adapt, args.bin_size,
                                    args.example_cid,
                                    out_dir / 'example_crossing_compare.pdf')
    if args.n_examples > 0:
        rendered, noise = render_example_gallery(
            df_test, q_const, q_adapt, args.bin_size, args.per_direction,
            out_dir, n=args.n_examples)
        print(f"\nGallery ({len(rendered)} crossings, calm to noisy):")
        for cid, dev in rendered:
            print(f"  {cid}: median dev {dev:.2f} m")
        print(f"\nCrossing noisiness (test set), overall:")
        print(f"  median dev across crossings: "
              f"med={noise['median_dev_m'].median():.2f} m, "
              f"p90={noise['median_dev_m'].quantile(0.9):.2f} m, "
              f"max={noise['median_dev_m'].max():.2f} m")
        ex_row = noise[noise['crossing_id'] == args.example_cid]
        if not ex_row.empty:
            pctile = (noise['median_dev_m'].rank(pct=True)
                      [noise['crossing_id'] == args.example_cid].iloc[0] * 100)
            print(f"  '{args.example_cid}' median dev = "
                  f"{ex_row['median_dev_m'].iloc[0]:.2f} m "
                  f"({pctile:.0f}th pct — {'noisy' if pctile > 75 else 'typical'})")
    fig_coverage_curves(curve_c, curve_a, out_dir / 'coverage_curves.pdf')
    fig_width_vs_arclength(df_all, q_const, q_adapt, args.bin_size,
                           out_dir / 'band_width_vs_arclength.pdf')
    fig_width_distribution(df_all, q_const, q_adapt,
                           out_dir / 'band_width_distribution.pdf')
    fig_route_map(df_all, q_adapt, args.bin_size,
                  out_dir / 'route_map_band_width.png')

    print(f"\nExample crossing used: {used_cid}")
    print(f"Outputs in {out_dir}")
    for f in sorted(out_dir.glob('*')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
