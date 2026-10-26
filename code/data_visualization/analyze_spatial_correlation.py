"""Spatial-correlation analysis of echograms motivating the U-Net.

Computes directional autocorrelation and method-of-moments variograms along the
depth and ping axes for a sample of echopype zarr stores, and extracts
correlation lengths. Output: Figures/spatial_correlation/ and Latex/Tables/.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import argparse

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from raw_sensor_integration.preprocessing import load_sv_from_zarr, normalize_echogram
from data_visualization.thesis_style import (
    apply_thesis_style, FIGSIZE_FULL, savefig,
)

FREQS = {'38 kHz': 38000, '200 kHz': 200000}
FREQ_COLORS = {'38 kHz': '#1f77b4', '200 kHz': '#d62728'}

CLIP_DB = (-80.0, -34.0)          # model input range
MAX_RANGE_SAMPLES = 1500          # cap depth rows
MAX_PINGS = 3000                  # cap ping columns
DEPTH_LAG_MAX_M = 18.0            # display/compute depth lag up to this
PING_LAG_MAX = 1000              # display/compute ping lag up to this
DEPTH_GRID_STEP_M = 0.05          # common depth-lag grid for averaging
MAX_NAN_FRAC = 0.55              # skip echograms with too much missing data


def directional_stats(field, max_lag):
    """Method-of-moments autocorrelation and semivariogram along an axis.

    `field` is mean-removed and oriented so the target axis is axis 1.
    Returns (acf, gamma) arrays of length max_lag+1 (lag 0..max_lag).
    """
    var = float(np.mean(field * field))
    acf = np.ones(max_lag + 1, dtype=np.float64)
    gamma = np.zeros(max_lag + 1, dtype=np.float64)
    for h in range(1, max_lag + 1):
        a = field[:, :-h]
        b = field[:, h:]
        cov = float(np.mean(a * b))
        acf[h] = cov / var if var > 0 else 0.0
        gamma[h] = 0.5 * float(np.mean((a - b) ** 2))
    return acf, gamma


def one_over_e_length(acf, lags):
    """First lag (interpolated) where the autocorrelation drops below 1/e."""
    thr = 1.0 / np.e
    below = np.where(acf <= thr)[0]
    if len(below) == 0:
        return float('nan')
    j = below[0]
    if j == 0:
        return 0.0
    x0, x1 = lags[j - 1], lags[j]
    y0, y1 = acf[j - 1], acf[j]
    if y0 == y1:
        return float(x1)
    return float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0))


def exp_fit_length(acf, lags):
    """Correlation length from a least-squares fit of rho(h) = exp(-h/ell)."""
    from scipy.optimize import curve_fit
    mask = acf > 1e-3
    if mask.sum() < 4:
        return float('nan')
    try:
        popt, _ = curve_fit(
            lambda h, ell: np.exp(-h / ell),
            lags[mask], acf[mask], p0=[lags[-1] / 3.0], maxfev=10000,
        )
        return float(popt[0])
    except Exception:
        return float('nan')


def variogram_range(gamma, lags):
    """Range parameter a of an exponential variogram with nugget."""
    from scipy.optimize import curve_fit
    if len(lags) < 5:
        return float('nan')
    sill0 = float(np.nanmax(gamma))
    try:
        popt, _ = curve_fit(
            lambda h, a, sill, nug: nug + (sill - nug) * (1.0 - np.exp(-h / a)),
            lags, gamma,
            p0=[lags[-1] / 3.0, sill0, 0.0],
            bounds=([1e-3, 0.0, 0.0], [lags[-1] * 5, sill0 * 5 + 1e-9, sill0 + 1e-9]),
            maxfev=10000,
        )
        return float(popt[0])
    except Exception:
        return float('nan')


def process_file(zarr_path, freq_hz):
    sv_db, range_array, ping_times, _, heave, _ = load_sv_from_zarr(
        zarr_path, freq_hz, max_range_samples=MAX_RANGE_SAMPLES,
    )
    if sv_db.shape[1] > MAX_PINGS:
        sv_db = sv_db[:, :MAX_PINGS]
        ping_times = ping_times[:MAX_PINGS]

    # Crop to the valid depth extent: deep range bins are NaN padding where the
    # recorded range falls short of the array length. Filling them with zeros
    # would inject a large artificial constant block and inflate correlation.
    row_nan = np.isnan(sv_db).mean(axis=1)
    valid = np.where(row_nan < 0.5)[0]
    if len(valid) < 50:
        return None
    last = int(valid.max()) + 1
    sv_db = sv_db[:last]
    range_array = range_array[:last]

    if sv_db.shape[0] < 50 or sv_db.shape[1] < 50:
        return None
    if np.isnan(sv_db).mean() > MAX_NAN_FRAC:
        return None

    norm = normalize_echogram(sv_db, method='minmax', vmin=CLIP_DB[0], vmax=CLIP_DB[1])
    norm = np.nan_to_num(norm, nan=0.0)
    field = norm - norm.mean()

    dz = float(np.median(np.diff(range_array)))
    if not np.isfinite(dz) or dz <= 0:
        return None

    # Depth axis -> orient so target axis is axis 1 (transpose)
    depth_lag_max = min(int(DEPTH_LAG_MAX_M / dz), field.shape[0] - 1)
    acf_d, gamma_d = directional_stats(field.T, depth_lag_max)
    depth_lags_m = np.arange(depth_lag_max + 1) * dz

    ping_lag_max = min(PING_LAG_MAX, field.shape[1] - 1)
    acf_p, gamma_p = directional_stats(field, ping_lag_max)
    ping_lags = np.arange(ping_lag_max + 1, dtype=np.float64)

    # Ping interval in seconds (physical horizontal scale without a speed assumption)
    dt_s = float('nan')
    try:
        secs = np.array([t.timestamp() for t in ping_times])
        d = np.diff(secs)
        d = d[(d > 0) & (d < 10)]
        if len(d):
            dt_s = float(np.median(d))
    except Exception:
        pass

    return {
        'depth_lags_m': depth_lags_m, 'acf_d': acf_d, 'gamma_d': gamma_d,
        'ping_lags': ping_lags, 'acf_p': acf_p, 'gamma_p': gamma_p,
        'dz': dz, 'dt_s': dt_s,
        'len_d_e': one_over_e_length(acf_d, depth_lags_m),
        'len_d_fit': exp_fit_length(acf_d, depth_lags_m),
        'range_d': variogram_range(gamma_d, depth_lags_m),
        'len_p_e': one_over_e_length(acf_p, ping_lags),
        'len_p_fit': exp_fit_length(acf_p, ping_lags),
        'range_p': variogram_range(gamma_p, ping_lags),
    }


def resample_to_grid(x, y, grid):
    return np.interp(grid, x, y, left=y[0], right=np.nan)


def mean_band(curves):
    arr = np.vstack(curves)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr-dir', type=Path,
                    default=PROJECT_ROOT / 'Datasets' / 'echopype_zarr')
    ap.add_argument('--n', type=int, default=24, help='number of files to sample')
    ap.add_argument('--output-dir', type=Path,
                    default=PROJECT_ROOT / 'Figures' / 'spatial_correlation')
    args = ap.parse_args()

    apply_thesis_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stores = sorted(args.zarr_dir.glob('*.zarr'))
    if not stores:
        raise SystemExit(f'No zarr stores under {args.zarr_dir}')
    idx = np.linspace(0, len(stores) - 1, min(args.n, len(stores))).round().astype(int)
    sample = [stores[i] for i in sorted(set(idx))]
    print(f'Sampling {len(sample)} of {len(stores)} stores')

    depth_grid = np.arange(0.0, DEPTH_LAG_MAX_M + DEPTH_GRID_STEP_M, DEPTH_GRID_STEP_M)
    agg = {f: {'acf_d': [], 'gamma_d': [], 'acf_p': [], 'gamma_p': [], 'rows': []}
           for f in FREQS}

    for f_name, f_hz in FREQS.items():
        for sp in sample:
            try:
                r = process_file(sp, f_hz)
            except Exception as e:
                print(f'  skip {sp.name} @ {f_name}: {e}')
                continue
            if r is None:
                continue
            agg[f_name]['acf_d'].append(resample_to_grid(r['depth_lags_m'], r['acf_d'], depth_grid))
            agg[f_name]['gamma_d'].append(resample_to_grid(r['depth_lags_m'], r['gamma_d'], depth_grid))
            agg[f_name]['acf_p'].append(r['acf_p'])
            agg[f_name]['gamma_p'].append(r['gamma_p'])
            agg[f_name]['rows'].append({'file': sp.name, **{
                k: r[k] for k in ('len_d_e', 'len_d_fit', 'range_d',
                                   'len_p_e', 'len_p_fit', 'range_p', 'dz', 'dt_s')}})
            print(f'  {sp.name} @ {f_name}: '
                  f"depth 1/e={r['len_d_e']:.2f} m, ping 1/e={r['len_p_e']:.1f} pings")

    # Common ping-lag length
    ping_lags = np.arange(PING_LAG_MAX + 1, dtype=np.float64)

    def stack_ping(curves):
        L = min(len(c) for c in curves)
        return np.vstack([c[:L] for c in curves]), ping_lags[:L]

    # Aggregate mean curves per frequency (used for both the figure and the table)
    mc = {}
    for f_name in FREQS:
        if not agg[f_name]['acf_d']:
            continue
        ap_arr, plags = stack_ping(agg[f_name]['acf_p'])
        vp_arr, _ = stack_ping(agg[f_name]['gamma_p'])
        mc[f_name] = {
            'acf_d': mean_band(agg[f_name]['acf_d']),
            'gamma_d': mean_band(agg[f_name]['gamma_d']),
            'plags': plags,
            'acf_p': (np.nanmean(ap_arr, 0), np.nanstd(ap_arr, 0)),
            'gamma_p': (np.nanmean(vp_arr, 0), np.nanstd(vp_arr, 0)),
            'n': len(agg[f_name]['acf_d']),
        }

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_FULL)
    (ax_ad, ax_ap), (ax_vd, ax_vp) = axes

    for f_name, c in mc.items():
        col = FREQ_COLORS[f_name]
        plags = c['plags']
        m, s = c['acf_d']
        ax_ad.plot(depth_grid, m, color=col, label=f_name)
        ax_ad.fill_between(depth_grid, m - s, m + s, color=col, alpha=0.18)
        m, s = c['acf_p']
        ax_ap.plot(plags, m, color=col, label=f_name)
        ax_ap.fill_between(plags, m - s, m + s, color=col, alpha=0.18)
        m, s = c['gamma_d']
        ax_vd.plot(depth_grid, m, color=col, label=f_name)
        ax_vd.fill_between(depth_grid, m - s, m + s, color=col, alpha=0.18)
        m, s = c['gamma_p']
        ax_vp.plot(plags, m, color=col, label=f_name)
        ax_vp.fill_between(plags, m - s, m + s, color=col, alpha=0.18)

    for ax in (ax_ad, ax_ap):
        ax.axhline(1.0 / np.e, color='grey', ls='--', lw=1.5)
        ax.set_ylabel('Autocorrelation')
        ax.set_ylim(-0.1, 1.02)
    ax_ad.text(depth_grid[-1], 1.0 / np.e + 0.02, '1/e', color='grey',
               ha='right', va='bottom', fontsize=14)
    ax_ad.set_title('Depth autocorrelation')
    ax_ap.set_title('Ping autocorrelation')
    ax_vd.set_title('Depth variogram')
    ax_vp.set_title('Ping variogram')
    for ax in (ax_ad, ax_vd):
        ax.set_xlabel('Depth lag (m)')
    for ax in (ax_ap, ax_vp):
        ax.set_xlabel('Ping lag (pings)')
    for ax in (ax_vd, ax_vp):
        ax.set_ylabel('Semivariance')
    legend_handles = [
        Line2D([0], [0], color=FREQ_COLORS[f], lw=2.0, label=f) for f in FREQS
    ]
    legend_handles.append(
        Patch(facecolor='grey', alpha=0.18, label='$\\pm$1 SD across crossings')
    )
    fig.suptitle('Spatial correlation in normalised echograms')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.legend(handles=legend_handles, loc='center left',
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    out_fig = args.output_dir / 'autocorr_variogram.pdf'
    savefig(fig, out_fig)
    savefig(fig, args.output_dir / 'autocorr_variogram.png')
    plt.close(fig)
    print(f'Wrote {out_fig}')

    # Per-file CSV + aggregate LaTeX table
    import csv
    csv_path = args.output_dir / 'correlation_lengths.csv'
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['frequency', 'file', 'len_d_e_m', 'len_d_fit_m', 'range_d_m',
                    'len_p_e_pings', 'len_p_fit_pings', 'range_p_pings', 'dz_m', 'dt_s'])
        for f_name in FREQS:
            for row in agg[f_name]['rows']:
                w.writerow([f_name, row['file'],
                            f"{row['len_d_e']:.3f}", f"{row['len_d_fit']:.3f}",
                            f"{row['range_d']:.3f}", f"{row['len_p_e']:.2f}",
                            f"{row['len_p_fit']:.2f}", f"{row['range_p']:.2f}",
                            f"{row['dz']:.4f}", f"{row['dt_s']:.3f}"])
    print(f'Wrote {csv_path}')

    # Table values are read off the aggregate mean curves, not from per-file
    # scalars: per-file 1/e estimates are noisy and become censored (NaN) for the
    # most strongly correlated crossings, which would bias a per-file average.
    def fmt_censored(value, window, fmt):
        if not np.isfinite(value):
            return f'$>{fmt.format(window)}$'
        return fmt.format(value)

    def table_cell(f_name, kind, fmt):
        if f_name not in mc:
            return '--'
        c = mc[f_name]
        if kind == 'len_d_e':
            return fmt_censored(one_over_e_length(c['acf_d'][0], depth_grid),
                                depth_grid[-1], fmt)
        if kind == 'len_d_fit':
            return fmt.format(exp_fit_length(c['acf_d'][0], depth_grid))
        if kind == 'range_d':
            return fmt.format(variogram_range(c['gamma_d'][0], depth_grid))
        if kind == 'len_p_e':
            return fmt_censored(one_over_e_length(c['acf_p'][0], c['plags']),
                                c['plags'][-1], fmt)
        if kind == 'len_p_fit':
            return fmt.format(exp_fit_length(c['acf_p'][0], c['plags']))
        if kind == 'range_p':
            return fmt.format(variogram_range(c['gamma_p'][0], c['plags']))
        return '--'

    median_dt = np.nanmedian([r['dt_s'] for f in FREQS for r in agg[f]['rows']])
    n_38 = mc.get('38 kHz', {}).get('n', 0)
    n_200 = mc.get('200 kHz', {}).get('n', 0)

    rows_spec = [
        ('Depth corr. length (1/e) [m]', 'len_d_e', '{:.1f}'),
        ('Depth corr. length (fit) [m]', 'len_d_fit', '{:.1f}'),
        ('Ping corr. length (1/e) [pings]', 'len_p_e', '{:.0f}'),
        ('Ping corr. length (fit) [pings]', 'len_p_fit', '{:.0f}'),
        ('Ping variogram range [pings]', 'range_p', '{:.0f}'),
    ]
    lines = [
        '% Auto-generated by code/data_visualization/analyze_spatial_correlation.py',
        '\\begin{table}[ht!]',
        '\\centering',
        '\\caption[Echogram spatial correlation lengths]{Directional spatial '
        'correlation lengths, read from the mean autocorrelation and semivariogram '
        f'averaged over {n_38} (38\\,kHz) and {n_200} (200\\,kHz) sampled crossings. '
        'Depth lengths are in metres; ping lengths are in pings, with a median ping '
        f'interval of {median_dt:.3f}\\,s. The 1/e length is the lag at which the '
        'autocorrelation falls to $1/e$ (reported as a lower bound when this lag '
        'exceeds the analysis window); the fit length is the decay parameter of an '
        'exponential model $\\rho(h)=e^{-h/\\ell}$; the variogram range is the '
        'range parameter of an exponential semivariogram.}',
        '\\label{tab:spatial-correlation}',
        '\\begin{tabular}{l c c}',
        '\\toprule',
        '\\toprule',
        '\\textbf{Quantity} & \\textbf{38 kHz} & \\textbf{200 kHz} \\\\',
        '\\midrule',
    ]
    for label, key, fmt in rows_spec:
        lines.append(f'{label} & {table_cell("38 kHz", key, fmt)} '
                     f'& {table_cell("200 kHz", key, fmt)} \\\\[1.0ex]')
    lines += ['\\bottomrule', '\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    tex_path = PROJECT_ROOT / 'Latex' / 'Tables' / 'spatial_correlation.tex'
    tex_path.write_text('\n'.join(lines))
    print(f'Wrote {tex_path}')


if __name__ == '__main__':
    main()
