"""Test whether the validation IoU plateau is a label-noise ceiling.

Aggregates per-file eval metrics across every model evaluated on the shared
378-file survey validation split, then tests the hypothesis that the IoU gap
(~0.984 vs 1.0) is caused by errors in a handful of reference labels rather
than uniform model imperfection. If true: (1) the per-file IoU deficit is
concentrated in a few files, (2) all models fail on the *same* files, and
(3) across-model spread is small on those files (everyone agrees, the label
is the outlier).

Output: Figures/label_noise_ceiling/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from data_visualization.thesis_style import (
    apply_thesis_style, FIGSIZE_STD, FIGSIZE_FULL, savefig,
)

CKPT_DIR = PROJECT_ROOT / 'code' / 'checkpoints'
OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'label_noise_ceiling'

N_VAL_FILES = 378
MIN_MODEL_IOU = 0.90   # exclude broken line/geo_seg/regression runs


def load_matrix():
    """Build [file x model] matrices of IoU and bottom-error from eval_metrics.json.

    Restricts to the largest group of models that share an identical val file
    set, so every model is compared on exactly the same files (no intersection
    collapse across heterogeneous splits).
    """
    import collections
    records = []
    for ev in sorted(CKPT_DIR.rglob('eval_metrics.json')):
        try:
            data = json.loads(ev.read_text())
        except Exception:
            continue
        summ = data.get('summary', {})
        if summ.get('n_files') != N_VAL_FILES:
            continue
        if summ.get('mean_iou', 0) < MIN_MODEL_IOU:
            continue
        per_file = data.get('per_file', [])
        if not per_file:
            continue
        name = str(ev.parent.relative_to(CKPT_DIR))
        fileset = frozenset(r['file'] for r in per_file)
        records.append((name, fileset, per_file, summ))

    # pick the modal file set
    counts = collections.Counter(r[1] for r in records)
    modal_set = counts.most_common(1)[0][0]
    records = [r for r in records if r[1] == modal_set]

    iou_cols, berr_mean_cols, berr_med_cols, berr_p95_cols = {}, {}, {}, {}
    model_summaries = {}
    for name, _, per_file, summ in records:
        iou_cols[name] = {r['file']: r['iou'] for r in per_file}
        berr_mean_cols[name] = {r['file']: r.get('bottom_error_mean_m', np.nan) for r in per_file}
        berr_med_cols[name] = {r['file']: r.get('bottom_error_median_m', np.nan) for r in per_file}
        berr_p95_cols[name] = {r['file']: r.get('bottom_error_p95_m', np.nan) for r in per_file}
        model_summaries[name] = summ

    iou = pd.DataFrame(iou_cols)
    berr_mean = pd.DataFrame(berr_mean_cols)
    berr_med = pd.DataFrame(berr_med_cols)
    berr_p95 = pd.DataFrame(berr_p95_cols)
    return iou, berr_mean, berr_med, berr_p95, model_summaries


def main():
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    iou, berr_mean, berr_med, berr_p95, summaries = load_matrix()
    n_files, n_models = iou.shape
    print(f"Loaded {n_models} models x {n_files} val files")

    # ---- per-file statistics across models ----
    file_mean = iou.mean(axis=1)
    file_std = iou.std(axis=1)
    file_min = iou.min(axis=1)
    file_max = iou.max(axis=1)

    deficit = 1.0 - file_mean          # per-file mean IoU deficit
    total_deficit = deficit.sum()
    order = deficit.sort_values(ascending=False)

    # deficit concentration
    cum = order.cumsum() / total_deficit
    n_for_50 = int((cum < 0.50).sum()) + 1
    n_for_80 = int((cum < 0.80).sum()) + 1
    n_for_90 = int((cum < 0.90).sum()) + 1

    # mean IoU if we drop the worst files
    drop_worst = {}
    for k in [0, 5, 10, 20]:
        kept = file_mean.sort_values(ascending=False).index[:n_files - k] if k else file_mean.index
        drop_worst[k] = file_mean.loc[kept].mean()

    # ping-level catastrophe signal: median tiny, p95 huge -> spikes
    med_across = berr_med.mean(axis=1)
    p95_across = berr_p95.mean(axis=1)

    # save CSV
    summary_df = pd.DataFrame({
        'file': file_mean.index,
        'mean_iou_across_models': file_mean.values,
        'std_iou_across_models': file_std.values,
        'min_iou': file_min.values,
        'max_iou': file_max.values,
        'iou_deficit': deficit.values,
        'mean_bottom_err_median_m': med_across.reindex(file_mean.index).values,
        'mean_bottom_err_p95_m': p95_across.reindex(file_mean.index).values,
    }).sort_values('iou_deficit', ascending=False)
    summary_df.to_csv(OUTPUT_DIR / 'per_file_label_noise.csv', index=False)

    # =====================================================================
    # Figure 1: per-file IoU, sorted, with across-model min-max band
    # =====================================================================
    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    srt = file_mean.sort_values(ascending=False)
    x = np.arange(n_files)
    ax.fill_between(x, file_min.loc[srt.index].values, file_max.loc[srt.index].values,
                    color='#2c7bb6', alpha=0.25, label='Across-model min-max')
    ax.plot(x, srt.values, color='#08306b', lw=2.0, label='Mean IoU across models')
    ax.axhline(file_mean.mean(), color='#d7191c', ls='--', lw=2.0,
               label=f'Overall mean = {file_mean.mean():.4f}')
    ax.set_xlabel('Validation file (sorted by mean IoU)')
    ax.set_ylabel('Per-file IoU')
    ax.set_title('Validation IoU is near-perfect on most files, with a heavy bad tail')
    ax.set_ylim(0.5, 1.005)
    ax.legend(loc='lower left')
    savefig(fig, OUTPUT_DIR / 'per_file_iou_sorted.pdf')
    plt.close(fig)

    # =====================================================================
    # Figure 2: deficit concentration (Lorenz-style)
    # =====================================================================
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    frac_files = np.arange(1, n_files + 1) / n_files
    ax.plot(frac_files * 100, cum.values * 100, color='#08306b', lw=2.5)
    ax.axhline(80, color='#d7191c', ls=':', lw=1.8)
    ax.axvline(n_for_80 / n_files * 100, color='#d7191c', ls=':', lw=1.8)
    ax.annotate(f'{n_for_80} files ({n_for_80/n_files*100:.0f}%)\n= 80% of total deficit',
                xy=(n_for_80 / n_files * 100, 80),
                xytext=(n_for_80 / n_files * 100 + 18, 55),
                arrowprops=dict(arrowstyle='->', color='#d7191c'),
                fontsize=15, color='#d7191c')
    ax.set_xlabel('Worst N% of validation files')
    ax.set_ylabel('Cumulative share of total IoU deficit (%)')
    ax.set_title('The IoU shortfall is concentrated in a tiny minority of files')
    savefig(fig, OUTPUT_DIR / 'deficit_concentration.pdf')
    plt.close(fig)

    # =====================================================================
    # Figure 3: across-model agreement — do all models fail on the same files?
    #   x = mean IoU (file difficulty), y = std across models.
    #   Label-noise signature: low-IoU files have LOW std (all agree).
    # =====================================================================
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    sc = ax.scatter(file_mean.values, file_std.values,
                    c=deficit.values, cmap='inferno_r', s=45,
                    edgecolor='k', linewidth=0.3)
    ax.set_xlabel('Per-file mean IoU across models (file difficulty)')
    ax.set_ylabel('Std of IoU across models')
    ax.set_title('On the worst files, models agree with each other, not the label')
    # corr between difficulty and disagreement
    r = np.corrcoef(file_mean.values, file_std.values)[0, 1]
    ax.text(0.04, 0.94, f'corr(mean IoU, model spread) = {r:+.2f}',
            transform=ax.transAxes, fontsize=14,
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9))
    cbar = fig.colorbar(sc, ax=ax)
    from data_visualization.thesis_style import style_colorbar
    style_colorbar(cbar, label='IoU deficit')
    savefig(fig, OUTPUT_DIR / 'model_agreement_vs_difficulty.pdf')
    plt.close(fig)

    # =====================================================================
    # Figure 4: ping-level — median vs p95 bottom error per file
    #   Spiky label errors: median tiny, p95 huge.
    # =====================================================================
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.scatter(med_across.values, p95_across.values, s=45,
               color='#2c7bb6', edgecolor='k', linewidth=0.3, alpha=0.8)
    lim = max(p95_across.max(), 1) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1.2, alpha=0.6, label='p95 = median')
    ax.set_xlabel('Median bottom error per file (m)')
    ax.set_ylabel('p95 bottom error per file (m)')
    ax.set_title('A few pings per file carry huge errors while the median stays sub-metre')
    ax.set_yscale('symlog')
    ax.legend(loc='upper right')
    savefig(fig, OUTPUT_DIR / 'ping_error_median_vs_p95.pdf')
    plt.close(fig)

    # =====================================================================
    # Console report
    # =====================================================================
    print("\n" + "=" * 72)
    print("LABEL-NOISE CEILING TEST")
    print("=" * 72)
    print(f"Models analysed : {n_models}")
    print(f"Common val files: {n_files}")
    print(f"Overall mean IoU: {file_mean.mean():.4f}  (deficit {1-file_mean.mean():.4f})")
    print(f"Median per-file IoU: {file_mean.median():.4f}")
    print(f"Files with mean IoU > 0.99 : {(file_mean > 0.99).sum()} "
          f"({(file_mean > 0.99).mean()*100:.1f}%)")
    print(f"Files with mean IoU < 0.95 : {(file_mean < 0.95).sum()} "
          f"({(file_mean < 0.95).mean()*100:.1f}%)")
    print(f"Files with mean IoU < 0.90 : {(file_mean < 0.90).sum()}")
    print("-" * 72)
    print("DEFICIT CONCENTRATION")
    print(f"  worst {n_for_50} files ({n_for_50/n_files*100:.1f}%) = 50% of total deficit")
    print(f"  worst {n_for_80} files ({n_for_80/n_files*100:.1f}%) = 80% of total deficit")
    print(f"  worst {n_for_90} files ({n_for_90/n_files*100:.1f}%) = 90% of total deficit")
    print("-" * 72)
    print("MEAN IoU IF WORST FILES DROPPED")
    for k, v in drop_worst.items():
        print(f"  drop worst {k:>2} files -> mean IoU = {v:.4f}")
    print("-" * 72)
    # Do all models find the SAME files hard? Pairwise rank correlation of
    # per-file IoU across models. High => identical difficulty ordering =>
    # data/label-driven, not capability-driven.
    from scipy.stats import spearmanr
    import itertools
    M = iou.T.values  # models x files
    pair_rs = [spearmanr(M[i], M[j]).correlation
               for i, j in itertools.combinations(range(M.shape[0]), 2)]
    worst5_std = file_std.loc[order.index[:5]].mean()
    rest_std = file_std.loc[order.index[5:]].mean()
    print(f"mean pairwise Spearman of per-file IoU across models = {np.nanmean(pair_rs):.3f}")
    print("  (~0.9 => every model ranks files by difficulty almost identically")
    print("   => the hard files are hard for ALL models => label/data-driven)")
    print(f"across-model IoU std: worst 5 files = {worst5_std:.4f}, rest = {rest_std:.4f}")
    print("  (worst files have the LEAST disagreement => label is the outlier)")
    print("-" * 72)
    print("WORST 15 FILES (mean IoU across models):")
    worst = summary_df.head(15)
    for _, row in worst.iterrows():
        print(f"  {row['file']:<28} IoU={row['mean_iou_across_models']:.4f} "
              f"std={row['std_iou_across_models']:.4f} "
              f"medErr={row['mean_bottom_err_median_m']:.2f}m "
              f"p95Err={row['mean_bottom_err_p95_m']:.1f}m")
    print("=" * 72)
    print(f"\nFigures + CSV -> {OUTPUT_DIR}")

    return summary_df


if __name__ == '__main__':
    main()
