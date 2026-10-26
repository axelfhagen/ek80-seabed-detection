"""Underfitting, good fit, and overfitting on a 1D regression toy problem.

Three panels fit polynomials of increasing degree to noisy samples of a smooth
target. A low-degree fit underfits (high bias), a moderate degree captures the
trend, and a high-degree fit overfits (passes through the noise, high variance).
A fourth panel shows the corresponding training and validation error against
model complexity, with the train/validation gap widening in the overfit regime.

Output: Figures/model_fit_regimes/
"""

from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from data_visualization.thesis_style import apply_thesis_style, savefig  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / 'Figures' / 'model_fit_regimes'

COL_TRUE = '#444444'
COL_FIT = '#1f77b4'
COL_PTS = '#d95f02'
COL_TRAIN = '#1f77b4'
COL_VAL = '#d95f02'

RNG = np.random.default_rng(7)


def target(x):
    return np.sin(1.6 * np.pi * x)


def make_data(n, noise=0.22):
    x = np.sort(RNG.uniform(0.0, 1.0, n))
    y = target(x) + RNG.normal(0.0, noise, n)
    return x, y


def poly_fit_predict(x_tr, y_tr, x_eval, degree):
    # Polynomial.fit scales the domain to [-1, 1] internally to keep the
    # Vandermonde system well conditioned. At degree near the sample count the
    # fit is an interpolant and warns regardless; that regime is the overfit
    # illustration, so the warning is expected and silenced.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = np.polynomial.Polynomial.fit(x_tr, y_tr, degree)
    return p(x_eval)


def panel_fit(ax, x_tr, y_tr, degree, title, subtitle):
    xs = np.linspace(0.0, 1.0, 400)
    y_pred = poly_fit_predict(x_tr, y_tr, xs, degree)
    ax.plot(xs, target(xs), color=COL_TRUE, lw=2.2, ls='--', label='true function')
    ax.plot(xs, y_pred, color=COL_FIT, lw=2.8, label=f'fit (degree {degree})')
    ax.scatter(x_tr, y_tr, s=55, color=COL_PTS, edgecolor='white', lw=1.0,
               zorder=5, label='training data')
    ax.set_ylim(-1.8, 1.8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel('input $x$')
    ax.set_ylabel('output $y$')
    ax.set_title(subtitle, fontsize=15, color='#555555')
    ax.text(0.5, 1.06, title, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=19)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_thesis_style()

    x_tr, y_tr = make_data(12)
    x_val, y_val = make_data(40)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    panel_fit(axes[0, 0], x_tr, y_tr, 1, 'Underfitting',
              'degree 1: high bias, misses the trend')
    panel_fit(axes[0, 1], x_tr, y_tr, 4, 'Good fit',
              'degree 4: captures the trend, ignores noise')
    panel_fit(axes[1, 0], x_tr, y_tr, 10, 'Overfitting',
              'degree 10: fits the noise, high variance')

    # one shared legend, placed in the fit panels
    axes[0, 0].legend(loc='lower left', framealpha=0.9, fontsize=14)

    # --- complexity vs error panel ---
    ax = axes[1, 1]
    degrees = np.arange(1, 13)
    train_err, val_err = [], []
    for d in degrees:
        train_err.append(np.mean((poly_fit_predict(x_tr, y_tr, x_tr, d) - y_tr) ** 2))
        val_err.append(np.mean((poly_fit_predict(x_tr, y_tr, x_val, d) - y_val) ** 2))
    train_err = np.array(train_err)
    val_err = np.array(val_err)

    ax.plot(degrees, train_err, color=COL_TRAIN, lw=2.8, marker='o', ms=7,
            label='training error')
    ax.plot(degrees, val_err, color=COL_VAL, lw=2.8, marker='s', ms=7,
            label='validation error')
    best = degrees[int(np.argmin(val_err))]
    ax.axvline(best, color='#444444', ls=':', lw=2.0)
    ax.set_yscale('log')
    ax.set_ylim(max(train_err.min() * 0.5, 1e-3), val_err.max() * 2.5)
    ax.set_xlabel('model complexity (polynomial degree)')
    ax.set_ylabel('mean squared error')
    ax.set_title('error vs complexity: train/validation gap', fontsize=15,
                 color='#555555')
    ax.text(0.5, 1.06, 'Bias-variance tradeoff', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=19)
    ax.text(0.04, 0.06, 'underfit', transform=ax.transAxes, fontsize=14,
            color='#555555', ha='left', va='bottom')
    ax.text(0.96, 0.06, 'overfit', transform=ax.transAxes, fontsize=14,
            color='#555555', ha='right', va='bottom')
    ax.legend(loc='upper center', framealpha=0.9, fontsize=14)

    fig.suptitle('Underfitting, good fit, and overfitting', fontsize=22)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    savefig(fig, OUTPUT_DIR / 'model_fit_regimes.pdf')
    plt.close(fig)
    print(f'wrote {OUTPUT_DIR / "model_fit_regimes.pdf"}')


if __name__ == '__main__':
    main()
