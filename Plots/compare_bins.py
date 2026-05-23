"""
Bin-width + KDE comparison PDFs for Figures 2, 3b, and 4.

Each PDF shows several rows:
  - Histogram variants at different bin counts
  - KDE variants at different bandwidths (Silverman, and scaled versions)

Outputs in Results/gwtc1_phasemarg/plots/:
  bin_comparison_fig2.pdf
  bin_comparison_fig3b.pdf
  bin_comparison_fig4.pdf
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import (OUT_DIR, COLORS, load_nested_csv, compute_hpd_samples)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def hist_panel(ax, runs, n_bins, xlim, planck_shoes=True):
    bins = np.linspace(xlim[0], xlim[1], n_bins + 1)
    for x, w, label, color, ls in runs:
        counts, _ = np.histogram(x, bins=bins, weights=w, density=True)
        xs = np.r_[bins[0], np.repeat(bins[1:-1], 2), bins[-1]]
        ys = np.r_[np.repeat(counts, 2)]
        ax.plot(xs, ys, color=color, lw=1.8, ls=ls, label=label)
    _add_bands(ax, xlim, planck_shoes)
    ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=9)
    ax.set_ylabel(r'$P(H_0)$', fontsize=9)


def kde_panel(ax, runs, xlim, bw_factor=1.0, planck_shoes=True):
    x_eval = np.linspace(xlim[0], xlim[1], 1000)
    for x, w, label, color, ls in runs:
        w = w / w.sum()
        kde = gaussian_kde(x, weights=w)
        kde.set_bandwidth(kde.factor * bw_factor)
        pdf = kde(x_eval)
        _trapz = getattr(np, 'trapezoid', np.trapz) if hasattr(np, 'trapz') else np.trapezoid
        pdf /= _trapz(pdf, x_eval)
        ax.plot(x_eval, pdf, color=color, lw=1.8, ls=ls, label=label)
    _add_bands(ax, xlim, planck_shoes)
    ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=9)
    ax.set_ylabel(r'$P(H_0)$', fontsize=9)


def _add_bands(ax, xlim, planck_shoes):
    if planck_shoes:
        ax.axvspan(65.7, 68.2,
                   color=COLORS['planck_outer'], alpha=0.3, zorder=0)
        ax.axvspan(66.93 - 0.62, 66.93 + 0.62,
                   color=COLORS['planck_inner'], alpha=0.3, zorder=0,
                   label='Planck')
        ax.axvspan(69.76, 76.72,
                   color=COLORS['shoes_outer'], alpha=0.3, zorder=0)
        ax.axvspan(73.24 - 1.74, 73.24 + 1.74,
                   color=COLORS['shoes_inner'], alpha=0.3, zorder=0,
                   label='SH0ES')
    ax.set_xlim(xlim); ax.set_ylim(bottom=0)
    for sp in ax.spines.values():
        sp.set_edgecolor('black'); sp.set_linewidth(1.2)


def _legend(ax, fontsize=7):
    ax.legend(frameon=False, fontsize=fontsize, loc='upper right')


def load_h0_w(csv):
    if 'reweighted' in csv.lower() or 'reweight' in os.path.basename(csv).lower():
        df = pd.read_csv(csv, low_memory=False)
        x = df['H_0'].to_numpy().astype(float)
        w = df['weight'].to_numpy().astype(float)
    else:
        s = load_nested_csv(csv)
        x = s['H_0'].to_numpy().astype(float)
        w = np.asarray(s.get_weights(), dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return x[ok], w[ok] / w[ok].sum()


# ===========================================================================
# Figure 2 — data
# ===========================================================================
print("=== Figure 2 data ===")
XAS = 'Results/test_suite/s14__gw170817__imrphenomxas_nrtidalv3'
S18 = 'Results/test_suite/s18__gw170817__imrphenomxas_nrtidalv3__baseline'

_ab_raw = pd.read_csv('Figure1.csv')['H0_samples'].to_numpy().astype(float)
_ab_w   = np.ones(len(_ab_raw)) / len(_ab_raw)

runs_a = [(_ab_raw, _ab_w, r'Abbott+2017', '0.45', '-')]
for csv, label, color in [
    (f'{XAS}__baseline__seed0000/samples.csv',
     r'Baseline ($\pi(d_L)\propto d_L^2$)', COLORS['imr_baseline']),
    (f'{XAS}__flatz__seed0000/samples.csv',
     r'Flat-in-$z$ (direct)',               COLORS['flatZ']),
    (f'{XAS}__reweighted_flatz__seed0000/samples.csv',
     r'Flat-in-$z$ (reweighted)',           COLORS['reweighted']),
    (f'{XAS}__vp250__seed0000/samples.csv',
     r'$\sigma_{v_p}=250$',                 COLORS['vp250']),
]:
    x, w = load_h0_w(csv)
    runs_a.append((x, w, label, color, '-'))

runs_b = []
for csv, label, color in [
    (f'{S18}__vpmean215__seed0000/samples.csv',
     r'$\langle v_p\rangle=215$', '#e07b39'),
    (f'{S18}__vpmean310__seed0000/samples.csv',
     r'$\langle v_p\rangle=310$ (baseline)', COLORS['imr_baseline']),
    (f'{S18}__vpmean405__seed0000/samples.csv',
     r'$\langle v_p\rangle=405$', '#5b78b5'),
]:
    x, w = load_h0_w(csv)
    runs_b.append((x, w, label, color, '-'))

XLIM2 = (40, 180)
# Rows: hist 75, hist 85, hist 80, KDE Silverman, KDE ×0.6, KDE ×0.4
row_specs_2 = [
    ('hist', 75,  '(a) Histogram  bins=75'),
    ('hist', 85,  '(a) Histogram  bins=85'),
    ('hist', 80,  '(a) Histogram  bins=80'),
    ('kde',  1.0, '(a) KDE  bw=Silverman'),
    ('kde',  0.6, '(a) KDE  bw=×0.6'),
    ('kde',  0.4, '(a) KDE  bw=×0.4'),
]

fig2, axes2 = plt.subplots(len(row_specs_2), 2,
                            figsize=(14, 4.0 * len(row_specs_2)),
                            sharey=False)
for row, (kind, param, title_a) in enumerate(row_specs_2):
    ax_a, ax_b = axes2[row]
    title_b = title_a.replace('(a)', '(b)')
    if kind == 'hist':
        hist_panel(ax_a, runs_a, param, XLIM2)
        hist_panel(ax_b, runs_b, param, XLIM2)
    else:
        kde_panel(ax_a, runs_a, XLIM2, bw_factor=param)
        kde_panel(ax_b, runs_b, XLIM2, bw_factor=param)
    ax_a.set_title(title_a, fontsize=9, loc='left')
    ax_b.set_title(title_b, fontsize=9, loc='left')
    _legend(ax_a); _legend(ax_b)

fig2.suptitle('Figure 2 — histogram vs KDE comparison', fontsize=12)
fig2.tight_layout()
out2 = os.path.join(OUT_DIR, 'bin_comparison_fig2.pdf')
fig2.savefig(out2, bbox_inches='tight')
print(f'  -> {out2}')
plt.close(fig2)


# ===========================================================================
# Figure 3b — data
# ===========================================================================
print("=== Figure 3b data ===")
CSV_A    = 'Results/test_suite/s10__gw170817__imrphenomd_nrtidalv2__flatz__dL30-75__refGWTC1__seed0000/samples.csv'
CSV_B    = 'Results/test_suite/s10__gw170817__imrphenomd_nrtidalv2__flatz__dL10-30__refGWTC1__seed0000/samples.csv'
CSV_FULL = 'Results/test_suite/s10__gw170817__imrphenomd_nrtidalv2__flatz__dL10-75__refModeB__seed0000/samples.csv'

MODE_A_COL   = '#E07B00'
MODE_B_COL   = '#2C7FB8'
COMBINED_COL = '#444444'

def _load_h0(csv):
    s = load_nested_csv(csv)
    h0 = s['H_0'].to_numpy().astype(float)
    w  = np.asarray(s.get_weights(), dtype=float)
    return h0, w / w.sum()

h0A, wA = _load_h0(CSV_A)
h0B, wB = _load_h0(CSV_B)
h0F, wF = _load_h0(CSV_FULL)

runs_3b = [
    (h0A, wA, 'Mode A',       MODE_A_COL,   '-'),
    (h0B, wB, 'Mode B',       MODE_B_COL,   '-'),
    (h0F, wF, 'Unrestricted', COMBINED_COL, '-'),
]

XLIM3b = (40, 230)
row_specs_3b = [
    ('hist', 70,  'bins=70'),
    ('hist', 75,  'bins=75'),
    ('hist', 80,  'bins=80'),
    ('kde',  1.0, 'KDE  bw=Silverman'),
    ('kde',  0.6, 'KDE  bw=×0.6'),
    ('kde',  0.4, 'KDE  bw=×0.4'),
]

fig3b, axes3b = plt.subplots(len(row_specs_3b), 1,
                              figsize=(7, 4.0 * len(row_specs_3b)))
for row, (kind, param, title) in enumerate(row_specs_3b):
    ax = axes3b[row]
    if kind == 'hist':
        hist_panel(ax, runs_3b, param, XLIM3b)
    else:
        kde_panel(ax, runs_3b, XLIM3b, bw_factor=param)
    ax.set_title(f'Figure 3b  {title}', fontsize=9, loc='left')
    _legend(ax, fontsize=8)

fig3b.tight_layout()
out3b = os.path.join(OUT_DIR, 'bin_comparison_fig3b.pdf')
fig3b.savefig(out3b, bbox_inches='tight')
print(f'  -> {out3b}')
plt.close(fig3b)


# ===========================================================================
# Figure 4 — data
# ===========================================================================
print("=== Figure 4 data ===")
CSV_XAS = 'Results/test_suite/s07__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds__seed0000/samples.csv'
CSV_TF2 = 'Results/test_suite/s07__gw170817__taylorf2__baseline_lvkbounds__seed0000/samples.csv'

ab4   = pd.read_csv('Figure1.csv')['H0_samples'].to_numpy()
ab4_w = np.ones(len(ab4)) / len(ab4)

xas_s  = load_nested_csv(CSV_XAS)
xas_h0 = xas_s['H_0'].to_numpy().astype(float)
xas_w  = np.asarray(xas_s.get_weights(), dtype=float); xas_w /= xas_w.sum()

tf2_s  = load_nested_csv(CSV_TF2)
tf2_h0 = tf2_s['H_0'].to_numpy().astype(float)
tf2_w  = np.asarray(tf2_s.get_weights(), dtype=float); tf2_w /= tf2_w.sum()

runs_4 = [
    (ab4,    ab4_w, r'Abbott+2017',             '0.45',                    '-'),
    (xas_h0, xas_w, r'this work (IMRX)',         COLORS['imr_baseline'],    '-'),
    (tf2_h0, tf2_w, r'this work (TaylorF2)',     COLORS['tf2_baseline'],    '-'),
]

XLIM4 = (40, 180)
row_specs_4 = [
    ('hist', 70,  'bins=70'),
    ('hist', 80,  'bins=80'),
    ('hist', 90,  'bins=90'),
    ('kde',  1.0, 'KDE  bw=Silverman'),
    ('kde',  0.6, 'KDE  bw=×0.6'),
    ('kde',  0.4, 'KDE  bw=×0.4'),
]

fig4, axes4 = plt.subplots(len(row_specs_4), 1,
                            figsize=(7, 4.0 * len(row_specs_4)))
for row, (kind, param, title) in enumerate(row_specs_4):
    ax = axes4[row]
    if kind == 'hist':
        hist_panel(ax, runs_4, param, XLIM4)
    else:
        kde_panel(ax, runs_4, XLIM4, bw_factor=param)
    ax.set_title(f'Figure 4  {title}', fontsize=9, loc='left')
    _legend(ax, fontsize=8)

fig4.tight_layout()
out4 = os.path.join(OUT_DIR, 'bin_comparison_fig4.pdf')
fig4.savefig(out4, bbox_inches='tight')
print(f'  -> {out4}')
plt.close(fig4)

print("\nDone.")
