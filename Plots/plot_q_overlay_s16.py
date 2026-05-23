"""
GW170817 q-marginal overlay across waveforms and spin priors.

Compares six q-marginals on a common grid and prints summary statistics +
ratios to stdout:

  REFERENCE
    LVK GWTC-1                 IMRPhenomPv2_NRTidal lowSpin (precessing+tides)

  PRECESSING (this work, no tides)
    s07 IMRPhenomPv2           precessing, low-spin uniform-magnitude prior

  ALIGNED-SPIN, default p(s_z) (s07 baseline_lvkbounds)
    s07 IMRPhenomD_NRTidalv2   uniform p(s_z) on [-0.05, 0.05] (square)
    s07 IMRPhenomXAS_NRTidalv3 uniform p(s_z) on [-0.05, 0.05] (square)

  ALIGNED-SPIN, LVK ball p(s_z) (s16 q05/q06 with --lvk-spin-ball)
    s16 IMRPhenomD_NRTidalv2   p(s_z) ∝ (chi_max^2 - s_z^2) (parabolic ball)
    s16 IMRPhenomXAS_NRTidalv3 p(s_z) ∝ (chi_max^2 - s_z^2) (parabolic ball)

Two panels:
  (a) Weighted step-histogram overlay of P(q).
  (b) Ratio P_run(q) / P_LVK(q) on a log y-axis, isolating where each run
      under- or over-shoots the LVK reference.

Stdout: per-run table of {median q, 68% HPD, 95% HPD, P(q>0.85), P(q>0.90),
P(q>0.95)} and the ratio of each tail probability to LVK.

Output: Results/gwtc1_phasemarg/plots/q_overlay_s16.{pdf,png}
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import (
    OUT_DIR, COLORS, load_nested_csv, load_gwtc1_gw170817,
)
import numpy as np
import matplotlib.pyplot as plt

# ── run inventory ──────────────────────────────────────────────────────────
RUNS = [
    # (label, color, linestyle, csv_glob)
    ('LVK GWTC-1 (Pv2_NRTidal)',
     COLORS['gwtc'],          '-',
     None),                    # special-case: load from GWTC-1 HDF5
    ('IMRPhenomPv2 (this work, prec.)',
     '#9467bd',               '-',
     'Results/test_suite/s07__gw170817__imrphenompv2__baseline_lvkbounds__seed0000/samples.csv'),
    ('IMR_NRTv2 (s07, default p(s_z))',
     COLORS['imr_baseline'],  '-',
     'Results/test_suite/s07__gw170817__imrphenomd_nrtidalv2__baseline_lvkbounds__seed0000/samples.csv'),
    ('XAS_NRTv3 (s07, default p(s_z))',
     COLORS['flatZ'],         '-',
     'Results/test_suite/s07__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds__seed0000/samples.csv'),
    ('IMR_NRTv2 (s16, ball p(s_z))',
     COLORS['imr_baseline'],  '--',
     'Results/test_suite/s16__gw170817__imrphenomd_nrtidalv2__qtest_spinball__seed0000/PhaseMarg_*.csv'),
    ('XAS_NRTv3 (s16, ball p(s_z))',
     COLORS['flatZ'],         '--',
     'Results/test_suite/s16__gw170817__imrphenomxas_nrtidalv3__qtest_spinball__seed0000/PhaseMarg_*.csv'),
]


def _load_q_w(csv_glob):
    """Return (q_array, normalised_weight_array)."""
    if csv_glob is None:
        lvk = load_gwtc1_gw170817(columns=['q'])
        q = lvk['q'].to_numpy().astype(float)
        w = np.ones_like(q)
        return q, w / w.sum()
    matches = sorted(glob.glob(csv_glob))
    if not matches:
        raise SystemExit(f"  no CSV match for {csv_glob}")
    s = load_nested_csv(matches[0])
    q = s['q'].to_numpy().astype(float)
    w = np.asarray(s.get_weights(), dtype=float)
    return q, w / w.sum()


def _hpd(q, w, level):
    """Sample-derived shortest-window HPD."""
    order = np.argsort(q)
    qs, ws = q[order], w[order]
    cdf = np.cumsum(ws)
    target = level
    n = len(qs)
    j = 0
    best = (np.inf, qs[0], qs[-1])
    for i in range(n):
        while j < n and cdf[j] - (cdf[i-1] if i > 0 else 0.0) < target:
            j += 1
        if j >= n:
            break
        width = qs[j] - qs[i]
        if width < best[0]:
            best = (width, qs[i], qs[j])
    return best[1], best[2]


def _weighted_median(q, w):
    order = np.argsort(q)
    qs, ws = q[order], w[order]
    cdf = np.cumsum(ws)
    return float(np.interp(0.5, cdf, qs))


# ── load all runs ──────────────────────────────────────────────────────────
print("\nLoading runs:")
data = []
for label, col, ls, src in RUNS:
    q, w = _load_q_w(src)
    data.append((label, col, ls, q, w))
    print(f"  {label:42s}  N={len(q):8d}  median q={_weighted_median(q, w):.4f}")

# ── summary table + ratios vs LVK ──────────────────────────────────────────
THRESHOLDS = [0.85, 0.90, 0.95]
print()
header = f"{'Run':<42} {'median':>8} {'68% HPD':>16} {'95% HPD':>16}"
for t in THRESHOLDS:
    header += f"  P(q>{t:.2f})"
print(header)
print('-' * len(header))

# LVK reference tail probabilities for ratio computation
_, _, _, q_lvk, w_lvk = data[0]
lvk_tails = {t: float(w_lvk[q_lvk > t].sum()) for t in THRESHOLDS}

rows = []
for label, col, ls, q, w in data:
    med = _weighted_median(q, w)
    lo68, hi68 = _hpd(q, w, 0.68269)
    lo95, hi95 = _hpd(q, w, 0.95450)
    tails = {t: float(w[q > t].sum()) for t in THRESHOLDS}
    rows.append((label, med, (lo68, hi68), (lo95, hi95), tails))
    line = (f"{label:<42} {med:>8.4f} "
            f"[{lo68:.3f},{hi68:.3f}]  [{lo95:.3f},{hi95:.3f}] ")
    for t in THRESHOLDS:
        line += f"  {tails[t]:.4f}"
    print(line)

# Ratios of tail probabilities vs LVK
print()
print("Tail-probability ratios vs LVK GWTC-1:")
ratio_header = f"{'Run':<42}"
for t in THRESHOLDS:
    ratio_header += f"  P(q>{t})/P_LVK"
print(ratio_header)
print('-' * len(ratio_header))
for label, _, _, _, tails in rows:
    line = f"{label:<42}"
    for t in THRESHOLDS:
        ratio = tails[t] / lvk_tails[t] if lvk_tails[t] > 0 else float('nan')
        line += f"   {ratio:9.3f}"
    print(line)

# ── overlay + ratio plot ───────────────────────────────────────────────────
bins = np.linspace(0.5, 1.0, 51)
centres = 0.5 * (bins[:-1] + bins[1:])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel (a): step-histogram overlay
ax = axes[0]
hist_lvk = np.histogram(q_lvk, bins=bins, weights=w_lvk, density=True)[0]
for label, col, ls, q, w in data:
    counts = np.histogram(q, bins=bins, weights=w, density=True)[0]
    edges = np.r_[bins[0], np.repeat(bins[1:-1], 2), bins[-1]]
    ys    = np.repeat(counts, 2)
    ax.plot(edges, ys, color=col, ls=ls, lw=2.0, label=label)
ax.set_xlabel(r'$q = m_2/m_1$', fontsize=13)
ax.set_ylabel(r'$P(q)$', fontsize=13)
ax.set_title('(a) GW170817 mass-ratio posteriors')
ax.set_xlim(0.5, 1.0); ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=8.5, loc='upper left')

# Panel (b): ratio vs LVK on log-y
ax = axes[1]
hist_lvk_safe = np.where(hist_lvk > 1e-6, hist_lvk, np.nan)
for label, col, ls, q, w in data[1:]:
    counts = np.histogram(q, bins=bins, weights=w, density=True)[0]
    ratio  = counts / hist_lvk_safe
    edges  = np.r_[bins[0], np.repeat(bins[1:-1], 2), bins[-1]]
    ys     = np.repeat(ratio, 2)
    ax.plot(edges, ys, color=col, ls=ls, lw=2.0, label=label)
ax.axhline(1.0, color='k', lw=0.8, ls=':')
ax.set_yscale('log')
ax.set_xlabel(r'$q = m_2/m_1$', fontsize=13)
ax.set_ylabel(r'$P_\mathrm{run}(q) / P_\mathrm{LVK}(q)$', fontsize=13)
ax.set_title('(b) Ratio to LVK GWTC-1 posterior')
ax.set_xlim(0.5, 1.0)
ax.legend(frameon=False, fontsize=8.5, loc='upper left')

for a in axes:
    for sp in a.spines.values():
        sp.set_edgecolor('black'); sp.set_linewidth(1.2)

fig.tight_layout()
out = os.path.join(OUT_DIR, 'q_overlay_s16')
plt.savefig(f'{out}.pdf', bbox_inches='tight')
plt.savefig(f'{out}.png', dpi=150, bbox_inches='tight')
print(f"\n  -> Saved {out}.pdf / .png")
plt.close(fig)
