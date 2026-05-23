"""
GW170817 mass-ratio (q) prior-sensitivity diagnostic.

NOTE: the s07 baseline_lvkbounds runs already apply the (M_c, q) -> (m1, m2)
Jacobian inside the launcher (GW170817_heterodyned_1.py, search log_jacobian),
so the project's *effective* q-prior already matches LVK lowSpin
(uniform-in-(m1, m2) on [0.87, 1.74]^2 with m2 <= m1). The
prior_comparison.csv produced by sH was generated against the *naive*
project prior (uniform-in-q), which the production sampler does not use;
panel (a) is therefore a "what-if" comparison rather than a real prior
mismatch. The dominant remaining axis between this work and LVK is
aligned-spin vs precessing waveforms; the secondary axis is the spin-prior
shape (square vs ball-projection), addressed by --lvk-spin-ball in the
launcher and the s16 q05/q06 runs.

Three panels:
  (a) Naive project vs LVK-equivalent q priors (from sH prior_comparison.csv).
  (b) Posterior q marginals — IMRPhenomD_NRTidalv2 baseline (this work),
      IMRPhenomXAS_NRTidalv3 baseline (this work), GWTC-1 LVK reference.
  (c) Same posteriors after post-hoc reweighting to the LVK-equivalent
      q prior — illustrates what such a reweighting would do if the project
      did sample uniform-in-q. Since it doesn't, this panel is for context
      only; the proper "correction" is to run s16 q05/q06.

Output: Results/gwtc1_phasemarg/plots/q_prior_sensitivity.{pdf,png}
Reads:
  Results/test_suite/sH__gw170817__prior_only_q__seed0000/prior_comparison.csv
  Results/test_suite/s07__gw170817__imrphenomd_nrtidalv2__baseline_lvkbounds__seed0000/samples.csv
  Results/test_suite/s07__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds__seed0000/samples.csv
  Results/GW170817_GWTC-1.hdf5 (via _plot_utils.load_gwtc1_gw170817)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import (
    OUT_DIR, COLORS, load_nested_csv, load_gwtc1_gw170817,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

PRIOR_CSV = 'Results/test_suite/sH__gw170817__prior_only_q__seed0000/prior_comparison.csv'
CSV_NRTV2 = 'Results/test_suite/s07__gw170817__imrphenomd_nrtidalv2__baseline_lvkbounds__seed0000/samples.csv'
CSV_XAS   = 'Results/test_suite/s07__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds__seed0000/samples.csv'

prior_df = pd.read_csv(PRIOR_CSV)
qg       = prior_df['q'].to_numpy()
proj_pdf = prior_df['project_pdf'].to_numpy().astype(float)
lvk_pdf  = prior_df['lvk_pdf'].to_numpy().astype(float)

proj_interp = interp1d(qg, proj_pdf, bounds_error=False, fill_value=0.0)
lvk_interp  = interp1d(qg, lvk_pdf,  bounds_error=False, fill_value=0.0)

def _post_q_w(csv):
    s = load_nested_csv(csv)
    q = s['q'].to_numpy().astype(float)
    w = np.asarray(s.get_weights(), dtype=float)
    w = w / w.sum()
    return q, w

q_imr,  w_imr  = _post_q_w(CSV_NRTV2)
q_xas,  w_xas  = _post_q_w(CSV_XAS)

lvk = load_gwtc1_gw170817(columns=['q'])
q_lvk = lvk['q'].to_numpy().astype(float)
w_lvk = np.ones_like(q_lvk) / len(q_lvk)

def _hist(q, w, bins):
    counts, _ = np.histogram(q, bins=bins, weights=w, density=True)
    return counts

def _step(centres, counts):
    bins_edge = np.r_[centres[0] - (centres[1]-centres[0])/2,
                      0.5 * (centres[:-1] + centres[1:]),
                      centres[-1] + (centres[-1]-centres[-2])/2]
    x = np.r_[bins_edge[0], np.repeat(bins_edge[1:-1], 2), bins_edge[-1]]
    y = np.r_[np.repeat(counts, 2)]
    return x, y

bins = np.linspace(0.5, 1.0, 51)
centres = 0.5 * (bins[:-1] + bins[1:])

# Reweight our posteriors to LVK-equivalent prior:
#   target_q_prior(q) / sampling_q_prior(q) = lvk_pdf(q) / project_pdf(q)
# Then renormalise.
def _reweight_to_lvk(q, w):
    factor = lvk_interp(q) / np.where(proj_interp(q) > 0, proj_interp(q), 1.0)
    w_new = w * factor
    return w_new / w_new.sum()

w_imr_rw = _reweight_to_lvk(q_imr, w_imr)
w_xas_rw = _reweight_to_lvk(q_xas, w_xas)

P_GT_95 = lambda q, w: float(w[q > 0.95].sum())
print(f"P(q>0.95):")
print(f"  Project prior      : {(proj_pdf[qg>0.95] / proj_pdf.sum() * len(proj_pdf) * (qg[1]-qg[0])).sum():.4f}")
print(f"  LVK-equivalent prior: {(lvk_pdf[qg>0.95] / lvk_pdf.sum() * len(lvk_pdf) * (qg[1]-qg[0])).sum():.4f}")
print(f"  IMR posterior      : {P_GT_95(q_imr, w_imr):.4f}")
print(f"  IMR posterior (reweighted to LVK prior): {P_GT_95(q_imr, w_imr_rw):.4f}")
print(f"  XAS posterior      : {P_GT_95(q_xas, w_xas):.4f}")
print(f"  XAS posterior (reweighted to LVK prior): {P_GT_95(q_xas, w_xas_rw):.4f}")
print(f"  GWTC-1 reference   : {P_GT_95(q_lvk, w_lvk):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

# Panel (a): priors
ax = axes[0]
ax.plot(qg, proj_pdf, color=COLORS['imr_baseline'], lw=2.0,
        label='project prior (uniform in $q$)')
ax.plot(qg, lvk_pdf,  color=COLORS['gwtc'],         lw=2.0,
        label='LVK-equivalent prior')
ax.set_xlabel(r'$q = m_2/m_1$', fontsize=13)
ax.set_ylabel(r'$\pi(q)$', fontsize=13)
ax.set_title('(a) Mass-ratio priors')
ax.set_xlim(0.5, 1.0)
ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=10, loc='upper right')

# Panel (b): posteriors as sampled
ax = axes[1]
for q, w, lab, col in [
    (q_xas, w_xas, r'XAS\_NRTv3 (this work)',  COLORS['flatZ']),
    (q_imr, w_imr, r'IMR\_NRTv2 (this work)',  COLORS['imr_baseline']),
    (q_lvk, w_lvk, 'LVK GWTC-1',                COLORS['gwtc']),
]:
    counts = _hist(q, w, bins)
    x, y = _step(centres, counts)
    ax.plot(x, y, color=col, lw=2.0, label=lab)
ax.set_xlabel(r'$q$', fontsize=13)
ax.set_ylabel(r'$P(q)$', fontsize=13)
ax.set_title(r'(b) Posterior $q$ — as sampled')
ax.set_xlim(0.5, 1.0); ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=10, loc='upper left')

# Panel (c): our posteriors reweighted to LVK-equivalent prior
ax = axes[2]
for q, w, lab, col in [
    (q_xas, w_xas_rw, r'XAS\_NRTv3 (reweighted)', COLORS['flatZ']),
    (q_imr, w_imr_rw, r'IMR\_NRTv2 (reweighted)', COLORS['imr_baseline']),
    (q_lvk, w_lvk,    'LVK GWTC-1',                COLORS['gwtc']),
]:
    counts = _hist(q, w, bins)
    x, y = _step(centres, counts)
    ax.plot(x, y, color=col, lw=2.0, label=lab)
ax.set_xlabel(r'$q$', fontsize=13)
ax.set_ylabel(r'$P(q)$', fontsize=13)
ax.set_title(r'(c) Posterior $q$ — reweighted to LVK prior')
ax.set_xlim(0.5, 1.0); ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=10, loc='upper left')

for a in axes:
    for sp in a.spines.values():
        sp.set_edgecolor('black'); sp.set_linewidth(1.2)

fig.tight_layout()
p = os.path.join(OUT_DIR, 'q_prior_sensitivity')
plt.savefig(f'{p}.pdf', bbox_inches='tight')
plt.savefig(f'{p}.png', dpi=150, bbox_inches='tight')
print(f"  -> Saved {p}.pdf / .png")
plt.close(fig)
