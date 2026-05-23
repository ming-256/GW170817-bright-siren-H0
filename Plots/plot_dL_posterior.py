"""
d_L marginal posterior comparison across prior variants and waveforms.
Shows how the distance prior drives the H_0 constraint.

Output: Results/gwtc1_phasemarg/plots/dL_posterior.{pdf,png}
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *
from scipy.stats import gaussian_kde, beta as beta_dist

RESULTS_PHASEMARG = os.path.join(RESULTS_DIR, 'gwtc1_phasemarg')

# CSV paths
IMR_BASELINE = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_baseline.csv')
IMR_FLATZ = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_flatZ.csv')
TF2_BASELINE = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_baseline.csv')
TF2_FLATZ = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_flatZ.csv')

# GWTC-1 reference
gwtc = load_gwtc1_gw170817(columns=['d_L'])
gwtc_dL = gwtc['d_L'].to_numpy()

# Collect runs
runs = []

# GWTC-1
runs.append((gwtc_dL, np.ones(len(gwtc_dL)), 'LVK (GWTC-1)', COLORS['gwtc']))

for csv_path, label, color in [
    (IMR_BASELINE, r'IMRPhenomD baseline ($\beta$(3,1))', COLORS['imr_baseline']),
    (TF2_BASELINE, r'TaylorF2 baseline ($\beta$(3,1))', COLORS['tf2_baseline']),
    (IMR_FLATZ, r'IMRPhenomD flat-in-$z$', COLORS['flatZ']),
    (TF2_FLATZ, r'TaylorF2 flat-in-$z$', COLORS['flatZ']),
]:
    if os.path.exists(csv_path):
        s = load_nested_csv(csv_path)
        dL = s['d_L'].to_numpy()
        w = np.asarray(s.get_weights())
        runs.append((dL, w, label, color))
    else:
        print(f"  WARNING: {csv_path} not found — skipping")

if not runs:
    print("  No data found.")
    raise SystemExit(1)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
x_eval = np.linspace(1, 75, 500)

for dL, w, label, color in runs:
    w = w / w.sum()
    kde = gaussian_kde(dL, weights=w)
    pdf = kde(x_eval)
    pdf = pdf / np.trapezoid(pdf, x_eval)
    ax.plot(x_eval, pdf, color=color, lw=2, label=label)

# Show Beta(3,1) prior for reference
d_lo, d_hi = 1.0, 75.0
u = (x_eval - d_lo) / (d_hi - d_lo)
prior_beta = beta_dist.pdf(u, 3, 1) / (d_hi - d_lo)
ax.plot(x_eval, prior_beta, 'k--', lw=1.5, alpha=0.5, label=r'$\beta(3,1)$ prior')

# Flat prior for reference
prior_flat = np.ones_like(x_eval) / (d_hi - d_lo)
ax.plot(x_eval, prior_flat, 'k:', lw=1.5, alpha=0.5, label='Flat prior')

ax.set_xlim(1, 75)
ax.set_ylim(bottom=0)
ax.set_xlabel(r'$d_L$ (Mpc)')
ax.set_ylabel(r'$P(d_L)$ (Mpc$^{-1}$)')

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

ax.legend(frameon=False, fontsize=12)
fig.tight_layout()

path = os.path.join(OUT_DIR, 'dL_posterior')
plt.savefig(f'{path}.pdf', bbox_inches='tight')
plt.savefig(f'{path}.png', dpi=150, bbox_inches='tight')
print(f"  -> Saved {path}.pdf / .png")
plt.close(fig)

print("\nDone.")
