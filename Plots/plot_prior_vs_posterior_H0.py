"""
Prior vs posterior overlay for H_0.
Shows how informative the GW data is by comparing the prior to the posterior.

Output: Results/gwtc1_phasemarg/plots/H0_prior_vs_posterior.{pdf,png}
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *
from scipy.stats import gaussian_kde

RESULTS_PHASEMARG = os.path.join(RESULTS_DIR, 'gwtc1_phasemarg')

IMR_BASELINE = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_baseline.csv')
TF2_BASELINE = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_baseline.csv')

fig, ax = plt.subplots(figsize=(10, 6))
x_eval = np.linspace(20, 250, 500)

# H_0 prior: log-uniform on [20, 250]
# p(H_0) = 1 / (H_0 * ln(250/20))
prior_pdf = 1.0 / (x_eval * np.log(250.0 / 20.0))
ax.plot(x_eval, prior_pdf, 'k--', lw=2, label=r'Prior: log-uniform [20, 250]')


# Posteriors
for csv_path, label, color in [
    (IMR_BASELINE, 'IMRPhenomD (this work)', COLORS['imr_baseline']),
    (TF2_BASELINE, 'TaylorF2 (this work)', COLORS['tf2_baseline']),
]:
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found — skipping")
        continue

    s = load_nested_csv(csv_path)
    h0 = s['H_0'].to_numpy()
    w = np.asarray(s.get_weights())
    w = w / w.sum()

    kde = gaussian_kde(h0, weights=w)
    pdf = kde(x_eval)
    pdf = pdf / np.trapezoid(pdf, x_eval)

    ax.plot(x_eval, pdf, color=color, lw=2, label=label)

# Planck and SHoES
ax.axvspan(66.93 - 0.62, 66.93 + 0.62, color=COLORS['planck_inner'],
           alpha=0.3, zorder=0, label='Planck')
ax.axvspan(73.24 - 1.74, 73.24 + 1.74, color=COLORS['shoes_inner'],
           alpha=0.3, zorder=0, label='SHoES')

from matplotlib.ticker import MultipleLocator
ax.yaxis.set_major_locator(MultipleLocator(0.005))
ax.set_xlim(20, 250)
ax.set_ylim(bottom=0)
ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
ax.set_ylabel(r'$P(H_0)$ (km$^{-1}$ s Mpc)')

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

ax.legend(frameon=False, fontsize=12)
fig.tight_layout()

path = os.path.join(OUT_DIR, 'H0_prior_vs_posterior')
plt.savefig(f'{path}.pdf', bbox_inches='tight')
plt.savefig(f'{path}.png', dpi=150, bbox_inches='tight')
print(f"  -> Saved {path}.pdf / .png")
plt.close(fig)

print("\nDone.")
