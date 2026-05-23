"""
Phase marginalization comparison: analytic (Bessel) vs grid approaches.

Creates a schematic figure illustrating the two approaches to marginalizing
over the coalescence phase:
  1. Analytic (log I_0): exact marginalisation via Bessel function identity
  2. Grid (logsumexp): numerical marginalisation over discrete phase grid

Includes a comparison of the resulting H_0 posteriors from heterodyned
(phase-marginalised) and unheterodyned (phase-sampled) runs.

Reference: GW-JAX-Team/jim PR #57 — grid marginalisation implementation.
See also Thrane & Talbot (2019) for the Bessel function identity.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0e
from matplotlib.gridspec import GridSpec

RESULTS_PHASEMARG = os.path.join(RESULTS_DIR, 'gwtc1_phasemarg')

# ----------------------------------------------------------------------- #
# Panel 1: Schematic of analytic vs grid phase marginalisation
# ----------------------------------------------------------------------- #
fig = plt.figure(figsize=(14, 5))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1.3], wspace=0.35)

# --- Panel A: log-likelihood vs phase_c ---
ax1 = fig.add_subplot(gs[0])
phase_c = np.linspace(0, 2*np.pi, 500)
# Simulated log-likelihood as a function of phase_c (cosine-like)
amp = 3.0
logL_phase = amp * np.cos(phase_c - 1.2) + 10.0

ax1.plot(phase_c, logL_phase, 'k-', lw=2)
ax1.set_xlabel(r'$\phi_c$ (rad)', fontsize=12)
ax1.set_ylabel(r'$\ln\mathcal{L}(\phi_c \mid \theta)$', fontsize=12)
ax1.set_title(r'(a) $\ln\mathcal{L}$ vs coalescence phase', fontsize=11)
ax1.set_xlim(0, 2*np.pi)
ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax1.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

for spine in ax1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

# --- Panel B: Analytic vs grid marginalisation ---
ax2 = fig.add_subplot(gs[1])

# Analytic: log I_0(|A|) where |A| is the complex SNR amplitude
abs_A = np.linspace(0.1, 15, 200)
log_I0_analytic = np.log(i0e(abs_A)) + abs_A  # log(I_0(x)) = log(I_0e(x)) + x

# Grid: logsumexp over N grid points
for N, ls, alpha in [(8, ':', 0.6), (32, '--', 0.7), (128, '-.', 0.8)]:
    grid_phases = np.linspace(0, 2*np.pi, N, endpoint=False)
    # For each |A|, compute logsumexp(|A| * cos(phi_k))
    log_grid = np.array([
        np.log(np.mean(np.exp(a * np.cos(grid_phases) -
                               np.max(a * np.cos(grid_phases))))) +
        np.max(a * np.cos(grid_phases))
        for a in abs_A
    ])
    ax2.plot(abs_A, log_grid, ls=ls, alpha=alpha, lw=1.5,
             label=f'Grid ($N={N}$)')

ax2.plot(abs_A, log_I0_analytic, 'k-', lw=2.5, label=r'Analytic $\ln I_0$')

ax2.set_xlabel(r'$|A|$ (complex SNR amplitude)', fontsize=12)
ax2.set_ylabel(r'$\ln \int \mathcal{L}\, d\phi_c$', fontsize=12)
ax2.set_title('(b) Marginalisation methods', fontsize=11)
ax2.legend(fontsize=8, frameon=False, loc='upper left')

for spine in ax2.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

# --- Panel C: H_0 posterior — phase-marg vs phase-sampled ---
ax3 = fig.add_subplot(gs[2])

HETERO_CSV = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_baseline.csv')
UNHETERO_CSV = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Unheterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1.csv')

from scipy.stats import gaussian_kde

x_eval = np.linspace(20, 250, 500)

for csv_path, label, color, loader in [
    (HETERO_CSV, r'Phase-marg. ($\ln I_0$)', COLORS['imr_baseline'],
     load_nested_csv),
    (UNHETERO_CSV, r'Phase-sampled ($\phi_c$)', COLORS['unhetero_imr'],
     load_nested_csv),
]:
    if not os.path.exists(csv_path):
        continue
    s = loader(csv_path)
    try:
        cols = s.columns.get_level_values(0)
    except AttributeError:
        cols = s.columns
    if 'H_0' not in cols:
        continue

    h0_vals = s['H_0'].to_numpy()
    weights = np.asarray(s.get_weights())
    weights = weights / weights.sum()
    kde = gaussian_kde(h0_vals, weights=weights)
    pdf_vals = kde(x_eval)
    pdf_vals = pdf_vals / np.trapezoid(pdf_vals, x_eval)

    ax3.plot(x_eval, pdf_vals, color=color, lw=2, label=label)

# Planck / SHoES bands
ax3.axvspan(66.93 - 0.62, 66.93 + 0.62, color=COLORS['planck_inner'],
            alpha=0.3, zorder=0, label='Planck')
ax3.axvspan(73.24 - 1.74, 73.24 + 1.74, color=COLORS['shoes_inner'],
            alpha=0.3, zorder=0, label='SHoES')

ax3.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=12)
ax3.set_ylabel(r'$P(H_0)$', fontsize=12)
ax3.set_title(r'(c) $H_0$ posterior consistency', fontsize=11)
ax3.set_xlim(20, 250)
ax3.set_ylim(bottom=0)
ax3.legend(fontsize=10, frameon=False)

for spine in ax3.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

fig.tight_layout()
path = os.path.join(OUT_DIR, 'phase_marginalization_schematic')
plt.savefig(f'{path}.pdf', bbox_inches='tight')
plt.savefig(f'{path}.png', dpi=150, bbox_inches='tight')
print(f"  -> Saved {path}.pdf / .png")
plt.close(fig)

print("\nDone.")
