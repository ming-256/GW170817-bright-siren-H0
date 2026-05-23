"""
Compute and print a summary table of Bayesian evidence, KL divergence,
and effective sample size for all nested sampling runs.

Outputs:
  - Console table (LaTeX-ready)
  - Results/gwtc1_phasemarg/evidence_table.csv
"""

import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *
from anesthetic import read_chains

RESULTS_PHASEMARG = os.path.join(RESULTS_DIR, 'gwtc1_phasemarg')
OUT_CSV = os.path.join(RESULTS_PHASEMARG, 'evidence_table.csv')

# Auto-discover all nested sampling CSVs (exclude reweighted files)
pattern = os.path.join(RESULTS_PHASEMARG, '*.csv')
all_csvs = sorted(glob.glob(pattern))
nested_csvs = [f for f in all_csvs if 'reweighted' not in os.path.basename(f)]

if not nested_csvs:
    print(f"No nested sampling CSVs found in {RESULTS_PHASEMARG}")
    raise SystemExit(1)

print(f"Found {len(nested_csvs)} nested sampling result(s)\n")

# Header
header = ['Run', 'log Z', 'log Z err', 'D_KL (nats)', 'N_dead', 'N_eff']
rows = []

for csv_path in nested_csvs:
    name = os.path.basename(csv_path).replace('.csv', '')
    # Strip common prefix for readability
    name = name.replace('PhaseMarg_Heterodyned_', 'Het ')
    name = name.replace('PhaseMarg_Unheterodyned_', 'Unhet ')
    name = name.replace('IMRPhenomD_NRTidalv2_local_', 'IMR ')
    name = name.replace('TaylorF2_local_', 'TF2 ')

    try:
        samples = read_chains(csv_path)
    except Exception as e:
        print(f"  SKIP {os.path.basename(csv_path)}: {e}")
        continue

    n_dead = len(samples)

    # Bayesian evidence
    try:
        logZ = samples.logZ()
        logZ_val = float(logZ)
        # Error estimate from anesthetic (if available)
        try:
            logZ_err = float(samples.logZ(nsamples=20).std())
        except Exception:
            logZ_err = float('nan')
    except Exception:
        logZ_val = float('nan')
        logZ_err = float('nan')

    # KL divergence (information gain)
    try:
        D_KL = float(samples.D_KL())
    except Exception:
        D_KL = float('nan')

    # Effective sample size from posterior weights
    try:
        w = np.asarray(samples.get_weights())
        w = w / w.sum()
        n_eff = 1.0 / np.sum(w**2)
    except Exception:
        n_eff = float('nan')

    rows.append([name, logZ_val, logZ_err, D_KL, n_dead, n_eff])
    print(f"  {name}")
    print(f"    log Z = {logZ_val:.2f} +/- {logZ_err:.2f}")
    print(f"    D_KL  = {D_KL:.2f} nats")
    print(f"    N_dead = {n_dead},  N_eff = {n_eff:.0f}")

# Save CSV
import csv
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
print(f"\n-> Saved {OUT_CSV}")

# Print LaTeX table
print("\n% LaTeX table:")
print(r"\begin{tabular}{lrrrr}")
print(r"  \hline")
print(r"  Run & $\ln\mathcal{Z}$ & $D_\mathrm{KL}$ (nats) & $N_\mathrm{dead}$ & $N_\mathrm{eff}$ \\")
print(r"  \hline")
for row in rows:
    name, logZ_val, logZ_err, D_KL, n_dead, n_eff = row
    logZ_str = f"${logZ_val:.1f} \\pm {logZ_err:.1f}$" if not np.isnan(logZ_err) else f"${logZ_val:.1f}$"
    print(f"  {name} & {logZ_str} & ${D_KL:.1f}$ & ${n_dead}$ & ${n_eff:.0f}$ \\\\")
print(r"  \hline")
print(r"\end{tabular}")

print("\nDone.")
