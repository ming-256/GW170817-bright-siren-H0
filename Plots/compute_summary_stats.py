"""
Compute summary statistics for all runs: MAP, median, and 68%/95% credible
intervals for key parameters.

Outputs:
  - Console table
  - Results/gwtc1_phasemarg/summary_stats.csv
"""

import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *
from anesthetic import read_chains

RESULTS_PHASEMARG = os.path.join(RESULTS_DIR, 'gwtc1_phasemarg')
OUT_CSV = os.path.join(RESULTS_PHASEMARG, 'summary_stats.csv')

PARAMS = ['H_0', 'M_c', 'q', 'd_L', 'iota', 's1_z', 's2_z']

# Discover all CSVs (nested + reweighted)
pattern = os.path.join(RESULTS_PHASEMARG, '*.csv')
all_csvs = sorted(glob.glob(pattern))
# Exclude our own output
all_csvs = [f for f in all_csvs if 'summary_stats' not in f and 'evidence_table' not in f]

if not all_csvs:
    print(f"No CSVs found in {RESULTS_PHASEMARG}")
    raise SystemExit(1)


def weighted_quantile(values, weights, quantiles):
    """Compute weighted quantiles."""
    idx = np.argsort(values)
    sorted_vals = values[idx]
    sorted_w = weights[idx]
    cumw = np.cumsum(sorted_w)
    cumw /= cumw[-1]
    return np.interp(quantiles, cumw, sorted_vals)


def compute_map_kde(values, weights, n_eval=500):
    """Estimate MAP via weighted KDE."""
    from scipy.stats import gaussian_kde
    mask = np.isfinite(values)
    values = values[mask]
    weights = weights[mask]
    if len(values) < 10:
        return float('nan')
    kde = gaussian_kde(values, weights=weights / weights.sum())
    lo, hi = np.percentile(values, [1, 99])
    x = np.linspace(lo, hi, n_eval)
    return x[np.argmax(kde(x))]


import csv
header = ['Run', 'Parameter', 'MAP', 'Median', '68% lo', '68% hi', '95% lo', '95% hi']
rows = []

for csv_path in all_csvs:
    basename = os.path.basename(csv_path).replace('.csv', '')
    is_reweighted = 'reweighted' in basename

    # Short name
    short = basename
    short = short.replace('PhaseMarg_Heterodyned_', 'Het ')
    short = short.replace('PhaseMarg_Unheterodyned_', 'Unhet ')
    short = short.replace('IMRPhenomD_NRTidalv2_local_', 'IMR ')
    short = short.replace('TaylorF2_local_', 'TF2 ')

    try:
        if is_reweighted:
            samples = load_reweighted_csv(csv_path)
        else:
            samples = load_nested_csv(csv_path)
    except Exception as e:
        print(f"  SKIP {basename}: {e}")
        continue

    weights = np.asarray(samples.get_weights())
    weights = weights / weights.sum()

    print(f"\n{'='*70}")
    print(f"  {short}")
    print(f"{'='*70}")
    print(f"  {'Param':>8s}  {'MAP':>8s}  {'Median':>8s}  {'68% CI':>18s}  {'95% CI':>18s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*18}  {'-'*18}")

    for param in PARAMS:
        if param not in samples.columns:
            continue

        vals = samples[param].to_numpy().astype(float)
        mask = np.isfinite(vals)
        vals_clean = vals[mask]
        w_clean = weights[mask]

        if len(vals_clean) < 10:
            continue

        map_val = compute_map_kde(vals_clean, w_clean)
        q_vals = weighted_quantile(vals_clean, w_clean,
                                   [0.025, 0.15865, 0.5, 0.84135, 0.975])
        lo95, lo68, median, hi68, hi95 = q_vals

        rows.append([short, param, f'{map_val:.4f}', f'{median:.4f}',
                     f'{lo68:.4f}', f'{hi68:.4f}', f'{lo95:.4f}', f'{hi95:.4f}'])
        print(f"  {param:>8s}  {map_val:8.3f}  {median:8.3f}  [{lo68:8.3f}, {hi68:8.3f}]  [{lo95:8.3f}, {hi95:8.3f}]")

# Save
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
print(f"\n-> Saved {OUT_CSV}")

# Print headline H_0 constraints
print("\n" + "="*70)
print("  H_0 CONSTRAINTS (symmetric credible intervals)")
print("="*70)
for csv_path in all_csvs:
    basename = os.path.basename(csv_path).replace('.csv', '')
    is_reweighted = 'reweighted' in basename
    short = basename
    short = short.replace('PhaseMarg_Heterodyned_', 'Het ')
    short = short.replace('PhaseMarg_Unheterodyned_', 'Unhet ')
    short = short.replace('IMRPhenomD_NRTidalv2_local_', 'IMR ')
    short = short.replace('TaylorF2_local_', 'TF2 ')

    try:
        if is_reweighted:
            samples = load_reweighted_csv(csv_path)
        else:
            samples = load_nested_csv(csv_path)
    except Exception:
        continue

    if 'H_0' not in samples.columns:
        continue

    vals = samples['H_0'].to_numpy().astype(float)
    weights = np.asarray(samples.get_weights())
    weights = weights / weights.sum()
    mask = np.isfinite(vals)
    vals, weights = vals[mask], weights[mask]

    map_val = compute_map_kde(vals, weights)
    q_vals = weighted_quantile(vals, weights, [0.15865, 0.84135])
    lo68, hi68 = q_vals
    print(f"  {short}")
    print(f"    H_0 = {map_val:.1f}  +{hi68-map_val:.1f} / -{map_val-lo68:.1f}  km/s/Mpc  (68% symmetric CI around MAP)")

print("\nDone.")
