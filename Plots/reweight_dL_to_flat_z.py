"""
Reweight d_L posterior samples from Beta(3,1) prior to flat-in-z prior.
========================================================================

Takes nested sampling CSV results (with Beta(3,1) ∝ d_L^2 prior on d_L)
and reweights them to a flat-in-redshift prior using importance sampling.

Derivation  (LVK convention — only change the d_L prior)
----------
Old prior:  p_old(d_L) = Beta(3,1) on [lo, hi] = 3u^2 / (hi - lo),
            where u = (d_L - lo) / (hi - lo).  This is ∝ d_L^2 (volumetric).

New prior:  p_new(d_L) = uniform on [lo, hi] = 1 / (hi - lo).
            At fixed H_0, flat-in-d_L is equivalent to flat-in-z
            since z = H_0 d_L / c  ⟹  dz ∝ dd_L.

The H_0 prior is kept unchanged (log-uniform).

Importance reweighting:

    w_new / w_old = p_new(d_L) / p_old(d_L)
                  = [1/(hi-lo)] / [3u^2/(hi-lo)]
                  = 1 / (3 u^2)
                  ∝ 1 / u^2

Since we normalise weights at the end, all multiplicative constants cancel:

    w_new ∝ w_old / u^2

Usage:
    # Reweight a single file:
    python reweight_dL_to_flat_z.py <input_csv> [--output <output_csv>]

    # Auto-discover and reweight all *_baseline.csv files in Results/gwtc1_phasemarg/:
    python reweight_dL_to_flat_z.py
"""

import argparse
import glob
import os
import numpy as np
from anesthetic import read_chains, NestedSamples

# Prior bounds for d_L (must match the sampling script)
D_L_LO = 1.0   # Mpc
D_L_HI = 75.0  # Mpc

# Default directory to scan for baseline results
DEFAULT_BASELINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Results', 'gwtc1_phasemarg',
)

parser = argparse.ArgumentParser(
    description='Reweight d_L samples from Beta(3,1) to flat-in-z prior')
parser.add_argument('input_csv', nargs='?', default=None,
                    help='Path to input CSV (nested sampling results). '
                         'If omitted, auto-discovers *_baseline.csv in Results/gwtc1_phasemarg/')
parser.add_argument('--output', '-o', default=None,
                    help='Output CSV path (default: replace _baseline with _reweighted_flatZ)')
parser.add_argument('--baseline-dir', default=DEFAULT_BASELINE_DIR,
                    help='Directory to scan when no input_csv is given '
                         f'(default: {DEFAULT_BASELINE_DIR})')
args = parser.parse_args()


def reweight_file(input_csv, output_path=None):
    """Reweight a single baseline CSV from Beta(3,1) d_L prior to flat-in-z."""

    # Safety: warn if the input filename suggests it already has a flat-in-z prior
    if 'flatZ' in input_csv or 'flat_z' in input_csv.lower():
        print(f"  WARNING: Input filename contains 'flatZ' — are you sure this has a Beta(3,1) d_L prior?")
        print(f"           Reweighting a file that already has flat-in-z prior will give incorrect results.")

    # Load samples
    samples = read_chains(input_csv)
    print(f"Loaded {len(samples)} samples from {input_csv}")

    # Extract needed columns
    d_L = samples['d_L'].to_numpy()

    # Original nested sampling weights (from evidence calculation)
    weights_old = np.asarray(samples.get_weights())

    # Reweighting factor: w_new ∝ w_old / u^2  (LVK convention: only change d_L prior)
    # Guard against d_L == d_L_lo (u=0) which would give infinite weight
    u = (d_L - D_L_LO) / (D_L_HI - D_L_LO)
    reweight_factor = 1.0 / (u**2 + 1e-30)

    weights_new = weights_old * reweight_factor

    # Normalise
    weights_new /= weights_new.sum()

    # Effective sample size (diagnostic)
    n_eff = 1.0 / np.sum(weights_new**2)
    print(f"Effective sample size after reweighting: {n_eff:.0f} / {len(samples)}")

    # Determine output path
    if output_path is None:
        # Replace _baseline with _reweighted_flatZ; fall back to appending
        if '_baseline' in input_csv:
            output_path = input_csv.replace('_baseline.csv', '_reweighted_flatZ.csv')
        else:
            output_path = input_csv.replace('.csv', '_reweighted_flatZ.csv')

    # Save as CSV with a weights column appended
    # Use numpy for fast writing (pandas to_csv is very slow for 200K+ rows
    # due to per-element string conversion of object-typed anesthetic columns)
    col_names = list(samples.columns.get_level_values(0)) + ['weight']
    data = np.asarray(samples.to_numpy(), dtype=np.float64)
    data = np.column_stack([data, weights_new])
    with open(output_path, 'w') as f:
        f.write(','.join(col_names) + '\n')
    with open(output_path, 'ab') as f:
        np.savetxt(f, data, delimiter=',', fmt='%.15g')
    print(f"Saved reweighted samples to {output_path}")

    # Print summary statistics for key parameters
    col_idx = {name: i for i, name in enumerate(col_names)}
    print(f"\n{'='*50}")
    print(f"Reweighted summary (flat-in-z prior)")
    print(f"{'='*50}")
    for param in ['d_L', 'H_0', 'M_c', 'q', 'iota']:
        if param in col_idx:
            vals = data[:, col_idx[param]]
            mean = np.average(vals, weights=weights_new)
            var = np.average((vals - mean)**2, weights=weights_new)
            sort_idx = np.argsort(vals)
            lo, med, hi = np.interp(
                [0.05, 0.50, 0.95],
                np.cumsum(weights_new[sort_idx]),
                vals[sort_idx],
            )
            print(f"  {param:8s}: mean={mean:.2f}, median={med:.2f}, 90% CI=[{lo:.2f}, {hi:.2f}]")
    print(f"{'='*50}")
    return output_path


# ---- Main: single file or auto-discover ----
if args.input_csv is not None:
    reweight_file(args.input_csv, args.output)
else:
    # Auto-discover all *_baseline.csv files in the baseline directory
    pattern = os.path.join(args.baseline_dir, '*_baseline.csv')
    baseline_files = sorted(glob.glob(pattern))

    if not baseline_files:
        print(f"No *_baseline.csv files found in {args.baseline_dir}")
        print(f"  (searched: {pattern})")
        raise SystemExit(1)

    print(f"Auto-discovered {len(baseline_files)} baseline file(s) in {args.baseline_dir}:")
    for f in baseline_files:
        print(f"  {os.path.basename(f)}")
    print()

    for f in baseline_files:
        print(f"{'─'*60}")
        reweight_file(f)
        print()
