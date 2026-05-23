"""
Quantify waveform systematics: Jensen-Shannon divergence and KL divergence
between IMRPhenomD and TaylorF2 posteriors for key parameters.

Output:
  - Console table
  - Results/gwtc1_phasemarg/waveform_systematics.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *
from scipy.stats import gaussian_kde

RESULTS_PHASEMARG = os.path.join(RESULTS_DIR, 'gwtc1_phasemarg')
OUT_CSV = os.path.join(RESULTS_PHASEMARG, 'waveform_systematics.csv')

# Pairs to compare (IMR vs TF2 for each prior variant)
PAIRS = [
    ('baseline',
     os.path.join(RESULTS_PHASEMARG,
         'PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_baseline.csv'),
     os.path.join(RESULTS_PHASEMARG,
         'PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_baseline.csv')),
    ('flat-in-z',
     os.path.join(RESULTS_PHASEMARG,
         'PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_flatZ.csv'),
     os.path.join(RESULTS_PHASEMARG,
         'PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_flatZ.csv')),
    ('vp250',
     os.path.join(RESULTS_PHASEMARG,
         'PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_vp250.csv'),
     os.path.join(RESULTS_PHASEMARG,
         'PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_vp250.csv')),
]

PARAMS = ['H_0', 'M_c', 'q', 'd_L', 'iota']


def kde_from_samples(vals, weights, x_eval):
    """Build a normalised KDE PDF on x_eval."""
    w = weights / weights.sum()
    kde = gaussian_kde(vals, weights=w)
    pdf = kde(x_eval)
    pdf = pdf / np.trapezoid(pdf, x_eval)
    return pdf


def js_divergence(p, q, x_eval):
    """Jensen-Shannon divergence between two PDFs (in nats)."""
    m = 0.5 * (p + q)
    dx = x_eval[1] - x_eval[0]
    # Avoid log(0)
    eps = 1e-30
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps))) * dx
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps))) * dx
    return 0.5 * (kl_pm + kl_qm)


def kl_divergence(p, q, x_eval):
    """KL divergence D_KL(P || Q) in nats."""
    dx = x_eval[1] - x_eval[0]
    eps = 1e-30
    return np.sum(p * np.log((p + eps) / (q + eps))) * dx


import csv
header = ['Prior variant', 'Parameter', 'JSD (nats)', 'JSD (bits)',
          'KL(IMR||TF2)', 'KL(TF2||IMR)']
rows = []

for variant_name, imr_csv, tf2_csv in PAIRS:
    if not os.path.exists(imr_csv) or not os.path.exists(tf2_csv):
        missing = imr_csv if not os.path.exists(imr_csv) else tf2_csv
        print(f"  WARNING: {os.path.basename(missing)} not found — skipping {variant_name}")
        continue

    imr = load_nested_csv(imr_csv)
    tf2 = load_nested_csv(tf2_csv)

    w_imr = np.asarray(imr.get_weights())
    w_tf2 = np.asarray(tf2.get_weights())

    print(f"\n{'='*60}")
    print(f"  {variant_name}: IMRPhenomD vs TaylorF2")
    print(f"{'='*60}")
    print(f"  {'Param':>8s}  {'JSD (nats)':>12s}  {'JSD (bits)':>12s}  {'KL(IMR||TF2)':>14s}  {'KL(TF2||IMR)':>14s}")

    for param in PARAMS:
        if param not in imr.columns or param not in tf2.columns:
            continue

        vals_imr = imr[param].to_numpy().astype(float)
        vals_tf2 = tf2[param].to_numpy().astype(float)

        # Common evaluation grid
        lo = min(np.percentile(vals_imr, 0.5), np.percentile(vals_tf2, 0.5))
        hi = max(np.percentile(vals_imr, 99.5), np.percentile(vals_tf2, 99.5))
        x_eval = np.linspace(lo, hi, 1000)

        pdf_imr = kde_from_samples(vals_imr, w_imr, x_eval)
        pdf_tf2 = kde_from_samples(vals_tf2, w_tf2, x_eval)

        jsd = js_divergence(pdf_imr, pdf_tf2, x_eval)
        jsd_bits = jsd / np.log(2)
        kl_imr_tf2 = kl_divergence(pdf_imr, pdf_tf2, x_eval)
        kl_tf2_imr = kl_divergence(pdf_tf2, pdf_imr, x_eval)

        rows.append([variant_name, param, f'{jsd:.6f}', f'{jsd_bits:.6f}',
                     f'{kl_imr_tf2:.6f}', f'{kl_tf2_imr:.6f}'])
        print(f"  {param:>8s}  {jsd:12.6f}  {jsd_bits:12.6f}  {kl_imr_tf2:14.6f}  {kl_tf2_imr:14.6f}")

# Save
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
print(f"\n-> Saved {OUT_CSV}")

print("\nDone.")
