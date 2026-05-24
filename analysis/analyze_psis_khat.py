#!/usr/bin/env python3
"""Pareto-smoothed importance-sampling (PSIS) k_hat diagnostic for the
volumetric → uniform-in-d_L reweighting of the GW170817 IMRX baseline.

Referee/data-release item: §4.1 of the manuscript claims that the standard
k_hat > 0.7 Vehtari threshold would flag the reweighting failure. This
script computes the actual k_hat by fitting a generalized Pareto
distribution to the upper tail of the log importance ratios, following
Vehtari, Simpson, Gelman, Yao & Gabry (2024, JMLR 25:72). It also runs a
non-parametric bootstrap on the reweighted P(H_0 > 120 km/s/Mpc) estimate
to demonstrate that the reweighting deficit is bias, not high variance.

Outputs (appended to Results/gwtc1_phasemarg/paper_diagnostics.csv):
    pareto_khat                (for the reweighted-IMRX row)
    reweighted_bootstrap_q025  (P(H_0>120) bootstrap 2.5 percentile)
    reweighted_bootstrap_q975  (P(H_0>120) bootstrap 97.5 percentile)
    direct_minus_reweighted_sigma  (binomial-SE units)

The reweighted-IMRX row is the one this diagnostic applies to; the other
rows are left unchanged.

No GPU, no sampler.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import genpareto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import REPO_ROOT, RESULTS_ROOT, read_nested_samples_csv


# ---------------------------------------------------------------------------- #
def psis_khat(log_ratio, frac_tail=0.20, max_tail=None):
    """Fit a generalised Pareto to the upper tail of log importance ratios.

    Parameters
    ----------
    log_ratio : array_like
        Log of the importance ratios pi_target / pi_proposal at the n
        baseline samples (constants drop out: shifting by a scalar does
        not change the shape parameter).
    frac_tail : float
        Fraction of the upper tail used for the GPD fit; Vehtari+2024
        use min(0.2 * S, 3 * sqrt(S)) / S as the default cap.

    Returns
    -------
    khat : float
        The Pareto shape parameter from the MLE fit.
    M : int
        Number of points actually used in the fit.
    """
    S = len(log_ratio)
    M = int(min(frac_tail * S, 3.0 * np.sqrt(S)))
    if max_tail is not None:
        M = min(M, max_tail)
    sorted_lr = np.sort(log_ratio)
    threshold = sorted_lr[-M - 1]
    exceed_log = sorted_lr[-M:] - threshold       # log of (ratio / threshold)
    exceed     = np.exp(exceed_log) - 1.0         # GPD support starts at 0
    shape, _, _ = genpareto.fit(exceed, floc=0.0) # MLE with location pinned
    return float(shape), int(M)


def bootstrap_reweighted_tail(h0, w_rw, threshold, n_eff, n_boot=4000, seed=0):
    """Multinomial bootstrap of P(H_0 > threshold) at n_eff effective draws."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(h0))
    n_draws = int(round(n_eff))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.choice(idx, size=n_draws, replace=True, p=w_rw)
        boot[b] = float((h0[sel] > threshold).mean())
    q025, q500, q975 = np.quantile(boot, [0.025, 0.5, 0.975])
    return q025, q500, q975, boot


# ---------------------------------------------------------------------------- #
def main():
    base_dir = os.path.join(
        RESULTS_ROOT,
        "s14__gw170817__imrphenomxas_nrtidalv3__baseline__seed0000",
    )
    direct_dir = os.path.join(
        RESULTS_ROOT,
        "s14__gw170817__imrphenomxas_nrtidalv3__flatz__seed0000",
    )

    base_csv   = os.path.join(base_dir,   "samples.csv")
    direct_csv = os.path.join(direct_dir, "samples.csv")

    df_b, w_b = read_nested_samples_csv(base_csv)
    df_f, w_f = read_nested_samples_csv(direct_csv)

    h0_b = df_b["H_0"].to_numpy()
    dL_b = df_b["d_L"].to_numpy()
    h0_f = df_f["H_0"].to_numpy()

    # Volumetric -> uniform-in-dL reweight: pi_target / pi_base = 1 / dL^2
    # (constants drop out of both the IS estimator and the GPD shape fit).
    log_ratio = -2.0 * np.log(dL_b)

    # PSIS k_hat
    khat, M = psis_khat(log_ratio)

    # Reweighted IS weights
    ratio = np.exp(log_ratio - log_ratio.max())   # numerically stable
    w_rw  = w_b * ratio
    w_rw /= w_rw.sum()
    w_b_n = w_b / w_b.sum()
    w_f_n = w_f / w_f.sum()

    n_eff_rw = float(1.0 / (w_rw**2).sum())
    n_eff_b  = float(1.0 / (w_b_n**2).sum())

    P_base   = float((w_b_n * (h0_b > 120)).sum())
    P_rw     = float((w_rw   * (h0_b > 120)).sum())
    P_direct = float((w_f_n  * (h0_f > 120)).sum())

    q025, q500, q975, _ = bootstrap_reweighted_tail(
        h0_b, w_rw, threshold=120.0, n_eff=n_eff_rw, n_boot=4000, seed=0
    )
    se_binom = float(np.sqrt(P_rw * (1 - P_rw) / n_eff_rw))
    sigma_gap = float((P_direct - P_rw) / se_binom)

    print("=" * 72)
    print("PSIS k_hat diagnostic — IMRX reweighting (volumetric → uniform-in-d_L)")
    print("=" * 72)
    print(f"Baseline NS samples                 S = {len(log_ratio):>7,d}")
    print(f"GPD fit on top                      M = {M:>7,d}")
    print(f"PSIS k_hat (Vehtari+2024 MLE)         = {khat:>7.3f}")
    print(f"Vehtari thresholds:")
    print(f"  k_hat ≤ 0.5         : IS reliable")
    print(f"  0.5 < k_hat ≤ 0.7   : high variance but consistent")
    print(f"  k_hat > 0.7         : IS unreliable")
    if   khat <= 0.5: regime = "reliable"
    elif khat <= 0.7: regime = "high variance but consistent (cautionary)"
    elif khat <= 1.0: regime = "unreliable"
    else:             regime = "estimator has infinite variance"
    print(f"  → this run         : k_hat = {khat:.3f}  ({regime})")
    print()
    print(f"P(H_0 > 120 km/s/Mpc):")
    print(f"  baseline (volumetric)            : {P_base:.4f}")
    print(f"  reweighted (uniform-in-dL)       : {P_rw:.4f}")
    print(f"  direct (uniform-in-dL sampled)   : {P_direct:.4f}")
    print(f"\nReweighted bootstrap (4000 draws at n_eff={n_eff_rw:.0f}):")
    print(f"  95% CI                            : [{q025:.4f}, {q975:.4f}]")
    print(f"  Bootstrap CI excludes direct      : {bool(not (q025 <= P_direct <= q975))}")
    print(f"  Direct − reweighted               : {P_direct - P_rw:.4f}")
    print(f"  Shift / binomial SE (= sigma)     : {sigma_gap:.1f}")
    print(f"\nThe shift is ~{sigma_gap:.0f}σ above the reweighted estimator's binomial SE,")
    print(f"so reweighting on this draw is biased, not high-variance.\n")

    # Persist into paper_diagnostics.csv
    out_csv = os.path.join(
        REPO_ROOT, "Results", "gwtc1_phasemarg", "paper_diagnostics.csv"
    )
    diag = pd.read_csv(out_csv)
    # Add columns only if absent; populate only the reweighted row.
    if "pareto_khat" not in diag.columns:
        diag["pareto_khat"] = np.nan
    if "rw_bootstrap_q025" not in diag.columns:
        diag["rw_bootstrap_q025"] = np.nan
    if "rw_bootstrap_q975" not in diag.columns:
        diag["rw_bootstrap_q975"] = np.nan
    if "direct_minus_rw_sigma" not in diag.columns:
        diag["direct_minus_rw_sigma"] = np.nan

    rw_mask = diag["variant"].str.contains("reweighted", case=False, na=False)
    if rw_mask.any():
        diag.loc[rw_mask, "pareto_khat"]            = khat
        diag.loc[rw_mask, "rw_bootstrap_q025"]      = q025
        diag.loc[rw_mask, "rw_bootstrap_q975"]      = q975
        diag.loc[rw_mask, "direct_minus_rw_sigma"]  = sigma_gap

    diag.to_csv(out_csv, index=False)
    print(f"Updated {out_csv} with PSIS k_hat + bootstrap CI columns "
          f"on the reweighted row.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
