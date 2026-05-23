#!/usr/bin/env python3
"""Analyse Session 10: d_L–iota bimodality characterisation.

Uses the targeted Mode-A (d_L ∈ [30, 75]) and Mode-B (d_L ∈ [10, 30]) runs
to compute local log evidences and hence the Mode-B Bayes factor, properly
normalised for the prior-volume restriction.

The heterodyne reference-swap run ([10, 75] with ref anchored at Mode B)
answers: is the Mode-B probability mass real, or an artefact of the Mode-A
heterodyne reference?

Output: Results/test_suite/bimodality_summary.csv and a human-readable
report on stdout.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (
    RESULTS_ROOT, load_catalog, load_run,
    weighted_median, weighted_quantiles, weighted_tail_prob,
    read_log_evidence_from_log,
)


def _lookup(df, pattern):
    hit = df[df["run_id"].str.contains(pattern)]
    return hit.iloc[0] if not hit.empty else None


def main() -> int:
    cat = load_catalog()
    s10 = cat[(cat["session"] == "10") & (cat["status"] == "done")]
    if s10.empty:
        print("No Session 10 runs marked 'done'.")
        return 0

    rows = []
    for _, cr in s10.iterrows():
        run_dir = os.path.join(RESULTS_ROOT, cr["run_id"])
        run = load_run(cr["run_id"])
        x = run.param("H_0"); w = run.weights
        d = run.param("d_L"); wD = run.weights
        log_z, sigma = read_log_evidence_from_log(run_dir)
        rows.append({
            "run_id": cr["run_id"],
            "variant": cr["variant"],
            "dL_median": weighted_median(d, wD),
            "H0_median": weighted_median(x, w),
            "P_H0_gt_120": weighted_tail_prob(x, w, 120.0),
            "log_Z": log_z,
            "sigma_log_Z": sigma,
        })
    df = pd.DataFrame(rows)

    out = os.path.join(RESULTS_ROOT, "bimodality_summary.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(df.to_string(index=False))

    # Bayes factor: Mode B vs Mode A. Account for prior volume:
    #   ln Z_raw is the evidence within the restricted prior;
    #   the LOG of the prior-volume ratio (vol_B / vol_A) should be added so
    #   that we recover the joint-prior-normalised evidence.
    mode_a = _lookup(df, "dL30-75")
    mode_b = _lookup(df, "dL10-30")
    if mode_a is not None and mode_b is not None:
        # For flat-in-z prior, the prior volumes are proportional to the
        # redshift interval, which at low z is approximately proportional
        # to the d_L interval. Use a first-order correction; refine with
        # the exact cosmology if needed.
        vol_a = 75.0 - 30.0
        vol_b = 30.0 - 10.0
        ln_vol_ratio_b_over_a = np.log(vol_b / vol_a)
        ln_bf = (mode_b["log_Z"] - mode_a["log_Z"]) + ln_vol_ratio_b_over_a
        print()
        print(f"Mode-A log Z (d_L ∈ [30,75]) = {mode_a['log_Z']:.3f}")
        print(f"Mode-B log Z (d_L ∈ [10,30]) = {mode_b['log_Z']:.3f}")
        print(f"ln-volume ratio (B/A)       = {ln_vol_ratio_b_over_a:.3f}")
        print(f"ln Bayes factor Mode-B / Mode-A = {ln_bf:.3f}")
        if ln_bf < -2.0:
            interp = "weakly disfavoured"
        elif ln_bf < 0:
            interp = "only mildly disfavoured"
        elif ln_bf < 2:
            interp = "comparable evidence"
        else:
            interp = "Mode B favoured"
        print(f"Interpretation: {interp}")

    ref_swap = _lookup(df, "refModeB")
    if ref_swap is not None:
        print(f"\nReference-swap run H0 median: {ref_swap['H0_median']:.2f} (flat-in-z, ref anchored in Mode B)")
        print("Compare to existing flatZ with GWTC-1 reference (H0 median = 93.6). If shift is small, heterodyne reference is not biasing mode weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
