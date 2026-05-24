#!/usr/bin/env python3
"""Analyse the reference-parameter swap run in Session 05.

Compares the IMR baseline run with --ref-params=optimize against the
existing IMR baseline (ref-params=gwtc1). If the posteriors agree to
within statistical variance, the heterodyne reference choice is not a
source of systematic bias at the paper's reported precision.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (
    REPO_ROOT, RESULTS_ROOT, load_catalog, load_run,
    weighted_median, weighted_tail_prob, weighted_wasserstein1,
    read_nested_samples_csv,
)


def main() -> int:
    cat = load_catalog()
    row = cat[(cat["run_id"] == "s05__gw170817__imrphenomd_nrtidalv2__baseline__refOptimize__seed0000")
              & (cat["status"] == "done")]
    if row.empty:
        print("ref-params=optimize run not done yet.")
        return 0

    opt_run = load_run(row.iloc[0]["run_id"])
    baseline_csv = os.path.join(
        REPO_ROOT, "results", "gwtc1_phasemarg",
        "PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_baseline.csv",
    )
    baseline, w_b = read_nested_samples_csv(baseline_csv)

    x_o = opt_run.param("H_0"); w_o = opt_run.weights
    x_b = baseline["H_0"].to_numpy()

    wd = weighted_wasserstein1(x_b, w_b, x_o, w_o)
    print(f"Wasserstein-1 (baseline gwtc1 vs optimize reference): {wd:.3f} km/s/Mpc")
    print(f"Baseline H0 median: {weighted_median(x_b, w_b):.2f}")
    print(f"Optimize-ref H0 median: {weighted_median(x_o, w_o):.2f}")
    print(f"Baseline P(H0>120): {weighted_tail_prob(x_b, w_b, 120):.4f}")
    print(f"Optimize-ref P(H0>120): {weighted_tail_prob(x_o, w_o, 120):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
