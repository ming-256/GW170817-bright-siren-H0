#!/usr/bin/env python3
"""Analyse Session 07: GW170817 with IMRPhenomPv2_NRTidal and IMRPhenomXAS_NRTidalv3.

The science question: does a precessing tidal waveform (matching the LVK
bright-siren analysis) change the H0 posterior, in particular the tail
probability P(H0 > 120)?

Output: Results/test_suite/gw170817_waveform_comparison.csv
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (
    RESULTS_ROOT, load_catalog, load_run,
    weighted_map, weighted_median, weighted_quantiles, weighted_tail_prob,
    weighted_wasserstein1, read_log_evidence_from_log,
)


def main() -> int:
    cat = load_catalog()
    s07 = cat[(cat["session"] == "07") & (cat["status"] == "done")]
    if s07.empty:
        print("No Session 07 runs marked 'done'.")
        return 0

    rows = []
    for _, cr in s07.iterrows():
        try:
            run = load_run(cr["run_id"])
        except FileNotFoundError:
            print(f"missing samples for {cr['run_id']}")
            continue
        x = run.param("H_0")
        w = run.weights
        log_z, sigma = read_log_evidence_from_log(os.path.join(RESULTS_ROOT, cr["run_id"]))
        q16, q84 = weighted_quantiles(x, w, [0.15865, 0.84135])
        rows.append({
            "run_id": cr["run_id"],
            "waveform": cr["waveform"],
            "variant": cr["variant"],
            "H0_MAP": weighted_map(x, w),
            "H0_median": weighted_median(x, w),
            "H0_q16": q16, "H0_q84": q84,
            "P_H0_gt_120": weighted_tail_prob(x, w, 120.0),
            "log_Z": log_z,
            "sigma_log_Z": sigma,
        })

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_ROOT, "gw170817_waveform_comparison.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(df.to_string(index=False))

    # Headline diagnostic: Wasserstein distance between primary IMR_NRTv2 baseline (from
    # main analysis) and each precessing-waveform baseline.
    main_baseline = os.path.join(os.path.dirname(RESULTS_ROOT), "gwtc1_phasemarg",
                                 "PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_baseline.csv")
    if os.path.exists(main_baseline):
        base = pd.read_csv(main_baseline)
        x1 = base["H_0"].to_numpy()
        w1 = base.get("weight", pd.Series([1.0] * len(base))).to_numpy()
        w1 = w1 / w1.sum()
        print("\nWasserstein-1 distance from IMR_NRTv2 baseline (km/s/Mpc):")
        for _, r in df[df["variant"] == "baseline"].iterrows():
            run = load_run(r["run_id"])
            x2 = run.param("H_0"); w2 = run.weights
            wd = weighted_wasserstein1(x1, w1, x2, w2)
            print(f"  {r['waveform']:30s}  W1 = {wd:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
