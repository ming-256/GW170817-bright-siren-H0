#!/usr/bin/env python3
"""Analyse Session 06: GW150914 with IMRPhenomPv2, XPHM, and repeated IMRPhenomD.

Compares d_L / M_c / q / iota marginals against the existing GWTC-2.1
reference posterior. The specific question we are answering:
  does adding precession (IMRPhenomPv2 or XPHM) shift the d_L peak from
  ~380 Mpc (our IMRPhenomD) to ~410 Mpc (LVK), as expected?

Output: Results/test_suite/gw150914_waveform_comparison.csv
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (
    RESULTS_ROOT, load_catalog, load_run,
    weighted_map, weighted_median, weighted_quantiles,
    weighted_wasserstein1, read_log_evidence_from_log,
)


def summarise_params(run):
    summary = {}
    for p, cols in [("M_c", ["M_c", "Mc", "chirp_mass"]),
                    ("q", ["q", "mass_ratio"]),
                    ("d_L", ["d_L", "dL", "luminosity_distance"]),
                    ("iota", ["iota", "inclination"])]:
        for c in cols:
            if c in run.samples.columns:
                x = run.samples[c].to_numpy()
                w = run.weights
                summary[f"{p}_MAP"] = weighted_map(x, w)
                summary[f"{p}_median"] = weighted_median(x, w)
                q16, q84 = weighted_quantiles(x, w, [0.15865, 0.84135])
                summary[f"{p}_q16"] = q16
                summary[f"{p}_q84"] = q84
                break
    return summary


def main() -> int:
    cat = load_catalog()
    s06 = cat[(cat["session"] == "06") & (cat["status"] == "done")]
    if s06.empty:
        print("No Session 06 runs marked 'done'.")
        return 0

    rows = []
    for _, cr in s06.iterrows():
        try:
            run = load_run(cr["run_id"])
        except FileNotFoundError:
            print(f"missing samples for {cr['run_id']}")
            continue
        row = {"run_id": cr["run_id"], "waveform": cr["waveform"]}
        row.update(summarise_params(run))
        log_z, sigma = read_log_evidence_from_log(os.path.join(RESULTS_ROOT, cr["run_id"]))
        row["log_Z"] = log_z
        row["sigma_log_Z"] = sigma
        rows.append(row)

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_ROOT, "gw150914_waveform_comparison.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(df.to_string(index=False))

    # Headline claim check: does IMRPhenomPv2 shift d_L_median closer to the LVK 410 Mpc?
    imr_d_row = df[df["waveform"].str.lower() == "imrphenomd"]
    pv2_row = df[df["waveform"].str.lower() == "imrphenompv2"]
    if not imr_d_row.empty and not pv2_row.empty:
        d_imr = imr_d_row["d_L_median"].iloc[0]
        d_pv2 = pv2_row["d_L_median"].iloc[0]
        print(f"\nd_L median shift (PhenomD -> PhenomPv2): {d_imr:.1f} -> {d_pv2:.1f} Mpc")
        print("LVK GWTC-2.1 reference ~410 Mpc; target is <10 Mpc offset from PhenomPv2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
