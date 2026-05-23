#!/usr/bin/env python3
"""Analyse the PSD-source sensitivity runs in Session 05 (kazewong, bilby).

Compares TaylorF2 baseline posteriors across three PSD sources (gwtc1, kazewong,
bilby) and reports whether H0 / d_L / log Z change at the systematic-bias level.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (
    REPO_ROOT, RESULTS_ROOT, load_catalog, load_run,
    weighted_median, weighted_tail_prob, read_log_evidence_from_log,
    read_nested_samples_csv,
)


def main() -> int:
    cat = load_catalog()
    s05 = cat[(cat["session"] == "05") & (cat["status"] == "done")]
    kz = s05[s05["variant"] == "baseline_psdKazewong"]
    bb = s05[s05["variant"] == "baseline_psdBilby"]
    if kz.empty and bb.empty:
        print("No PSD-sensitivity runs done yet.")
        return 0

    gwtc1_csv = os.path.join(
        REPO_ROOT, "Results", "gwtc1_phasemarg",
        "PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_baseline.csv",
    )
    rows = []
    gwtc1, w1 = read_nested_samples_csv(gwtc1_csv)
    rows.append({
        "psd_source": "gwtc1",
        "H0_median": weighted_median(gwtc1["H_0"].to_numpy(), w1),
        "P_H0_gt_120": weighted_tail_prob(gwtc1["H_0"].to_numpy(), w1, 120.0),
        "log_Z": None,
    })
    for df_sub, label in [(kz, "kazewong"), (bb, "bilby")]:
        if df_sub.empty:
            continue
        run = load_run(df_sub.iloc[0]["run_id"])
        x = run.param("H_0"); w = run.weights
        log_z, _ = read_log_evidence_from_log(os.path.join(RESULTS_ROOT, df_sub.iloc[0]["run_id"]))
        rows.append({
            "psd_source": label,
            "H0_median": weighted_median(x, w),
            "P_H0_gt_120": weighted_tail_prob(x, w, 120),
            "log_Z": log_z,
        })
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_ROOT, "psd_sensitivity_summary.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
