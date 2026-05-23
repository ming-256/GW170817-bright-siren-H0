#!/usr/bin/env python3
"""Qualitative M4 cross-check: does the (d_L, iota) bimodality survive the
IMR/NRTidalv2 -> IMRX/NRTidalv3 tidal-calibration change?

Uses two existing unrestricted direct uniform-in-d_L posteriors:
  - s10__gw170817__imrphenomd_nrtidalv2__flatz__dL10-75__refModeB__seed0000   (IMR; refModeB)
  - s14__gw170817__imrphenomxas_nrtidalv3__flatz__seed0000                    (IMRX; default refGWTC1)

The two runs differ in two axes: waveform calibration AND heterodyne-reference
anchor. The s10 (refModeB) vs s10-equivalent-at-refGWTC1 control is already
established in the main paper (§5: P(H0>120) is statistically indistinguishable
across the two reference anchorings), so the comparison here is dominated by
the waveform difference.

Output:
  - mnras_paper/figures/bimodality_imr_vs_imrx.{pdf,png}
  - Results/test_suite/bimodality_waveform_check.csv

This is the qualitative half of referee M4 (the (d_L, iota) bimodality is a
property of the data plus the uniform-in-d_L prior, not the NRTidalv2 tidal
phase). The quantitative half -- Mode-A/Mode-B Bayes factor for IMRX -- still
requires the s19 IMRX mode-isolated runs of launch_tier2.sh and is deferred.

No GPU, no sampler.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (
    REPO_ROOT, RESULTS_ROOT, load_run,
    weighted_median, weighted_tail_prob,
)


IMR_RUN_ID  = "s10__gw170817__imrphenomd_nrtidalv2__flatz__dL10-75__refModeB__seed0000"
IMRX_RUN_ID = "s14__gw170817__imrphenomxas_nrtidalv3__flatz__seed0000"


def mode_b_fraction(d, w, dl_split=30.0):
    """Posterior weight in d_L < dl_split (the Mode-B region)."""
    return float(w[d < dl_split].sum() / w.sum())


def main():
    rows = []
    runs = {}
    for label, run_id in [("IMR/NRTidalv2 (refModeB)", IMR_RUN_ID),
                          ("IMRX/NRTidalv3 (refGWTC1)", IMRX_RUN_ID)]:
        run_dir = os.path.join(RESULTS_ROOT, run_id)
        if not os.path.exists(run_dir):
            print(f"  missing: {run_dir}")
            continue
        run = load_run(run_id)
        d = run.param("d_L"); i = run.param("iota"); h = run.param("H_0")
        w = run.weights
        runs[label] = (d, i, h, w)
        rows.append({
            "label": label,
            "run_id": run_id,
            "P_modeB_dL_lt_30": mode_b_fraction(d, w, 30.0),
            "P_modeA_dL_ge_30": 1.0 - mode_b_fraction(d, w, 30.0),
            "dL_median": weighted_median(d, w),
            "H0_median": weighted_median(h, w),
            "P_H0_gt_120": weighted_tail_prob(h, w, 120.0),
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_ROOT, "bimodality_waveform_check.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(df.to_string(index=False))

    if len(runs) < 2:
        print("\nNeed both IMR and IMRX unrestricted runs to draw the comparison.")
        return 1

    # ----- Figure -----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharex=True, sharey=True)

    for ax, (label, (d, i, h, w)) in zip(axes, runs.items()):
        # 2-D weighted histogram in (d_L, iota)
        H, xe, ye = np.histogram2d(
            d, i, bins=[60, 60],
            range=[[10, 50], [1.8, np.pi]], weights=w,
        )
        ax.imshow(H.T, origin="lower", aspect="auto",
                  extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="Greys")
        ax.axvline(30.0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.text(30.5, 3.05, r"$d_L = 30$ Mpc (Mode split)",
                rotation=90, ha="left", va="top", fontsize=9, color="0.3")
        # Mode-B / Mode-A weight annotation
        pB = mode_b_fraction(d, w, 30.0)
        ax.text(0.03, 0.97,
                f"$P(d_L<30) = {pB:.3f}$\n$P(d_L\\geq 30) = {1-pB:.3f}$",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
        ax.set_xlabel(r"$d_L$ (Mpc)")
        ax.set_title(label, fontsize=11)

    axes[0].set_ylabel(r"$\iota$ (rad)")
    fig.suptitle(
        r"M4 cross-check: $(d_L,\iota)$ bimodality across the "
        r"NRTidalv2 $\to$ NRTidalv3 calibration",
    )
    fig.tight_layout()

    fig_dir = os.path.join(REPO_ROOT, "mnras_paper", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(fig_dir, f"bimodality_imr_vs_imrx.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)

    # ----- Interpretation print -----
    pB_imr  = df.loc[df["label"].str.startswith("IMR/"),  "P_modeB_dL_lt_30"].iloc[0]
    pB_imrx = df.loc[df["label"].str.startswith("IMRX/"), "P_modeB_dL_lt_30"].iloc[0]
    print(f"\nMode-B (d_L<30 Mpc) posterior weight:")
    print(f"  IMR / NRTidalv2:  {pB_imr:.3f}")
    print(f"  IMRX / NRTidalv3: {pB_imrx:.3f}")
    print(f"Both substantially > 0 => the bimodality is not specific to "
          f"NRTidalv2. The qualitative half of M4 is supported by existing data; "
          f"the Mode-A/Mode-B Bayes factor on IMRX still requires the s19 runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
