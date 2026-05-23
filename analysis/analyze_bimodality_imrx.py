#!/usr/bin/env python3
"""Analyse the M4 IMRX (NRTidalv3) mode-isolated bimodality runs.

Sibling to analyze_bimodality.py, which does the same on IMR/NRTidalv2
(session s10). The referee's M4 complaint is that every mode-isolated run
is on the legacy IMR (NRTidalv2) waveform but the manuscript prose attributes
the (d_L, iota) bimodality mechanism to the locked primary IMRX (NRTidalv3).
This script confirms (or refutes) that bridging argument on the locked
primary by computing the Mode-B / Mode-A Bayes factor for IMRX.

Runs (glob, no catalog required):
  Results/test_suite/s19__gw170817__imrphenomxas_nrtidalv3__flatz__dL30-75__refGWTC1__seed0000  (Mode A IMRX)
  Results/test_suite/s19__gw170817__imrphenomxas_nrtidalv3__flatz__dL10-30__refGWTC1__seed0000  (Mode B IMRX)
  Results/test_suite/s14__gw170817__imrphenomxas_nrtidalv3__flatz__seed0000                     (unrestricted IMRX, default ref)

Output:
  - Results/test_suite/bimodality_imrx_summary.csv
  - mnras_paper/figures/bimodality_imrx_dL_iota.{pdf,png}

The s14 unrestricted run is the existing IMRX direct uniform-in-d_L run from
the main analysis; it has no d_L window and uses the default GWTC-1
heterodyne reference (matching the s10 figure-overlay convention).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (
    REPO_ROOT, RESULTS_ROOT, load_run,
    weighted_median, weighted_tail_prob,
    read_log_evidence_from_log,
)


# ------------------------------------------------------------------ #
# HPD (inline; same algorithm as build_paper_tables.hpd)             #
# ------------------------------------------------------------------ #
def hpd(x, w, frac=0.68):
    idx = np.argsort(x)
    xs, ws = x[idx], w[idx] / w.sum()
    cdf = np.cumsum(ws)
    n = len(xs)
    best = (np.inf, float(xs[0]), float(xs[-1]))
    for i in range(n):
        target = cdf[i] + frac
        if target > 1.0:
            break
        j = int(np.searchsorted(cdf, target))
        if j >= n:
            break
        width = xs[j] - xs[i]
        if width < best[0]:
            best = (float(width), float(xs[i]), float(xs[j]))
    return best[1], best[2]


def map_from_hist(x, w, bin_width=1.0, lo=40.0, hi=230.0):
    edges = np.arange(lo, hi + bin_width, bin_width)
    counts, _ = np.histogram(x, bins=edges, weights=w)
    i = int(np.argmax(counts))
    return float(0.5 * (edges[i] + edges[i + 1]))


# ------------------------------------------------------------------ #
RUNS_IMRX = [
    ("ModeA",        "s19__gw170817__imrphenomxas_nrtidalv3__flatz__dL30-75__refGWTC1__seed0000", 30, 75),
    ("ModeB",        "s19__gw170817__imrphenomxas_nrtidalv3__flatz__dL10-30__refGWTC1__seed0000", 10, 30),
    ("Unrestricted", "s14__gw170817__imrphenomxas_nrtidalv3__flatz__seed0000",                    10, 75),
]


def main():
    missing = []
    rows = []
    for label, run_id, dl_lo, dl_hi in RUNS_IMRX:
        run_dir = os.path.join(RESULTS_ROOT, run_id)
        if not os.path.exists(run_dir):
            missing.append(run_id)
            continue
        run = load_run(run_id)
        x = run.param("H_0"); w = run.weights
        try:
            d = run.param("d_L")
        except KeyError:
            d = None
        log_z, sigma = read_log_evidence_from_log(run_dir)
        lo68, hi68 = hpd(x, w, 0.68)
        rows.append({
            "label": label,
            "run_id": run_id,
            "dl_lo": dl_lo, "dl_hi": dl_hi,
            "H0_MAP": map_from_hist(x, w),
            "H0_median": weighted_median(x, w),
            "HPD68_lo": lo68, "HPD68_hi": hi68,
            "P_H0_gt_120": weighted_tail_prob(x, w, 120.0),
            "log_Z": log_z, "sigma_log_Z": sigma,
            "n_samples": len(w),
            "dL_median": float(weighted_median(d, w)) if d is not None else float("nan"),
        })

    if missing:
        print("Missing IMRX bimodality runs (launch the M4 block in Tier 2):")
        for r in missing:
            print(f"  {os.path.join(RESULTS_ROOT, r)}")
        if not rows:
            return 1

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_ROOT, "bimodality_imrx_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(df.to_string(index=False))

    # ----- Bayes factor -----
    mode_a = df[df["label"] == "ModeA"]
    mode_b = df[df["label"] == "ModeB"]
    if len(mode_a) and len(mode_b):
        ln_vol_ratio = float(np.log(20.0 / 45.0))  # width_B / width_A
        ln_diff = float(mode_b.iloc[0]["log_Z"] - mode_a.iloc[0]["log_Z"])
        ln_bf = ln_diff + ln_vol_ratio
        print()
        print(f"Mode-A log Z (d_L in [30,75]) = {mode_a.iloc[0]['log_Z']:.3f}")
        print(f"Mode-B log Z (d_L in [10,30]) = {mode_b.iloc[0]['log_Z']:.3f}")
        print(f"ln(volume_B/volume_A) = ln(20/45) = {ln_vol_ratio:.4f}  "
              "(exact: prior is uniform in d_L)")
        print(f"ln BF Mode-B / Mode-A = {ln_bf:.3f}")
        verdict = "comparable evidence" if abs(ln_bf) < 1.0 else (
            "Mode B favoured" if ln_bf > 1.0 else "Mode B disfavoured"
        )
        print(f"Interpretation (Jeffreys): |lnBF|<1 -> {verdict}")
        print()
        print("Referee M4: if |lnBF|<1 on IMRX as well as IMR, the bridging "
              "argument is supported -- the (d_L, iota) bimodality is a "
              "property of the data + uniform-in-d_L prior, not the tidal "
              "phase model.")

    # ----- Figure: (d_L, iota) joint scatter for the unrestricted run + per-mode H0 -----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(df) < 3:
        print("\nNot enough runs to draw the joint (d_L, iota) figure; skipping plot.")
        return 0

    unrestricted_id = df[df["label"] == "Unrestricted"].iloc[0]["run_id"]
    run_u = load_run(unrestricted_id)
    d_u = run_u.param("d_L"); i_u = run_u.param("iota"); w_u = run_u.weights

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    # left: (d_L, iota) joint as a weighted 2D histogram
    h, xe, ye = np.histogram2d(d_u, i_u, bins=[60, 60],
                                range=[[10, 50], [1.8, np.pi]], weights=w_u)
    axes[0].imshow(h.T, origin="lower", aspect="auto",
                   extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="Greys")
    axes[0].axvline(30, color="k", lw=0.7, ls="--", alpha=0.6)
    axes[0].text(30.5, 3.05, r"$d_L = 30$ Mpc (prior boundary)",
                 rotation=90, ha="left", va="top", fontsize=9, color="0.3")
    axes[0].set_xlabel(r"$d_L$ (Mpc)")
    axes[0].set_ylabel(r"$\iota$ (rad)")
    axes[0].set_title(r"(a) IMRX joint $(d_L,\iota)$ posterior, uniform-in-$d_L$")

    # right: per-mode H0 histograms
    colors = {"ModeA": "#cd5c5c", "ModeB": "#3a5fcd", "Unrestricted": "0.3"}
    h0_edges = np.arange(40, 200 + 2, 2.0)
    for label, color in colors.items():
        sub = df[df["label"] == label]
        if not len(sub):
            continue
        r = load_run(sub.iloc[0]["run_id"])
        h, e = np.histogram(r.param("H_0"), bins=h0_edges, weights=r.weights)
        axes[1].step(0.5 * (e[:-1] + e[1:]), h / h.sum(), where="mid",
                     color=color, lw=2.0, label=label)
    axes[1].set_xlabel(r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)")
    axes[1].set_ylabel(r"$P(H_0)$ (per bin)")
    axes[1].set_title("(b) Per-mode IMRX $H_0$ marginals")
    axes[1].legend(fontsize=9)

    fig.suptitle("M4: IMRX (NRTidalv3) mode-isolated bimodality cross-check")
    fig.tight_layout()

    fig_dir = os.path.join(REPO_ROOT, "mnras_paper", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(fig_dir, f"bimodality_imrx_dL_iota.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
