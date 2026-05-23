#!/usr/bin/env python3
"""Analyse Session 15: heterodyned full-sky vs narrow-sky matched pairs.

Each s15 run is matched to its s07 sibling at identical configuration
(waveform, priors, n_live, n_bins, seed, PSD/reference) — the only
varying axis is the (RA, dec) prior. We summarise:

 * wall-clock per run (from config.json) and the narrow/full-sky ratio,
 * H0 and chirp-mass weighted median + 68% intervals on each side,
 * Wasserstein-1 distance between posteriors as a scalar measure of how
   much the EM-localised sky prior shifts the inferred parameters.

The wall-clock figure produced by Plots/plot_sky_prior_runtime.py is
the headline; this script answers the secondary question "does the
narrow-sky prior bias H0 or M_chirp?" so the paper can claim sky-prior
restriction is purely a sampling-cost trick, not a physics knob.

Companion: Plots/build_scaling_table.py + plot_sky_prior_runtime.py.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (
    RESULTS_ROOT, load_catalog, load_run,
    weighted_median, weighted_quantiles, weighted_wasserstein1,
)


PAIRS = [
    ("IMRPhenomD_NRTidalv2",
     "s07__gw170817__imrphenomd_nrtidalv2__baseline_lvkbounds__seed0000",
     "s15__gw170817__imrphenomd_nrtidalv2__baseline_lvkbounds_narrow__seed0000"),
    ("IMRPhenomXAS_NRTidalv3",
     "s07__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds__seed0000",
     "s15__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds_narrow__seed0000"),
]


def _runtime_seconds(run) -> float | None:
    cfg = run.config or {}
    for key in ("wall_clock_seconds", "runtime_seconds", "total_seconds", "elapsed_seconds"):
        if key in cfg:
            try:
                return float(cfg[key])
            except (TypeError, ValueError):
                pass
    return None


def _summarise(run, param: str):
    x = run.param(param)
    w = run.weights
    q16, q84 = weighted_quantiles(x, w, [0.15865, 0.84135])
    return weighted_median(x, w), float(q16), float(q84)


def main() -> int:
    cat = load_catalog()
    done = set(cat[cat["status"] == "done"]["run_id"])

    rows = []
    for waveform, full_id, narrow_id in PAIRS:
        if full_id not in done:
            print(f"skip {waveform}: full-sky run {full_id} not marked done")
            continue
        if narrow_id not in done:
            print(f"skip {waveform}: narrow-sky run {narrow_id} pending GPU time")
            continue

        full = load_run(full_id)
        narrow = load_run(narrow_id)

        t_full = _runtime_seconds(full)
        t_narrow = _runtime_seconds(narrow)
        ratio = (t_narrow / t_full) if (t_full and t_narrow) else None

        row = {"waveform": waveform,
               "t_full_s": t_full, "t_narrow_s": t_narrow,
               "ratio_narrow_full": ratio}

        for param in ("H_0", "M_chirp"):
            try:
                m_f, lo_f, hi_f = _summarise(full, param)
                m_n, lo_n, hi_n = _summarise(narrow, param)
                d_w1 = weighted_wasserstein1(
                    full.param(param), full.weights,
                    narrow.param(param), narrow.weights,
                )
            except KeyError:
                continue
            row.update({
                f"{param}_median_full": m_f,
                f"{param}_q16_full": lo_f,
                f"{param}_q84_full": hi_f,
                f"{param}_median_narrow": m_n,
                f"{param}_q16_narrow": lo_n,
                f"{param}_q84_narrow": hi_n,
                f"{param}_wasserstein1": d_w1,
            })
        rows.append(row)

    if not rows:
        print("No matched s07/s15 pairs available yet. Run session_15_sky_prior_runtime.sh first.")
        return 0

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_ROOT, "sky_prior_runtime_summary.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(df.to_string(index=False))

    print("\nInterpretation:")
    print("  ratio_narrow_full < 1  => narrow-sky finishes faster (expected)")
    print("  H_0/M_chirp Wasserstein-1 small relative to the 68% interval width")
    print("  => narrow-sky prior does not bias the headline parameters; the")
    print("     speedup is purely a sampling-cost effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
