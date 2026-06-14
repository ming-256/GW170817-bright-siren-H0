#!/usr/bin/env python3
"""Regenerate the cross-waveform H0 table (Table 4, tab:waveform-h0) from the
released sample CSVs, so every number in that table traces to data + this
script. The row values were previously hand-typed in main.tex, which let the
95 per cent HPD upper bounds drift from the released chains.

Rows (both are the volumetric d_L^2 "LVK-matched" baseline; only the waveform
differs):
  IMRPhenomXAS_NRTidalv3 (primary)
      <- results/test_suite/s14__gw170817__imrphenomxas_nrtidalv3__baseline__seed0000
         (lnZ parsed from that run's sampler.log; chains regenerated via run_chains.sh)
  TaylorF2 (family check)
      <- results/gwtc1_phasemarg/PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_baseline.csv
         lnZ 486.90 +/- 0.10 (from the TaylorF2-baseline sampler log)

Output: results/gwtc1_phasemarg/tableW_waveform.tex
        (regenerate.sh mirrors it to paper/tables/tableW_waveform.tex, which
         main.tex \\input{}s)

MAP / HPD / tail conventions are imported from build_paper_tables.py so this
table uses the identical estimators and LaTeX formatting as Tables 1/5/6.
"""
import os

import numpy as np
from anesthetic import read_chains

import build_paper_tables as bpt

ROOT = bpt.ROOT
TF_CSV = os.path.join(
    ROOT, 'results', 'gwtc1_phasemarg',
    'PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_baseline.csv')
TF_LNZ, TF_DLNZ = 486.90, 0.10
IMRX_RUN = 's14__gw170817__imrphenomxas_nrtidalv3__baseline__seed0000'
OUT = os.path.join(ROOT, 'results', 'gwtc1_phasemarg', 'tableW_waveform.tex')


def _row(label, x, w, lnz, dlnz):
    finite = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[finite], w[finite]
    lo68, hi68 = bpt.hpd(x, w, 0.68269)
    lo95, hi95 = bpt.hpd(x, w, 0.95450)
    print(f"  {label}: MAP={bpt.map_h0(x, w):.1f}  68%={bpt._hpd_str(lo68, hi68)}  "
          f"95%={bpt._hpd_str(lo95, hi95)}  P>120={bpt.tail(x, w, 120):.3f}  "
          f"lnZ={lnz:.2f}±{dlnz:.2f}")
    return (f"    {label:<28} & {bpt.map_h0(x, w):.1f} & "
            f"{bpt._hpd_str(lo68, hi68)} & {bpt._hpd_str(lo95, hi95)} & "
            f"{bpt._tail_str(bpt.tail(x, w, 120))} & {bpt._lnz_str(lnz, dlnz)} \\\\\n")


def main():
    xi, wi = bpt.load_samples(IMRX_RUN)
    lnzi, dlnzi = bpt.parse_lnZ(IMRX_RUN)

    s = read_chains(TF_CSV)
    xt = s['H_0'].to_numpy().astype(float)
    wt = np.asarray(s.get_weights(), dtype=float)

    body = ''
    body += _row(r'\IMRX', xi, wi, lnzi, dlnzi)
    body += _row(r'\TF', xt, wt, TF_LNZ, TF_DLNZ)
    body += '    \\bottomrule\n'

    with open(OUT, 'w') as f:
        f.write(body)
    print('\nWrote', OUT)


if __name__ == '__main__':
    main()
