"""Shared helpers for test_suite analysis scripts.

All analysis scripts read from Results/test_suite/<run_id>/{samples.csv, config.json}
and write derived tables alongside the session outputs. No GPU, no sampler.

Intended usage:

    from _helpers import (
        REPO_ROOT, RESULTS_ROOT, load_run, weighted_quantiles,
        weighted_tail_prob, weighted_map, weighted_median, weighted_wasserstein1,
    )

The CSVs produced by BlackJAX-NS here are anesthetic-compatible nested-sample
files with a 'weight' (or 'w') column. We support both.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_SUITE_ROOT = os.path.join(REPO_ROOT, "results", "test_suite")
RESULTS_ROOT = TEST_SUITE_ROOT
CATALOG = os.path.join(TEST_SUITE_ROOT, "run_catalog.csv")


@dataclass
class Run:
    run_id: str
    samples: pd.DataFrame
    weights: np.ndarray
    config: dict

    def param(self, name: str) -> np.ndarray:
        # Column name may be 'H_0' / 'h0' / 'H0' — try each.
        for alias in (name, name.upper(), name.lower(), name.replace("_", "")):
            if alias in self.samples.columns:
                return self.samples[alias].to_numpy()
        raise KeyError(f"{name} not found in samples for {self.run_id} (cols: {list(self.samples.columns)})")


def _locate_weight_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("weight", "weights", "w", "nested_sample_weight", "posterior_weight"):
        if candidate in df.columns:
            return candidate
    return None


def read_nested_samples_csv(csv_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a nested-samples CSV in anesthetic format or plain format.

    Anesthetic format (produced by the project's sampler) has two metadata rows:
    row 0: column names, with first column blank (index) and second column also blank
    row 1: 'labels,' + LaTeX labels for each column
    row 2: ',weights,,,,,...'  (the literal word 'weights' in column 1)
    row 3: first data row — column 0 is the integer sample index, column 1 is weight.

    Plain format: one header row, optional 'weight' column.
    """
    # Peek at the first couple of lines to determine format.
    with open(csv_path, "r") as fh:
        first_lines = [fh.readline() for _ in range(4)]

    is_anesthetic = (
        len(first_lines) >= 3
        and first_lines[0].startswith(",")
        and "labels" in first_lines[1]
    )
    if is_anesthetic:
        # Row 0 has header; rows 1 and 2 are label/weights metadata.
        df = pd.read_csv(csv_path, skiprows=[1, 2], low_memory=False)
        # Column 0 is the sample index (unnamed). Column 1 is weight (unnamed).
        cols = list(df.columns)
        # The first two unnamed columns typically render as 'Unnamed: 0', 'Unnamed: 1'.
        if cols[0].startswith("Unnamed") and cols[1].startswith("Unnamed"):
            df = df.rename(columns={cols[0]: "sample_index", cols[1]: "weight"})
        weights = df["weight"].to_numpy(dtype=float)
    else:
        df = pd.read_csv(csv_path, low_memory=False)
        wcol = _locate_weight_column(df)
        if wcol is None:
            weights = np.ones(len(df), dtype=float) / len(df)
        else:
            weights = df[wcol].to_numpy(dtype=float)
    # Normalise weights (tolerate log-weights with wide dynamic range).
    if weights.sum() <= 0 or not np.isfinite(weights.sum()):
        # Some runs may store log-weights; convert if we detect very negative values.
        lw = np.asarray(weights, dtype=float)
        weights = np.exp(lw - np.nanmax(lw))
    weights = weights / weights.sum()
    return df, weights


def load_run(run_id: str, results_root: str = RESULTS_ROOT) -> Run:
    """Load a single test-suite run (samples.csv + config.json)."""
    run_dir = os.path.join(results_root, run_id)
    csv_path = os.path.join(run_dir, "samples.csv")
    cfg_path = os.path.join(run_dir, "config.json")

    samples, weights = read_nested_samples_csv(csv_path)

    config = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            config = json.load(fh)

    return Run(run_id=run_id, samples=samples, weights=weights, config=config)


def load_runs(run_ids: Iterator[str]) -> list[Run]:
    return [load_run(rid) for rid in run_ids]


# ----- Weighted statistics -----

def weighted_quantiles(x: np.ndarray, w: np.ndarray, qs: list[float]) -> np.ndarray:
    order = np.argsort(x)
    x_s, w_s = x[order], w[order]
    cum = np.cumsum(w_s) / w_s.sum()
    return np.interp(qs, cum, x_s)


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    return float(weighted_quantiles(x, w, [0.5])[0])


def weighted_map(x: np.ndarray, w: np.ndarray, bins: int = 128) -> float:
    hist, edges = np.histogram(x, bins=bins, weights=w)
    i = int(np.argmax(hist))
    return float(0.5 * (edges[i] + edges[i + 1]))


def weighted_tail_prob(x: np.ndarray, w: np.ndarray, threshold: float, above: bool = True) -> float:
    mask = (x > threshold) if above else (x < threshold)
    return float(w[mask].sum() / w.sum())


def weighted_wasserstein1(x1: np.ndarray, w1: np.ndarray, x2: np.ndarray, w2: np.ndarray, grid_n: int = 4000) -> float:
    lo = float(min(x1.min(), x2.min()))
    hi = float(max(x1.max(), x2.max()))
    grid = np.linspace(lo, hi, grid_n)
    cdf1 = np.interp(grid, np.sort(x1), np.cumsum(w1[np.argsort(x1)]) / w1.sum())
    cdf2 = np.interp(grid, np.sort(x2), np.cumsum(w2[np.argsort(x2)]) / w2.sum())
    return float(np.trapezoid(np.abs(cdf1 - cdf2), grid))


# ----- Evidence extraction -----

def read_log_evidence_from_log(run_dir: str) -> tuple[Optional[float], Optional[float]]:
    """Return (log_z, sigma_log_z) for a run.

    First looks at config.json for 'log_evidence'/'sigma_log_evidence' (set when
    importing externally completed runs). Falls back to parsing sampler.log for
    common log-Z formats: 'Log Evidence: X +/- Y', 'log Z = X +/- Y', etc."""
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as fh:
                cfg = json.load(fh)
            if "log_evidence" in cfg:
                return float(cfg["log_evidence"]), float(cfg.get("sigma_log_evidence", 0.0)) or None
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    log_path = os.path.join(run_dir, "sampler.log")
    if not os.path.exists(log_path):
        return None, None
    import re
    log_z, sigma = None, None
    with open(log_path) as fh:
        for line in fh:
            low = line.lower()
            if "log z" in low or "log_z" in low or "log-evidence" in low or "log evidence" in low:
                # Match: "log Z = 486.67 +/- 0.09", "Log Evidence: 490.51 +/- 0.14"
                m = re.search(
                    r"log[_ ]?(?:z|evidence)[:=\s]+(-?[\d.]+)\s*(?:\+/-|±|\+\-)?\s*(-?[\d.]+)?",
                    low,
                )
                if m:
                    try:
                        log_z = float(m.group(1))
                        sigma = float(m.group(2)) if m.group(2) else sigma
                    except ValueError:
                        pass
    return log_z, sigma


# ----- Catalog helpers -----

def load_catalog() -> pd.DataFrame:
    return pd.read_csv(CATALOG)


def runs_by_session(catalog: pd.DataFrame, session: str) -> list[str]:
    return catalog.loc[catalog["session"] == session, "run_id"].tolist()


def runs_done(catalog: pd.DataFrame) -> list[str]:
    return catalog.loc[catalog["status"] == "done", "run_id"].tolist()
