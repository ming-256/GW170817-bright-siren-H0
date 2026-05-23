#!/usr/bin/env python3
"""Walk Results/test_suite/ and verify manifest consistency.

For each directory under Results/test_suite/:
 - check samples.csv exists (or prior_samples.csv for session H)
 - check config.json exists
 - check run_id matches the directory name
 - look up the row in run_catalog.csv and flag status mismatches

For each catalog row with status=done, check that the expected CSV exists.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import REPO_ROOT, RESULTS_ROOT, CATALOG


def main() -> int:
    problems = []

    with open(CATALOG, newline="") as fh:
        catalog = list(csv.DictReader(fh))
    by_id = {row["run_id"]: row for row in catalog}

    # Files on disk -> catalog
    for entry in sorted(os.listdir(RESULTS_ROOT)):
        run_dir = os.path.join(RESULTS_ROOT, entry)
        if not os.path.isdir(run_dir):
            continue
        cfg_path = os.path.join(run_dir, "config.json")
        samples_pattern = any(
            os.path.exists(os.path.join(run_dir, name))
            for name in ("samples.csv", "prior_samples.csv")
        )
        has_cfg = os.path.exists(cfg_path)
        if not has_cfg:
            problems.append(f"missing config.json: {run_dir}")
        if not samples_pattern:
            problems.append(f"missing samples.csv (or prior_samples.csv): {run_dir}")
        if has_cfg:
            try:
                with open(cfg_path) as fh:
                    cfg = json.load(fh)
                if cfg.get("run_id") != entry:
                    problems.append(f"config.run_id mismatch: {entry} vs {cfg.get('run_id')}")
            except json.JSONDecodeError as exc:
                problems.append(f"invalid config.json: {run_dir}: {exc}")

        if entry not in by_id:
            problems.append(f"orphan run on disk, not in catalog: {entry}")

    # Catalog rows with status=done must have their CSVs on disk.
    for row in catalog:
        if row["status"] == "done":
            expected_csv = os.path.join(REPO_ROOT, row["expected_csv"])
            if not os.path.exists(expected_csv):
                problems.append(f"status=done but file missing: {row['run_id']} -> {expected_csv}")

    if problems:
        print(f"Found {len(problems)} issue(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Manifest consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
