#!/usr/bin/env python3
"""Update a single run's status field in run_catalog.csv.

Usage: _update_catalog_status.py <run_id> <status> <catalog_csv>

status is one of: pending, in_progress, done, failed.
Idempotent; rewrites the CSV in place with a simple atomic swap.
"""
import csv
import os
import sys
import tempfile

VALID_STATUSES = {"pending", "in_progress", "done", "failed"}


def main(run_id: str, status: str, catalog_path: str) -> int:
    if status not in VALID_STATUSES:
        print(f"invalid status: {status}", file=sys.stderr)
        return 2
    if not os.path.exists(catalog_path):
        print(f"catalog not found: {catalog_path}", file=sys.stderr)
        return 1

    with open(catalog_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("empty catalog", file=sys.stderr)
        return 1

    fieldnames = rows[0].keys()
    updated = False
    for row in rows:
        if row["run_id"] == run_id:
            row["status"] = status
            updated = True
            break

    if not updated:
        print(f"run_id not found in catalog: {run_id}", file=sys.stderr)
        return 3

    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(os.path.abspath(catalog_path)),
        prefix=".catalog.",
        suffix=".csv.tmp",
    )
    os.close(fd)
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, catalog_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
