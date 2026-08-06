#!/usr/bin/env python3
"""Regenerate results/CHAIN_MANIFEST.csv from the chains on disk.

The manifest is the contract between this repository and the Zenodo
deposit: fetch_data.sh verify checks a download against it, and
make_chain_bundle.sh takes its file list from it. So it must never record
anything that is not a real chain.

It once did. In two run directories samples.csv is a symlink to the
PhaseMarg_*.csv beside it, and in a third it points at a scaling-study
chain; when those targets were still Git LFS pointers, hashing the
symlink recorded a 133-byte pointer as though it were a 45 MB chain.
verify then passed, because it faithfully compared against a wrong
manifest. The checks below exist so that cannot happen again:

  - refuse symlinks (resolve them before recording),
  - refuse Git LFS pointer files,
  - refuse anything under MIN_CHAIN_BYTES, since the smallest real chain
    in this release is 5.5 MB,
  - refuse to write a manifest that is empty or that loses chains the
    existing one records, so running this in a clone without the chains
    cannot silently erase the contract.

Usage:
  python make_chain_manifest.py            # rewrite results/CHAIN_MANIFEST.csv
  python make_chain_manifest.py --check    # verify it is current; do not write
"""
import csv
import glob
import hashlib
import os
import sys

MIN_CHAIN_BYTES = 1_000_000
MANIFEST = "results/CHAIN_MANIFEST.csv"

SOURCES = [
    ("results/test_suite/*/samples.csv", "per-run nested-sampling chain"),
    ("results/test_suite/*/prior_samples.csv", "prior-only draw"),
    ("results/gwtc1_phasemarg/*PhaseMarg*.csv", "host-localised prior-variant chain"),
    ("results/scaling_study/*PhaseMarg*.csv", "scaling-study chain"),
]


def looks_like_lfs_pointer(path):
    if os.path.getsize(path) >= 1024:
        return False
    with open(path, "rb") as fh:
        return b"git-lfs.github.com" in fh.read(200)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect():
    rows, problems = [], []
    for pattern, note in SOURCES:
        for path in sorted(glob.glob(pattern)):
            if os.path.islink(path):
                problems.append(f"symlink (resolve it first): {path}")
                continue
            if looks_like_lfs_pointer(path):
                problems.append(f"Git LFS pointer, not data: {path}")
                continue
            size = os.path.getsize(path)
            if size < MIN_CHAIN_BYTES:
                problems.append(f"too small to be a chain ({size} B): {path}")
                continue
            rows.append(dict(path=path, bytes=size, sha256=sha256(path), note=note))
    rows.sort(key=lambda r: r["path"])
    return rows, problems


def existing_rows():
    if not os.path.exists(MANIFEST):
        return None
    with open(MANIFEST) as fh:
        return list(csv.DictReader(fh))


def main():
    check_only = "--check" in sys.argv[1:]
    rows, problems = collect()
    existing = existing_rows()

    # No chains on disk at all is the ordinary state of a fresh clone, not a
    # manifest that has drifted. Say so, and never treat it as a reason to
    # rewrite: without this guard, running the script here would replace 58
    # rows with a header and nothing else.
    if not rows:
        print(
            "no chains found on disk — nothing to record.\n"
            "This is expected in a fresh clone; run 'bash fetch_data.sh chains'\n"
            f"to download them. {MANIFEST} left untouched.",
            file=sys.stderr,
        )
        return 1

    if problems:
        print("refusing to write the manifest:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nIf this is a Git LFS checkout, run 'git lfs pull'. If a path is a\n"
            "symlink, copy the real file over it (cp -L) before regenerating.",
            file=sys.stderr,
        )
        return 1

    if check_only:
        if existing is None:
            print(f"{MANIFEST} does not exist", file=sys.stderr)
            return 1
        current = [{k: str(v) for k, v in r.items()} for r in rows]
        if existing != current:
            print(f"{MANIFEST} is out of date; rerun without --check", file=sys.stderr)
            return 1
        print(f"{MANIFEST} is current: {len(rows)} chains")
        return 0

    # A partial checkout must not quietly shrink the published set.
    if existing is not None and len(rows) < len(existing):
        lost = {r["path"] for r in existing} - {r["path"] for r in rows}
        print(
            f"refusing to write: {MANIFEST} records {len(existing)} chains but only "
            f"{len(rows)} are on disk.\nMissing {len(lost)}, including:",
            file=sys.stderr,
        )
        for p in sorted(lost)[:5]:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nFetch the full set first, or delete the manifest deliberately if the\n"
            "release really is losing these chains.",
            file=sys.stderr,
        )
        return 1

    with open(MANIFEST, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "bytes", "sha256", "note"])
        writer.writeheader()
        writer.writerows(rows)

    total = sum(r["bytes"] for r in rows)
    print(f"wrote {MANIFEST}")
    print(f"  chains: {len(rows)}")
    print(f"  total:  {total / 2**30:.2f} GB")
    print(f"  range:  {min(r['bytes'] for r in rows) / 2**20:.1f} MB "
          f"– {max(r['bytes'] for r in rows) / 2**20:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
