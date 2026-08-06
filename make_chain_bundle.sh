#!/usr/bin/env bash
# Yang et al. (2026) MNRAS — build the chain bundle for the Zenodo deposit.
#
# The chains are not in git: 58 files, 3.7 GB, one of them 1.2 GB, which is
# past GitHub's 100 MB per-file limit. They are published on Zenodo instead,
# and fetch_data.sh pulls them back down.
#
# Run this where the chains actually live (the working repository), with
# results/ populated. It produces:
#
#   gw170817-chains-<version>.tar.gz   the bundle to upload to Zenodo
#   gw170817-chains-<version>.sha256   its checksum
#
# The name must contain "chain": fetch_data.sh matches on that, and
# deliberately ignores the auto-generated GitHub release snapshot that sits
# in the same deposit.
#
# Usage:
#   bash make_chain_bundle.sh                 # version defaults to today's date
#   bash make_chain_bundle.sh v1.1.0
#
# Then, on Zenodo: open the existing deposit (concept DOI
# 10.5281/zenodo.21038511) -> "New version" -> upload the tarball -> publish.
# That mints a new version DOI under the same concept DOI, so the DOI cited
# in the paper keeps resolving to the newest version and needs no change.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VERSION="${1:-$(date -u +%Y%m%d)}"
OUT="gw170817-chains-${VERSION}.tar.gz"
MANIFEST="results/CHAIN_MANIFEST.csv"

if [[ ! -f "$MANIFEST" ]]; then
    echo "error: $MANIFEST not found — run this from the repository root." >&2
    exit 1
fi

# Take the file list from the manifest, so the bundle and the checksums that
# ship in git can never drift apart.
mapfile -t FILES < <(tail -n +2 "$MANIFEST" | cut -d, -f1)

echo "Chain files listed in the manifest: ${#FILES[@]}"

missing=0
for f in "${FILES[@]}"; do
    if [[ ! -s "$f" ]]; then
        echo "  MISSING: $f" >&2
        missing=$((missing + 1))
    fi
done
if (( missing )); then
    echo >&2
    echo "error: $missing chain file(s) missing. If this is a Git LFS checkout," >&2
    echo "       run 'git lfs pull' first." >&2
    exit 1
fi

echo "Building $OUT ..."
tar -czf "$OUT" "${FILES[@]}"

sha256sum "$OUT" > "gw170817-chains-${VERSION}.sha256"

echo
echo "=== done ==="
ls -lh "$OUT" "gw170817-chains-${VERSION}.sha256" | awk '{printf "  %6s  %s\n", $5, $9}'
echo
echo "Upload $OUT to Zenodo as a new version of 10.5281/zenodo.21038511."
echo "Consumers then get it with:  bash fetch_data.sh chains"
echo "and check it with:           bash fetch_data.sh verify"
