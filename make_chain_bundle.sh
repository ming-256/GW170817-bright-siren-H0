#!/usr/bin/env bash
# Yang et al. (2026) MNRAS — build the chain bundle for the Zenodo deposit.
#
# The chains are not in git: 58 files, 3.8 GB, one of them 1.2 GB, which is
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

VERSION=""
SOURCE=""
while (( $# )); do
    case "$1" in
        --source) SOURCE="${2:?--source needs a directory}"; shift 2 ;;
        --source=*) SOURCE="${1#*=}"; shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) VERSION="$1"; shift ;;
    esac
done
VERSION="${VERSION:-$(date -u +%Y%m%d)}"
OUT="gw170817-chains-${VERSION}.tar.gz"
MANIFEST="results/CHAIN_MANIFEST.csv"

if [[ ! -f "$MANIFEST" ]]; then
    echo "error: $MANIFEST not found — run this from the repository root." >&2
    exit 1
fi

# Take the file list from the manifest, so the bundle and the checksums that
# ship in git can never drift apart.
FILES=()
while IFS= read -r line; do
    [[ -n "$line" ]] && FILES+=("$line")
done < <(tail -n +2 "$MANIFEST" | cut -d, -f1)

echo "Chain files listed in the manifest: ${#FILES[@]}"

# Resolve one manifest path to a real file. With --source, look in that tree
# too, and accept the working repository's capitalised Results/ as well as
# this repository's results/ — the two layouts differ only in that letter,
# which matters on every filesystem except a case-insensitive macOS one.
resolve() {
    local rel="$1" c
    for c in "$rel" \
             ${SOURCE:+"$SOURCE/$rel"} \
             ${SOURCE:+"$SOURCE/${rel/#results\//Results/}"}; do
        if [[ -s "$c" ]]; then printf '%s' "$c"; return 0; fi
    done
    return 1
}

# A Git LFS pointer is a ~130-byte text file starting with this line. Catching
# it here is worth the trouble: tarring pointers would produce a bundle that
# looks the right shape and is entirely useless.
is_lfs_pointer() {
    [[ $(stat -c%s "$1" 2>/dev/null || stat -f%z "$1") -lt 1024 ]] &&
        head -c 40 "$1" | grep -q 'git-lfs.github.com'
}

missing=0; pointers=0
declare -a SRC
for f in "${FILES[@]}"; do
    if src=$(resolve "$f"); then
        if is_lfs_pointer "$src"; then
            echo "  LFS POINTER (not the real file): $src" >&2
            pointers=$((pointers + 1))
        fi
        SRC+=("$src")
    else
        echo "  MISSING: $f" >&2
        missing=$((missing + 1))
        SRC+=("")
    fi
done

if (( pointers )); then
    echo >&2
    echo "error: $pointers file(s) are Git LFS pointers, not data." >&2
    echo "       Run 'git lfs pull' in the checkout holding the chains, then retry." >&2
    exit 1
fi
if (( missing )); then
    echo >&2
    echo "error: $missing chain file(s) not found." >&2
    if [[ -z "$SOURCE" ]]; then
        echo "       If the chains live in the working repository rather than here," >&2
        echo "       point at it:  bash make_chain_bundle.sh $VERSION --source /path/to/repo" >&2
    else
        echo "       Looked in this repository and in $SOURCE (both results/ and Results/)." >&2
        echo "       If it is a Git LFS checkout, run 'git lfs pull' there first." >&2
    fi
    exit 1
fi

# When the files come from elsewhere, stage them under the canonical
# lowercase paths so the tarball unpacks straight into a clone. Hardlink
# where the filesystem allows it; 3.8 GB is not worth copying twice.
STAGE=""
if [[ -n "$SOURCE" ]]; then
    STAGE="$(mktemp -d "${TMPDIR:-/tmp}/gw170817-bundle.XXXXXX")"
    trap 'rm -rf "$STAGE"' EXIT
    echo "Staging into $STAGE ..."
    for i in "${!FILES[@]}"; do
        mkdir -p "$STAGE/$(dirname "${FILES[$i]}")"
        ln "${SRC[$i]}" "$STAGE/${FILES[$i]}" 2>/dev/null ||
            cp "${SRC[$i]}" "$STAGE/${FILES[$i]}"
    done
    OUT="$ROOT/$OUT"
    cd "$STAGE"
fi

echo "Building $OUT ..."
# On macOS, bsdtar stores AppleDouble "._" companions for extended attributes
# unless told otherwise. They would end up in the deposit and confuse anyone
# unpacking it on Linux.
COPYFILE_DISABLE=1 tar -czf "$OUT" "${FILES[@]}"

# Absolute, because with --source the cwd is a staging dir that is about to
# be removed.
[[ "$OUT" = /* ]] || OUT="$ROOT/$OUT"
SUM="${OUT%.tar.gz}.sha256"

if command -v sha256sum >/dev/null 2>&1; then
    ( cd "$(dirname "$OUT")" && sha256sum "$(basename "$OUT")" ) > "$SUM"
else
    # macOS
    ( cd "$(dirname "$OUT")" && shasum -a 256 "$(basename "$OUT")" ) > "$SUM"
fi

echo
echo "=== done ==="
ls -lh "$OUT" "$SUM" | awk '{printf "  %6s  %s\n", $5, $9}'
echo
echo "Upload $OUT to Zenodo as a new version of 10.5281/zenodo.21038511."
echo "Consumers then get it with:  bash fetch_data.sh chains"
echo "and check it with:           bash fetch_data.sh verify"
