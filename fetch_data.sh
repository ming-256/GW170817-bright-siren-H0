#!/usr/bin/env bash
# Yang et al. (2026) MNRAS — fetch the bulk data this repository does not carry.
#
# Git holds the code, the per-run provenance (config.json, sampler.log), the
# derived summary tables and the figures. The nested-sampling chains live on
# Zenodo instead: 58 files, 3.8 GB, one of them 1.2 GB, which is well past
# what belongs in a git repository (and past GitHub's 100 MB per-file limit).
#
#   chains   the 58 chain CSVs, from the Zenodo deposit. Needed by
#            regenerate.sh and by everything in analysis/.
#   figures  the 287 MB LVK GWTC-2.1 GW150914 PE release, needed only by
#            Figure 1 (scripts/plot_GW150914_waveform_comparison.py).
#   strain   the LVK GW170817 and GW150914 strain + PSD inputs, needed only
#            if you are re-running the sampler via run_chains.sh.
#
# Usage:
#   bash fetch_data.sh chains       # 3.8 GB  — required to rebuild the paper
#   bash fetch_data.sh figures      # 287 MB  — to rebuild Figure 1
#   bash fetch_data.sh strain       # ~440 MB — to re-run the sampler
#   bash fetch_data.sh all
#
# Downloads are skipped if the target already exists, so re-running is cheap.
# After fetching chains, verify them against results/CHAIN_MANIFEST.csv:
#   bash fetch_data.sh verify

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GW170817_DIR="${GWOSC_GW170817_DIR:-data/GWOSC/GW170817}"
GW150914_DIR="${GWOSC_GW150914_DIR:-data/GWOSC/GW150914}"

get() {
    local url="$1" dest="$2"
    if [[ -s "$dest" ]]; then
        echo "  have  $dest"
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    echo "  get   $dest"
    # --fail so an HTML error page never lands on disk masquerading as data.
    if ! curl -fL --retry 3 --retry-delay 2 -o "$dest.part" "$url"; then
        rm -f "$dest.part"
        echo "  error: download failed: $url" >&2
        return 1
    fi
    mv -f "$dest.part" "$dest"
}

fetch_figures() {
    echo "--- LVK GWTC-2.1 GW150914 PE release (Zenodo 10.5281/zenodo.6513631) ---"
    get "https://zenodo.org/record/6513631/files/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_nocosmo.h5" \
        "results/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_nocosmo.h5"
}

fetch_strain() {
    echo "--- GW170817 strain, cleaned 2048 s @ 4096 Hz (GWOSC, LIGO-P1700349) ---"
    local base="https://gwosc.org/s/events/GW170817"
    get "${base}/H-H1_LOSC_CLN_4_V1-1187007040-2048.hdf5" "${GW170817_DIR}/H-H1_LOSC_CLN_4_V1-1187007040-2048.hdf5"
    get "${base}/L-L1_LOSC_CLN_4_V1-1187007040-2048.hdf5" "${GW170817_DIR}/L-L1_LOSC_CLN_4_V1-1187007040-2048.hdf5"
    get "${base}/V-V1_LOSC_CLN_4_V1-1187007040-2048.hdf5" "${GW170817_DIR}/V-V1_LOSC_CLN_4_V1-1187007040-2048.hdf5"

    echo "--- GW170817 BayesWave PSDs (LIGO-P1900011), for --psd-source gwtc1 ---"
    get "https://dcc.ligo.org/public/0158/P1900011/001/GWTC1_GW170817_PSDs.dat" \
        "${GW170817_DIR}/GWTC1_GW170817_PSDs.dat"

    echo "--- GW150914 strain, 4096 s @ 4096 Hz (GWOSC O1 archive) ---"
    local o1="https://gwosc.org/archive/data/O1/1126170624"
    get "${o1}/H-H1_LOSC_4_V1-1126256640-4096.hdf5" "${GW150914_DIR}/H-H1_LOSC_4_V1-1126256640-4096.hdf5"
    get "${o1}/L-L1_LOSC_4_V1-1126256640-4096.hdf5" "${GW150914_DIR}/L-L1_LOSC_4_V1-1126256640-4096.hdf5"
}

# Concept DOI: always resolves to the newest version of the deposit, so this
# keeps working across releases without a hardcoded record ID.
ZENODO_CONCEPT_ID="21038511"

fetch_chains() {
    echo "--- nested-sampling chains (Zenodo concept DOI 10.5281/zenodo.${ZENODO_CONCEPT_ID}) ---"
    local api="https://zenodo.org/api/records/${ZENODO_CONCEPT_ID}"
    local json
    if ! json="$(curl -fsSL --retry 3 "$api")"; then
        echo "  error: could not reach the Zenodo API at ${api}" >&2
        return 1
    fi

    # Match only files whose name marks them as a chain bundle. The deposit
    # also carries the auto-generated GitHub release snapshot of this
    # repository, which must NOT be matched: it is the code, not the data,
    # and unpacking it over results/ would be wrong.
    local urls
    urls="$(printf '%s' "$json" | python3 -c '
import json, re, sys
d = json.load(sys.stdin)
print("  resolved to version:", d.get("metadata", {}).get("version", "?"), file=sys.stderr)
# The GitHub release snapshot is named "<owner>/<repo>-v<x.y.z>.zip".
snapshot = re.compile(r"GW170817-bright-siren-H0-v[\d.]+\.zip$", re.I)
for f in d.get("files", []):
    key = f.get("key", "")
    if snapshot.search(key):
        continue
    if not re.search(r"chain", key, re.I):
        continue
    link = (f.get("links") or {}).get("self") or ""
    if link:
        print(f"{key}\t{link}")
')"
    if [[ -z "$urls" ]]; then
        echo "  error: no chain bundle found in the deposit." >&2
        echo "         Expected a file with \"chain\" in its name (e.g." >&2
        echo "         gw170817-chains-v1.tar.gz). If the chains have not been" >&2
        echo "         uploaded yet, see docs/chain_regeneration.md to regenerate" >&2
        echo "         them on a GPU instead." >&2
        return 1
    fi

    # The bundle stores repository-relative paths (results/test_suite/...), so
    # it unpacks at the repository root, not inside results/. Extracting into
    # results/ would produce results/results/... and leave every chain where
    # nothing looks for it.
    mkdir -p .cache
    while IFS=$'\t' read -r key link; do
        [[ -z "$key" ]] && continue
        local archive=".cache/$(basename "$key")"
        get "$link" "$archive"
        case "$key" in
            *.tar.gz)  echo "  unpacking $key"; tar -xzf "$archive" -C . ;;
            *.tar.zst) echo "  unpacking $key"; tar --zstd -xf "$archive" -C . ;;
            *.zip)     echo "  unpacking $key"; unzip -q -o "$archive" -d . ;;
        esac
    done <<< "$urls"

    echo
    echo "  now verify: bash fetch_data.sh verify"
}

verify_chains() {
    echo "--- verifying chains against results/CHAIN_MANIFEST.csv ---"
    python3 - <<'PY'
import csv, hashlib, os, sys
man = 'results/CHAIN_MANIFEST.csv'
if not os.path.exists(man):
    sys.exit("  error: results/CHAIN_MANIFEST.csv not found")
ok = missing = bad = 0
for r in csv.DictReader(open(man)):
    p = r['path']
    if not os.path.exists(p):
        missing += 1
        continue
    if os.path.getsize(p) != int(r['bytes']):
        print(f"  SIZE MISMATCH {p}"); bad += 1; continue
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(8 << 20), b''):
            h.update(c)
    if h.hexdigest() != r['sha256']:
        print(f"  CHECKSUM MISMATCH {p}"); bad += 1
    else:
        ok += 1
print(f"  verified {ok}, missing {missing}, corrupt {bad}")
sys.exit(1 if bad else 0)
PY
}

case "${1:-all}" in
    chains)  fetch_chains ;;
    verify)  verify_chains ;;
    figures) fetch_figures ;;
    strain)  fetch_strain ;;
    all)     fetch_chains; fetch_figures; fetch_strain ;;
    *)
        echo "Usage: bash fetch_data.sh [chains | verify | figures | strain | all]" >&2
        exit 1
        ;;
esac

echo
echo "=== fetch_data.sh: complete ==="
