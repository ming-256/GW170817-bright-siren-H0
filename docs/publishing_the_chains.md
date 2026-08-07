# Publishing the chains to Zenodo

A runbook for the one step that cannot be done from a code repository:
getting the 3.8 GB of nested-sampling chains into the Zenodo deposit, so
that `bash fetch_data.sh chains` works for everyone else.

You only need this when the chains change. Ordinary readers never run it.

**Time:** ~15 minutes of your attention, plus a 1.8 GB upload.

---

## 0. What you need

| | |
|---|---|
| The **chains** | wherever the working repository's `Results/` tree lives, with Git LFS objects fetched |
| This **release repository** | `GW170817-bright-siren-H0` |
| A **bash shell** | Git Bash on Windows (ships with Git for Windows), or WSL; any shell on macOS/Linux |
| `git-lfs` | included in Git for Windows; else `brew install git-lfs` / `apt install git-lfs`, then `git lfs install` |
| Python 3 | for the manifest tooling; no third-party packages needed |
| Disk | ~9 GB free (3.8 GB of chains + a 1.8 GB tarball + staging) |

The scripts are written for bash 3.2, so the stock `/bin/bash` on macOS is
fine, and they find `python` when `python3` does not exist, which is the
usual Windows situation.

### Windows: paths and shell

Run everything from **Git Bash**, not `cmd` or PowerShell. Git Bash
rewrites drive letters, so translate Windows paths like this:

| Windows | Git Bash |
|---|---|
| `C:\Users\Ming\Documents\CambridgeProject` | `/c/Users/Ming/Documents/CambridgeProject` |
| `F:\ProjectLatest\CambridgeProject` | `/f/ProjectLatest/CambridgeProject` |

(Under WSL it is `/mnt/c/...` and `/mnt/f/...` instead.)

One performance note: the bundler hardlinks files into a staging directory
rather than copying 3.8 GB, but hardlinks cannot cross volumes. If the
chains are on `F:` and your temp directory is on `C:`, it silently falls
back to copying. Put the staging directory on the same drive to avoid
that:

```bash
export TMPDIR=/f/tmp && mkdir -p "$TMPDIR"
```

---

## 1. Get the chains onto disk, for real

In the tree that holds the chains:

```bash
cd /f/ProjectLatest/CambridgeProject          # or wherever Results/ lives
git lfs install
git lfs pull
```

`git lfs pull` is the step people skip. Without it the chain files are
~130-byte pointer stubs, and a bundle built from them looks the right
shape and is worthless. The tooling refuses to build in that state, but it
is quicker to just pull first.

Three of the chains are symlinks rather than files — in the two `s19` runs
`samples.csv` points at the `PhaseMarg_*.csv` beside it, and in `s04` at a
scaling-study chain elsewhere. The bundler resolves them for you.

> **Windows and symlinks.** Git only creates real symlinks on Windows when
> `core.symlinks` is on and the account may create them; otherwise it
> writes a small text file containing the target path. If step 3 reports
> those three as missing or too small, that is what happened. Fix it by
> copying the target over the stub, e.g.
> `cp results/test_suite/s19__*imrphenomd*fixedsky*/PhaseMarg_*.csv results/test_suite/s19__*imrphenomd*fixedsky*/samples.csv`,
> or clone with `git clone -c core.symlinks=true`.

## 2. Get the release repository

```bash
cd /c/Users/Ming/Documents
git clone https://github.com/ming-256/GW170817-bright-siren-H0
cd GW170817-bright-siren-H0
```

## 3. Build the bundle

Point it at the tree holding the chains. It maps that tree's capitalised
`Results/` onto the release layout's `results/`, hardlinks where it can,
and refuses to proceed on a pointer or a missing file.

```bash
bash make_chain_bundle.sh v1.1.0 --source /f/ProjectLatest/CambridgeProject
```

Expect roughly:

```
Chain files listed in the manifest: 58
Staging into /f/tmp/gw170817-bundle.XXXX ...
Building gw170817-chains-v1.1.0.tar.gz ...

=== done ===
   1.8G  gw170817-chains-v1.1.0.tar.gz
    96   gw170817-chains-v1.1.0.sha256
```

About five minutes, nearly all of it gzip.

**If it stops with `LFS POINTER (not the real file)`** — go back to step 1
and run `git lfs pull`. It fails in under a second rather than producing a
bad 1.8 GB artefact.

**If it stops with `MISSING`** — the path it names is in neither tree.
Check the `--source` argument, and see the symlink note in step 1.

### Sanity check before uploading

```bash
tar -tzf gw170817-chains-v1.1.0.tar.gz | wc -l      # expect 58
tar -tzf gw170817-chains-v1.1.0.tar.gz | head -3    # paths start with results/
```

Paths must begin `results/…`, not `./results/…` or an absolute path, or it
will not unpack into a clone correctly.

## 4. Upload to Zenodo

1. Go to <https://doi.org/10.5281/zenodo.21038511>. That is the **concept
   DOI** and always lands on the newest version.
2. **New version** — not a new upload. A new version keeps the concept DOI,
   and so keeps the citation in the paper valid.
3. Upload `gw170817-chains-v1.1.0.tar.gz`.
4. Leave the existing GitHub release snapshot in place. `fetch_data.sh`
   knows to ignore it and take only the chain bundle.
5. In the description, say what the file is — for example: *"Nested-sampling
   chains for all 58 runs (3.8 GB uncompressed). Per-file checksums are in
   `results/CHAIN_MANIFEST.csv` in the code repository; verify a download
   with `bash fetch_data.sh verify`."*
6. **Publish.**

> **Keep the word `chain` in the filename.** `fetch_data.sh` selects the
> bundle by matching `chain` in the name, and explicitly skips the
> `GW170817-bright-siren-H0-v*.zip` snapshot. A file named
> `data-v1.1.0.tar.gz` would not be found.

Publishing mints a new version DOI under the same concept DOI. **Nothing in
the paper needs to change** — it cites the concept DOI.

## 5. Check it from the outside

The real test is whether a stranger can use it. From a clean clone:

```bash
cd /c/Users/Ming/Documents
git clone https://github.com/ming-256/GW170817-bright-siren-H0 check
cd check

bash fetch_data.sh chains     # resolves the concept DOI, downloads, unpacks
bash fetch_data.sh verify     # expect: verified 58, missing 0, corrupt 0
```

Then confirm the paper still comes out:

```bash
conda env create -f environment.yml
conda activate gw170817-bright-siren-H0
bash regenerate.sh tables
```

The prior-sensitivity table should print

```
Baseline ($\pi(d_L)\propto d_L^{2}$): MAP=70.5, ..., P>120=0.017
Uniform-in-$d_L$, direct:             MAP=70.5, ..., P>120=0.159
Uniform-in-$d_L$, reweighted:         MAP=73.5, ..., P>120=0.041
```

Those three numbers are the paper's headline result. If they match, the
deposit is good.

Optional, for the figures:

```bash
bash fetch_data.sh figures    # 287 MB LVK GW150914 PE release, Figure 1 only
bash regenerate.sh figures    # 8 PDFs
```

## 6. Afterwards

Update the GitHub **About** blurb on the release repository — it still
describes a data-and-analysis release without mentioning that the sampling
pipeline is now included.

---

## If the chains ever change

Regenerate the manifest **before** building a bundle, or `verify` will
disagree with the data:

```bash
python make_chain_manifest.py          # rewrite results/CHAIN_MANIFEST.csv
python make_chain_manifest.py --check  # assert it matches what is on disk
```

It refuses to record a symlink, a Git LFS pointer, or anything under 1 MB
when the smallest genuine chain here is 5.5 MB, and it refuses to shrink
the recorded set if you run it somewhere the chains are only partly
present. Commit the regenerated manifest, then build and upload as above.

## Reference

| Command | What it does |
|---|---|
| `bash fetch_data.sh chains` | download + unpack the chains from Zenodo |
| `bash fetch_data.sh verify` | check every chain against `CHAIN_MANIFEST.csv` |
| `bash fetch_data.sh figures` | the 287 MB LVK GW150914 PE release (Figure 1) |
| `bash fetch_data.sh strain` | LVK strain + PSDs (only to re-run the sampler) |
| `bash make_chain_bundle.sh [ver] [--source DIR]` | build the Zenodo tarball |
| `python make_chain_manifest.py [--check]` | rebuild / check the checksum manifest |
| `bash regenerate.sh [tables\|figures\|pdf]` | rebuild the paper |
| `bash run_chains.sh list` | list the GPU sampling sessions |

`PYTHON=/path/to/python` overrides interpreter detection if the scripts
pick the wrong one.
