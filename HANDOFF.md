# Handoff: building the release dataset from raw recordings

Minimal steps to go from this repo + the private raw recordings archive to a
built `release/` dataset, extracted metrics, and reproduced figures. See
[`dataset_tools/SCHEMA.md`](dataset_tools/SCHEMA.md) for what each file
contains, and [`QUICKSTART.md`](QUICKSTART.md) if you only have the public
Zenodo deposit (no raw recordings).

## 0. Environment

```powershell
conda create -n glusnfr python=3.11
conda activate glusnfr
pip install -r requirements.txt
```

## 1. Set your paths

```powershell
$UPPER = "C:\path\to\your\working\folder"        # release/ is built here
$MESSY = "C:\path\to\the\raw\recordings\folder"   # e.g. "$UPPER\MESSY RAW DATA"
```

`$MESSY` must contain the per-condition folders (`Stability_Before`,
`WT_Theo`, etc. — see `ALL_CONDITIONS` in `dataset_tools/parse_ids.py`) plus,
if available, `ID_and_sex.csv`, `Target_WT_pooled.xlsx`, and
`Saturation_data.xlsx`. `$MESSY` can live anywhere — inside `$UPPER`, next to
it, or on another drive — it's read-only from here on and never has to be
`$UPPER` itself.

## 2. Build the release dataset

```powershell
# Manifest — scans $MESSY, writes into $UPPER's release/
python dataset_tools\build_manifest.py --data-root $UPPER --source-root $MESSY

# Organized raw (release\raw\<uid>.csv + release\f0.csv) — same split
python dataset_tools\reorganize_raw.py --data-root $UPPER --source-root $MESSY

# Saturation tables (optional, only if Saturation_data.xlsx is present)
python dataset_tools\convert_saturation.py --xlsx "$MESSY\Saturation_data.xlsx" --out "$UPPER\release"
```

Check the console output: `build_manifest.py` prints row/uid/physical-bouton
counts, `reorganize_raw.py` prints `raw=<N> f0 rows=<N> (failures: 0)`. Any
failures or unexpectedly low counts mean a condition folder was missed or a
file failed to parse — fix before continuing.

## 3. Run the batch extraction

Reads only `release\raw\` + `release\boutons_manifest.csv` — never touches
`$MESSY` again from here on.

```powershell
$env:GLUSNFR_DATA_ROOT = $UPPER
python Feature_extraction\demo_batch_process.py
```

`WRITE_TIDY = True` (the default in that script) writes the metrics-enriched
`release\boutons.csv` / `trials.csv` / `null_amps.csv` / `traces.csv` straight
into `$UPPER\release\` in the same run.

## 4. Run the notebook

With `GLUSNFR_DATA_ROOT` still set to `$UPPER` (same PowerShell session, or
set it again before launching Jupyter/VS Code), open
[`Support_figure.ipynb`](Support_figure.ipynb), select the `glusnfr` kernel,
and run all cells. `BASE_DIR` resolves to `$UPPER`, so figures and statistics
land in `$UPPER\output\` — never inside `$MESSY`.
