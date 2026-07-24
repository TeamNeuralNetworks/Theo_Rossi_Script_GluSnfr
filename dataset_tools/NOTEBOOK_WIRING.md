# Wiring Support_figure.ipynb to the release/ folder

The notebook keeps **all** its grouping/normalisation (BaseID/FiberID/PairBaseID on
`legacy_id`) — only the *data source* changes. A validated shim (`dataset_tools/
release_io.py`) reshapes `release/*.csv` back into the exact old dataframes.

Controls run before this wiring:
- features / trials / null / traces reproduce the old files to ~1e-14.
- `target` matches the old join **exactly** on the PCA cohort (0 disagreements).
- `sex` and `date` are the manifest's (deliberately more complete: +179 sexed
  boutons, +130 six-digit-dated recordings) — **Fig S8 and some n-day counts will
  change** from the published version (you chose to adopt these).

Apply each edit, then re-run that cell (and the cells that depend on it) to confirm.

---

## 1. New setup cell — INSERT immediately AFTER cell 8 (id `9a55b667`)

```python
# === Wire notebook to the consolidated release/ folder ===
import sys as _sys
_release_tools = REPO_ROOT / "dataset_tools" if "REPO_ROOT" in globals() else BASE_DIR.parent
_sys.path.insert(0, str(_release_tools))
import release_io as rio
rio.RELEASE_DIR = BASE_DIR / "release"

# Route the function-based loaders (used by ~8 cells) through the shim.
def load_summary_trials_dataframe():
    return rio.trials()

def load_target_mapping_dataframe():
    return rio.target_mapping()

# Manifest-derived date map (adopted: covers 6-digit dates the old regex missed).
_RELEASE_DATE_BY_ID = rio.date_map()
```
*(Set `REPO_ROOT` if the notebook doesn't already define it — it's the repo root
containing `dataset_tools/`.)*

This alone fixes every cell that calls `load_summary_trials_dataframe()`
(30, 32, 34, 160, 179, 207) and `load_target_mapping_dataframe()` (20).

---

## 2. Cell 12 (id `b5d40622`) — traces

Replace the whole Excel-loading body with:

```python
raw_traces_data = []
_traces, _times = rio.traces_by_condition()
experimental_conditions = list(_traces.keys())
for condition_name in experimental_conditions:
    trace_df = _traces[condition_name]; time_df = _times[condition_name]
    for bouton_id in trace_df.columns:
        raw_traces_data.append({
            'ID': str(bouton_id), 'Condition': condition_name,
            'Time': time_df[bouton_id].to_numpy(float).tolist(),
            'Avg': trace_df[bouton_id].to_numpy(float).tolist(),
            'n_trials': 1, 'FilePath': str(rio.RELEASE_DIR / 'traces.csv'),
        })
RAW_TRACES_DF = pd.DataFrame(raw_traces_data)
CONDITIONS = experimental_conditions
print(f"=== Loaded {len(RAW_TRACES_DF)} preprocessed traces from release/traces.csv ===")
```

## 3. Cell 16 (id `99205a0b`) — features

Replace the top (through `FEATURES_DATAFRAME = pd.concat(...)`) with:

```python
_feat_all = rio.features()
available_conditions = [c for c in experimental_conditions if c in set(_feat_all['condition'])]
CONDITIONS = available_conditions
feature_dataframes = []
for condition_name in CONDITIONS:
    cf = _feat_all[_feat_all['condition'] == condition_name].drop(columns=['condition']).copy()
    cf['ID'] = cf['ID'].apply(lambda x: clean_bouton_id(str(x)) if pd.notnull(x) else x)
    cf['Condition'] = condition_name
    feature_dataframes.append(cf)
FEATURES_DATAFRAME = pd.concat(feature_dataframes, ignore_index=True)
```
Keep the rest of the cell (missing-value dropping, AMP-column drop) unchanged.

## 4. Cell 28 (id `2d4da25f`) — sex (+ date)

Replace the `SEX_FILENAME`/`_sex_path` block with:

```python
_sex_map = rio.sex_map()   # columns ['ID','Sexe'] (manifest, more complete)
for _name in ('FEATURES_DATAFRAME', 'FEATURES_DATAFRAME_RAW'):
    if _name in globals():
        _df = globals()[_name]
        _df['ID'] = _df['ID'].astype(str).str.strip()
        if 'Sexe' in _df.columns: _df = _df.drop(columns=['Sexe'])
        globals()[_name] = _df.merge(_sex_map, on='ID', how='left')
_refresh_shared_datasets()
print(f"Loaded sex annotation for {len(_sex_map)} IDs from the manifest")
```
To adopt the manifest date in the cohort counts, change `_cohort_recording_day`:
```python
def _cohort_recording_day(value):
    d = _RELEASE_DATE_BY_ID.get(_normalize_bouton_id(value))
    if d and d != '<NA>': return d
    m = re.match(r'^(\d{8})', str(value).strip()); return m.group(1) if m else np.nan
```

## 5. Cell 212 (id `s5-paired-saturation-recordings`) — saturation

`Saturation_data.xlsx` (4 sheets) → `rio.saturation()` returns `(amps_df, traces_long_df)`.
`amps_df` has `ca_mM, button_id, AMP1..`; `traces_long_df` is long `ca_mM, button_id,
time_s, dff`. Adapt the sheet-based reads to these two frames.

## 6. Cells 215 & 218 (`s5b-…`, `s5d-…`) — summary 'All'

Replace `pd.read_excel(summary_path, sheet_name='All')` with `rio.features()`
(same columns: `condition`, `ID`, metric columns).

---

## Files no longer needed by the notebook
`summary.xlsx`, `summary_trials.xlsx`, `summary_trials_nnls_null.xlsx`,
`summary_traces.xlsx`, `summary_times.xlsx`, `Target_WT_pooled.xlsx`,
`ID_and_sex.csv`, `Saturation_data.xlsx` — all now served from `release/`.
