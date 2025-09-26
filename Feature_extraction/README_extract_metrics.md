# Streamlined iGluSnFR Analysis (extract_metrics)

This guide mirrors the demo scripts with generic paths and options for the simplified API built around `extract_metrics`.

The function keeps the math equivalent to the main pipeline while exposing a small, explicit options dictionary and minimal plotting controls.

- Import path: `from extract_metrics import extract_metrics`
- Plot controls: `options['plot'] = {'enabled': True, 'traces': ['raw','savgol','nnls'], 'show_decay': True, 'trials': False, 'baseline': False}`
- Always enforces non‑decreasing τd across pulses for stability.
- Core controls:
  - `measurement`: `'NNLS'|'SAVGOL'|'RAW'` (default `'NNLS'`) — p‑values use this amplitude series
  - `fail_method`: `'NNLS'|'SAVGOL'|'RAW'` (default: same as `measurement`) — controls null/threshold rule
    - `threshold_mode`: `'auto'|'mad'|'sd'` (default `'auto'`) — failure rates compare
      pulses 1–3 against a single baseline threshold; pulses 2/3 amplitudes are
      corrected for residual pre‑stim currents
    - `allow_shift`: bool (default True) — enable per‑pulse micro‑shifts
    - `event_model`: kernel used for fitting. Default `'double_exp'` (one rise τ and one decay τ). If you pass a value here it is respected; there is no auto‑replacement.
    - Extras for `'cooperative'`: `event_model_settings={'n_coop': 2.0}`
  - `fit_source`: `'global'|'average'|'individual'` (default `'global'`)
    - `global`: fit a single template from all trials (recut median) then apply progression
    - `average`: fit kinetics on the average trace per event then smooth via progression
    - `individual`: fit per trial then aggregate (median) and smooth
  - `decay_progression_mode`: `'fixed'|'free_monotonic'|'linear'` (default `'linear'`)
    - `fixed`: one τd for the whole train (median)
    - `free_monotonic`: interpolate τd between first and last event (non‑decreasing)
    - `linear`: non‑negative‑slope linear trend across pulses

Related demo scripts in this folder (with concrete paths): `demo_single_file.py` and `demo_batch_process.py` (works for one or many folders).

---

## 1) Analyze a single file

Minimal usage (last column = time, other columns = trials):

```python
import numpy as np, pandas as pd
from extract_metrics import extract_metrics

xlsx_path = r"C:\\path\\to\\fiber.xlsx"
df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
# time in last column; trials in all columns except last
_time = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
_trials = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
valid = np.isfinite(_time)
time = _time[valid]
trials = _trials[valid, :]

res = extract_metrics(
    time, trials,
    train_start=0.5,   # seconds
    isi=0.05,          # seconds
    n_pulses=10,
    options={
        'normalize_dff': True,
        'bleach': True,
        'measurement': 'NNLS',        # or 'SAVGOL', 'RAW'
        'fail_method': 'NNLS',        # default: same as measurement
        'threshold_mode': 'auto',     # or 'mad', 'sd'
        'allow_shift': True,
        'fit_source': 'global',
        'decay_progression_mode': 'linear',
        'plot': {
            'enabled': True,
            'traces': ['raw','savgol','nnls'],  # show all average overlays
            'show_decay': True,
            'trials': True,
            'baseline': True
        }
    }
)

# Inspect results
print("Averages (NNLS):", res['average']['amp_nnls'])
print("PPR (NNLS):", res['average']['ppr_nnls'])
print("A1 thresholds per trial:", res['threshold_amp1'])
print("A1 p-values per trial (using", 'NNLS', "):", res['pval_amp1'])

# Save plot if enabled
fig = res.get('figure')
if fig is not None:
    fig.savefig(r"C:\\path\\to\\fiber_plot.png", dpi=150)
```

Override the train start (seconds) for that file by passing `train_start=...`.

---

## 2) Analyze one folder

Process all `.xlsx` files in a directory, save one image per file and a summary CSV:

```python
import os, glob, zipfile, numpy as np, pandas as pd
from extract_metrics import extract_metrics

in_dir  = r"C:\\data\\groupA"
out_dir = r"C:\\out\\groupA"; os.makedirs(out_dir, exist_ok=True)
rows = []

def _is_valid_xlsx(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return '[Content_Types].xml' in z.namelist()
    except Exception:
        return False

for xlsx_path in glob.glob(os.path.join(in_dir, "*.xlsx")):
    if not _is_valid_xlsx(xlsx_path):
        print(f"[skip] Not a valid .xlsx package: {xlsx_path}")
        continue
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    except Exception as e:
        print(f"[skip] Failed to read Excel: {xlsx_path} -> {e}")
        continue
    t_raw = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    ok = np.isfinite(t_raw)
    time = t_raw[ok]
    trials = X[ok, :]

    res = extract_metrics(
        time, trials,
        train_start=0.5, isi=0.05, n_pulses=10,
        options={'normalize_dff': True, 'bleach': True,
                 'fit_source': 'global', 'decay_progression_mode': 'linear',
                 'plot': {'enabled': True, 'traces': ['raw','savgol','nnls'], 'show_decay': True, 'trials': False}}
    )

    base = os.path.splitext(os.path.basename(xlsx_path))[0]
    if res.get('figure') is not None:
        res['figure'].savefig(os.path.join(out_dir, f"{base}.png"), dpi=150)

    row = {'file': base}
    amp = res['average']['amp_nnls']; ppr = res['average']['ppr_nnls']
    for i, v in enumerate(amp): row[f'amp_{i+1}'] = float(v)
    for i, v in enumerate(ppr): row[f'ppr_{i+1}'] = float(v)
    rows.append(row)

pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
```

Tips:
- Vary `train_start` per directory: compute it from the folder name with a small map.
- Control which traces are drawn via `options['plot']['traces']`.

---

## 4) Multi‑folder → multi‑sheet Excel

Use the built‑in helper to export one sheet per folder with AMP and PPR columns (and `%Fail1..3` if available):

```python
from Feature_extraction.extract_metrics import export_folders_to_excel

folders = [r"C:\\data\\groupA", r"C:\\data\\groupB"]
export_folders_to_excel(
    folders,
    out_file=r"C:\\out\\ppr_results.xlsx",
    train_start=0.5,
    isi=0.05,
    n_pulses=10,
    options={'measurement': 'NNLS', 'fail_method': 'NNLS', 'threshold_mode': 'auto'}
)
```

---

## 3) Batch process multiple folders

Loop over several input folders; create one output subfolder per input. The last two below use a different `train_start` (0.5 s):

```python
import os, glob, zipfile, numpy as np, pandas as pd
from extract_metrics import extract_metrics

folders = [
    r"C:\\data\\groupA",
    r"C:\\data\\groupB",
    r"C:\\data\\groupA_05",
    r"C:\\data\\groupB_05",
]
root_out = r"C:\\out"; os.makedirs(root_out, exist_ok=True)

train_start_by_folder = {
    folders[2]: 0.5,
    folders[3]: 0.5,
}

def _is_valid_xlsx(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return '[Content_Types].xml' in z.namelist()
    except Exception:
        return False

for in_dir in folders:
    out_dir = os.path.join(root_out, os.path.basename(in_dir))
    os.makedirs(out_dir, exist_ok=True)

    train_start = train_start_by_folder.get(in_dir, 1.0)
    isi = 0.05
    n_pulses = 10

    rows = []
    for xlsx_path in glob.glob(os.path.join(in_dir, "*.xlsx")):
        if not _is_valid_xlsx(xlsx_path):
            print(f"[skip] Not a valid .xlsx package: {xlsx_path}")
            continue
        try:
            df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
        except Exception as e:
            print(f"[skip] Failed to read Excel: {xlsx_path} -> {e}")
            continue
        t_raw = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
        X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        ok = np.isfinite(t_raw)
        time = t_raw[ok]
        trials = X[ok, :]

        res = extract_metrics(
            time, trials,
            train_start=train_start, isi=isi, n_pulses=n_pulses,
            options={'normalize_dff': True, 'bleach': True,
                     'plot': {'enabled': True, 'traces': ['raw','savgol','nnls'], 'show_decay': True, 'trials': False}}
        )

        base = os.path.splitext(os.path.basename(xlsx_path))[0]
        if res.get('figure') is not None:
            res['figure'].savefig(os.path.join(out_dir, f"{base}.png"), dpi=150)

        row = {'file': base}
        amp = res['average']['amp_nnls']; ppr = res['average']['ppr_nnls']
        for i, v in enumerate(amp): row[f'amp_{i+1}'] = float(v)
        for i, v in enumerate(ppr): row[f'ppr_{i+1}'] = float(v)
        rows.append(row)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
```

Headless batch (no windows) still writes figures if you call `figure.savefig(...)`. To suppress figure windows entirely, set `options['plot']['enabled'] = False`.

---

## 4) Adjust/overlay fitted model by recutting events (adjust_fit_model)

Use `adjust_fit_model.py` to re-cut windows around selected events (with optional peak alignment), average them, and overlay the current fitted model of the average trace.

CLI examples (paths are generic):

```bash
# Single folder, save figures
python adjust_fit_model.py C:\data\groupA \
  --train-start 0.5 --isi 0.05 --n-pulses 10 \
  --events 1-5,7 --align-by-peak \
  --pre-ms 2 --post-ms 200 \
  --out-dir C:\out --save

# Mix files and folders; disable ΔF/F0 and bleach if needed
python adjust_fit_model.py C:\data\groupA C:\data\fiber.xlsx \
  --train-start 1.0 --isi 0.05 --n-pulses 10 \
  --events all --pre-ms 2 --post-ms 200 \
  --no-dff --no-bleach --out-dir C:\out --save
```

Programmatic usage:

```python
from adjust_fit_model import process_file, process_folder, process_inputs

# Single file
process_file(
    r"C:\\data\\fiber.xlsx",
    train_start=0.5, isi=0.05, n_pulses=10,
    events_spec="1-5,7", peak_recenter=5,
    pre_ms=2.0, post_ms=200.0, out_dir=r"C:\\out", save=True
)

# Folder
process_folder(
    r"C:\\data\\groupA",
    train_start=1.0, isi=0.05, n_pulses=10,
    events_spec="all", peak_recenter=0,
    pre_ms=2.0, post_ms=200.0, out_dir=r"C:\\out", save=True
)
```

---

## Programmatic call (inside Python)

```python
from extract_metrics import extract_metrics

res = extract_metrics(
    time, trials,
    train_start=0.5, isi=0.05, n_pulses=10,
    options={
        # Only override what you need; the rest comes from DEFAULTS
        'normalize_dff': True,
        'bleach': True,
        'peak_window_ms': 25.0,
        'plot': {'enabled': True, 'traces': ['raw','savgol','nnls'], 'show_decay': True, 'trials': False, 'baseline': False}
    }
)
```

### Output structure
- `tau_r_s` (float): rise time (s)
- `tau_d_s` (array): per‑pulse decay times (s), non‑decreasing by construction
- `stim_times_s` (array): stimulus times (s)
- `average` (dict): `y_avg`, `yhat_avg`, `amp_raw`, `amp_savgol`, `amp_nnls`, `ppr_nnls`
- `per_trial` (list of dict): for each trial, the same amplitude/PPR triplets plus fitted `a_coeff`, `delta_s`, shared threshold (`thr_shared`), and `pval_amp1/2/3`
- `threshold_amp1` (array): MAD‑rule thresholds for pulse 1 per trial
- `pval_amp1` (array): empirical p‑values for pulse 1 per trial
- `figure` (matplotlib Figure or None): average trace with selected overlays

### Configuration via options
All defaults live in a single dictionary inside `extract_metrics.py` named `DEFAULTS`. You can override any of these keys in the `options` you pass to `extract_metrics`:

- Smoothing: `sg_window`, `sg_poly`
- Fit/plot window: `pre_zoom_s`, `post_zoom_s`
- Peak window: `peak_window_ms`, `peak_avg_points`, `pre_peak_ms`
- Baseline/null: `f0_window_s`, `null_sim_max_points`, `null_min_post_zoom_s`, `null_N`
- Kinetics grids (ms): `kin_taur_grid_ms`, `kin_taud0_grid_ms`, `kin_slope_grid_ms`
- Robust NNLS + shifts: `huber_delta`, `irls_iters`, `delta_max_s`, `delta_step_s`, `shift_min_s`
- Bleach correction: `bleach_huber_delta`, `bleach_tau_range_factor`, `bleach_n_tau`
- Plot control: `plot = {'enabled': bool, 'traces': [...], 'show_decay': bool, 'trials': bool, 'baseline': bool}`

Notes:
- Provide `train_start`, `isi`, and `n_pulses` appropriate to each dataset.
- ΔF/F0 is used by default; disable by `options['normalize_dff'] = False` if you need raw ΔF.
- τd is always enforced non‑decreasing across pulses to avoid non‑physical regressions and improve stability.
