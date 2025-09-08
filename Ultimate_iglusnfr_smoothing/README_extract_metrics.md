# Streamlined iGluSnFR Analysis (extract_metrics)

This guide mirrors the original README’s outline, but for the new, simplified API built around `extract_metrics`.

The function keeps the math equivalent to the main pipeline while exposing a small, explicit options dictionary and minimal plotting controls.

- Import path: `from extract_metrics import extract_metrics`
- Plot controls: `options['plot'] = {'enabled': True, 'traces': ['nnls','savgol','raw'], 'show_decay': True}`
- Always enforces non‑decreasing τd across pulses for stability.

---

## 1) Analyze a single file

Minimal usage (last column = time, other columns = trials):

```python
import numpy as np, pandas as pd
from extract_metrics import extract_metrics

xlsx_path = r"C:\\path\\to\\fiber.xlsx"
df = pd.read_excel(xlsx_path, sheet_name=0)
# time in last column; trials in all columns except last
_time = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
_trials = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
valid = np.isfinite(_time)
time = _time[valid]
trials = _trials[valid, :]

res = extract_metrics(
    time, trials,
    train_start=1.0,   # seconds
    isi=0.05,          # seconds
    n_pulses=10,
    options={
        'normalize_dff': True,
        'bleach': True,
        'plot': {
            'enabled': True,
            'traces': ['nnls','savgol'],  # any of ['raw','savgol','nnls']
            'show_decay': True
        }
    }
)

# Inspect results
print("Averages (NNLS):", res['average']['amp_nnls'])
print("PPR (NNLS):", res['average']['ppr_nnls'])
print("A1 thresholds per trial:", res['threshold_amp1'])
print("A1 p-values per trial:", res['pval_amp1'])

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
import os, glob, numpy as np, pandas as pd
from extract_metrics import extract_metrics

in_dir  = r"C:\\data\\groupA"
out_dir = r"C:\\out\\groupA"; os.makedirs(out_dir, exist_ok=True)
rows = []

for xlsx_path in glob.glob(os.path.join(in_dir, "*.xlsx")):
    df = pd.read_excel(xlsx_path, sheet_name=0)
    t_raw = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    ok = np.isfinite(t_raw)
    time = t_raw[ok]
    trials = X[ok, :]

    res = extract_metrics(
        time, trials,
        train_start=1.0, isi=0.05, n_pulses=10,
        options={'normalize_dff': True, 'bleach': True,
                 'plot': {'enabled': True, 'traces': ['nnls'], 'show_decay': True}}
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

## 3) Batch process multiple folders

Loop over several input folders; create one output subfolder per input:

```python
import os
folders = [
    r"C:\\data\\groupA",
    r"C:\\data\\groupB",
    r"C:\\data\\groupC",
]
root_out = r"C:\\out"; os.makedirs(root_out, exist_ok=True)

for d in folders:
    out_dir = os.path.join(root_out, os.path.basename(d))
    os.makedirs(out_dir, exist_ok=True)
    # Paste the per-folder loop from section (2) here with in_dir=d, out_dir=out_dir
```

Headless batch (no windows) still writes figures if you call `figure.savefig(...)`. To suppress figure windows entirely, set `options['plot']['enabled'] = False`.

---

## Programmatic call (inside Python)

```python
from extract_metrics import extract_metrics

res = extract_metrics(
    time, trials,
    train_start=1.0, isi=0.05, n_pulses=10,
    options={
        # Only override what you need; the rest comes from DEFAULTS
        'normalize_dff': True,
        'bleach': True,
        'peak_window_ms': 25.0,
        'plot': {'enabled': True, 'traces': ['nnls'], 'show_decay': True}
    }
)
```

### Output structure
- `tau_r_s` (float): rise time (s)
- `tau_d_s` (array): per‑pulse decay times (s), non‑decreasing by construction
- `stim_times_s` (array): stimulus times (s)
- `average` (dict): `amp_raw`, `amp_savgol`, `amp_nnls`, `ppr_nnls`
- `per_trial` (list of dict): for each trial, the same amplitude/PPR triplets plus fitted `a_coeff` and `delta_s`
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
- Plot control: `plot = {'enabled': bool, 'traces': [...], 'show_decay': bool}`

Notes:
- Provide `train_start`, `isi`, and `n_pulses` appropriate to each dataset.
- ΔF/F0 is used by default; disable by `options['normalize_dff'] = False` if you need raw ΔF.
- τd is always enforced non‑decreasing across pulses to avoid non‑physical regressions and improve stability.

