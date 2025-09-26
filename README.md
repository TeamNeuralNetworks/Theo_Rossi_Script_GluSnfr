# iGluSnFR Analysis: Extraction, Batch Processing, and Model Calibration

This repository provides a practical toolkit to analyze iGluSnFR stimulation trains from Excel workbooks, extract robust per-pulse metrics, batch export results, and compare simple kinetic models on averaged responses. It combines feature extraction utilities, a configurable batch pipeline, and model calibration demos that share a consistent input format.

## Toolkit Overview

- **Feature extraction**: `Feature_extraction/extract_metrics.py` plus demos (`demo_single_file.py`, `demo_batch_process.py`) and snippets. The batch demo covers both single- and multi-folder workflows.
- **Batch analysis**: `batch_measure_complex.py` with CLI flags, runtime overrides, and helper scripts (`smoothing_test.py`, `whitenng_interactive.py`).
- **Model calibration**: `Model_Calibration/demo_adjust_fit_events.py`, `Model_Calibration/Demo_two_good_model_fitting.py`, and the notebook `Model_Calibration/Demo_Different_model_fitting.ipynb` (edit code/markdown only).
- **Utilities**: `smoothing.py` and helpers for detrending, kernels, NNLS, plotting, and NNLS weight visualization.

## Quick Start

Run the simplified extractor API (update paths to your dataset):

```
python Feature_extraction/demo_single_file.py
```

Batch across folders and export Excel/plots using default directories defined in the script:

```
python batch_measure_complex.py --no-show --save-plots
```

Compare models on averaged responses (directory can be passed positionally):

```
python Model_Calibration/Demo_two_good_model_fitting.py "C:\path\to\folder"
```

## Excel Input Format

Provide an `.xlsx` file with these columns on the first worksheet (index 0):

- Columns 1..N‑2: individual trials
- Column N‑1: optional average trace (ignored for per‑trial processing)
- Column N: timestamps in seconds

Non-numeric cells are treated as `NaN`. Reading Excel requires `pandas` (and `openpyxl`).

## Processing Workflows

Step-by-step recipes for single files, folders, and batches using the existing demo scripts and code snippets already in this repo.

### Single File (extract_metrics API)

Minimal usage (last column = time, other columns = trials):

```python
import numpy as np, pandas as pd
from Feature_extraction.extract_metrics import extract_metrics

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
        'plot': {
            'enabled': True,
            'traces': ['raw','savgol','nnls'],  # show all average overlays
            'show_decay': True,
            'trials': True,
            'baseline': True
        }
    }
)

print("Averages (NNLS):", res['average']['amp_nnls'])
print("PPR (NNLS):", res['average']['ppr_nnls'])
```

### One Folder (extract_metrics API)

Process all `.xlsx` files in a directory, save one image per file and a summary CSV:

```python
import os, glob, zipfile, numpy as np, pandas as pd
from Feature_extraction.extract_metrics import extract_metrics

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
        continue
    df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    t_raw = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    ok = np.isfinite(t_raw)
    time = t_raw[ok]
    trials = X[ok, :]

    res = extract_metrics(
        time, trials,
        train_start=0.5, isi=0.05, n_pulses=10,
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

### Multiple Folders (extract_metrics API)

Loop over several input folders; create one output subfolder per input:

```python
import os, glob, zipfile, numpy as np, pandas as pd
from Feature_extraction.extract_metrics import extract_metrics

folders = [
    r"C:\\data\\groupA",
    r"C:\\data\\groupB",
]
root_out = r"C:\\out"; os.makedirs(root_out, exist_ok=True)

for in_dir in folders:
    out_dir = os.path.join(root_out, os.path.basename(in_dir))
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for xlsx_path in glob.glob(os.path.join(in_dir, "*.xlsx")):
        # ... load and call extract_metrics as above ...
        pass
```

See the full implementation in `Feature_extraction/demo_batch_process.py`, including per-folder `train_start` overrides.

### Batch Pipeline (batch_measure_complex.py)

Use the consolidated batch script for folder/multi-folder export, headless plotting, and Excel summaries.

- Single file quick run:
  ```
  python batch_measure_complex.py --single "C:\\path\\to\\fiber.xlsx"
  ```
- Save a one-row Excel summary and suppress windows:
  ```
  python batch_measure_complex.py --single "C:\\path\\to\\fiber.xlsx" --out "C:\\path\\to\\fiber_summary.xlsx" --save-plots --no-show
  ```
- Batch process default folders (configure in file):
  ```
  python batch_measure_complex.py --no-show --save-plots
  ```
- Override train start for a single file:
  ```
  python batch_measure_complex.py --single "C:\\path\\to\\fiber.xlsx" --train-start 0.5
  ```
- Choose the measurement saved in the summary (`NNLS` | `RAW` | `SAVGOL` | `KALMAN`, default `NNLS`):
  ```
  python batch_measure_complex.py --single "C:\\path\\to\\fiber.xlsx" --measurement RAW
  ```
- Baseline-only quick preview (no fitting):
  ```
  python batch_measure_complex.py --single "C:\\path\\to\\fiber.xlsx" --baseline-only
  ```
- Force saving plots even if disabled in the script:
  ```
  python batch_measure_complex.py --save-plots
  ```
- Force not saving plots regardless of script settings:
  ```
  python batch_measure_complex.py --no-save
  ```

Programmatic use for precise control:

```python
from batch_measure_complex import batch_measure_complex
batch_measure_complex(
    paths=[r"C:\\data\\groupA", r"C:\\data\\groupB"],
    out_file=r"C:\\out\\ppr_results.xlsx",
    measurement="NNLS",   # or "RAW", "SAVGOL", "KALMAN"
    max_files=None         # or an int to limit per folder
)
```

## Batch CLI Reference

### Runtime Flags

| Flag | Effect |
|------|--------|
| `--single <file>` | Single-file mode; skips batch. |
| `--out <excel>` | Write single-file Excel summary. |
| `--train-start <s>` | Override stimulation train start for that file. |
| `--measurement <M>` | Choose NNLS, RAW, SAVGOL, or KALMAN (default NNLS). |
| `--baseline-only` | Only baseline preview plot (no fitting). |
| `--no-show` | Disable showing interactive plots. |
| `--save-plots` | Force saving plots (PNG). |
| `--no-save` | Force not saving plots. |

Priority rules:
1. `--no-save` overrides `--save-plots` if both given.
2. Runtime flags override the boolean constants in the script for that run only.

### Output Locations and Exports

- Single-file Excel: specified with `--out`.
- Batch Excel workbook: path set in `run_batch_export()` (e.g., `before_after_metrics.xlsx`).
- Plots: `<BATCH_EXPORT_DIR>/<PLOTS_SUBDIR>`.
- Baseline-only preview PNG naming: `baseline_<parent>_<file>.png`.
- Batch workbook layout: one sheet per input folder with one row per fibre (file) plus a final average row; columns include `AMP1..AMPn`, `PPR2/1..PPRn/1`, and `%Fail1..%Fail3`.

### Settings Overview

All main settings live at the top of `batch_measure_complex.py` and are grouped by theme:

- Core data: `sheet_index`, `train_start_s`, `isi_s`, `n_pulses`
- Feature toggles: `ENABLE_*` flags, `SHOW_PLOTS_DURING_BATCH`, `SAVE_PLOTS`, `USE_GUI`, `RUN_BATCH_EXPORT`
- Plotting: `PLOT_MODE` (`replace|keep|none`), `PLOTS_SUBDIR`, `sg_window`, `sg_poly`, `pre_zoom`, `post_zoom`, `peak_win_ms`, `avg_N_points`
- Robust NNLS & shifts: `ROBUST_LOSS`, `HUBER_DELTA`, `IRLS_ITERS`, `DELTA_MAX_MS`, `DELTA_STEP_MS`, `SHIFT_MIN_MS`, `ENABLE_CONTINUOUS_SHIFTS`
- Baseline/null sampling: `F0_WINDOW_S`, `NULL_FAIL_THRESHOLD_PARAM`, `SHUFFLE_BASELINE_BOOTSTRAP`, `NULL_SIM_MAX_POINTS`
- Kinetics constraints: `USE_LINEAR_TAUD`, `SLOPE_BOUNDS`, `FORCE_TAUD_MS`
- Bleach correction: `ENABLE_BLEACH_CORRECTION`, `BLEACH_MAX_ITER`, `BLEACH_TAU_GRID_FACTORS`, `BLEACH_N_TAU`, `BLEACH_HUBER_DELTA`, `SHOW_BLEACH_PLOTS`
- Normalization & ratios: `USE_DF_OVER_F0` (ΔF/F0 vs ΔF), `PPR_NORMALIZE_TO_TRAIN_MEAN` (divide by A1 vs train mean)
- Batch/export: `BATCH_EXPORT_DIR`, `DEFAULT_BATCH_OUTPUT_FILE`, `DEFAULT_BATCH_INPUT_DIRS`, `BATCH_MEASUREMENT`, `BATCH_FILE_LIMIT`
- Train start overrides: `TRAIN_START_OVERRIDE_MAP` for folder-specific `train_start` values; CLI `--train-start` overrides for single-file runs

Tip: runtime flags (`--no-show`, `--save-plots`, `--no-save`, `--plot-mode`) take precedence over the booleans in the file for that run only.

### Baseline-only vs Full Analysis

- **Baseline-only**: loads, interpolates, optional bleach correction, baseline subtract / ΔF/F₀, SG smoothing, and plotting. No NNLS fitting or PPR metrics.
- **Full analysis**: adds kinetics estimation, robust NNLS with micro-shifts (and optional Bayes), amplitude correction, null sampling for failure thresholds, and exports metrics.

### Common Recipes

- Minimal quick look (single file, interactive):
  ```
  python batch_measure_complex.py --single "C:\\data\\fiber.xlsx" --baseline-only
  ```
- High-throughput headless batch (save plots, no windows):
  ```
  python batch_measure_complex.py --save-plots --no-show
  ```
- Single file full analysis saving summary + plots, no GUI windows:
  ```
  python batch_measure_complex.py --single "C:\\data\\fiber.xlsx" --out "C:\\data\\fiber_summary.xlsx" --save-plots --no-show
  ```

### Troubleshooting

- If no sheets are written, verify your folders actually contain `.xlsx` files and that none are the output workbook itself.
- If plots don’t appear, check `--no-show` and whether `SAVE_PLOTS` closed figures; use `--save-plots` to write PNGs or call `fig.show()` manually when both showing and saving are required.
- If the train start differs across datasets, update `TRAIN_START_OVERRIDE_MAP` or pass `--train-start` for single-file runs.
- Ensure at least one input workbook has ≥3 columns; a placeholder sheet is added automatically if none processed.

## NNLS Weight Control

The NNLS (Non-Negative Least Squares) fitter supports configurable weighting patterns for event amplitude estimation.

### Supported Modes

1. **Uniform** (default): All timepoints have equal weight.
2. **Linear**: Weights decrease linearly from 1 to 0 between each stimulus and the next (sawtooth pattern).
3. **Exponential**: Exponential decay weights for each event.
4. **Savgol**: Data-driven weights derived from the Savitzky–Golay smoothed trace.

### Options

- `nnls_weight_mode` (string, default `'uniform'`): choose `'uniform'`, `'linear'`, `'exponential'`, or `'savgol'`.
- `nnls_weight_tau_s` (float or `None`, default `None`): time constant for weight decay in seconds.
  - Linear: controls the decay slope (default ISI) and is clipped at 0.
  - Exponential: controls the exponential decay time constant.
    - `fit_source='global'`: uses estimated tau_d from kinetics fitting (default 10 ms if unavailable).
    - Other fit sources: default 10 ms.
- `nnls_show_weights` (bool, default `False`): when `True`, displays a plot showing the weight pattern for the train.
  - When `'savgol'` is selected the smoothing trace is computed automatically if not already requested.

### Weight Pattern Behavior

- **Uniform**: `weight = 1.0` for all timepoints.
- **Linear**: creates a sawtooth pattern across the stimulus train.
- **Exponential**: `weight = exp(-time_since_stimulus / tau)` per event.
- **Savgol**: normalized |ΔF| from the Savitzky–Golay average (0.1–1.0 range).

### Usage Examples

```python
# Basic exponential weighting with auto tau
options = {
    'nnls_weight_mode': 'exponential',
    'nnls_show_weights': True,
}

# Linear weighting with custom slope
options = {
    'nnls_weight_mode': 'linear',
    'nnls_weight_tau_s': 0.025,  # 25 ms time constant
    'nnls_show_weights': True,
}

# Exponential weighting with explicit time constant
options = {
    'nnls_weight_mode': 'exponential',
    'nnls_weight_tau_s': 0.015,  # 15 ms decay
    'nnls_show_weights': True,
}

# Savgol weighting driven by the smoothed trace
options = {
    'nnls_weight_mode': 'savgol',
    'nnls_show_weights': True,
}
```

### Implementation Notes

- `_calculate_nnls_weights()` computes weight patterns for the supported modes (including Savitzky–Golay derived weights).
- `_nnls_weighted()` performs weighted NNLS solving.
- `estimate_kinetics_from_average()` accepts weight parameters (including SG window/polynomial for `'savgol'` mode), and `extract_metrics()` threads them through.
- Weight visualization integrates with the existing plotting system when `nnls_show_weights=True`.

### Automatic Parameter Selection

- Linear mode: default tau = ISI (inter-stimulus interval).
- Exponential mode with global fit: default tau = estimated tau_d from kinetics.
- Exponential mode with other fits: default tau = 10 ms.
- Savgol mode: no tau parameter; weights follow the normalized smoothed trace.

### Testing and Compatibility

- Tested with synthetic data, weight calculation verification, weighted NNLS solver checks, and integration with `extract_metrics`.
- Changes are backward compatible; existing code continues to use uniform weights by default.

## Model Suitability for Biphasic (Fast + Slow) Waveforms

Models that are explicitly multi-component or have two clear decay terms will fit a fast + slow (double-exponential) waveform well; single-timescale / single-shape models will struggle.

- **Good choices** (can capture a fast component + slower decay):
  - `double_exp` — explicit two-exponential form (best match)
  - `two_component` / `two_component_shared_rise` — explicit fast + slow amplitude/tau
  - `double_cooperative` / cooperative variants — two-component cooperative forms
  - `diffusion_clearance` — has two clearance taus + fraction (can produce biphasic decay)
  - `binding_kinetics` — can show multi-timescale behaviour depending on kon/koff/tau_clear

- **Models that will struggle** (poor fit for a true double exponential):
  - `single_exp` — single tau_decay only
  - `alpha` — single timescale (alpha-function shape)
  - `gamma` — single dominant time constant (shape parameter controls onset)
  - `cooperative` (coop) — nonlinear amplitude but single tau_decay
  - `bilinear` — piecewise linear rise/decay, won't reproduce exponential tails accurately
  - `coop_plus_linear` / `desensitization` — add other dynamics but not two independent exponential decay terms

## Model Calibration Demos

Build median event waveforms across files, then compare two kinetic models.

- Median waveform aggregation (folder can be passed or inferred via `GLUSNFR_IN_DIR`):
  ```
  python Model_Calibration/demo_adjust_fit_events.py "C:\\path\\to\\folder"
  ```
- Compare Double-Exponential vs Cooperative Binding on averaged and individual traces:
  ```
  python Model_Calibration/Demo_two_good_model_fitting.py "C:\\path\\to\\folder"
  ```

## Repository Layout

- `Feature_extraction/extract_metrics.py`: small, explicit interface to compute amplitudes (NNLS and alternatives), PPR, and plots.
- `Feature_extraction/demo_single_file.py`, `Feature_extraction/demo_batch_process.py`: runnable demos covering single-file and single-/multi-folder scenarios.
- `batch_measure_complex.py`: consolidated settings and CLI for folder and multi-folder export; supports plotting and Excel output.
- `Model_Calibration/demo_adjust_fit_events.py`, `Model_Calibration/Demo_two_good_model_fitting.py`: model calibration demos and comparisons.
- `smoothing.py`: shared smoothing, detrending, NNLS, and plotting helpers.

## Notes

- Jupyter notebooks: when editing, ignore output cells; focus on code and markdown cells only.
- No external dependencies should be added beyond what is already used in the repo.
