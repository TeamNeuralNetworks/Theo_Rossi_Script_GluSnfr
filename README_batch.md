# Ultimate iGluSnFR Smoothing and Batch Analysis

Tools to analyze iGluSnFR stimulation trains from Excel workbooks: baseline/bleach correction, robust NNLS per‑pulse amplitudes with micro‑shifts, paired‑pulse ratios (PPR), failure rates, plotting, and batch export to Excel.

Primary entrypoint: `batch_measure_complex.py`. Utilities: `smoothing_test.py` (interactive exploration), `whitenng_interactive.py` (per‑trial smoothing preview).

## Excel Input Format

Provide an `.xlsx` file with these columns on the first worksheet (index 0):

- Columns 1..N‑2: individual trials
- Column N‑1: optional average trace (ignored for per‑trial processing)
- Column N: time stamps in seconds

Non‑numeric cells are treated as `NaN`. Reading Excel requires `pandas` (and `openpyxl`).

## Quick Start

Run commands from your Python environment. Windows paths are shown; replace with POSIX paths on macOS/Linux.

### 1) Analyze a single file

- Minimal baseline preview (no fitting):
  `python batch_measure_complex.py --single "C:\path\to\fiber.xlsx" --baseline-only`
- Full analysis, write a one‑row Excel summary and save plots without showing windows:
  `python batch_measure_complex.py --single "C:\path\to\fiber.xlsx" --out "C:\path\to\fiber_summary.xlsx" --save-plots --no-show`
- Override the stimulation train start (seconds) for that file:
  `python batch_measure_complex.py --single "C:\path\to\fiber.xlsx" --train-start 0.5`
- Choose the measurement saved in the summary (`NNLS` | `RAW` | `SAVGOL`, default `NNLS`):
  `python batch_measure_complex.py --single "C:\path\to\fiber.xlsx" --measurement RAW`

### 2) Analyze one folder

Edit the folder list inside `DEFAULT_BATCH_INPUT_DIRS` in `batch_measure_complex.py` to include just your folder, then run:
`python batch_measure_complex.py`

Useful runtime switches:

- Show plots during batch: default is controlled by `SHOW_PLOTS_DURING_BATCH`; override with `--no-show` to suppress.
- Save plots to disk: set `SAVE_PLOTS = True` in the file or pass `--save-plots`.

### 3) Batch process multiple folders

Add all desired directories to `DEFAULT_BATCH_INPUT_DIRS`, set an output path in `DEFAULT_BATCH_OUTPUT_FILE`, then:
`python batch_measure_complex.py`

Headless batch that saves plots:
`python batch_measure_complex.py --no-show --save-plots`

Programmatic call (inside Python) for precise control:

```python
from batch_measure_complex import batch_measure_complex
batch_measure_complex(
    paths=[r"C:\data\groupA", r"C:\data\groupB"],
    out_file=r"C:\out\ppr_results.xlsx",
    measurement="NNLS",   # or "RAW", "SAVGOL"
    max_files=None         # or an int to limit per folder
)
```

## What Gets Exported

`batch_measure_complex.py` writes a multi‑sheet Excel workbook:

- One sheet per input folder: one row per fibre (file) + a final average row
- Columns include `AMP1..AMPn`, `PPR2/1..PPRn/1`, and `%Fail1..%Fail3`
- Plots (if enabled) go to `<BATCH_EXPORT_DIR>/<PLOTS_SUBDIR>`

## Settings (what they do)

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
- Train start overrides: `TRAIN_START_OVERRIDE_MAP` lets you specify folder‑specific `train_start` values; CLI `--train-start` overrides for single‑file runs

Tip: runtime flags (`--no-show`, `--save-plots`, `--no-save`, `--plot-mode`) take precedence over the booleans in the file for that run only.

## Troubleshooting

- If no sheets are written, verify your folders actually contain `.xlsx` files and that none are the output workbook itself.
- If plots don’t appear, check `--no-show` and whether `SAVE_PLOTS` closed figures; use `--save-plots` to write PNGs.
- If train start differs across datasets, update `TRAIN_START_OVERRIDE_MAP` or pass `--train-start` for single‑file runs.

## See Also

- `USAGE_GUIDE.md` for more command variants and flag details.
- `smoothing_test.py` for interactive exploration of smoothing and NNLS fitting on a single workbook.
- `whitenng_interactive.py` for per‑trial ΔF/F0 smoothing previews.
