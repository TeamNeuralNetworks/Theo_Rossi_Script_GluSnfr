# Usage Guide: batch_measure_complex

This guide shows concrete command examples for typical workflows.
Assumes you run commands inside the activated Python environment (PowerShell examples).

## 1. Process a single Excel file

Analyze one workbook and print summary (NNLS default):
```
python batch_measure_complex.py --single "C:\path\to\fiber.xlsx"
```
Save a one-row Excel summary:
```
python batch_measure_complex.py --single "C:\path\to\fiber.xlsx" --out "C:\path\to\fiber_summary.xlsx"
```
Override stimulation train start (seconds):
```
python batch_measure_complex.py --single "C:\path\to\fiber.xlsx" --train-start 0.5
```
Disable interactive plot windows (still saves if SAVE_PLOTS True):
```
python batch_measure_complex.py --single "C:\path\to\fiber.xlsx" --no-show
```
Force saving plots even if SAVE_PLOTS was False in the script:
```
python batch_measure_complex.py --single "C:\path\to\fiber.xlsx" --save-plots
```
Run baseline-only quick preview (no fitting):
```
python batch_measure_complex.py --single "C:\path\to\fiber.xlsx" --baseline-only
```
Baseline-only and save the PNG:
```
python batch_measure_complex.py --single "C:\path\to\fiber.xlsx" --baseline-only --save-plots
```

## 2. Process all files in one folder

Edit the constants list inside `run_batch_export()` to include just that folder OR run the batch and limit to one folder by commenting others. Then:
```
python batch_measure_complex.py
```
(Uses the folder list embedded in the script.)

### Variants for the one-folder case

Analyze and show plots (don’t save): set in file:
```
SHOW_PLOTS_DURING_BATCH = True
SAVE_PLOTS = False
```
Then run:
```
python batch_measure_complex.py
```

Analyze and show and also save:
```
SHOW_PLOTS_DURING_BATCH = True
SAVE_PLOTS = True
```
OR override at runtime (save only applies to new figures):
```
python batch_measure_complex.py --save-plots
```

Analyze and save only (no windows):
```
SHOW_PLOTS_DURING_BATCH = False
SAVE_PLOTS = True
```
Or runtime:
```
python batch_measure_complex.py --save-plots --no-show
```

Just plot baseline (quick preview) for a representative file in the folder:
```
python batch_measure_complex.py --single "C:\folder\fiber1.xlsx" --baseline-only
```

Analyze but do not save plots (and optionally still not show):
```
SHOW_PLOTS_DURING_BATCH = False
SAVE_PLOTS = False
python batch_measure_complex.py
```
Or runtime override when they were True in file:
```
python batch_measure_complex.py --no-show --no-save
```

## 3. Process multiple folders

Ensure all desired folder paths are in the list inside `run_batch_export()`. Example run:
```
python batch_measure_complex.py
```
Add plot saving only for this run:
```
python batch_measure_complex.py --save-plots
```
Suppress any windows for a headless batch run:
```
python batch_measure_complex.py --no-show --save-plots
```

## 4. Flags summary

| Flag | Effect |
|------|--------|
| `--single <file>` | Single-file mode; skips batch. |
| `--out <excel>` | Write single-file Excel summary. |
| `--train-start <s>` | Override stimulation train start for that file. |
| `--measurement <M>` | Choose NNLS, RAW, SAVGOL, KALMAN (default NNLS). |
| `--baseline-only` | Only baseline preview plot (no fitting). |
| `--no-show` | Disable showing interactive plots. |
| `--save-plots` | Force saving plots (PNG). |
| `--no-save` | Force not saving plots. |

Priority rules:
1. `--no-save` overrides `--save-plots` if both given.
2. Runtime flags override the boolean constants in the script for that run only.

## 5. Output locations

- Batch Excel workbook: path set in `run_batch_export()` (e.g. `before_after_metrics.xlsx`).
- Single-file Excel: specified with `--out`.
- Plots: `<BATCH_EXPORT_DIR>/<PLOTS_SUBDIR>`.
- Baseline-only preview PNG naming: `baseline_<parent>_<file>.png`.

## 6. Baseline-only vs Full Analysis

Baseline-only: loads, interpolates, optional bleach correction, baseline subtract / ΔF/F0, SG smoothing, plot. No NNLS fitting, no PPR metrics.

Full analysis: adds kinetics estimation, robust NNLS with micro-shifts (and optional Bayes), amplitude correction, null sampling for failure thresholds, exports metrics.

## 7. Common recipes

Minimal quick look (single file, interactive):
```
python batch_measure_complex.py --single "C:\data\fiber.xlsx" --baseline-only
```
High-throughput headless batch (save plots, no windows):
```
python batch_measure_complex.py --save-plots --no-show
```
Single file full analysis saving summary + plots, no GUI windows:
```
python batch_measure_complex.py --single "C:\data\fiber.xlsx" --out "C:\data\fiber_summary.xlsx" --save-plots --no-show
```

## 8. Troubleshooting

- No sheets / Excel error: ensure at least one input workbook has ≥3 columns; placeholder sheet is added automatically if none processed.
- No plots: confirm you didn’t pass `--no-show` and that `SAVE_PLOTS` isn’t causing figures to be closed (use `--save-plots` plus manual `fig.show()` if both needed).
- Train start mismatch: use `--train-start` or edit `TRAIN_START_OVERRIDE_MAP`.

---
Generated documentation. Update as needed when adding new flags or behaviors.
