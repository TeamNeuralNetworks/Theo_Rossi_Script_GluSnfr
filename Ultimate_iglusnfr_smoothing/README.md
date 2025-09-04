# Ultimate_iglusnfr_smoothing

Advanced smoothing for trains.

## Excel input format

All scripts expect an `.xlsx` file with the following column layout:

- Columns 1..N-2 — individual trials.
- Column N-1 — optional average trace (ignored for per-trial processing).
- Column N — time stamps in seconds.

Non‑numeric cells are treated as `NaN`. The first worksheet (index 0) is used
by default. Reading `.xlsx` files requires `pandas` and `openpyxl`.

## Platform notes

Examples in the repository use Windows-style paths (e.g. `C:\\Users\\...`).
On macOS or Linux replace them with standard POSIX paths such as
`/path/to/data.xlsx`. Plotting requires a GUI backend.

## Scripts

### `smoothing_test.py`

Interactive iGluSnFR denoising and paired‑pulse‑ratio exploration.

**Libraries:** `numpy`, `pandas`, `matplotlib`,
`scipy.signal`, `scipy.linalg`, `scipy.optimize`.

**Run:** `python smoothing_test.py` (edit `file_path` in the script).

### `batch_measure.py`

Single‑trial/average plotter with ε‑band paired‑pulse ratios using
non‑negative least squares (NNLS) deconvolution.

**Libraries:** `numpy`, `pandas`, `matplotlib`,
`scipy.signal`, `scipy.linalg`, `scipy.optimize`.

**Run:** `python batch_measure.py` (set `file_path` to your Excel file).

### `batch_measure_complex.py`

Robust NNLS deconvolution with micro‑shift optimization,
auto AR(p) whitening, optional Bayesian amplitude inference,
and batch export support. Includes an optional Tkinter‑based navigation GUI
for stepping through recordings and adjusting filtering/display settings.

**Libraries:** `numpy`, `pandas`, `matplotlib`,
`scipy.signal`, `scipy.linalg`, `scipy.optimize`,
plus standard `os`, `glob`, `re`, `time`, and `warnings`.

**Run:** `python batch_measure_complex.py`.
Set `USE_GUI = True` in the script (or call `run_batch_export(gui=True)`) to
launch the interactive navigator.

Programmatic batch processing is available via
`batch_measure_complex.batch_measure_complex`. Provide one or more directory
paths and it will write a multi-sheet Excel workbook where each sheet contains
one row per fibre plus a final average row. Each row reports amplitudes
(`AMP1`..`AMP10`), paired-pulse ratios relative to the first pulse
(`PPR2/1`..`PPR10/1`), and failure-rate estimates for the first three pulses
(`%Fail1`..`%Fail3`) computed from the selected measurement method (``NNLS`` by
default, but ``RAW``, ``Savgol`` or ``kalman`` are also accepted). A
`BATCH_FILE_LIMIT` toggle limits processing to the first *N* files in each
directory (set it to ``None`` to scan all files).

### `Baysian_approach.py`

Exploratory script featuring continuous micro‑shifts, AR(p) whitening and
empirical‑Bayes amplitude inference with numerous performance tuning flags.

**Libraries:** `numpy`, `pandas`, `matplotlib`,
`scipy.signal`, `scipy.linalg`, `scipy.optimize`,
`os`, `glob`, `re`, `time`, `warnings`.

**Run:** `python Baysian_approach.py`.

### `smoothing_simplified.py`

Streamlined single‑trial plotter demonstrating NNLS‑based smoothing and
ε‑band paired‑pulse ratios.

**Libraries:** `numpy`, `pandas`, `matplotlib`,
`scipy.signal`, `scipy.linalg`, `scipy.optimize`.

**Run:** `python smoothing_simplified.py`
(update the `file_path` variable).

### `whitenng_interactive.py`

Per‑trial ΔF/F₀ computation and Kalman–RTS smoothing with visualisation of
raw and smoothed traces plus pulse amplitudes.

**Libraries:** `numpy`, `pandas`, `matplotlib`, optional `scipy.ndimage`.

**Run:** `python whitenng_interactive.py`
(the `__main__` section illustrates a full workflow).

## Example

```bash
python batch_measure.py
```

Ensure the `file_path` variable points to an Excel file using the format
described above.

