# iGluSnFR Analysis: Extraction, Batch Processing, and Model Calibration

This repository provides a practical toolkit to analyze iGluSnFR stimulation trains from Excel workbooks, extract robust per‑pulse metrics, batch export results, and compare simple kinetic models on averaged responses.

It combines three complementary pieces:

- Feature extraction API with demos: `Feature_extraction/extract_metrics.py` and `Feature_extraction/demo_*.py`
- Batch processing pipeline and CLI: `batch_measure_complex.py` with `USAGE_GUIDE.md`
- Model calibration demos: `Model_Calibration/demo_adjust_fit_events.py` and `Model_Calibration/Demo_two_good_model_fitting.py`

See `PROCESSING_README.md` for end‑to‑end recipes using existing demo scripts and copied code snippets.

## Model suitability for biphasic (fast + slow) waveforms

Models that are explicitly multi-component or have two clear decay terms will fit a fast + slow (double-exponential) waveform well; single-timescale / single-shape models will struggle.

- Good choices (can capture a fast component + slower decay):
  - `double_exp` — explicit two-exponential form (best match)
  - `two_component` / `two_component_shared_rise` — explicit fast + slow amplitude/tau
  - `double_cooperative` / cooperative variants — two-component cooperative forms
  - `diffusion_clearance` — has two clearance taus + fraction (can produce biphasic decay)
  - `binding_kinetics` — can show multi-timescale behaviour depending on kon/koff/tau_clear

- Models that will struggle (poor fit for a true double exponential):
  - `single_exp` — single tau_decay only
  - `alpha` — single timescale (alpha-function shape)
  - `gamma` — single dominant time constant (shape parameter controls onset)
  - `cooperative` (coop) — nonlinear amplitude but single tau_decay
  - `bilinear` — piecewise linear rise/decay, won't reproduce exponential tails accurately
  - `coop_plus_linear` / `desensitization` — add other dynamics but not two independent exponential decay terms

## Repository Layout

- Feature extraction:
  - `Feature_extraction/extract_metrics.py`: small, explicit interface to compute amplitudes (NNLS and alternatives), PPR, and plots.
  - Demos: `Feature_extraction/demo_single_file.py`, `Feature_extraction/demo_single_folder.py`, `Feature_extraction/demo_batch_process.py`.
  - Extra doc: `Feature_extraction/README_extract_metrics.md`.
- Batch analysis:
  - `batch_measure_complex.py`: consolidated settings and CLI for folder and multi‑folder export; supports plotting and Excel output.
  - Docs: `USAGE_GUIDE.md` (commands), `README_batch.md` (overview).
- Model calibration:
  - `Model_Calibration/demo_adjust_fit_events.py`: build median event waveforms across files.
  - `Model_Calibration/Demo_two_good_model_fitting.py`: compare Double‑Exponential vs Cooperative Binding on average and individual traces.
  - Notebook: `Model_Calibration/Demo_Different_model_fitting.ipynb` (edit code/markdown only — ignore outputs).
- Utilities: `smoothing.py` and helpers for detrending, kernels, NNLS, and plotting.

## Quick Start

Start with the simplified extractor API (copy paths accordingly):

```
python Feature_extraction/demo_single_file.py
```

Batch across folders and export Excel/plots:

```
python batch_measure_complex.py --no-show --save-plots
```

Compare models on averaged responses (directory can be passed positionally):

```
python Model_Calibration/Demo_two_good_model_fitting.py "C:\path\to\folder"
```

More end‑to‑end examples are in `PROCESSING_README.md`.

## Input Format (Excel)

- First worksheet (index 0)
- Last column: time (seconds)
- All preceding columns: trials (non‑numeric → NaN)

## Documentation

- Processing guide with full recipes: `PROCESSING_README.md`
- Batch pipeline overview: `README_batch.md`
- Command variants and flags: `USAGE_GUIDE.md`
- Extractor API guide and snippets: `Feature_extraction/README_extract_metrics.md`
- Agent guidelines: `AGENTS.md`

## Notes

- Jupyter notebooks: when editing, ignore output cells; focus on code and markdown cells only.
- No external dependencies should be added beyond what is already used in the repo.
