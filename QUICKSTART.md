# Quickstart

Two common starting points: reproducing the published figures from the
released dataset, and re-running the extraction pipeline on your own
recordings with different settings.

## What do I need to generate the paper figures?

1. **Data** — download the dataset from Zenodo,
   DOI [10.5281/zenodo.21554049](https://doi.org/10.5281/zenodo.21554049),
   and extract it. The archive contains one folder, `release/`, with the
   consolidated CSV tables described in
   [`dataset_tools/SCHEMA.md`](dataset_tools/SCHEMA.md).

2. **Code** — clone or download this repository at the tagged release linked
   from the Zenodo record (the `main`/`prepublication` branch may have moved
   on since the deposit was minted).

3. **Python environment**
   ```bash
   conda create -n glusnfr python=3.11
   conda activate glusnfr
   pip install -r requirements.txt
   ```

4. **Point the code at your data.** Either:
   - extract/copy the `release/` folder so it ends up at
     `<repo_root>/PPR_DATA_FINAL/release/` (no path edits needed), **or**
   - set the `GLUSNFR_DATA_ROOT` environment variable to wherever you put the
     data (the folder that directly *contains* `release/`):
     ```powershell
     $env:GLUSNFR_DATA_ROOT = "C:\path\to\your\data"
     ```
     ```bash
     export GLUSNFR_DATA_ROOT="/path/to/your/data"
     ```

   Both `Support_figure.ipynb` and the `Feature_extraction`/`Model_Calibration`
   demo scripts read this same variable, so you only need to set it once. (Only
   `Support_figure.ipynb` and the other paper-support notebooks can run from
   the public Zenodo deposit alone — see the note below.)

5. **Run the notebook.** Open
   [`Support_figure.ipynb`](Support_figure.ipynb) in Jupyter or VS Code, select
   the `glusnfr` kernel, and run all cells in order. Figures and statistics
   tables are written to `<data_root>/output/`.

   [`docs/model_fitting/Model_Fitting_Gallery.ipynb`](docs/model_fitting/Model_Fitting_Gallery.ipynb)
   and [`docs/nnls_lecture/NNLS_Lecture_Demo.ipynb`](docs/nnls_lecture/NNLS_Lecture_Demo.ipynb)
   are self-contained and don't require the dataset.

**Scope note:** the Zenodo deposit ships only the *converted* recordings
(`release/*.csv`) — not the original raw per-condition `.xlsx` files or the
`.tif` microscopy stacks (see `SCHEMA.md` §4). That's sufficient to reproduce
every figure. It is **not** sufficient to run the `Feature_extraction`/
`Model_Calibration` demo scripts, which extract pulse amplitudes from raw
recordings — those demos expect the pre-conversion `.xlsx` tree and are meant
to be run against your own raw data (next section).

## How do I re-extract data differently?

Use this if you want to change extraction settings (fitting model, template
options, thresholds, ...) and regenerate the derived tables, rather than just
reproducing the published figures.

This requires your own raw recordings, organized as
`<DATA_ROOT>/<condition>/<recording>.xlsx` (one subfolder per experimental
condition) — the layout `dataset_tools/build_manifest.py` expects. This raw
tree is not part of the Zenodo deposit.

1. **Extract pulse-by-pulse metrics** with your chosen options, either for one
   file ([`Feature_extraction/demo_single_file.py`](Feature_extraction/demo_single_file.py))
   or the whole dataset
   ([`Feature_extraction/demo_batch_process.py`](Feature_extraction/demo_batch_process.py)).
   Set `GLUSNFR_DATA_ROOT` as above and edit the `options={}` dict / preset
   name in the demo to match the extraction you want. Set `WRITE_TIDY=True` in
   `demo_batch_process.py` to write the tidy tables straight into
   `<DATA_ROOT>/release/`, or keep the legacy `summary_*` outputs and convert
   them in a separate step (next).

2. **Rebuild the manifest and consolidated tables** from the raw folders and
   your extraction outputs:
   ```bash
   python dataset_tools/build_manifest.py   --data-root <DATA_ROOT>
   python dataset_tools/reorganize_raw.py   --data-root <DATA_ROOT> --out <DATA_ROOT>/release
   python dataset_tools/convert_saturation.py --xlsx <DATA_ROOT>/Saturation_data.xlsx --out <DATA_ROOT>/release
   python dataset_tools/consolidate.py --data-root <DATA_ROOT> \
          --summary-dir <dir with summary_*> --out <DATA_ROOT>/release
   ```
   See [`dataset_tools/SCHEMA.md`](dataset_tools/SCHEMA.md) §5 for what each
   step produces and how the columns are defined.

3. **Re-run `Support_figure.ipynb`** against the regenerated `release/`
   folder — no notebook changes needed, since it reads through
   `dataset_tools/release_io.py` the same way regardless of how `release/`
   was produced.
