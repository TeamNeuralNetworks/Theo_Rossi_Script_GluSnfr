# Processing Guide

Step‑by‑step recipes for single files, folders, and batches using the existing demo scripts and code snippets already in this repo.

See also: `USAGE_GUIDE.md`, `README_batch.md`, and `Feature_extraction/README_extract_metrics.md`.

## Single File (extract_metrics API)

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

Source: `Feature_extraction/README_extract_metrics.md` and `Feature_extraction/demo_single_file.py`.

## One Folder (extract_metrics API)

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

Source: `Feature_extraction/README_extract_metrics.md` and `Feature_extraction/demo_single_folder.py`.

## Multiple Folders (extract_metrics API)

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

See complete example in `Feature_extraction/demo_batch_process.py` (includes per‑folder `train_start`).

## Batch Pipeline (batch_measure_complex.py)

Use the consolidated batch script for folder/multi‑folder export, headless plotting, and Excel summaries.

- Single file quick run:
  ```
  python batch_measure_complex.py --single "C:\\path\\to\\fiber.xlsx"
  ```
- Save a one‑row Excel summary and suppress windows:
  ```
  python batch_measure_complex.py --single "C:\\path\\to\\fiber.xlsx" --out "C:\\path\\to\\fiber_summary.xlsx" --save-plots --no-show
  ```
- Batch process default folders (configure in file):
  ```
  python batch_measure_complex.py --no-show --save-plots
  ```

More variants and flags: `USAGE_GUIDE.md` and `README_batch.md`.

## Model Calibration Demos

Build median event waveforms across files, then compare two kinetic models.

- Median waveform aggregation (folder can be passed or inferred via `GLUSNFR_IN_DIR`):
  ```
  python Model_Calibration/demo_adjust_fit_events.py "C:\\path\\to\\folder"
  ```
- Compare Double‑Exponential vs Cooperative Binding on averaged and individual traces:
  ```
  python Model_Calibration/Demo_two_good_model_fitting.py "C:\\path\\to\\folder"
  ```

Notes:
- Excel input format: last column time (s); preceding columns trials.
- When editing notebooks, ignore output cells — update only code and markdown.

