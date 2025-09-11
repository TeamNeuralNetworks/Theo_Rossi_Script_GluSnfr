import os, sys, numpy as np, pandas as pd

# Ensure repo root is on sys.path when running this script from the subfolder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics

xlsx_path = r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Theo_4Ca\20211125_linescan1_20Hz_10pulses_4mMCa_bouton1_traces_converted.xlsx"
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

# Inspect results
print("Averages (NNLS):", res['average']['amp_nnls'])
print("PPR (NNLS):", res['average']['ppr_nnls'])
print("A1 thresholds per trial:", res['threshold_amp1'])
print("A1 p-values per trial:", res['pval_amp1'])

# Save plot if enabled
fig = res.get('figure')
if fig is not None:
    fig.savefig(r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\fiber_plot.png", dpi=150)
