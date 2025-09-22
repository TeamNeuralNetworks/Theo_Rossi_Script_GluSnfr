import os, sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt

"""
Compact model & options reference (from `Model_Calibration/event_models.py`)

For a full list of canonical models, aliases, and parameter names see
`Model_Calibration/event_models.py`. Common models include `double_exp`,
`two_component`, `binding_kinetics`, `cooperative`, and `single_exp`.

Notes:
 - Use `event_model` (preferred) or `model` (backwards-compatible alias) in the `options` dict.
 - Some models accept extra model-specific settings (e.g. `n_coop` for cooperative models).

Decay progression modes (options['decay_progression_mode']):
 - 'fixed'         : a single tau_d applied to whole train (median)
 - 'free_monotonic': interpolate per-pulse tau_d non-decreasingly
 - 'linear'        : non-negative linear slope across pulses (default)

Kinetics fit source (options['fit_source']):
 - 'global'     : fit one template from all trials (recut median) (default)
 - 'average'    : fit kinetics on the multi-trial average trace
 - 'individual' : fit kinetics per trial then aggregate (median)

Examples (usage):
        options = {
                'event_model': 'cooperative',
                'decay_progression_mode': 'free_monotonic',
                'fit_source': 'global',
                'event_model_settings': {'n_coop': 2.0},
        }

"""

# Ensure repo root is on sys.path when running this script from the subfolder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics

xlsx_path = r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Theo_4Ca\20211125_linescan1_20Hz_10pulses_4mMCa_bouton1_traces_converted.xlsx"
xlsx_path = r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Theo_1_5Ca\20220726_linescan3_20Hz_10pulses_1.5mMCa_bouton3_traces_converted.xlsx"


out_dir = r"C:\Users\Antoine.Valera\Desktop\Testout"
os.makedirs(out_dir, exist_ok=True)
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
        # Kinetics source and progression
        'fit_source': 'global',
        'decay_progression_mode': 'linear',
        'model': 'two_component',  # backwards-compatible alias
        'recut_projection': 'robust_mean',  # 'mean'|'median'|'std'
        'recut_oversample': 20,     # integer >=1
        'plot': {
            'enabled': True,
            'traces': ['raw','savgol','nnls'],  # show all average overlays
            'show_decay': True,
            'trials': True,
            'baseline': True,
            'residuals': False,
            

        }
    }
)

# Inspect results
print("Averages (NNLS):", res['average']['amp_nnls'])
print("PPR (NNLS):", res['average']['ppr_nnls'])
print("A1 thresholds per trial:", res['threshold_amp1'])
print("A1 p-values per trial:", res['pval_amp1'])

base = os.path.splitext(os.path.basename(xlsx_path))[0]
row = {'measurement': 'NNLS', 'ID': base}
amp = res['average']['amp_nnls']; ppr = res['average']['ppr_nnls']
for i, v in enumerate(amp, 1):
    row[f'AMP{i}'] = float(v)
for i in range(2, len(ppr) + 1):
    row[f'PPR{i}/1'] = float(ppr[i - 1])

per_trial_rows, fail_counts = [], {i: [0, 0] for i in range(1, 4)}
for idx_trial, rtrial in enumerate(res.get('per_trial', [])):
    amp_trial = np.asarray(rtrial.get('amp_nnls'), float)
    thr = float(rtrial.get('thr_shared', np.nan))
    a1 = amp_trial[0] if amp_trial.size else np.nan
    status = 'NA'
    if np.isfinite(a1) and np.isfinite(thr):
        status = 'success' if a1 > thr else 'failure'
        per_trial_rows.append({'AMP1': float(a1), 'status': status, 'file': base, 'trial': idx_trial + 1})
    for p in range(1, min(3, amp_trial.size) + 1):
        val = amp_trial[p - 1]
        if np.isfinite(val) and np.isfinite(thr):
            fail_counts[p][1] += 1
            if val <= thr:
                fail_counts[p][0] += 1
for p in range(1, 4):
    n_fail, n_valid = fail_counts[p]
    if n_valid:
        row[f'%Fail{p}'] = round((n_fail / n_valid) * 100.0, 2)

df_rows = pd.DataFrame([row])
ordered = [f'AMP{i}' for i in range(1, 11)] \
    + [f'PPR{i}/1' for i in range(2, 11)] \
    + [f'%Fail{i}' for i in range(1, 4)]
for col in ['measurement', 'ID', *ordered]:
    if col not in df_rows.columns:
        df_rows[col] = np.nan
df_rows = df_rows[['ID', *ordered, 'measurement']]

csv_out = os.path.join(out_dir, f"{base}_summary.csv")
df_rows.to_csv(csv_out, index=False)
xl_out = os.path.splitext(csv_out)[0] + ".xlsx"
df_rows.to_excel(xl_out, index=False)
if per_trial_rows:
    pd.DataFrame(per_trial_rows).to_excel(os.path.splitext(xl_out)[0] + "_trials.xlsx", index=False)

# Save plot if enabled
fig = res.get('figure')
if fig is not None:
    fig.savefig(r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\fiber_plot.png", dpi=150)
    try:
        plt.show()
    except Exception:
        # If running in an environment without an interactive backend, ignore.
        pass
