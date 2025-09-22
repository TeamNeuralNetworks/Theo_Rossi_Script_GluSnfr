import os, sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt

"""
Compact model & options reference (from `Model_Calibration/event_models.py`)

Canonical models (accepted aliases) and their fit parameter names:
 - 'double_exp'  (aliases: 'double','double-exponential','biexp')
     params: ['amp', 'tau_rise', 'tau_decay', 't_peak']

 - 'cooperative' (aliases: 'coop','cooperative_binding')
     params: ['amp', 'tau_rise', 'tau_decay', 'n_coop', 't_peak']

 - 'single_exp'  (aliases: 'single','single-exponential')
     params: ['amp', 'tau_decay', 't_peak']

 - 'alpha'
     params: ['amp', 'tau', 't_peak']

 - 'gamma'
     params: ['amp', 'n', 'tau', 't_peak']

 - 'bilinear'
     params: ['amp', 't_rise', 't_decay', 't_peak']

 - 'binding_kinetics' (alias: 'binding')
     params: ['amp', 'kon', 'koff', 'tau_clear', 't_peak']

 - 'two_component' (aliases: 'two-component','two_component_shared_rise')
     params: ['amp_fast', 'tau_rise', 'tau_fast', 'amp_slow', 'tau_slow', 't_peak']

 - 'desensitization' (aliases: 'desens')
     params: ['amp', 'tau_rise', 'tau_decay', 'tau_recovery', 'desens_factor', 't_peak']

 - 'coop_plus_linear' (alias: 'cooperative_plus_linear')
     params: ['amp_coop', 'tau_rise_coop', 'tau_decay_coop', 'n_coop', 'amp_linear', 'tau_decay_linear', 't_peak']

 - 'diffusion_clearance' (alias: 'diffusion')
     params: ['amp', 'tau_diff', 'tau_clear1', 'tau_clear2', 'frac_clear1', 't_peak']

 - 'double_cooperative' (alias: 'double_coop')
     params: ['amp', 'tau_rise1', 'tau_decay1', 'n1', 'tau_rise2', 'tau_decay2', 'n2', 't_peak']

 - 'hetero_coop' (alias: 'heterogeneous_cooperative')
     params: ['amp', 'tau_rise1', 'tau_decay1', 'n1', 'frac1', 'tau_rise2', 'tau_decay2', 'n2', 't_peak']

 - 'two_comp_coop' (alias: 'two_component_cooperative')
     params: ['amp_fast', 'tau_rise_fast', 'tau_decay_fast', 'n_fast', 'amp_slow', 'tau_rise_slow', 'tau_decay_slow', 'n_slow', 't_peak']

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
    train_start=0.5-0.001,   # seconds
    isi=0.05,          # seconds
    n_pulses=10,
    options={
        'normalize_dff': True,
        'bleach': True,
        # Kinetics source and progression
        'fit_source': 'global',
        'decay_progression_mode': 'free_monotonic',
        'model': 'coop_plus_linear',
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
