import os, sys, glob, zipfile, numpy as np, pandas as pd

# Ensure repo root is on sys.path when running this script from the subfolder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics

in_dir  = r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\WT_Theo\\"
out_dir = r"C:\Users\Antoine.Valera\Desktop\Testout"
os.makedirs(out_dir, exist_ok=True)
rows, per_trial_rows = [], []

def _is_valid_xlsx(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return '[Content_Types].xml' in z.namelist()
    except Exception:
        return False

for xlsx_path in glob.glob(os.path.join(in_dir, "*.xlsx")):
    if not _is_valid_xlsx(xlsx_path):
        print(f"[skip] Not a valid .xlsx package: {xlsx_path}")
        continue
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    except Exception as e:
        print(f"[skip] Failed to read Excel: {xlsx_path} -> {e}")
        continue
    t_raw = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    ok = np.isfinite(t_raw)
    time = t_raw[ok]
    trials = X[ok, :]

    res = extract_metrics(
        time, trials,
        train_start=0.5-0.001, isi=0.05, n_pulses=10,
        options={
            'normalize_dff': True,
            'bleach': True,
            # Kinetics source and progression
            'fit_source': 'global',
            'decay_progression_mode': 'free_monotonic',
            'model': 'binding_kinetics',

            'plot': {
                'enabled': True,
                'traces': ['raw','nnls'],
                'show_decay': True,
                'trials': True,
                'residuals': True
            }
        }
    )

    """
    Available models:
    - 'double_exp' (default): classic double exponential (constrained)
    - 'cooperative': cooperative binding (Hill-like rise, exp decay)
    - 'single_exp': single exponential decay (constrained)
    - 'alpha': alpha function (constrained)
    - 'gamma': gamma function (constrained)
    - 'bilinear': bilinear rise + exp decay (constrained)
    - 'binding_kinetics': binding kinetics model with on/off rates + clearance (constrained)
    - 'two_component': two-component model with shared rise time (constrained)
    - 'desensitization': model with desensitization term (constrained)
    - 'coop_plus_linear': cooperative binding + linear component (constrained)
    - 'diffusion_clearance': diffusion rise + bi-exponential clearance (constrained)
    - 'double_cooperative': sum of two cooperative binding components (constrained)
    - 'hetero_coop': heterogeneous cooperative binding (constrained)
    """




    base = os.path.splitext(os.path.basename(xlsx_path))[0]
    if res.get('figure') is not None:
        res['figure'].savefig(os.path.join(out_dir, f"{base}.png"), dpi=150)
    # Save auto-selected event model aggregated fit (if available)
    if res.get('figure_event_model') is not None:
        res['figure_event_model'].savefig(os.path.join(out_dir, f"{base}_event_model.png"), dpi=150)

    row = {'measurement': 'NNLS', 'ID': base}
    amp = res['average']['amp_nnls']; ppr = res['average']['ppr_nnls']
    for i, v in enumerate(amp, 1):
        row[f'AMP{i}'] = float(v)
    for i in range(2, len(ppr) + 1):
        row[f'PPR{i}/1'] = float(ppr[i - 1])

    fail_counts = {i: [0, 0] for i in range(1, 4)}
    for idx_trial, rtrial in enumerate(res.get('per_trial', [])):
        amp_trial = np.asarray(rtrial.get('amp_nnls'), float)
        thr = float(rtrial.get('thr_shared', np.nan))
        a1 = amp_trial[0] if amp_trial.size else np.nan
        status = 'NA'
        if np.isfinite(a1) and np.isfinite(thr):
            status = 'success' if a1 > thr else 'failure'
            per_trial_rows.append({
                'AMP1': float(a1),
                'status': status,
                'file': base,
                'trial': idx_trial + 1,
            })
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
    rows.append(row)

df_rows = pd.DataFrame(rows)
ordered = [f'AMP{i}' for i in range(1, 11)] \
    + [f'PPR{i}/1' for i in range(2, 11)] \
    + [f'%Fail{i}' for i in range(1, 4)]
for col in ['measurement', 'ID', *ordered]:
    if col not in df_rows.columns:
        df_rows[col] = np.nan
df_rows = df_rows[['ID', *ordered, 'measurement']]

csv_out = os.path.join(out_dir, "summary.csv")
df_rows.to_csv(csv_out, index=False)
xl_out = os.path.splitext(csv_out)[0] + ".xlsx"
df_rows.to_excel(xl_out, index=False)
if per_trial_rows:
    pd.DataFrame(per_trial_rows).to_excel(os.path.splitext(xl_out)[0] + "_trials.xlsx", index=False)


# Decay progression across train (doc snippet retained for reference)
#  - 'fixed': single τd across pulses
#  - 'free_monotonic': interpolate between first and last τd (non-decreasing)
#  - 'linear': non-negative-slope linear regression across pulses

