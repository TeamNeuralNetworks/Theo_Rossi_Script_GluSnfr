import os, sys, glob, zipfile, numpy as np, pandas as pd

# Ensure repo root is on sys.path when running this script from the subfolder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics

in_dir  = r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Theo_4Ca\\"
out_dir = r"C:\Users\Antoine.Valera\Desktop\Testout"
os.makedirs(out_dir, exist_ok=True)
rows = []

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
        train_start=0.499, isi=0.05, n_pulses=10,
        options={
            'normalize_dff': True,
            'bleach': True,
            # Kinetics source and progression
            'fit_source': 'global',
            'decay_progression_mode': 'free_monotonic',
            'model': 'double_exp',

            'plot': {
                'enabled': True,
                'traces': ['raw','nnls'],
                'show_decay': True,
                'trials': False
            }
        }
    )


    base = os.path.splitext(os.path.basename(xlsx_path))[0]
    if res.get('figure') is not None:
        res['figure'].savefig(os.path.join(out_dir, f"{base}.png"), dpi=150)
    # Save auto-selected event model aggregated fit (if available)
    if res.get('figure_event_model') is not None:
        res['figure_event_model'].savefig(os.path.join(out_dir, f"{base}_event_model.png"), dpi=150)

    row = {'file': base}
    amp = res['average']['amp_nnls']; ppr = res['average']['ppr_nnls']
    for i, v in enumerate(amp): row[f'amp_{i+1}'] = float(v)
    for i, v in enumerate(ppr): row[f'ppr_{i+1}'] = float(v)
    rows.append(row)

pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)


# Decay progression across train (doc snippet retained for reference)
#  - 'fixed': single τd across pulses
#  - 'free_monotonic': interpolate between first and last τd (non-decreasing)
#  - 'linear': non-negative-slope linear regression across pulses

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
