import os, sys, glob, zipfile, numpy as np, pandas as pd

# Ensure repo root is on sys.path when running this script from the subfolder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics


def _safe_sheet_name(name: str) -> str:
    """Return a workbook‑safe Excel sheet name."""
    cleaned = "".join(c for c in name if c not in ":\\/?*[]")
    return (cleaned or "Sheet")[:31]



# Input listed above
folders = [
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\Stability_Before",
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\Stability_After",
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\Stability_Before_05",
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\Stability_After_05",
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\Theo_4Ca",
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\Theo_1_5Ca",
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\WT_Theo",
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\WT_Theo_1scd",
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\WT_Anthime",
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\SynII",    
]
root_out = r"C:\\Users\\Antoine.Valera\\Desktop\\Testout"; os.makedirs(root_out, exist_ok=True)

# Per-folder train_start (seconds). Default 1.0; override last two to 0.5
train_start_by_folder = {
    folders[2]: 0.499,
    folders[3]: 0.499,
    folders[4]: 0.499,
    folders[5]: 0.499,
    folders[6]: 0.499,
}

summaries = {}
per_trial_rows = []

for in_dir in folders:
    out_dir = os.path.join(root_out, os.path.basename(in_dir))
    os.makedirs(out_dir, exist_ok=True)

    # Per-folder timing
    train_start = train_start_by_folder.get(in_dir, 0.999)
    isi = 0.05
    n_pulses = 10

    rows = []
    def _is_valid_xlsx(path: str) -> bool:
        # Basic OOXML sanity check to avoid openpyxl errors on misnamed/corrupt files
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
            train_start=train_start, isi=isi, n_pulses=n_pulses,
            options={
                'normalize_dff': True,
                'bleach': True,
                'fit_source': 'global',
                'decay_progression_mode': 'free_monotonic',
                'model': 'double_exp',
                'plot': {
                    'enabled': True,
                    'traces': ['raw', 'nnls'],
                    'show_decay': True,
                    'trials': False,
                    'residuals': True,
                },
            },
        )

        base = os.path.splitext(os.path.basename(xlsx_path))[0]
        if res.get('figure') is not None:
            res['figure'].savefig(os.path.join(out_dir, f"{base}.png"), dpi=150)

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
                    'folder': os.path.basename(in_dir),
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

    # Save per-folder summary and collect for global workbook
    df_rows = pd.DataFrame(rows)
    ordered = [f'AMP{i}' for i in range(1, n_pulses + 1)] \
        + [f'PPR{i}/1' for i in range(2, n_pulses + 1)] \
        + [f'%Fail{i}' for i in range(1, 4)]
    for col in ['measurement', 'ID', *ordered]:
        if col not in df_rows.columns:
            df_rows[col] = np.nan
    df_rows = df_rows[['ID', *ordered, 'measurement']]
    df_rows.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    summaries[os.path.basename(in_dir)] = df_rows

# Save a multi-sheet workbook with one sheet per input folder
main_out = os.path.join(root_out, "summary.xlsx")
with pd.ExcelWriter(main_out) as writer:
    for folder_name, df in summaries.items():
        df.to_excel(writer, sheet_name=_safe_sheet_name(folder_name), index=False)
if per_trial_rows:
    pd.DataFrame(per_trial_rows).to_excel(os.path.splitext(main_out)[0] + "_trials.xlsx", index=False)
