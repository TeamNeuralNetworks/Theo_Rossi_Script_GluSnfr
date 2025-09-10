import os, glob, zipfile, numpy as np, pandas as pd
from Feature_extraction.extract_metrics import extract_metrics

# Input folders (last two use a different train_start)
folders = [
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\Stability_Before",
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\Stability_After",
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\Stability_Before_05",
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\Stability_After_05",
]
root_out = r"C:\\Users\\Antoine.Valera\\Desktop\\Testout"; os.makedirs(root_out, exist_ok=True)

# Per-folder train_start (seconds). Default 1.0; override last two to 0.5
train_start_by_folder = {
    folders[2]: 0.5,
    folders[3]: 0.5,
}

for in_dir in folders:
    out_dir = os.path.join(root_out, os.path.basename(in_dir))
    os.makedirs(out_dir, exist_ok=True)

    # Per-folder timing
    train_start = train_start_by_folder.get(in_dir, 1.0)
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

    # Save per-folder summary
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
