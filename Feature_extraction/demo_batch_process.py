import os, sys, glob, zipfile, numpy as np, pandas as pd

"""
Compact model & options reference (from `Model_Calibration/event_models.py`)

For a full list of canonical models, aliases, and parameter names see
`Model_Calibration/event_models.py`. Common models include `double_exp`,
`two_component`, `binding_kinetics`, `cooperative`, and `single_exp`.

Notes:
 - Use `event_model` in the `options` dict.
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
        options={
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

# Per-folder train_start (seconds). Default 1.0; override last two to 0.498
train_start_by_folder = {
    folders[2]: 0.498,
    folders[3]: 0.498,
    folders[4]: 0.498,
    folders[5]: 0.498,
    folders[6]: 0.498,
}

summaries = {}
per_trial_rows = []

for in_dir in folders:
    out_dir = os.path.join(root_out, os.path.basename(in_dir))
    os.makedirs(out_dir, exist_ok=True)

    # Per-folder timing
    train_start = train_start_by_folder.get(in_dir, 0.998)
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

        # Define option presets
        options_presets = {
            'double_exp_default': {
                # === Preprocessing ===
                'normalize_dff': True,  # bool (default: True) - apply ΔF/F0 normalization
                'bleach': True,  # bool (default: True) - correct slow bleaching
                'sg_window': 9,  # int (default: 9) - Savitzky-Golay window size
                'sg_poly': 2,  # int (default: 2) - Savitzky-Golay polynomial order
                
                # === Kinetics Estimation ===
                'fit_source': 'global',  # (default: 'global') 'global'|'average'|'individual'
                'decay_progression_mode': 'free_monotonic',  # (default: 'linear') 'fixed'|'free_monotonic'|'linear'
                
                # === Event Model ===
                'event_model': 'two_comp_coop',  # (default: 'double_exp') 'double_exp'|'cooperative'|'bilinear'|'single_exp'|'two_step_binding'|'alpha'|'gamma'|'binding_kinetics'|'two_component'|'desensitization'|'coop_plus_linear'|'diffusion_clearance'|'double_cooperative'|'hetero_coop'|'two_comp_coop'
                'event_model_settings': {},  # dict (default: {}) - model-specific params - see event_models.py for details (e.g., {'n_coop': 2.0})
                
                # === Recut/Averaging ===
                'recut_projection': 'median',  # (default: 'median') 'mean'|'median'|'std'|'robust_mean'
                'recut_oversample': 50,  # int ≥1 (default: 1) - interpolation factor
                'recut_peak_recenter': 0,  # int|tuple|None (default: 0) - peak realignment (0=disabled)
                'recut_snippets': True,  # bool (default: False) - return snippets for plotting
                
                # === NNLS Fitting ===
                'nnls_weight_mode': 'savgol',  # (default: 'uniform') 'uniform'|'linear'|'exponential'|'savgol'
                'nnls_weight_tau_s': None,  # float|None (default: None=auto) - time constant (for linear or exponential modes)
                'fit_diagnostic_plot': False,  # bool (default: False) - weight + τd diagnostics
                'allow_shift': True,  # bool (default: True) - enable per-pulse micro-shifts
                'huber_delta': 5.5,  # float (default: 5.5) - robust fitting threshold
                'irls_iters': 20,  # int (default: 6) - IRLS iterations
                'delta_max_s': 0.002,  # float (default: 0.002) - max shift in seconds
                'delta_step_s': 0.00025,  # float (default: 0.00025) - shift step size in seconds
                'shift_min_s': 0.00005,  # float (default: 0.00005) - minimum shift in seconds
                
                # === Time Windows ===
                'pre_zoom_s': 0.20,  # float (default: 0.15) - pre-train window
                'post_zoom_s': 0.20,  # float (default: 0.60) - post-train window
                'f0_window_s': 1.0,  # float (default: 0.4) - baseline window
                
                # === Peak Detection ===
                'peak_window_ms': 25.0,  # float (default: 25.0) - peak search window
                'peak_avg_points': 5,  # int (default: 5) - points to average at peak
                'pre_peak_ms': 0.0,  # float (default: 0.0) - pre-peak offset
                
                # === Thresholding ===
                'measurement': 'NNLS',  # (default: 'NNLS') 'NNLS'|'SAVGOL'|'RAW' - series for p-values
                'fail_method': 'SAVGOL',  # (default: None) 'NNLS'|'SAVGOL'|'RAW'|None - failure classification (None=use measurement)
                'threshold_mode': 'auto',  # (default: 'auto') 'auto'|'mad'|'sd' - threshold rule (auto=MAD for NNLS, SD for SAVGOL)
                'null_N': 1.0,  # float (default: 3.0) - threshold multiplier
                'null_sim_max_points': 1000,  # int (default: 1000) - max null samples
                'null_min_post_zoom_s': 0.05,  # float (default: 0.05) - min post window for null
                
                # === Kinetics Grids ===
                'kin_taur_grid_ms': [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],  # list[float] (default: [0.6, 0.8, 1.0, 1.2, 1.5, 2.0])
                'kin_taud0_grid_ms': [1.6, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0, 12.5, 15.0, 18.0, 22.0, 28.0, 35.0, 45.0, 60.0],  # list[float] (default: [1.6, 2.0, ..., 60.0])
                'kin_slope_grid_ms': [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0],  # list[float] (default: [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
                
                # === Bleach Correction ===
                'bleach_huber_delta': 3.0,  # float (default: 3.0) - robust fitting threshold
                'bleach_tau_range_factor': (0.25, 4.0),  # tuple[float,float] (default: (0.25, 4.0)) - tau range multipliers
                'bleach_n_tau': 25,  # int (default: 25) - number of tau values to test
                
                # === Plotting ===
                'plot': {
                    'enabled': True,  # bool (default: False) - create plots
                    'traces': ['raw', 'nnls'],  # list[str] (default: ['nnls']) - traces to show
                    'show_decay': True,  # bool (default: True) - show decay components
                    'trials': False,  # bool (default: False) - plot individual trials
                    'baseline': False,  # bool (default: False) - show baseline diagnostics
                    'residuals': True,  # bool (default: False) - show residual analysis
                    'plot_peaks_details': True,  # bool (default: False) - show peak markers and residuals
                }
            },
        }
    


        # Choose which preset to use
        preset_name = 'double_exp_default'
        options = options_presets[preset_name]

        res = extract_metrics(
            time, trials,
            train_start=train_start,   # seconds
            isi=isi,          # seconds
            n_pulses=n_pulses,
            options=options
        )

        base = os.path.splitext(os.path.basename(xlsx_path))[0]
        if res.get('figure') is not None:
            res['figure'].savefig(os.path.join(out_dir, f"{base}.png"), dpi=150)

        row = {'measurement': 'NNLS', 'ID': base}
        amp = res['average'].get('amp_nnls_corr', res['average']['amp_nnls'])
        ppr = res['average'].get('ppr_nnls_corr')
        if ppr is None:
            a1 = float(amp[0]) if len(amp) else np.nan
            ppr = (amp / a1) if np.isfinite(a1) and abs(a1) > 1e-12 else amp * np.nan
        for i, v in enumerate(amp, 1):
            row[f'AMP{i}'] = float(v)
        for i in range(2, len(ppr) + 1):
            row[f'PPR{i}/1'] = float(ppr[i - 1])

        fail_counts = {i: [0, 0] for i in range(1, 4)}
        for idx_trial, rtrial in enumerate(res.get('per_trial', [])):
            amp_trial = np.asarray(rtrial.get('amp_nnls_corr', rtrial.get('amp_nnls')), float)
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
