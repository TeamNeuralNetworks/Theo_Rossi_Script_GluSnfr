import os, sys, glob, zipfile, numpy as np, pandas as pd

"""
Compact model & options reference (from `Model_Calibration/event_models.py`)

For a full list of canonical models, aliases, and parameter names see
`Model_Calibration/event_models.py`. Common models include `double_exp`,
`two_component`, `binding_kinetics`, `cooperative`, and `single_exp`.

Notes:
 - Use `event_model` in the `options` dict.
 - Some models accept extra model-specific settings (e.g. `n_coop` for cooperative models).

ISI-Aware Parameters (IMPORTANT for 50Hz and fast stimulation):
 - This script automatically adjusts critical parameters based on ISI to prevent
   capturing overlapping events during single-event analysis (template fitting).
 - Set ISI in isi_by_folder or default_isi and the following are computed per-file:
   * peak_window_ms: Limited to ~60% of ISI for fast stim (avoids next pulse)
   * post_zoom_s: Plotting window scaled to show ~5 pulses
   * Recut window for template fitting: Automatically limited by extract_metrics.py
 - These adjustments prevent template contamination at 50Hz while preserving
   accuracy at 20Hz and slower stimulation frequencies.

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
    """Return a workbook-safe Excel sheet name."""
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
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\Theo_1_5_50Hz",
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\Theo_2_5_50Hz",
    r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL\\Theo_4_50Hz",
]
root_out = r"C:\\Users\\Antoine.Valera\\Desktop\\Testout"; os.makedirs(root_out, exist_ok=True)

# Per-folder train_start (seconds). Default 0.998; override selected folders to 0.498
default_start = 0.998
train_start_by_folder = {
    folders[2]: 0.498,
    folders[3]: 0.498,
    folders[4]: 0.498,
    folders[5]: 0.498,
    folders[6]: 0.498,
    folders[10]: 0.498,
    folders[11]: 0.498,
    folders[12]: 0.498,
}

# ISI control: default_isi applies to all unless overridden in isi_by_folder.
# The ISI automatically adjusts critical parameters (peak_window, post_zoom, recut_window)
# to prevent capturing overlapping events at high frequencies (50Hz).
# Example ISI values:
#   0.05 = 20Hz (default for most datasets)
#   0.02 = 50Hz (high frequency, uses shorter analysis windows)
#   0.01 = 100Hz (ultra-high frequency)
default_isi = 0.05
isi_by_folder = {
    folders[10]: 0.02,  # Theo_1_5_50Hz
    folders[11]: 0.02,  # Theo_2_5_50Hz
    folders[12]: 0.02,  # Theo_4_50Hz
}

# add a debug skip that would select one condition and adjust isis_by_folder and train_start_by_folder accordingly, given the index to keep
keep_expe_idx = None
if keep_expe_idx is not None:
    folders = [folders[keep_expe_idx]]
    train_start_by_folder = {folders[0]: train_start_by_folder.get(folders[0], default_start)}
    isi_by_folder = {folders[0]: isi_by_folder.get(folders[0], default_isi)}


summaries = {}
per_trial_rows = []

for in_dir in folders:
    out_dir = os.path.join(root_out, os.path.basename(in_dir))
    os.makedirs(out_dir, exist_ok=True)

    # Per-folder timing
    train_start = train_start_by_folder.get(in_dir, default_start)
    isi = isi_by_folder.get(in_dir, default_isi)
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

        # === ISI-Dependent Parameter Calculation ===
        # Automatically adjust critical parameters based on ISI to prevent
        # capturing overlapping events during template fitting and peak detection
        ISI_MS = isi * 1000.0  # Convert to milliseconds

        # Peak detection window: should be < ISI to avoid next pulse
        # Use 60% of ISI for fast stim, capped at 25ms for slow stim
        if ISI_MS < 30.0:
            PEAK_WINDOW_MS = max(8.0, ISI_MS * 0.6)  # 60% of ISI, min 8ms
        else:
            PEAK_WINDOW_MS = min(25.0, ISI_MS * 0.7)  # Standard window for slow stim

        # Zoom windows for plotting and analysis
        # For fast stim: limit to avoid excessive overlap visualization
        # For slow stim: use standard windows
        if ISI_MS < 30.0:
            POST_ZOOM_S = max(0.10, isi * 5)  # Show ~5 pulses or 100ms minimum
        else:
            POST_ZOOM_S = 0.20  # Standard 200ms post-train window

        PRE_ZOOM_S = 0.20  # Pre-train window (constant)

        # Define option presets
        options_presets = {
            'iglusnfr_optimized': {
                # === Preprocessing ===
                'normalize_dff': True,
                'bleach': True,
                'sg_window': 9,
                'sg_poly': 2,

                # === Kinetics Estimation ===
                'fit_source': 'global',
                'decay_progression_mode': 'none',  # Allow non-linear but still monotonic progression
                'anchor_final_tau': False,  # Don't over-constrain - let the model fit naturally
                'anchor_first_tau': False,

                # === Event Model ===
                'event_model': 'iglusnfr',  # Specifically optimized for iGluSnFR S72A
                'event_model_settings': {},

                # === Recut/Averaging ===
                'recut_projection': 'mean',
                'recut_oversample': 20,
                'recut_peak_recenter': 0,
                'recut_snippets': True,

                # === Onset Detection for High-Frequency Trains ===
                # Method for excluding contaminated pre-onset baseline:
                # - 'inflection': Find inflection point (minimum derivative) - default
                # - 'baseline_threshold': Exclude all points below baseline + threshold * peak
                # - 'none': No onset masking
                'onset_method': 'baseline_threshold',  # Use aggressive baseline masking for 50Hz
                'onset_baseline_threshold': 0.15,  # 15% above baseline (adjustable 0.1-0.3)

                # === NNLS Fitting ===
                'nnls_weight_mode': 'savgol',
                'nnls_weight_tau_s': None,
                'fit_diagnostic_plot': False,
                'allow_shift': True,
                'huber_delta': 2.5,
                'irls_iters': 20,
                'delta_max_s': 0.002,
                'delta_step_s': 0.00025,
                'shift_min_s': 0.00005,

                # === Time Windows (ISI-aware) ===
                'pre_zoom_s': PRE_ZOOM_S,  # Computed above based on ISI
                'post_zoom_s': POST_ZOOM_S,  # Automatically adjusted for fast/slow stim
                'f0_window_s': 1.0,

                # === Peak Detection (ISI-aware) ===
                'peak_window_ms': PEAK_WINDOW_MS,  # Automatically scaled to avoid next pulse
                'peak_avg_points': 1,  # Capture sharp peaks without averaging
                'pre_peak_ms': 1.0,

                # === Thresholding ===
                'measurement': 'NNLS',
                'fail_method': 'SAVGOL',
                'threshold_mode': 'auto',
                'null_N': 1.0,
                'null_sim_max_points': 1000,
                'null_min_post_zoom_s': 0.05,

                # === Kinetics Grids ===
                # Ultra-fast rise times for sharp iGluSnFR peaks
                'kin_taur_grid_ms': [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
                # Bi-exponential decay: fast and slow components
                'kin_taud0_grid_ms': [2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0, 80.0, 120.0],
                'kin_slope_grid_ms': [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0],

                # === Bleach Correction ===
                'bleach_huber_delta': 3.0,
                'bleach_tau_range_factor': (0.25, 4.0),
                'bleach_n_tau': 25,

                # === Plotting ===
                'plot': {
                    'enabled': True,
                    'traces': ['raw', 'nnls'],
                    'show_decay': True,
                    'trials': False,
                    'baseline': False,
                    'residuals': True,
                    'plot_peaks_details': True,
                },

                # === Template Variants (Experimental) ===
                # Enable multi-template NNLS: test multiple slow/fast ratios per event
                # NNLS automatically selects best combination based on residuals
                'use_template_variants': True,  # Set to True to enable
                'template_variant_ratios': np.arange(0.0, 1.0, 0.1),  # Slow component fractions to test
            },
            'double_exp_default': {
                # === Preprocessing ===
                'normalize_dff': True,
                'bleach': True,
                'sg_window': 9,
                'sg_poly': 2,

                # === Kinetics Estimation ===
                'fit_source': 'global',
                'decay_progression_mode': 'linear',
                'anchor_final_tau': True,
                'anchor_first_tau': False,

                # === Event Model ===
                'event_model': 'double_exp',
                'event_model_settings': {},

                # === Recut/Averaging ===
                'recut_projection': 'median',
                'recut_oversample': 50,
                'recut_peak_recenter': 0,
                'recut_snippets': True,

                # === Onset Detection for High-Frequency Trains ===
                # Method for excluding contaminated pre-onset baseline:
                # - 'inflection': Find inflection point (minimum derivative) - default
                # - 'baseline_threshold': Exclude all points below baseline + threshold * peak
                # - 'none': No onset masking
                'onset_method': 'inflection',  # Use default inflection method for 20Hz
                'onset_baseline_threshold': 0.15,  # 15% above baseline (adjustable 0.1-0.3)

                # === NNLS Fitting ===
                'nnls_weight_mode': 'savgol',
                'nnls_weight_tau_s': None,
                'fit_diagnostic_plot': False,
                'allow_shift': True,
                'huber_delta': 5.5,
                'irls_iters': 20,
                'delta_max_s': 0.002,
                'delta_step_s': 0.00025,
                'shift_min_s': 0.00005,

                # === Time Windows (ISI-aware) ===
                'pre_zoom_s': PRE_ZOOM_S,  # Computed above based on ISI
                'post_zoom_s': POST_ZOOM_S,  # Automatically adjusted for fast/slow stim
                'f0_window_s': 1.0,

                # === Peak Detection (ISI-aware) ===
                'peak_window_ms': PEAK_WINDOW_MS,  # Automatically scaled to avoid next pulse
                'peak_avg_points': 1,  # Reduced from 5 to capture sharp peaks better
                'pre_peak_ms': 1.0,  # Increased from 0.0 to ensure full peak capture

                # === Thresholding ===
                'measurement': 'NNLS',
                'fail_method': 'SAVGOL',
                'threshold_mode': 'auto',
                'null_N': 1.0,
                'null_sim_max_points': 1000,
                'null_min_post_zoom_s': 0.05,

                # === Kinetics Grids ===
                'kin_taur_grid_ms': [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0],  # Added faster rise times
                'kin_taud0_grid_ms': [1.6, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0, 12.5, 15.0, 18.0, 22.0, 28.0, 35.0, 45.0, 60.0],
                'kin_slope_grid_ms': [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0],

                # === Bleach Correction ===
                'bleach_huber_delta': 3.0,
                'bleach_tau_range_factor': (0.25, 4.0),
                'bleach_n_tau': 25,

                # === Plotting ===
                'plot': {
                    'enabled': True,
                    'traces': ['raw', 'nnls'],
                    'show_decay': True,
                    'trials': False,
                    'baseline': False,
                    'residuals': True,
                    'plot_peaks_details': True,
                }
            },
        }

        # Choose which preset to use
        preset_name = 'iglusnfr_optimized'  # Use iGluSnFR-specific model for better peak capture
        options = options_presets[preset_name]

        base = os.path.splitext(os.path.basename(xlsx_path))[0]
        res = extract_metrics(
            time, trials,
            train_start=train_start,  # seconds
            isi=isi,                  # seconds (per-folder override supported)
            n_pulses=n_pulses,
            options=options,
            filename=base  # Add filename for plot title
        )
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
