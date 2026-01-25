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


def _build_data_folders(base_dir: str, subfolders: list[str]) -> list[str]:
    return [os.path.join(base_dir, name) for name in subfolders]


# Input listed above
VIEW_ONLY = False
TARGET_BOUTON = ''#"20210721_linescan1_50Hz_10pulses_4mMCa_bouton4_traces_converted"
DATA_ROOT = r"C:\\Users\\Antoine.Valera\\Desktop\\PPR_DATA_FINAL"
SUBFOLDERS = [
    "Stability_Before",
    "Stability_After",
    "Stability_Before_05",
    "Stability_After_05",
    "Theo_4Ca",
    "Theo_1_5Ca",
    "WT_Theo",
    "WT_Theo_1scd",
    "WT_Anthime",
    "SynII",
    "Theo_1_5_50Hz",
    "Theo_2_5_50Hz",
    "Theo_4_50Hz",
]
folders = _build_data_folders(DATA_ROOT, SUBFOLDERS)
root_out = os.path.join(DATA_ROOT, "Testout_BiExp")  # BiExp results
os.makedirs(root_out, exist_ok=True)

# Per-folder train_start (seconds). Default 0.998; override selected folders to 0.498
default_start = 0.998
train_start_by_name = {
    "Stability_Before_05": 0.498,
    "Stability_After_05": 0.498,
    "Theo_4Ca": 0.498,
    "Theo_1_5Ca": 0.498,
    "WT_Theo": 0.498,
    "Theo_1_5_50Hz": 0.498,
    "Theo_2_5_50Hz": 0.498,
    "Theo_4_50Hz": 0.498,
}

# ISI control: default_isi applies to all unless overridden in isi_by_folder.
# The ISI automatically adjusts critical parameters (peak_window, post_zoom, recut_window)
# to prevent capturing overlapping events at high frequencies (50Hz).
# Example ISI values:
#   0.05 = 20Hz (default for most datasets)
#   0.02 = 50Hz (high frequency, uses shorter analysis windows)
#   0.01 = 100Hz (ultra-high frequency)
default_isi = 0.05
isi_by_name = {
    "Theo_1_5_50Hz": 0.02,
    "Theo_2_5_50Hz": 0.02,
    "Theo_4_50Hz": 0.02,
}

# add a debug skip that would select one condition and adjust isis_by_folder and train_start_by_folder accordingly, given the index to keep
keep_expe_idx = None
if keep_expe_idx is not None:
    folders = [folders[keep_expe_idx]]
    keep_name = os.path.basename(folders[0])
    train_start_by_name = {keep_name: train_start_by_name.get(keep_name, default_start)}
    isi_by_name = {keep_name: isi_by_name.get(keep_name, default_isi)}


summaries = {}
per_trial_rows = []
traces_by_folder = {}  # Store average traces for companion file

for in_dir in folders:
    out_dir = os.path.join(root_out, os.path.basename(in_dir))
    os.makedirs(out_dir, exist_ok=True)

    # Per-folder timing
    in_name = os.path.basename(in_dir)
    train_start = train_start_by_name.get(in_name, default_start)
    isi = isi_by_name.get(in_name, default_isi)
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
        # Use 50-70% of ISI for fast stim, capped at 25ms for slow stim
        if ISI_MS < 30.0:
            PEAK_WINDOW_MS = max(8.0, ISI_MS * 0.5)  # 50% of ISI, min 8ms
        else:
            PEAK_WINDOW_MS = min(25.0, ISI_MS * 0.3)  # Standard window for slow stim

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
                'event_model': 'iglusnfr_tri',  # Recut fit is bi-exp; superslow reserved for train

                # === Event Model Settings (initial tau values for NNLS kernels) ===
                'event_model_settings': {
                    'tau_decay_fast': 0.003,     # 3ms fast component (reasonable for iGluSnFR3v)
                    'tau_decay_slow': 0.015,     # 15ms intermediate component
                    # tau_decay_superslow: comes from post-train decay fitting
                },

                # === Parameter Bounds (auto-configured based on model) ===
                'parameter_bounds': {
                    'tau_decay_fast': (0.003, 0.010),     # 3-10ms fast component
                    'tau_decay_slow': (0.010, 0.035),     # 10-35ms intermediate
                    # tau_decay_superslow: auto from post-train decay (typically 30-50ms)
                },

                # Use all events for averaging (0 = all, N = first N only)
                'early_events_only': 0,

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

                # === PPR Safety ===
                # Floor amplitudes to noise threshold before PPR calculation
                # Prevents division by near-zero values and unrealistic PPR ratios
                'amplitude_floor_to_noise': True,  # Set to True for iGluSnFR to prevent giant PPR values

                # === NNLS Fitting ===
                'nnls_weight_mode': 'savgol',
                'nnls_weight_tau_s': None,
                'fit_diagnostic_plot': False,
                'huber_delta': 2.5,
                'irls_iters': 20,
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
                    'nnls_residual': True,
                    'nnls_n_minus_1': True,
                    'plot_peaks_details': True,
                },

                # === Template Variants (Tri-exp fractions) ===
                # NNLS selects best slow/superslow fraction pairs per event
                'use_template_variants': True,  # Set to True to enable
                # Slow fraction grid (fast = 1 - slow - superslow)
                'template_variant_ratios': [0.2, 0.4, 0.6, 0.8],
                # Superslow fraction at final event (ramps up monotonically across the train)
                'template_variant_superslow_fracs': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                # Disable superslow if it is too close to the slow tau
                'superslow_min_ratio': 1.2,
                # === Jitter Variants (NEW) ===
                # Enable temporal jitter search in milliseconds
                # Reduced range to prevent NNLS convergence issues
                'jitter_variant_ms': np.arange(-3.0, 3.1, 1.0),  # ±3ms in 1ms steps (7 values)
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

                # === PPR Safety ===
                # Floor amplitudes to noise threshold before PPR calculation
                # Prevents division by near-zero values and unrealistic PPR ratios
                'amplitude_floor_to_noise': False,  # Optional safety feature (not needed for 20Hz)

                # === NNLS Fitting ===
                'nnls_weight_mode': 'savgol',
                'nnls_weight_tau_s': None,
                'fit_diagnostic_plot': False,
                'huber_delta': 5.5,
                'irls_iters': 20,

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
                },

                # === Jitter Variants (Optional) ===
                # For non-variant models like double_exp, you can still use jitter search
                # Set to None to use legacy allow_shift mode, or specify a range for grid search
                'jitter_variant_ms': None,  # Example: np.arange(-1.0, 1.1, 0.2)
            },
        }

        # Choose which preset to use
        preset_name = 'iglusnfr_optimized'  # Use iGluSnFR-specific model for better peak capture
        options = options_presets[preset_name]

        base = os.path.splitext(os.path.basename(xlsx_path))[0]
        if VIEW_ONLY and base != TARGET_BOUTON:
            continue
        res = extract_metrics(
            time, trials,
            train_start=train_start,  # seconds
            isi=isi,                  # seconds (per-folder override supported)
            n_pulses=n_pulses,
            options=options,
            filename=base  # Add filename for plot title
        )
        if res.get('figure') is not None:
            if VIEW_ONLY:
                res['figure'].show()
            else:
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

        # Store average trace for companion traces file
        y_avg = res['average'].get('y_avg')
        time_s = res.get('time_s')
        if y_avg is not None and time_s is not None:
            folder_name = os.path.basename(in_dir)
            if folder_name not in traces_by_folder:
                traces_by_folder[folder_name] = {}
            traces_by_folder[folder_name][base] = (np.asarray(time_s, float), np.asarray(y_avg, float))

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

if not VIEW_ONLY:
    # Save a multi-sheet workbook with one sheet per input folder
    main_out = os.path.join(root_out, "summary.xlsx")
    with pd.ExcelWriter(main_out) as writer:
        for folder_name, df in summaries.items():
            df.to_excel(writer, sheet_name=_safe_sheet_name(folder_name), index=False)
    if per_trial_rows:
        pd.DataFrame(per_trial_rows).to_excel(os.path.splitext(main_out)[0] + "_trials.xlsx", index=False)

    # Save companion traces and times files (separate files, no interpolation)
    # - summary_traces.xlsx: amplitude values only (one column per bouton ID)
    # - summary_times.xlsx: time vectors (one column per bouton ID, same order)
    traces_out = os.path.splitext(main_out)[0] + "_traces.xlsx"
    times_out = os.path.splitext(main_out)[0] + "_times.xlsx"
    with pd.ExcelWriter(traces_out) as trace_writer, pd.ExcelWriter(times_out) as time_writer:
        wrote_traces = False
        for folder_name, id_traces in traces_by_folder.items():
            if not id_traces:
                continue
            # Build DataFrames: one column per bouton ID (no Time column in traces)
            # Each trace keeps its original time vector (no interpolation)
            # Use pd.Series to handle different lengths per column
            trace_dict = {}
            time_dict = {}
            for bid, (t_vec, y_avg) in sorted(id_traces.items()):
                trace_dict[bid] = pd.Series(y_avg)
                time_dict[bid] = pd.Series(t_vec)
            trace_df = pd.DataFrame(trace_dict)
            time_df = pd.DataFrame(time_dict)
            trace_df.to_excel(trace_writer, sheet_name=_safe_sheet_name(folder_name), index=False)
            time_df.to_excel(time_writer, sheet_name=_safe_sheet_name(folder_name), index=False)
            wrote_traces = True
        if not wrote_traces:
            pd.DataFrame({"info": ["No traces found"]}).to_excel(trace_writer, sheet_name="Summary", index=False)
            pd.DataFrame({"info": ["No traces found"]}).to_excel(time_writer, sheet_name="Summary", index=False)
    print(f"[export] Saved average traces to: {traces_out}")
    print(f"[export] Saved time vectors to: {times_out}")
