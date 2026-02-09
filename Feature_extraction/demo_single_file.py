"""
demo_single_file.py - Single-file analysis for iGluSnFR pulse trains

Quick reference for options (see Model_Calibration/event_models.py for details):
  - event_model: 'double_exp', 'iglusnfr', 'single_exp', 'cooperative', etc.
  - decay_progression_mode: 'fixed', 'linear', 'free_monotonic', 'none'
  - fit_source: 'global', 'average', 'individual'
  - nnls_weight_mode: 'uniform', 'linear', 'exponential', 'savgol', 'peak'
"""

import os, sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
#                         USER CONFIGURATION - EDIT HERE
# =============================================================================

# --- Data paths ---
DATA_ROOT = r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL"
OUT_DIR = os.path.join(DATA_ROOT, "Testout")

# files = [
#     "250128_Fibre2_Bouton_5.xlsx",
#     "250305_Fibre4_Bouton_4.xlsx",
#     "20190801_linescan1_20Hz_10pulses_2.5mMCa_bouton3_traces_converted.xlsx",
#     "20191017_linescan3_50Hz_10pulses_2.5mMCa_bouton1_traces_converted.xlsx",
#     "20191017_linescan3_50Hz_10pulses_2.5mMCa_bouton2_traces_converted.xlsx",
#     "20191017_linescan3_50Hz_10pulses_2.5mMCa_bouton4_traces_converted.xlsx",
#     "20201022_linescan1_50Hz_10pulses_2.5mMCa_bouton1_traces_converted.xlsx",
#     "20201022_linescan1_50Hz_10pulses_2.5mMCa_bouton2_traces_converted.xlsx",
#     "20201022_linescan1_50Hz_10pulses_2.5mMCa_bouton4_traces_converted.xlsx",
#     "20201030_linescan2_50Hz_10pulses_2.5mMCa_bouton3_traces_converted.xlsx",
#     "20210128_linescan5_20Hz_10pulses_2.5mMCa_bouton2_traces_converted.xlsx",
#     "20210512_linescan2_50Hz_10pulses_2.5mMCa_bouton6_traces_converted.xlsx",
#     "20210518_linescan1_50Hz_10pulses_2.5mMCa_bouton1_traces_converted.xlsx",
#     "20210518_linescan1_50Hz_10pulses_2.5mMCa_bouton2_traces_converted.xlsx",
#     "20210722_linescan3_50Hz_10pulses_1.5mMCa_bouton3_traces_converted.xlsx",
#     "20210722_linescan3_50Hz_10pulses_1.5mMCa_bouton12_traces_converted.xlsx",
#     "20220726_linescan5_50Hz_10pulses_1.5mMCa_bouton2_traces_converted.xlsx",
#     "20220726_linescan5_50Hz_10pulses_4mMCa_bouton2_traces_converted.xlsx",
#     "20220726_linescan6_50Hz_10pulses_4mMCa_bouton1_traces_converted.xlsx",
# ]



# --- Select condition and file ---
# If CONDITION is empty/None, the script will search all folders for TARGET_FILE
CONDITION = ""  # Leave empty to auto-detect from file location
TARGET_FILE = "20191017_linescan3_50Hz_10pulses_2.5mMCa_bouton2_traces_converted.xlsx"
#TARGET_FILE = "20210722_linescan3_50Hz_10pulses_1.5mMCa_bouton12_traces_converted.xlsx"
#TARGET_FILE = "20210721_linescan1_50Hz_10pulses_4mMCa_bouton1_traces_converted.xlsx"
# --- Select analysis preset ---
PRESET_NAME = 'iglusnfr_optimized'  # Options: 'iglusnfr_optimized', 'double_exp', 'single_exp_fixed_8ms'

# --- Manual overrides (set to None to use lookup tables) ---
OVERRIDE_ISI = None         # e.g., 0.02 for 50Hz, 0.05 for 20Hz
OVERRIDE_BASELINE = None    # e.g., 0.498 or 0.998
OVERRIDE_N_PULSES = None    # e.g., 10

# =============================================================================
#                         CONDITION LOOKUP TABLES
# =============================================================================
# These define default ISI and baseline for each folder. Override above if needed.

ALL_CONDITIONS = [
    # --- 20Hz conditions (ISI=0.05s) ---
    "Stability_Before",      # baseline 0.998
    "Stability_After",       # baseline 0.998
    "Stability_Before_05",   # baseline 0.498
    "Stability_After_05",    # baseline 0.498
    "Theo_4Ca",              # baseline 0.498
    "Theo_1_5Ca",            # baseline 0.498
    "WT_Theo",               # baseline 0.498
    "WT_Theo_1scd",          # baseline 0.998
    "WT_Anthime",            # baseline 0.998
    "SynII",                 # baseline 0.998
    
    # --- 50Hz conditions (ISI=0.02s) ---
    "Theo_1_5_50Hz",         # baseline 0.498
    "Theo_2_5_50Hz",         # baseline 0.498
    "Theo_4_50Hz",           # baseline 0.498
]

DEFAULT_ISI =                   0.05        # 20Hz
DEFAULT_BASELINE =              0.998
DEFAULT_N_PULSES =              10

ISI_BY_CONDITION = {
    "Theo_1_5_50Hz":            0.02,
    "Theo_2_5_50Hz":            0.02,
    "Theo_4_50Hz":              0.02,
}

BASELINE_BY_CONDITION = {
    "Stability_Before_05":      0.498,
    "Stability_After_05":       0.498,
    "Theo_4Ca":                 0.498,
    "Theo_1_5Ca":               0.498,
    "WT_Theo":                  0.498,
    "Theo_1_5_50Hz":            0.498,
    "Theo_2_5_50Hz":            0.498,
    "Theo_4_50Hz":              0.498,
}

# =============================================================================
#                         ANALYSIS OPTIONS PRESETS
# =============================================================================

def _build_options_presets(peak_window_ms, pre_zoom_s, post_zoom_s):
    """Build options presets with ISI-aware parameters."""
    return {
        'iglusnfr_optimized': {
            # --- Preprocessing ---
            'normalize_dff': True,                                          # Normalize to dF/F0
            'bleach': True,                                                 # Apply bleach correction
            'sg_window': 9,                                                 # Savitzky-Golay filter window (must be odd)
            'sg_poly': 2,                                                   # Savitzky-Golay filter polynomial order
            
            # --- Kinetics ---
            'fit_source': 'global',                                         # 'global', 'average', 'individual' ; this controls the source of data for kinetics fitting
            'decay_progression_mode': 'linear',                               # 'fixed', 'linear', 'free_monotonic', 'none' ; this controls how decay kinetics evolve over pulses
            'anchor_final_tau': False,                                      # Whether to anchor the final event tau to the last-event estimate
            'anchor_first_tau': False,                                      # Whether to anchor the first event tau to a fixed value
            
            # --- Event Model ---
            'event_model': 'iglusnfr_tri',                                  # 'double_exp', 'iglusnfr', 'iglusnfr_tri', 'single_exp', 'cooperative'
            'parameter_bounds': {
                'tau_decay_fast': (0.003, 0.008),                           # fast decay bounds (s)
                'tau_decay_slow': (0.008, 0.035),                           # slow decay bounds (s)
                'tau_superslow': (0.035, 0.090),                            # superslow decay bounds (s) - only for tri-exponential
                'amplitude_ratio': (0.0, 1.0),                              # amplitude ratio bounds (0 to 1) ; 0 means all fast, 1 means all slow
            },
            'early_events_only': 0,                                         # Use only first N events for kinetics fitting (0 = all events)
            
            # --- Recut/Averaging ---
            'recut_projection': 'mean',                                     # 'mean', 'median'
            'recut_oversample': 20,                                         # Oversampling factor for recut snippets ; data is projected onto a finer time grid
            'recut_peak_recenter': 0,                                       # Recenter recut snippets on peak (0 = no recentering)
            'recut_snippets': True,                                         # Whether to extract recut snippets for visualization
            
            # --- Onset Detection ---
            'onset_method': 'baseline_threshold',                           # 'baseline_threshold', 'derivative' ; method for onset detection of recut snippets
            'onset_baseline_threshold': 0.10,                               # Threshold (fraction of peak) for baseline_threshold onset detection
            
            # --- PPR Safety ---
            'amplitude_floor_to_noise': True,                               # Floor all pulse amplitudes to the per-trial A1 threshold (thr1) before PPR; average uses median(thr1)
            
            # --- NNLS Fitting ---
            'nnls_weight_mode': 'savgol',                                   # 'uniform', 'linear', 'exponential', 'savgol', 'peak' ; weighting scheme for NNLS fitting
            'nnls_weight_tau_s': None,                                      # Time constant for exponential weighting (s) ; only used if nnls_weight_mode is 'exponential'
            'nnls_peak_window_s': 0.010,                                    # Peak-emphasis window after each stimulus (s)
            'nnls_peak_weight': 3.0,                                        # Weight multiplier inside the peak window
            'fit_diagnostic_plot': False,                                   # Whether to generate fit diagnostic plots
            'huber_delta': 2.5,                                             # Huber loss delta for robust fitting (in std units); set to None to disable robust fitting
            'irls_iters': 20,                                               # Number of IRLS iterations for robust fitting ; only used if huber_delta is set
            'nnls_last_event_tail_tau_s': 'best',                           # Last event tail downweight tau (s); None=off, 'auto'=ISI, 'best'=search for optimal, or float

            # --- Time Windows (ISI-aware) ---
            'pre_zoom_s': pre_zoom_s,                                       # Pre-event snippet duration (s); controls how much data before each event is shown ; does not affect fitting
            'post_zoom_s': post_zoom_s,                                     # Post-event snippet duration (s); controls how much data after each event is shown ; does not affect fitting
            'f0_window_s': 1.0,                                             # F0 baseline window duration (s) ; controls how baseline F0 is computed for dF/F0 normalization
            
            # --- Peak Detection (ISI-aware) ---
            'peak_window_ms': peak_window_ms,                               # Peak detection window duration (ms) ; controls how peaks are identified within each event
            'peak_avg_points': 1,                                           # Number of points to average around peak for amplitude measurement
            'pre_peak_ms': 1.0,                                             # Pre-peak baseline window (ms) ; controls how local baseline before each peak is computed ;
            
            # --- Thresholding ---
            'measurement': 'NNLS',                                          # 'NNLS', 'SAVGOL', 'RAW' ; amplitude series used for p-values/classification
            'fail_method': 'SAVGOL',                                        # 'NNLS', 'SAVGOL', 'RAW' ; method used to build null/noise amplitudes for thresholding
            'threshold_mode': 'auto',                                       # 'auto', 'mad', 'sd' ; auto => mad for NNLS null, sd for SAVGOL/RAW null
            'null_N': 1.0,                                                  # Multiplier for null distribution to set threshold ; only used if threshold_mode is 'auto'
            'null_sim_max_points': 1000,                                    # Max points for null distribution simulation
            'null_min_post_zoom_s': 0.05,                                   # Minimum post-zoom duration (s) to use for null distribution simulation
            
            # --- Bleach Correction ---
            'bleach_huber_delta': 3.0,                                      # Huber loss delta for bleach fitting (in std units); set to None to disable robust fitting
            'bleach_tau_range_factor': (0.25, 4.0),                         # Range factor for bleach tau fitting ; multiplied by initial estimate to get min and max bounds
            'bleach_n_tau': 25,     
            'template_variant_select': 'soft',                                        # Number of tau candidates for bleach fitting
            
            # --- Plotting ---
            'plot': {
                'enabled': True,                                            # Master plot enable/disable
                'traces': ['raw', 'nnls'],                                  # 'raw', 'bleach_corrected', 'nnls', 'nnls_corr' ; which traces to plot
                'figsize': (10, 6),                                         # Figure size
                'show_decay': True,                                         # Show decay fits on average plot
                'show_onsets': True,                                        # Show detected onsets on recut snippets
                'trials': True,                                            # Whether to generate per-trial figures
                'baseline': True,                                          # Whether to show baseline F0 levels on traces
                'residuals': True,                                          # Whether to show residuals on average plot
                'nnls_residual': False,                                     # Whether to show NNLS residuals on average plot
                'nnls_n_minus_1': False,                                    # Whether to show NNLS n-1 fit on average plot
                'plot_peaks_details': True,                                 # Show peak detection details on traces
                'param_evolution': True,                                    # Show parameter evolution across events
            },
            
            # --- Template Variants ---
            'use_template_variants': True,                                  # Enable variant testing, where we try multiple tau combinations
            'template_variant_ratios': np.linspace(0.1, 0.9, 10),           # bi-exp and tri-exp
            'template_variant_superslow_fracs': np.linspace(0.1, 0.9, 11),  # tri-exp only
            'superslow_min_ratio': 1.0,                                     # Minimum ratio between slow and superslow taus for tri-exp variants
            'allow_tau_slow_override': True,                                # If True, allow tau_slow = tau_superslow in tri-exp models if it improves fit
            'force_tau_slow_override': False,                                # If True, force tau_slow = tau_superslow in tri-exp models ; unlike allow_tau_slow_override, this enforces the equality rather than just allowing it
            
            # --- Jitter Variants ---
            'jitter_variant_ms': np.linspace(-1.0, 1.0, 5),                # Jitter variants to try (ms) ; set to None to disable jitter variants ; jitter means we shift event times by +/- jitter to test robustness
        },
    }


# =============================================================================
#                              INTERNAL SETUP
# =============================================================================
# (No need to edit below unless debugging)

try:
    _here = os.path.dirname(__file__)
except NameError:
    _here = os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(_here, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics

# --- Auto-detect condition from filename ---
def _find_condition(data_root, conditions, filename):
    """Search all condition folders for filename, return condition name or None."""
    for cond in conditions:
        if os.path.isfile(os.path.join(data_root, cond, filename)):
            return cond
    return None

# --- Resolve parameters (use overrides if set, otherwise lookup) ---
if not CONDITION:
    CONDITION = _find_condition(DATA_ROOT, ALL_CONDITIONS, TARGET_FILE)
    if not CONDITION:
        raise FileNotFoundError(f"'{TARGET_FILE}' not found in any condition folder")

ISI = OVERRIDE_ISI if OVERRIDE_ISI is not None else ISI_BY_CONDITION.get(CONDITION, DEFAULT_ISI)
START = OVERRIDE_BASELINE if OVERRIDE_BASELINE is not None else BASELINE_BY_CONDITION.get(CONDITION, DEFAULT_BASELINE)
N_PULSES = OVERRIDE_N_PULSES if OVERRIDE_N_PULSES is not None else DEFAULT_N_PULSES

xlsx_path = os.path.join(DATA_ROOT, CONDITION, TARGET_FILE)

# --- Compute ISI-dependent parameters ---
ISI_MS = ISI * 1000.0
MARGIN_MS = 2.0  # Fixed margin before next event (ms)
PEAK_WINDOW_MS = max(5.0, ISI_MS - MARGIN_MS)  # Use all data minus 2ms margin
POST_ZOOM_S = 0.3  # Show ~5 pulses
PRE_ZOOM_S = 0.20

# --- Print configuration ---
print("=" * 60)
print(f"  Condition:  {CONDITION}")
print(f"  File:       {TARGET_FILE}")
print(f"  ISI:        {ISI_MS:.0f}ms ({1/ISI:.0f}Hz)")
print(f"  Baseline:   {START:.3f}s")
print(f"  N pulses:   {N_PULSES}")
print(f"  Preset:     {PRESET_NAME}")
print(f"  Peak win:   {PEAK_WINDOW_MS:.1f}ms | Post zoom: {POST_ZOOM_S:.3f}s")
print("=" * 60)

# --- Build presets and select ---
options_presets = _build_options_presets(PEAK_WINDOW_MS, PRE_ZOOM_S, POST_ZOOM_S)
options = options_presets[PRESET_NAME]

# --- Load data ---
os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
_time = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
_trials = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
valid = np.isfinite(_time)
time = _time[valid]
trials = _trials[valid, :]

base = os.path.splitext(os.path.basename(xlsx_path))[0]

# =============================================================================
#                              RUN ANALYSIS
# =============================================================================
res = extract_metrics(
    time, trials,
    train_start=START,
    isi=ISI,
    n_pulses=N_PULSES,
    options=options,
    filename=base
)

# =============================================================================
#                              RESULTS OUTPUT
# =============================================================================

# --- Extract amplitudes and PPR ---
amp_avg = res['average'].get('amp_nnls_corr', res['average']['amp_nnls'])
ppr_avg = res['average'].get('ppr_nnls_corr')
if ppr_avg is None:
    a1 = float(amp_avg[0]) if len(amp_avg) else np.nan
    ppr_avg = (amp_avg / a1) if np.isfinite(a1) and abs(a1) > 1e-12 else amp_avg * np.nan

print("\n--- Results ---")
print("Amplitudes (NNLS):", amp_avg)
print("PPR (NNLS):", ppr_avg)
print("A1 thresholds:", res['threshold_amp1'])
print("A1 p-values:", res['pval_amp1'])

# --- Build summary row ---
row = {'measurement': 'NNLS', 'ID': base}
for i, v in enumerate(amp_avg, 1):
    row[f'AMP{i}'] = float(v)
for i in range(2, len(ppr_avg) + 1):
    row[f'PPR{i}/1'] = float(ppr_avg[i - 1])

# --- Per-trial failure counts ---
per_trial_rows, fail_counts = [], {i: [0, 0] for i in range(1, 4)}
for idx_trial, rtrial in enumerate(res.get('per_trial', [])):
    amp_trial = np.asarray(rtrial.get('amp_nnls_corr', rtrial.get('amp_nnls')), float)
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

# --- Save to files ---
df_rows = pd.DataFrame([row])
ordered = [f'AMP{i}' for i in range(1, 11)] + [f'PPR{i}/1' for i in range(2, 11)] + [f'%Fail{i}' for i in range(1, 4)]
for col in ['measurement', 'ID', *ordered]:
    if col not in df_rows.columns:
        df_rows[col] = np.nan
df_rows = df_rows[['ID', *ordered, 'measurement']]

csv_out = os.path.join(OUT_DIR, f"{base}_summary.csv")
df_rows.to_csv(csv_out, index=False)
xl_out = os.path.splitext(csv_out)[0] + ".xlsx"
df_rows.to_excel(xl_out, index=False)
if per_trial_rows:
    pd.DataFrame(per_trial_rows).to_excel(os.path.splitext(xl_out)[0] + "_trials.xlsx", index=False)

# =============================================================================
#                              PLOTTING
# =============================================================================

fig = res.get('figure')
if fig is not None:
    try:
        fig.savefig(os.path.join(OUT_DIR, "traces_converted_plot.png"), dpi=150)
    except Exception:
        pass

    try:
        plt.show()
    except Exception:
        pass

# Per-trial figures
figs_trials = res.get('figures_trials') or []
for i, ftri in enumerate(figs_trials, 1):
    try:
        outp = os.path.join(OUT_DIR, f"{base}_trialfig_{i:02d}.png")
        ftri.tight_layout()
        ftri.savefig(outp, dpi=120)
    except Exception:
        pass
if figs_trials:
    try:
        plt.show()
    except Exception:
        pass

# Parameter evolution figure
fig_param = res.get('figure_param_evolution')
if fig_param is not None:
    try:
        fig_param.savefig(os.path.join(OUT_DIR, f"{base}_param_evolution.png"), dpi=150)
        print(f"[demo] Saved param evolution figure")
        plt.show()
    except Exception as e:
        print(f"[demo] Error saving param evolution figure: {e}")
