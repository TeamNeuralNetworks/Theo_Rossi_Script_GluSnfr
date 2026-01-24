import os, sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt

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
 - Set ISI variable (in seconds) and the following are computed:
   * peak_window_ms: Limited to ~60% of ISI for fast stim (avoids next pulse)
   * post_zoom_s: Plotting window scaled to show ~5 pulses
   * Recut window for template fitting: Automatically limited by extract_metrics.py
 - These adjustments prevent template contamination at 50Hz while preserving
   accuracy at 20Hz and slower stimulation frequencies.

Decay progression modes (options['decay_progression_mode']):
 - 'fixed'         : a single tau_d applied to whole train (median)
 - 'free_monotonic': monotonic spline through per-pulse tau_d (non-decreasing)
 - 'linear'        : non-negative linear regression across pulses (default)

Anchor constraints (options['anchor_first_tau'], options['anchor_final_tau']):
 - anchor_first_tau : False (default) - if True, anchor first event's tau as minimum
 - anchor_final_tau : True (default) - if True, anchor last event's tau as maximum
                      (most reliable estimate, no following events)
 - Both can be enabled together for fully constrained progression between first and last

Kinetics fit source (options['fit_source']):
 - 'global'     : fit one template from all trials (recut median) (default)
                  For linear/monotonic: anchor global tau at middle event, then
                  fit each event on average, clip to respect anchor, then apply progression
 - 'average'    : fit each event individually on the multi-trial average trace
                  For linear/monotonic: fit per-event taus then apply progression
 - 'individual' : fit kinetics per trial then aggregate (median, can be noisy)

NNLS weight control (options['nnls_weight_mode']):
 - 'uniform'     : all timepoints have equal weight (default)
 - 'linear'      : weights decrease linearly from 1 to 0 between each stim and next
 - 'exponential' : exponential decay weights for each event
 - 'savgol'      : normalized |ΔF| from the Savitzky-Golay smoothed trace

NNLS weight time constant (options['nnls_weight_tau_s']):
 - None (auto)   : uses ISI for linear, tau_d for exponential in global mode
 - float         : explicit time constant in seconds

Examples (usage):
        options = {
                'event_model': 'cooperative',
                'decay_progression_mode': 'free_monotonic',
                'fit_source': 'global',
                'nnls_weight_mode': 'exponential',
                'nnls_weight_tau_s': 0.015,  # 15ms decay
                'fit_diagnostic_plot': True,
                'event_model_settings': {'n_coop': 2.0},
        }

"""

# Ensure repo root is on sys.path when running this script from the subfolder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics

# Test on the specific problematic 50Hz file
DATA_ROOT = r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL"
xlsx_path = os.path.join(
    DATA_ROOT,
    "Theo_4_50Hz",
    "20210721_linescan1_50Hz_10pulses_4mMCa_bouton4_traces_converted.xlsx",
)


START = 0.498
# START = START + 0.5

# === Inter-Stimulus Interval (ISI) ===
# Set this based on your stimulation frequency:
# ISI = 0.05  # 20Hz stimulation
# ISI = 0.02  # 50Hz stimulation
# ISI = 0.01  # 100Hz stimulation
ISI = 0.02  # Current: 50Hz

# === TRI-EXPONENTIAL FLAG ===
# Set to True for tri-exponential model (3 decay components: fast, slow, superslow)
# Set to False for bi-exponential model (2 decay components: fast, slow) - more stable
USE_TRI_EXPONENTIAL = True  # Testing tri-exponential

out_dir = os.path.join(DATA_ROOT, "Testout")

# === ISI-Dependent Parameter Calculation ===
# The following parameters are automatically adjusted based on ISI to prevent
# capturing overlapping events during template fitting and peak detection.
ISI_MS = ISI * 1000.0  # Convert to milliseconds

# Peak detection window: should be < ISI to avoid next pulse
# Use 50-70% of ISI for fast stim, capped at 25ms for slow stim
if ISI_MS < 30.0:
    PEAK_WINDOW_MS = max(8.0, ISI_MS * 0.6)  # 60% of ISI, min 8ms
else:
    PEAK_WINDOW_MS = min(25.0, ISI_MS * 0.7)  # Standard window for slow stim

# Zoom windows for plotting and analysis
# For fast stim: limit to avoid excessive overlap visualization
# For slow stim: use standard windows
if ISI_MS < 30.0:
    POST_ZOOM_S = max(0.10, ISI * 5)  # Show ~5 pulses or 100ms minimum
else:
    POST_ZOOM_S = 0.20  # Standard 200ms post-train window

PRE_ZOOM_S = 0.20  # Pre-train window (constant)

print(f"[ISI-aware] ISI={ISI_MS:.1f}ms, peak_window={PEAK_WINDOW_MS:.1f}ms, post_zoom={POST_ZOOM_S:.3f}s")

os.makedirs(out_dir, exist_ok=True)
df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
# time in last column; trials in all columns except last
_time = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
_trials = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
valid = np.isfinite(_time)
time = _time[valid]
trials = _trials[valid, :]

# Define option presets with defaults clearly indicated
options_presets = {
    'double_exp_default': {
        # === Preprocessing ===
        'normalize_dff': True,  # bool (default: True) - apply ΔF/F0 normalization
        'bleach': True,  # bool (default: True) - correct slow bleaching
        'sg_window': 9,  # int (default: 9) - Savitzky-Golay window size
        'sg_poly': 2,  # int (default: 2) - Savitzky-Golay polynomial order
        
        # === Kinetics Estimation ===
        'fit_source': 'average',  # (default: 'global') 'global'|'average'|'individual'
        'decay_progression_mode': 'linear',  # (default: 'linear') 'fixed'|'free_monotonic'|'linear'
        'anchor_final_tau': True,  # bool (default: True) - anchor final tau in progression fitting
        'anchor_first_tau': False,  # bool (default: False) - anchor first tau in progression fitting
        
        # === Event Model ===
        'event_model': 'iglusnfr',  # (default: 'double_exp') 'double_exp'|'cooperative'|'bilinear'|'single_exp'|'two_step_binding'|'alpha'|'gamma'|'binding_kinetics'|'two_component'|'desensitization'|'coop_plus_linear'|'diffusion_clearance'|'double_cooperative'|'hetero_coop'|'two_comp_coop'
        'event_model_settings': {},  # dict (default: {}) - model-specific params - see event_models.py for details (e.g., {'n_coop': 2.0})
        
        # === Recut/Averaging ===
        'recut_projection': 'mean',  # (default: 'median') 'mean'|'median'|'std'|'robust_mean'
        'recut_oversample': 50,  # int ≥1 (default: 1) - interpolation factor
        'recut_peak_recenter': 0,  # int|tuple|None (default: 0) - peak realignment (0=disabled)
        'recut_snippets': True,  # bool (default: False) - return snippets for plotting
        
        # === NNLS Fitting ===
        'nnls_weight_mode': 'savgol',  # (default: 'uniform') 'uniform'|'linear'|'exponential'|'savgol'
        'nnls_weight_tau_s': None,  # float|None (default: None=auto) - time constant (for linear or exponential modes)
        'fit_diagnostic_plot': True,  # bool (default: False) - weight + τd diagnostics
        'huber_delta': 5.5,  # float (default: 5.5) - robust fitting threshold
        'irls_iters': 20,  # int (default: 6) - IRLS iterations
        
        # === Time Windows ===
        'pre_zoom_s': 0.20,  # float (default: 0.15) - pre-train window
        'post_zoom_s': 0.20,  # float (default: 0.60) - post-train window
        'f0_window_s': 1.0,  # float (default: 0.4) - baseline window
        
        # === Peak Detection ===
        'peak_window_ms': 20.0,  # float (default: 25.0) - peak search window
        'peak_avg_points': 5,  # int (default: 5) - points to average at peak
        'pre_peak_ms': 0.0,  # float (default: 0.0) - pre-peak offset
        
        # === Thresholding ===
        'measurement': 'NNLS',  # (default: 'NNLS') 'NNLS'|'SAVGOL'|'RAW' - series for p-values
        'fail_method': 'NNLS',  # (default: None) 'NNLS'|'SAVGOL'|'RAW'|None - failure classification (None=use measurement)
        'threshold_mode': 'sd',  # (default: 'auto') 'auto'|'mad'|'sd' - threshold rule (auto=MAD for NNLS, SD for SAVGOL)
        'null_N': 3.0,  # float (default: 3.0) - threshold multiplier
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
            'trials': True,  # bool (default: False) - plot individual trials
            'baseline': True,  # bool (default: False) - show baseline diagnostics
            'residuals': True,  # bool (default: False) - show residual analysis
            'plot_peaks_details': True,  # bool (default: False) - show peak markers and residuals
        },

        # === Jitter Variants (Optional) ===
        # For grid search of temporal shifts in milliseconds
        # Set to None to disable jitter search
        'jitter_variant_ms': None,  # Example: np.arange(-1.0, 1.1, 0.2)
    },

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

        # === Event Model (controlled by USE_TRI_EXPONENTIAL flag) ===
        'event_model': 'iglusnfr_tri' if USE_TRI_EXPONENTIAL else 'iglusnfr',
        # iglusnfr: Bi-exponential (fast 1-10ms, slow from post-train decay)
        # iglusnfr_tri: Tri-exponential (fast 1-10ms, slow 10-25ms, superslow from post-train)

        # === Parameter Bounds (auto-configured based on model) ===
        'parameter_bounds': {
            'tau_decay_fast': (0.001, 0.010),      # 1-10ms fast component
            'tau_decay_slow': (0.010, 0.025),      # 10-25ms (only used for tri-exp)
            # tau_decay_slow (bi-exp) or tau_decay_superslow (tri-exp): auto from post-train
        } if USE_TRI_EXPONENTIAL else {
            'tau_decay_fast': (0.001, 0.010),      # 1-10ms fast component
            # tau_decay_slow: auto from post-train decay (no constraint needed)
        },
        
        # Use all events for averaging
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

        # === Template Variants (Experimental) ===
        # Enable multi-template NNLS: test multiple slow/fast ratios per event
        # NNLS automatically selects best combination based on residuals
        'use_template_variants': True,  # Set to True to enable
        'template_variant_ratios': np.arange(0.0, 1.0, 0.1),  # Slow component fractions to test

        # === Jitter Variants (NEW) ===
        # Enable temporal jitter search in milliseconds
        # Wide range to handle template timing mismatch
        'jitter_variant_ms': np.arange(-5.0, 5.1, 0.5),  # ±5ms in 0.5ms steps
    },
    
    # Minimal preset showing only changed values (others use defaults)
    'default_example': {
        'plot': {'enabled': True}  # changed from default False
    },

    # Example alternative preset (single-exp with fixed tau)
    'single_exp_fixed_8ms': {
        'normalize_dff': True,
        'bleach': True,
        'fit_source': 'global',
        'decay_progression_mode': 'fixed',
        'event_model': 'single_exp',
        'recut_projection': 'robust_mean',
        'recut_oversample': 5,
        'peak_recenter': 5,
        'recut_snippets': True,
        'nnls_weight_mode': 'exponential',
        'nnls_weight_tau_s': 0.02,
        'fit_diagnostic_plot': True,
        'event_model_settings': {'tau_decay': 0.008},  # valid for single_exp
        'plot': {
            'enabled': True,
            'traces': ['raw','savgol','nnls'],
            'show_decay': True,
            'trials': True,
            'baseline': True,
            'residuals': True,
        }
    },

    # Example: Using jitter variants WITHOUT template variants
    # (for models that don't need template ratio search)
    'jitter_only_example': {
        'normalize_dff': True,
        'bleach': True,
        'event_model': 'double_exp',
        'use_template_variants': False,  # Disable template variants
        'jitter_variant_ms': np.arange(-2.0, 2.1, 0.2),  # Only search temporal jitter
        'plot': {'enabled': True}
    },
}

# Choose which preset to use (set to the one you want to visualize)
preset_name = 'iglusnfr_optimized'  # e.g., 'single_exp_fixed_8ms'
options = options_presets[preset_name]

# Extract base filename for display
base = os.path.splitext(os.path.basename(xlsx_path))[0]

res = extract_metrics(
    time, trials,
    train_start=START,   # seconds
    isi=ISI,          # seconds
    n_pulses=10,
    options=options,
    filename=base
)


# Inspect results
amp_avg = res['average'].get('amp_nnls_corr', res['average']['amp_nnls'])
ppr_avg = res['average'].get('ppr_nnls_corr')
if ppr_avg is None:
    a1 = float(amp_avg[0]) if len(amp_avg) else np.nan
    ppr_avg = (amp_avg / a1) if np.isfinite(a1) and abs(a1) > 1e-12 else amp_avg * np.nan
print("Averages (NNLS, peak-baseline):", amp_avg)
print("PPR (NNLS, peak-baseline):", ppr_avg)
print("A1 thresholds per trial:", res['threshold_amp1'])
print("A1 p-values per trial:", res['pval_amp1'])

row = {'measurement': 'NNLS', 'ID': base}
amp = amp_avg
ppr = ppr_avg
for i, v in enumerate(amp, 1):
    row[f'AMP{i}'] = float(v)
for i in range(2, len(ppr) + 1):
    row[f'PPR{i}/1'] = float(ppr[i - 1])

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

# Save/show average plot FIRST if enabled
fig = res.get('figure')
if fig is not None:
    # If the recutter returned snippets, ensure overlay is enabled in the figure
    try:
        fig.savefig(os.path.join(out_dir, "fiber_plot.png"), dpi=150)
    except Exception:
        pass
    # If recut snippets were returned, create and display the average/overlay
    try:
        snips = res.get('recut_snippets')
        t_rel_rec = res.get('recut_t_rel')
        avg_rec = res.get('recut_avg')
        if snips is not None and t_rel_rec is not None and avg_rec is not None:
            from smoothing import build_median_recut_figure
            # If the main figure exists and has axes, plot recut overlay into its first subplot
            try:
                ax_target = None
                if fig is not None:
                    axes = getattr(fig, 'axes', None)
                    if axes:
                        ax_target = axes[0]
                # Build recut figure into existing axes (plot median first so it controls the visual)
                fig2 = build_median_recut_figure(t_rel_rec, avg_rec, snippets=snips, ax=ax_target, plot_median_first=True)
                try:
                    # If fig2 is the same as fig (we plotted into existing axes), save the main fig
                    outpath = os.path.join(out_dir, "fiber_recuts_overlay.png")
                    saved_fig = None
                    if fig is not None and getattr(fig, 'axes', None) and fig.axes[0] is ax_target:
                        fig.savefig(outpath, dpi=150)
                        saved_fig = fig
                    else:
                        fig2.savefig(outpath, dpi=150)
                        saved_fig = fig2
                    print('[demo] saved overlay to', outpath)
                    # Ensure the displayed figure is updated (refresh canvas)
                    try:
                        if saved_fig is not None:
                            saved_fig.canvas.draw()
                            plt.pause(0.001)
                    except Exception:
                        pass
                except Exception as e:
                    print('[demo] failed saving overlay:', e)
            except Exception as e:
                print('[demo] error building overlay:', e)
    except Exception:
        pass
    # Show the average plot FIRST
    try:
        plt.show()
    except Exception:
        pass

# Now save/show per-trial figures (including residual/baseline panels when enabled)
figs_trials = res.get('figures_trials') or []
if figs_trials:
    for i, ftri in enumerate(figs_trials, 1):
        try:
            outp = os.path.join(out_dir, f"{base}_trialfig_{i:02d}.png")
            ftri.tight_layout()
            ftri.savefig(outp, dpi=120)
        except Exception:
            pass
    try:
        plt.show()
    except Exception:
        pass
