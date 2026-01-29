"""
demo_batch_process.py - Batch analysis for iGluSnFR pulse trains

Quick reference for options (see Model_Calibration/event_models.py for details):
  - event_model: 'double_exp', 'iglusnfr', 'single_exp', 'cooperative', etc.
  - decay_progression_mode: 'fixed', 'linear', 'free_monotonic', 'none'
  - fit_source: 'global', 'average', 'individual'
  - nnls_weight_mode: 'uniform', 'linear', 'exponential', 'savgol'
"""

import os, sys, glob, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# =============================================================================
#                         USER CONFIGURATION - EDIT HERE
# =============================================================================

# --- Data paths ---
DATA_ROOT = r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL"
OUT_DIR = os.path.join(DATA_ROOT, "Testout")

# --- Select conditions and files ---
# If CONDITIONS_TO_RUN is empty/None, the script will process all conditions
CONDITIONS_TO_RUN = ["Theo_4_50Hz"]  # e.g., ["Theo_4_50Hz"]
FILE_GLOB = "*.xlsx"

# --- Select analysis preset ---
PRESET_NAME = 'iglusnfr_optimized'  # Options: 'iglusnfr_optimized', 'double_exp', 'single_exp_fixed_8ms'

# --- Manual overrides (set to None to use lookup tables) ---
OVERRIDE_ISI = None       # e.g., 0.02 for 50Hz, 0.05 for 20Hz
OVERRIDE_BASELINE = None  # e.g., 0.498 or 0.998
OVERRIDE_N_PULSES = None  # e.g., 10

# --- Plot output ---
SAVE_PLOTS = True
SHOW_PLOTS = False

# --- Parallel batch processing ---
PARALLEL_FILES = True
FILE_WORKERS = max(1, (os.cpu_count() or 1))

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

DEFAULT_ISI = 0.05       # 20Hz
DEFAULT_BASELINE = 0.998
DEFAULT_N_PULSES = 10

ISI_BY_CONDITION = {
    "Theo_1_5_50Hz": 0.02,
    "Theo_2_5_50Hz": 0.02,
    "Theo_4_50Hz": 0.02,
}

BASELINE_BY_CONDITION = {
    "Stability_Before_05": 0.498,
    "Stability_After_05": 0.498,
    "Theo_4Ca": 0.498,
    "Theo_1_5Ca": 0.498,
    "WT_Theo": 0.498,
    "Theo_1_5_50Hz": 0.498,
    "Theo_2_5_50Hz": 0.498,
    "Theo_4_50Hz": 0.498,
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
                'tau_superslow': (0.035, 0.150),                            # superslow decay bounds (s) - only for tri-exponential
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
            'amplitude_floor_to_noise': True,                               # Floor amplitudes to noise level before computing PPR; defined as 1*std of baseline
            
            # --- NNLS Fitting ---
            'nnls_weight_mode': 'savgol',                                   # 'uniform', 'linear', 'exponential', 'savgol' ; weighting scheme for NNLS fitting
            'nnls_weight_tau_s': None,                                      # Time constant for exponential weighting (s) ; only used if nnls_weight_mode is 'exponential'
            'fit_diagnostic_plot': False,                                   # Whether to generate fit diagnostic plots
            'huber_delta': 2.5,                                             # Huber loss delta for robust fitting (in std units); set to None to disable robust fitting
            'irls_iters': 20,                                               # Number of IRLS iterations for robust fitting ; only used if huber_delta is set
            
            # --- Time Windows (ISI-aware) ---
            'pre_zoom_s': pre_zoom_s,                                       # Pre-event snippet duration (s); controls how much data before each event is shown ; does not affect fitting
            'post_zoom_s': post_zoom_s,                                     # Post-event snippet duration (s); controls how much data after each event is shown ; does not affect fitting
            'f0_window_s': 1.0,                                             # F0 baseline window duration (s) ; controls how baseline F0 is computed for dF/F0 normalization
            
            # --- Peak Detection (ISI-aware) ---
            'peak_window_ms': peak_window_ms,                               # Peak detection window duration (ms) ; controls how peaks are identified within each event
            'peak_avg_points': 1,                                           # Number of points to average around peak for amplitude measurement
            'pre_peak_ms': 1.0,                                             # Pre-peak baseline window (ms) ; controls how local baseline before each peak is computed ;
            
            # --- Thresholding ---
            'measurement': 'NNLS',                                          # 'NNLS', 'AMP1' ; measurement used for thresholding the first event (to estimate failures)
            'fail_method': 'SAVGOL',                                        # 'SAVGOL', 'STD' ; method for estimating noise level for thresholding
            'threshold_mode': 'auto',                                       # 'auto', 'fixed' ; whether to use automatic or fixed thresholding; auto means threshold is computed from estimated noise; fixed means user provides threshold value
            'null_N': 1.0,                                                  # Multiplier for null distribution to set threshold ; only used if threshold_mode is 'auto'
            'null_sim_max_points': 1000,                                    # Max points for null distribution simulation
            'null_min_post_zoom_s': 0.05,                                   # Minimum post-zoom duration (s) to use for null distribution simulation
            
            # --- Bleach Correction ---
            'bleach_huber_delta': 3.0,                                      # Huber loss delta for bleach fitting (in std units); set to None to disable robust fitting
            'bleach_tau_range_factor': (0.25, 4.0),                         # Range factor for bleach tau fitting ; multiplied by initial estimate to get min and max bounds
            'bleach_n_tau': 25,                                             # Number of tau candidates for bleach fitting
            
            # --- Plotting ---
            'plot': {
                'enabled': True,                                            # Master plot enable/disable
                'traces': ['raw', 'nnls'],                                  # 'raw', 'bleach_corrected', 'nnls', 'nnls_corr' ; which traces to plot
                'figsize': (10, 6),                                         # Figure size
                'show_decay': True,                                         # Show decay fits on average plot
                'show_onsets': True,                                        # Show detected onsets on recut snippets
                'trials': False,                                            # Whether to generate per-trial figures
                'baseline': False,                                          # Whether to show baseline F0 levels on traces
                'residuals': True,                                          # Whether to show residuals on average plot
                'nnls_residual': False,                                     # Whether to show NNLS residuals on average plot
                'nnls_n_minus_1': False,                                    # Whether to show NNLS n-1 fit on average plot
                'plot_peaks_details': True,                                 # Show peak detection details on traces
                'param_evolution': True,                                    # Show parameter evolution across events
            },
            
            # --- Template Variants ---
            'use_template_variants': True,                                  # Enable variant testing, where we try multiple tau combinations
            'template_variant_ratios': np.linspace(0.0, 1.0, 10),           # bi-exp and tri-exp
            'template_variant_superslow_fracs': np.linspace(0.0, 1.0, 11),  # tri-exp only
            'superslow_min_ratio': 1.0,                                     # Minimum ratio between slow and superslow taus for tri-exp variants
            'allow_tau_slow_override': False,                                # If True, allow tau_slow = tau_superslow in tri-exp models if it improves fit
            'force_tau_slow_override': False,                                # If True, force tau_slow = tau_superslow in tri-exp models ; unlike allow_tau_slow_override, this enforces the equality rather than just allowing it
            
            # --- Jitter Variants ---
            'jitter_variant_ms': np.linspace(-1.0, 1.0, 5), 
        },
    }


# =============================================================================
#                              INTERNAL SETUP
# =============================================================================
# (No need to edit below unless debugging)

if not SHOW_PLOTS and "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "MPLBACKEND" in os.environ:
    import matplotlib
    matplotlib.use(os.environ["MPLBACKEND"], force=True)
import matplotlib.pyplot as plt

try:
    _here = os.path.dirname(__file__)
except NameError:
    _here = os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(_here, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics


def _safe_sheet_name(name: str) -> str:
    """Return a workbook-safe Excel sheet name."""
    cleaned = "".join(c for c in name if c not in ":\\/?*[]")
    return (cleaned or "Sheet")[:31]


def _iter_xlsx_files(folder: str, pattern: str) -> list[str]:
    return sorted(glob.glob(os.path.join(folder, pattern)))

def _save_figure(fig, outpath: str, *, dpi=None, label: str = "figure") -> bool:
    try:
        fig.savefig(outpath, dpi=dpi)
    except Exception as e:
        print(f"[save] Failed {label}: {outpath} ({e})")
        return False
    try:
        if not os.path.isfile(outpath):
            print(f"[save] Missing {label}: {outpath}")
            return False
        if os.path.getsize(outpath) <= 0:
            print(f"[save] Empty {label}: {outpath}")
            return False
    except Exception as e:
        print(f"[save] Could not validate {label}: {outpath} ({e})")
        return False
    print(f"[save] {label} -> {outpath}")
    return True

def _process_one_file(task: tuple[str, str], *, show_plots: bool) -> dict:
    condition, xlsx_path = task
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    except Exception as e:
        return {'error': f"[skip] Failed to read Excel: {xlsx_path} -> {e}"}

    _time = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
    _trials = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    valid = np.isfinite(_time)
    time = _time[valid]
    trials = _trials[valid, :]

    # --- Resolve parameters (use overrides if set, otherwise lookup) ---
    isi = OVERRIDE_ISI if OVERRIDE_ISI is not None else ISI_BY_CONDITION.get(condition, DEFAULT_ISI)
    start = OVERRIDE_BASELINE if OVERRIDE_BASELINE is not None else BASELINE_BY_CONDITION.get(condition, DEFAULT_BASELINE)
    n_pulses = OVERRIDE_N_PULSES if OVERRIDE_N_PULSES is not None else DEFAULT_N_PULSES

    # --- Compute ISI-dependent parameters ---
    isi_ms = isi * 1000.0
    margin_ms = 2.0  # Fixed margin before next event (ms)
    peak_window_ms = max(5.0, isi_ms - margin_ms)  # Use all data minus 2ms margin
    post_zoom_s = 0.3  # Show ~5 pulses
    pre_zoom_s = 0.20

    # --- Print configuration ---
    print("=" * 60)
    print(f"  Condition:  {condition}")
    print(f"  File:       {os.path.basename(xlsx_path)}")
    print(f"  ISI:        {isi_ms:.0f}ms ({1/isi:.0f}Hz)")
    print(f"  Baseline:   {start:.3f}s")
    print(f"  N pulses:   {n_pulses}")
    print(f"  Preset:     {PRESET_NAME}")
    print(f"  Peak win:   {peak_window_ms:.1f}ms | Post zoom: {post_zoom_s:.3f}s")
    print("=" * 60)

    # --- Build presets and select ---
    options_presets = _build_options_presets(peak_window_ms, pre_zoom_s, post_zoom_s)
    options = options_presets[PRESET_NAME]

    base = os.path.splitext(os.path.basename(xlsx_path))[0]
    res = extract_metrics(
        time, trials,
        train_start=start,
        isi=isi,
        n_pulses=n_pulses,
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
    row = {'measurement': 'NNLS', 'ID': base, 'condition': condition}
    for i, v in enumerate(amp_avg, 1):
        row[f'AMP{i}'] = float(v)
    for i in range(2, len(ppr_avg) + 1):
        row[f'PPR{i}/1'] = float(ppr_avg[i - 1])

    # --- Per-trial failure counts ---
    per_trial_rows = []
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
                'condition': condition,
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

    # =============================================================================
    #                              PLOTTING
    # =============================================================================

    condition_out_dir = os.path.join(OUT_DIR, condition)
    os.makedirs(condition_out_dir, exist_ok=True)

    fig = res.get('figure')
    if fig is None:
        if SAVE_PLOTS:
            print(f"[warn] No main figure returned for {base}")
    else:
        if SAVE_PLOTS:
            _save_figure(fig, os.path.join(condition_out_dir, f"{base}_plot.png"), dpi=150, label="plot")

        # Overlay recut snippets if available
        snips = res.get('recut_snippets')
        t_rel_rec = res.get('recut_t_rel')
        avg_rec = res.get('recut_avg')
        if snips is not None and t_rel_rec is not None and avg_rec is not None and SAVE_PLOTS:
            try:
                from smoothing import build_median_recut_figure
                ax_target = fig.axes[0] if (fig is not None and fig.axes) else None
                fig2 = build_median_recut_figure(t_rel_rec, avg_rec, snippets=snips, ax=ax_target, plot_median_first=True)
                outpath = os.path.join(condition_out_dir, f"{base}_recuts_overlay.png")
                target_fig = fig if ax_target is not None else fig2
                if target_fig is not None:
                    _save_figure(target_fig, outpath, dpi=150, label="recut overlay")
            except Exception as e:
                print(f"[demo] error building overlay: {e}")

        if show_plots:
            try:
                plt.show()
            except Exception:
                pass

    # Per-trial figures
    figs_trials = res.get('figures_trials') or []
    for i, ftri in enumerate(figs_trials, 1):
        if SAVE_PLOTS:
            try:
                outp = os.path.join(condition_out_dir, f"{base}_trialfig_{i:02d}.png")
                ftri.tight_layout()
                _save_figure(ftri, outp, dpi=120, label=f"trial {i:02d}")
            except Exception:
                pass
    if figs_trials and show_plots:
        try:
            plt.show()
        except Exception:
            pass

    # Parameter evolution figure
    fig_param = res.get('figure_param_evolution')
    if fig_param is not None and SAVE_PLOTS:
        try:
            _save_figure(fig_param, os.path.join(condition_out_dir, f"{base}_param_evolution.png"), dpi=150, label="param evolution")
            if show_plots:
                plt.show()
        except Exception as e:
            print(f"[demo] Error saving param evolution figure: {e}")
    if not show_plots:
        try:
            plt.close('all')
        except Exception:
            pass

    return {
        'row': row,
        'per_trial_rows': per_trial_rows,
        'max_pulses': len(amp_avg),
    }


# =============================================================================
#                              RUN ANALYSIS
# =============================================================================

def run_batch():
    conditions = CONDITIONS_TO_RUN or ALL_CONDITIONS
    os.makedirs(OUT_DIR, exist_ok=True)

    tasks = []
    for condition in conditions:
        in_dir = os.path.join(DATA_ROOT, condition)
        if not os.path.isdir(in_dir):
            print(f"[skip] Missing folder: {in_dir}")
            continue
        for xlsx_path in _iter_xlsx_files(in_dir, FILE_GLOB):
            tasks.append((condition, xlsx_path))

    summary_rows = []
    per_trial_rows = []
    max_pulses_seen = 0

    use_parallel = bool(PARALLEL_FILES) and len(tasks) > 1
    if use_parallel and SHOW_PLOTS:
        print("[warn] SHOW_PLOTS=True disables parallel file processing. Falling back to sequential.")
        use_parallel = False

    if use_parallel:
        max_workers = min(FILE_WORKERS, len(tasks))
        print(f"[info] Parallel file processing: {max_workers} worker(s) for {len(tasks)} file(s).")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_one_file, task, show_plots=False) for task in tasks]
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"[skip] Worker failure: {e}")
                    continue
                if result.get('error'):
                    print(result['error'])
                    continue
                summary_rows.append(result['row'])
                per_trial_rows.extend(result['per_trial_rows'])
                max_pulses_seen = max(max_pulses_seen, result['max_pulses'])
    else:
        for task in tasks:
            result = _process_one_file(task, show_plots=SHOW_PLOTS)
            if result.get('error'):
                print(result['error'])
                continue
            summary_rows.append(result['row'])
            per_trial_rows.extend(result['per_trial_rows'])
            max_pulses_seen = max(max_pulses_seen, result['max_pulses'])

    # =============================================================================
    #                              SUMMARY OUTPUT
    # =============================================================================

    if summary_rows:
        df_rows = pd.DataFrame(summary_rows)
        ordered = [f'AMP{i}' for i in range(1, max_pulses_seen + 1)] \
            + [f'PPR{i}/1' for i in range(2, max_pulses_seen + 1)] \
            + [f'%Fail{i}' for i in range(1, 4)]
        for col in ['condition', 'measurement', 'ID', *ordered]:
            if col not in df_rows.columns:
                df_rows[col] = np.nan
        df_rows = df_rows[['condition', 'ID', *ordered, 'measurement']]

        csv_out = os.path.join(OUT_DIR, "summary.csv")
        df_rows.to_csv(csv_out, index=False)
        xl_out = os.path.splitext(csv_out)[0] + ".xlsx"
        with pd.ExcelWriter(xl_out) as writer:
            df_rows.to_excel(writer, sheet_name="All", index=False)
            for condition in sorted(set(df_rows['condition'])):
                df_rows[df_rows['condition'] == condition].to_excel(
                    writer, sheet_name=_safe_sheet_name(condition), index=False
                )

        if per_trial_rows:
            pd.DataFrame(per_trial_rows).to_excel(os.path.splitext(xl_out)[0] + "_trials.xlsx", index=False)

        print(f"[export] Saved summary to: {csv_out}")
        print(f"[export] Saved summary workbook to: {xl_out}")
        if per_trial_rows:
            print(f"[export] Saved per-trial file to: {os.path.splitext(xl_out)[0] + '_trials.xlsx'}")
    else:
        print("[export] No files processed; no summary written.")


if __name__ == "__main__":
    run_batch()
