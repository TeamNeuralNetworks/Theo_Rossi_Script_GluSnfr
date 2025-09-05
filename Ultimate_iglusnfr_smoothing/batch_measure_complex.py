"""Batch iGluSnFR train analysis

Key tasks:
    * Baseline & bleaching correction
    * Kinetics estimation (rise τr / per-pulse decay τd)
    * Robust NNLS amplitude + micro-shift fitting (optionally segmented)
    * Deterministic null distribution & failure thresholds
    * Per-trial & average plotting, export to Excel

Configuration is consolidated below in themed blocks; adjust as needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import os, glob, re, time, warnings, argparse
from typing import Optional, Dict
from matplotlib import animation
from utils.smoothing import (
    progress_print,
    fill_nans_timewise,
    sg_smooth,
    iglusnfr_kernel,
    make_design_with_shifts,
    time_zoom_mask,
    windowed_max,
    pick_peak_on_series,
)

# Optional GUI support (tkinter is standard library)
try:
    import tkinter as tk
except Exception:
    tk = None


def normalize_amplitudes(amp: np.ndarray, tiny: float = 1e-12):
    """Return amplitudes normalized to first pulse (legacy train-mean option removed)."""
    amp = np.array(amp, dtype=float)
    if amp.size == 0:
        return amp
    denom = amp[0]
    if not np.isfinite(denom) or abs(denom) < tiny:
        return amp * np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        return amp / denom

###############################
#  A. CORE DATA / EXPERIMENT  #
###############################
sheet_index      = 0          # Excel sheet index
train_start_s    = 1.0        # Default train start (overridden by map or CLI)
isi_s            = 0.05       # Inter-stimulus interval (s)
n_pulses         = 10         # Expected number of pulses

###############################
#  B. FEATURE TOGGLES         #
###############################
ENABLE_FAST_KINETICS          = True   # Estimate kinetics on average trace
ENABLE_CONTINUOUS_SHIFTS      = True   # Per-pulse micro-shift optimization
ENABLE_ROBUST_FITTING         = True   # IRLS Huber weighting for NNLS
ENABLE_PER_TRIAL_PLOTS        = True   # Generate individual trial plots
ENABLE_AVERAGE_PLOTS          = True   # Generate average trace plot
ENABLE_SEGMENTED_NO_OVERLAP   = True   # Backward segmented fitting (truncate at next stim)
SHOW_PROGRESS                 = True  # Verbose progress lines
SHOW_PLOTS_DURING_BATCH       = True   # Display figures interactively
SAVE_PLOTS                    = False  # Save plots to disk
KEEP_FIGS_OPEN_ON_FINISH      = True
USE_GUI                       = False  # Optional Tk navigation GUI
RUN_BATCH_EXPORT              = True

# Kinetics search grids (ms); broadened to better capture long tails
KIN_TAUR_GRID_MS    = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
# Include shorter decays to avoid overly slow tails
KIN_TAUD0_GRID_MS   = [1.6, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0]
# Nonnegative slopes only (later pulses same or slower)
KIN_SLOPE_GRID_MS   = [0.0, 0.25, 0.5, 1.0, 2.0]  # per pulse (nonnegative)

# Constraint: keep τd non-decreasing across pulses (off by default to allow faster end-of-train)
ENFORCE_NONDECREASING_TAUD = True

###############################
#  C. PLOTTING / VISUAL       #
###############################
PLOT_MODE             = 'replace'  # 'replace' | 'keep' | 'none'
PAUSE_PLOTS_DURING_BATCH = 0.1
PLOTS_SUBDIR          = 'plots'
sg_window             = 9          # Savitzky–Golay window
sg_poly               = 2          # Savitzky–Golay polynomial order
pre_zoom              = 0.15       # Seconds before first pulse to show / fit
post_zoom             = 0.60       # Seconds after last pulse to show / fit
peak_win_ms           = 25.0       # Local window to search for peak (ms)
avg_N_points          = 5          # Local averaging points around peak center
peak_search_pre_ms    = 0.0        # No pre-stimulus lookback (enforce causal peak >= stim time)
peak_search_post_ms   = 10.0        # Post-stimulus peak search span (ms)
SHOW_NULL_FITS_ON_MAIN    = True   # If True, overlay baseline null fits directly on main per-trial trace plot (extends window)
SHUFFLE_BASELINE_BOOTSTRAP = False  # When True, null distribution uses random bootstrap windows (1000 draws) instead of deterministic scan
# Threshold multiplier (N):
#  - For SAVGOL nulls: threshold = mean(null) + N * std(null)
#  - For NNLS   nulls: threshold = median(null) + N * (1.4826 * MAD(null))
# Set N=2.0 to emulate a classic "2 SD" rule for SG and a robust
#   "2 MAD-equiv" rule for NNLS.
NULL_FAIL_THRESHOLD_PARAM = 1.0

# Individual trace visibility toggles (cannot hide a trace if its data were computed)
SHOW_TRACE_RAW    = False
SHOW_TRACE_SAVGOL = False
SHOW_TRACE_NNLS   = False

# Measurement role split:
# - AMP_MEASUREMENT_METHOD: which method provides reported amplitudes in export
# - FAILURE_MEASUREMENT_METHOD: which method generates null + thresholds for failure status
# Valid values: 'NNLS', 'SAVGOL', 'RAW' (failure method typically 'NNLS' or 'SAVGOL')
AMP_MEASUREMENT_METHOD = "NNLS"
FAILURE_MEASUREMENT_METHOD = "NNLS"

###############################
#  D. ROBUST NNLS / SHIFTS    #
###############################
ROBUST_LOSS   = True
HUBER_DELTA   = 5.5            # Huber delta for IRLS
IRLS_ITERS    = 6
DELTA_MAX_MS  = 2.0            # Max micro-shift magnitude (ms)
DELTA_STEP_MS = 0.25           # Shift resolution (ms)
SHIFT_MIN_MS  = 0.05           # Minimal positive shift guard
SHIFT_MODE    = "continuous" if ENABLE_CONTINUOUS_SHIFTS else "off"

###############################
#  E. NULL / BASELINE WINDOWS #
###############################
NULL_USE_SIMPLE_RULE       = True   # Simplified null sampling rule flag (always True)
F0_WINDOW_S                = 0.4    # Length of F0 & null window
NULL_SIM_MAX_POINTS        = 1000   # Max simulated null points
NULL_MIN_POST_ZOOM_S       = 0.05   # Minimum post window allowed near boundary

###############################
#  F. PEAK SEARCH DERIVED     #
###############################
PEAK_SEARCH_POST_S = peak_search_post_ms / 1000.0

###############################
#  G. KINETICS CONSTRAINTS    #
###############################
USE_LINEAR_TAUD   = True
SLOPE_BOUNDS      = (0.0, 0.015)    # Bounds on τd slope per pulse (s)
TAUD_MIN_MARGIN   = 0.0002          # Min separation guard (s); allow slightly faster decays
FORCE_TAUD_MS     = None            # Force all τd if not None (debug)

###############################
#  H. BLEACH CORRECTION       #
###############################
ENABLE_BLEACH_CORRECTION    = True
BLEACH_MAX_ITER             = 4
BLEACH_TAU_GRID_FACTORS     = (0.25, 4.0)
BLEACH_N_TAU                = 25
BLEACH_HUBER_DELTA          = 3.0
SHOW_BLEACH_PLOTS           = False
BLEACH_INTERRUPT            = False

###############################
#  I. BATCH / EXPORT          #
###############################
BATCH_EXPORT_DIR          = r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL"
DEFAULT_BATCH_OUTPUT_FILE = r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\ppr_NNLS.xlsx"
DEFAULT_BATCH_INPUT_DIRS  = [ r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Stability_After",
                              #r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Stability_After_05",
                              #r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Stability_Before",
                              #r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Stability_Before_05",
                              #r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\SynII",
                              #r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Theo_1_5Ca",
                              #r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Theo_4Ca",
                              #r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\WT_Anthime",
                              #r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\WT_Theo",
                              #r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\WT_Theo_1scd"
]
BATCH_FILE_LIMIT        = 4     # Set small int for quick tests during development (e.g., 3)

###############################
#  J. TRAIN START OVERRIDES   #
###############################
TRAIN_START_OVERRIDE_MAP: Dict[str, float] = { r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Stability_After_05": 0.5,
                                               r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Stability_Before_05": 0.5,
                                               r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Theo_1_5Ca": 0.5,
                                               r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Theo_4Ca": 0.5,
                                               r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\WT_Theo": 0.5
}

###############################
#  K. MISC / INTERNAL         #
###############################
F0_EPS = 1e-12  # small guard for F0 denominator

###############################
#  M. DOCUMENTATION (NULL RULE)
###############################
# 1. Baseline correction over time < train_start_s.
# 2. F0 computed over [train_start_s - F0_WINDOW_S, train_start_s).
# 3. Null (failure) amplitudes use same window with simulated t_sim spanning
#       [train_start_s - F0_WINDOW_S, train_start_s - PEAK_SEARCH_POST_S].
# 4. Each null amplitude is max over [t_sim, t_sim + PEAK_SEARCH_POST_S].
# 5. Directory overrides shift entire schedule.
# 6. Deterministic thinning limits to NULL_SIM_MAX_POINTS.

def compute_null_sim_times_simple(time: np.ndarray, train_start: float,
                                  window_s: float = F0_WINDOW_S,
                                  post_s: float = PEAK_SEARCH_POST_S,
                                  max_points: int = NULL_SIM_MAX_POINTS) -> np.ndarray:
    """Return simulated null stimulus times under the simplified rule.

    time: full time axis (seconds)
    train_start: real train start time (seconds)
    window_s: duration of pre-train window used for F0 & null sampling
    post_s: peak search post duration; ensures windows do not cross real train start
    max_points: optional cap (uniform down-sample if exceeded)
    """
    t0 = train_start - window_s
    t1 = train_start - post_s  # inclusive last simulated time
    if t1 < t0:
        return np.array([], dtype=float)
    mask = (time >= t0) & (time <= t1)
    sims = time[mask]
    if sims.size == 0:
        return sims
    if sims.size > max_points:
        idx = np.linspace(0, sims.size - 1, max_points).round().astype(int)
        sims = sims[idx]
    return sims

def resolve_train_start(xlsx_path: str, explicit: Optional[float] = None) -> float:
    """Resolve train start with longest path match precedence.

    Precedence:
      1. explicit not None
      2. Longest matching key (case-insensitive, normalized path) such that parent dir
         == key, startswith key (descendant), or endswith key (for legacy partial keys)
      3. global default train_start_s.
    Emits a progress line describing the decision.
    """
    if explicit is not None:
        progress_print(f"  Using explicit train_start override: {explicit}s")
        return float(explicit)
    try:
        parent = os.path.normcase(os.path.normpath(os.path.dirname(xlsx_path)))
        best_key=None; best_val=None; best_len=-1
        for k,v in TRAIN_START_OVERRIDE_MAP.items():
            nk = os.path.normcase(os.path.normpath(k))
            if parent == nk or parent.startswith(nk + os.sep) or parent.endswith(nk):
                L=len(nk)
                if L>best_len:
                    best_len=L; best_key=k; best_val=v
        if best_key is not None:
            progress_print(f"  train_start override matched '{best_key}': {best_val}s")
            return float(best_val)
    except Exception:
        pass
    progress_print(f"  Using default train_start: {train_start_s}s")
    return float(train_start_s)

# ==========================
# Amplitude normalization mode
# ==========================
# When True, all downstream analyses operate on ΔF/F0 (per-trial median F0) rather than
# raw baseline-subtracted fluorescence (ΔF). This affects:
#  - Stored Y_all signals
#  - Null amplitude sampling (works on normalized signal automatically)
#  - Exported AMP* values (now represent ΔF/F0)
#  - Plot y-axis label
USE_DF_OVER_F0 = True
F0_EPS = 1e-9  # safety to avoid divide-by-zero

if SHOW_PLOTS_DURING_BATCH and not SAVE_PLOTS:
    try: plt.ion()
    except Exception: pass

# Keep track of the last shown figure so we can close it when a new one is displayed.
_last_shown_fig = None

def _show_now(fig, pause=0.05):
    """Draw and pause on `fig`, closing the previously shown diagnostic figure first.

    This keeps only one diagnostic figure (e.g. baseline/bleach plot) visible at a time.
    """
    global _last_shown_fig
    if PLOT_MODE == 'none':
        try: plt.close(fig)
        except Exception: pass
        return
    try:
        if PLOT_MODE == 'replace':
            try:
                if _last_shown_fig is not None and _last_shown_fig is not fig:
                    try:
                        plt.close(_last_shown_fig)
                    except Exception:
                        pass
            except Exception:
                pass
        # if 'keep' do not close previous
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(pause)
        _last_shown_fig = fig
    except Exception:
        pass


def empirical_threshold(null_amps, alpha=0.05):
    """
    Return (thr_(1-alpha), p_value_func) where p_value_func(x)=Pr(null >= x).
    """
    if null_amps is None or null_amps.size < 10:
        return np.nan, lambda x: np.nan
    thr = float(np.quantile(null_amps, 1 - alpha))
    def pval(x):
        a = null_amps
        return float((np.sum(a >= x) + 1) / (a.size + 1))
    return thr, pval

# ---------------------------------------------------------------------------
# Consistent single-pulse estimator and null sampler (matches amp1 pipeline)
# ---------------------------------------------------------------------------
def _fit_single_pulse_amp_consistent(
    y, time, st, tau_r, tau_d, *,
    pre_zoom=0.15, post_zoom=0.60,
    robust=True, huber_delta=5.5, irls_iters=6,
    allow_shift=True, delta_max_s=0.001, delta_step_s=0.00025
):
    """Estimate single-pulse amplitude at time 'st' with optional micro-shift.

    Returns (a_hat, best_delta)."""
    zmask = (time >= (st - pre_zoom)) & (time <= (st + post_zoom))
    if not np.any(zmask):
        return 0.0, max(SHIFT_MIN_MS/1000.0, 0.0)

    # Local segment
    y_seg = y[zmask]

    def _nnls_irls_singlecol(y_orig, k_col, robust_flag, delta_h, iters):
        a = max(0.0, nnls(k_col[:, None], y_orig)[0][0])
        if not robust_flag:
            return a
        for _ in range(max(1, iters)):
            r = y_orig - a * k_col
            absr = np.abs(r)
            w = np.where(absr <= delta_h, 1.0, delta_h / np.maximum(absr, 1e-12))
            Wsqrt = np.sqrt(w)
            kw = k_col * Wsqrt
            yw = y_orig * Wsqrt
            a = max(0.0, nnls(kw[:, None], yw)[0][0])
        return a

    best_a, best_d = 0.0, max(SHIFT_MIN_MS/1000.0, 0.0)
    shift_min = best_d
    if allow_shift:
        shifts = np.arange(shift_min, delta_max_s + 1e-12, delta_step_s)
    else:
        shifts = np.array([shift_min])
    for d in shifts:
        k_full = iglusnfr_kernel(time - (st + d), tau_r, tau_d)
        if not np.any(k_full):
            continue
        # Direct kernel segment
        k_wz = k_full[zmask]
        if k_wz.size < 3 or np.all(k_wz == 0):
            continue
        a_loc = _nnls_irls_singlecol(
            y_seg, k_wz,
            robust and ENABLE_ROBUST_FITTING,
            huber_delta, irls_iters
        )
        if a_loc > best_a:
            best_a, best_d = a_loc, d
    return float(best_a), float(best_d)


def sample_null_amplitudes_consistent(
    y, time, baseline_mask, tau_r, tau_d, * ,
    train_start=None, f0_window_s=None,
    pre_zoom=0.15, post_zoom=0.60,
    robust=True, huber_delta=5.5, irls_iters=6,
    allow_shift=True, delta_max_s=0.001, delta_step_s=0.00025,
    n_samples=1000, seed=0, deterministic_step=True,
    shuffle_bootstrap: bool = None
):
    """Null distribution via the SAME single-pulse estimator used for Amp1.

    Changes vs earlier version:
      * Baseline_mask now represents the FULL pre-train baseline (time < train_start).
      * Null sampling is RESTRICTED to the final F0 window immediately preceding
        the (possibly overridden) train start: [train_start - f0_window_s, train_start).

    Strategy:
      1. Identify full baseline interval (baseline_mask True).
      2. Derive null window = intersection of baseline with final f0_window_s before train_start.
      3. Exhaustively (deterministically) scan candidate pseudo-stimulus start times inside the null window
         such that each fitting window [t - pre_zoom, t + post_zoom] lies entirely within the full baseline
         and ends strictly before train_start.
      4. For each candidate, run `_fit_single_pulse_amp_consistent` with identical
         robust / shifting settings as for pulse 1.
      5. Collect fitted amplitudes -> null amplitude samples.

    If train_start is None we attempt to infer it as the first time point after the last
    True in baseline_mask; as a fallback we set it just beyond the last timestamp.
    """
    if shuffle_bootstrap is None:
        shuffle_bootstrap = SHUFFLE_BASELINE_BOOTSTRAP

    # Legacy path removed; always use simplified rule (with optional bootstrap).

    idx = np.flatnonzero(baseline_mask)
    if idx.size < 10:
        return np.array([])
    baseline_start = time[idx[0]]
    baseline_end   = time[idx[-1]]
    if train_start is None:
        # Infer as first time after baseline; if baseline reaches the end, place beyond.
        after_mask = np.flatnonzero(~baseline_mask & (time > baseline_end))
        train_start = time[after_mask[0]] if after_mask.size else (baseline_end + 1e-3)
    if f0_window_s is None:
        f0_window_s = F0_WINDOW_S

    # Null window (F0 window) bounds
    null_window_start = max(baseline_start, train_start - f0_window_s)
    null_window_end   = min(baseline_end, train_start)
    # We no longer require the full (pre_zoom + post_zoom) to fit inside the null window.
    # Instead we allow truncation of post window per start (down to NULL_MIN_POST_ZOOM_S).
    # Candidate starts must allow at least minimal post segment before train start.
    st_min = null_window_start + pre_zoom
    st_max = null_window_end   - NULL_MIN_POST_ZOOM_S
    if st_max <= st_min:
        return np.array([])

    rng = np.random.default_rng(int(seed))
    if shuffle_bootstrap:
        # Bootstrap: draw n_samples random start times uniformly in [st_min, st_max]
        starts = rng.uniform(st_min, st_max, size=int(n_samples))
        starts.sort()  # purely cosmetic ordering
    else:
        # Deterministic exhaustive sampling at native sample spacing (1-point increment)
        cand_mask = (time >= st_min) & (time <= st_max)
        starts_full = time[cand_mask]
        if starts_full.size == 0:
            return np.array([])
        limit = int(min(NULL_SIM_MAX_POINTS, n_samples))
        if starts_full.size > limit:
            idx = np.linspace(0, starts_full.size-1, limit).round().astype(int)
            starts = starts_full[idx]
        else:
            starts = starts_full

    amps = []
    for st in starts:
        # Truncate post window so it does not cross train_start or null_window_end
        avail_post = min(post_zoom, train_start - st - 1e-6, null_window_end - st)
        if avail_post < NULL_MIN_POST_ZOOM_S:
            continue
        a_hat, d_hat = _fit_single_pulse_amp_consistent(
            y, time, st, tau_r, tau_d,
            pre_zoom=pre_zoom, post_zoom=avail_post,
            robust=robust, huber_delta=huber_delta, irls_iters=irls_iters,
            allow_shift=allow_shift, delta_max_s=delta_max_s, delta_step_s=delta_step_s
        )
        t_fit = time[(time >= st - pre_zoom) & (time <= st + avail_post)]
        if t_fit.size:
            k_fit = iglusnfr_kernel(t_fit - (st + d_hat), tau_r, tau_d)
            peak_amp = float(a_hat * np.max(k_fit))
            amps.append(peak_amp)
    return np.asarray(amps, float)
def baseline_threshold_and_pval(null_amps, N: float, mode: str):
    """Return (thr, pval_func) over baseline null amplitudes using a single rule.

    mode:
      - 'sd'  => thr = mean(null)   + N * std(null)
      - 'mad' => thr = median(null) + N * (1.4826 * MAD(null))

    If null_amps empty -> (NaN, lambda -> NaN).
    pval_func(x) = empirical one-sided p-value P(null >= x) with +1 smoothing.
    """
    if null_amps is None or np.size(null_amps) == 0 or not np.isfinite(N):
        return np.nan, (lambda x: np.nan)
    a = np.asarray(null_amps, float)
    with np.errstate(all='ignore'):
        mode_u = (mode or '').strip().lower()
        if mode_u == 'sd':
            mu = float(np.nanmean(a))
            sd = float(np.nanstd(a))
            thr = float(mu + float(N) * sd) if np.isfinite(sd) else np.nan
        elif mode_u == 'mad':
            med = float(np.nanmedian(a))
            mad = float(np.nanmedian(np.abs(a - med)))
            sigma_hat = 1.4826 * mad  # Normal-equiv scale
            thr = float(med + float(N) * sigma_hat) if np.isfinite(sigma_hat) else np.nan
        else:
            return np.nan, (lambda x: np.nan)
    def pval(x):
        return float((np.sum(a >= x) + 1) / (a.size + 1))
    return thr, pval

# Helper: human-readable threshold label based on NULL_FAIL_THRESHOLD_PARAM
def _format_fail_threshold_label(N: float, mode: str) -> str:
    """Return label string matching the active threshold rule.

    Modes:
      * 'sd'  -> 'thr{N}SD'  (e.g., N=2 => 'thr2SD')
      * 'mad' -> 'thr{N}MAD' (N times 1.4826*MAD above median)
    """
    try:
        if not np.isfinite(N):
            return "thr"
        mode_u = (mode or '').lower()
        if mode_u == 'sd':
            return f"thr{N:g}SD"
        if mode_u == 'mad':
            return f"thr{N:g}MAD"
        return "thr"
    except Exception:
        return "thr"

# --------------------------
# Null fit animation helpers
# --------------------------
# --------------------------
# Null fit baseline candidate enumeration (used for overlay)
# --------------------------
def _enumerate_null_candidates(time, baseline_mask, train_start, f0_window_s, pre_zoom, post_zoom):
    idx = np.flatnonzero(baseline_mask)
    if idx.size == 0:
        return []
    baseline_start = time[idx[0]]; baseline_end = time[idx[-1]]
    null_start = max(baseline_start, train_start - f0_window_s)
    null_end   = min(baseline_end, train_start)
    st_min = null_start + pre_zoom
    st_max = null_end - NULL_MIN_POST_ZOOM_S
    if st_max <= st_min:
        return []
    cand_mask = (time >= st_min) & (time <= st_max)
    return list(time[cand_mask])

## Legacy animate_null_fits removed (baseline null fits drawn directly on main trace)

## Removed obsolete plot_null_fits_overlay (baseline null fits now drawn directly on main trace when SHOW_NULL_FITS_ON_MAIN)

# --------------------------
# Robust mono-exponential bleaching correction
# --------------------------
def _fit_monoexp_robust(time, y, mask_fit, n_iter=4, huber_delta=3.0,
                        tau_candidates=None):
    t = np.asarray(time, float)
    y = np.asarray(y, float)
    msk = mask_fit & np.isfinite(y)
    if msk.sum() < 10:
        return np.zeros_like(y), np.nan  # no correction
    t_fit = t[msk]
    y_fit = y[msk]
    if tau_candidates is None:
        dur = t[-1] - t[0]
        tmin = max(1e-6, BLEACH_TAU_GRID_FACTORS[0] * dur)
        tmax = max(tmin*1.01, BLEACH_TAU_GRID_FACTORS[1] * dur)
        tau_candidates = np.geomspace(tmin, tmax, BLEACH_N_TAU)
    best = None
    for tau in tau_candidates:
        try:
            e = np.exp(-t_fit / tau)
            # Design matrix for A + B*e
            X = np.column_stack([np.ones_like(e), e])
            if X.shape[0] < 2:
                continue
            w = np.ones_like(e)
            for _ in range(max(1, n_iter)):
                # Guard against pathological weights
                w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
                Wsqrt = np.sqrt(w)
                Xw = X * Wsqrt[:, None]
                yw = y_fit * Wsqrt
                try:
                    coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
                except Exception:
                    # Fallback simple robust location (flat trend)
                    coef = np.array([np.nan, np.nan])
                A, B = coef
                if not np.isfinite(B) or B < 0.0:
                    B = 0.0
                    try:
                        A = float(np.nanmedian(y_fit))
                    except Exception:
                        A = float(np.nanmean(y_fit)) if np.isfinite(y_fit).any() else 0.0
                y_pred = A + B * e
                r = y_fit - y_pred
                absr = np.abs(r)
                w = np.where(absr <= huber_delta, 1.0, huber_delta / np.maximum(absr, 1e-12))
            rss = np.nansum((y_fit - (A + B*e))**2)
            if best is None or rss < best[0]:
                best = (rss, tau, A, B)
        except Exception as _ex:
            # Continue trying other taus; leave a lightweight trace if verbose
            if SHOW_PROGRESS:
                progress_print(f"[bleach-fit] tau={tau:.4g} failed: {_ex}")
            continue
    if best is None:
        return np.zeros_like(y), np.nan
    _, tau_b, A_b, B_b = best
    trend = A_b + B_b * np.exp(-t / tau_b)
    return trend, float(tau_b)

def apply_bleach_correction(time, y, train_start, stim_times, train_end_margin=0.1):
    """Return bleaching-corrected trace using robust mono-exponential trend.

    The correction uses only time points outside the stimulation train window
    [train_start, train_end] for fitting, where train_end is last stim + margin.
    Correction: y_corr = y - (trend - trend_ref) with trend_ref the median
    pre-train fitted trend, so baseline is flattened without lowering it.
    Safeguards prevent over-correction yielding values far below original.
    """
    y = np.asarray(y, float)
    if y.size != len(time):
        return y
    if stim_times is None or len(stim_times) == 0:
        return y
    # Fit only on pre-stimulation time points (strictly before train start)
    train_end = stim_times[-1] + train_end_margin
    mask_fit = (time < train_start)
    # Require a minimum amount of pre-train data for a stable fit
    if np.sum(mask_fit & np.isfinite(y)) < 10:
        return y
    trend, _ = _fit_monoexp_robust(time, y, mask_fit,
                                   n_iter=BLEACH_MAX_ITER,
                                   huber_delta=BLEACH_HUBER_DELTA)
    # Reference level
    pre_mask = (time < train_start) & np.isfinite(trend)
    if not np.any(pre_mask):
        return y
    trend_ref = np.nanmedian(trend[pre_mask])
    corr = y - (trend - trend_ref)
    # Safeguard: avoid large positive shifts creating negative dips inside train
    train_mask = (time >= train_start) & (time <= train_end)
    min_train = np.nanmin(corr[train_mask]) if np.any(train_mask) else np.nan
    if np.isfinite(min_train) and min_train < -0.05 * max(1.0, np.nanstd(y[pre_mask])):
        # Blend back towards original
        alpha = np.clip(min_train / (-0.05 * max(1.0, np.nanstd(y[pre_mask]))), 0.0, 1.0)
        corr = alpha * corr + (1 - alpha) * y
        # Prevent upward overshoot in the last 100 ms before the train start:
        try:
            last_window_start = train_start - 0.1
            last100_mask = (time >= last_window_start) & (time < train_start) & np.isfinite(corr) & np.isfinite(y)
            if np.any(last100_mask):
                med_last = float(np.nanmedian(y[last100_mask]))
                delta = corr - y
                # Only consider positive deltas (which raise the signal)
                delta_pos = delta[last100_mask]
                y_pos = y[last100_mask]
                pos_local = delta_pos > 0
                if np.any(pos_local):
                    pos_delta = delta_pos[pos_local]
                    allowed = med_last - y_pos[pos_local]
                    # If allowed <= 0 everywhere, disallow upward correction
                    if np.all(allowed <= 0):
                        s = 0.0
                    else:
                        # Compute per-point scaling factors, clamp to [0,1]
                        with np.errstate(divide='ignore', invalid='ignore'):
                            s_vals = allowed / pos_delta
                        s_vals = np.where(np.isfinite(s_vals), s_vals, 0.0)
                        s_vals = np.clip(s_vals, 0.0, 1.0)
                        s = float(np.nanmin(s_vals)) if s_vals.size else 1.0
                    if s < 1.0:
                        delta = delta * s
                        corr = y + delta
        except Exception:
            pass
    if SHOW_BLEACH_PLOTS:
        try:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(time, y, label='raw', alpha=0.5, linewidth=1.0, color='tab:gray')
            ax.plot(time, corr, label='corrected', linewidth=1.2, color='tab:blue')
            # Plot fitted trend last so it appears on top
            ax.plot(time, trend, label='fitted trend', linewidth=2.0, color='tab:red', zorder=10)
            ax.axvspan(train_start, train_end, color='orange', alpha=0.15, label='train window')
            ax.legend(); ax.set_title('Bleach correction (mono-exp)')
            _show_now(fig, pause=0.05)
        except Exception:
            pass
    return corr

# --------------------------
# Parameters (derived)
# --------------------------
# Derived parameters (read-only)
# --------------------------------
# Time axis for analysis (train + inter-stimulus interval)
time_s = train_start_s + isi_s * np.arange(n_pulses + 1)

# Baseline mask (train period)
baseline_mask = None

# Zoom mask for robust fitting (train + first inter-stimulus interval)
zoom_mask = None

## Preprocessing helper removed (was unused)
## Fitting functions

# --------------------------
# Robust NNLS without baseline (optional Δa smoothness removed)
# --------------------------

def nnls_huber(y, X, time, zoom_mask, robust=True, delta=1.5, irls_iters=6):
    """Robust (Huber) NNLS on a zoomed window."""
    if not ENABLE_ROBUST_FITTING:
        robust = False
        irls_iters = 1

    y = np.asarray(y, float)
    X = np.asarray(X, float)
    y_z = y[zoom_mask]
    X_z = X[zoom_mask, :]

    a = np.maximum(0.0, nnls(X_z, y_z)[0]) if X_z.size else np.zeros(X.shape[1], float)

    iters = irls_iters if robust else 1
    for _ in range(max(1, iters)):
        if X_z.size == 0:
            break
        r_z = y_z - X_z @ a
        if robust:
            absr = np.abs(r_z)
            w_full = np.where(absr <= delta, 1.0, delta/np.maximum(absr, 1e-12))
        else:
            w_full = np.ones_like(r_z)
        Wsqrt = np.sqrt(w_full)
        Xw = X_z * Wsqrt[:, None]
        yw = y_z * Wsqrt
        a = np.maximum(0.0, nnls(Xw, yw)[0]) if Xw.size else a

    yhat = X @ a if X.size else np.zeros_like(y)
    return a, yhat

# --------------------------
# Continuous shifts alternating optimizer (NNLS path)
# --------------------------

def fit_amplitudes_and_shifts(y, time, stim_times, tau_r, tau_d_vec, baseline_mask,
                              zoom_mask,
                              delta_max_s=0.001, delta_step_s=0.00025,
                              robust=True, delta=1.5, irls_iters=6,
                              alt_iters=4):
    """Alternating optimization of amplitudes and micro-shifts.

    A more efficient shift evaluation strategy is used here.  Baseline design
    components are precomputed once.  During shift
    searches only the affected column is updated via interpolation on the
    precomputed kernels, and the residual is updated incrementally.
    This avoids rebuilding the full design matrix
    for every candidate shift.
    """

    if not ENABLE_CONTINUOUS_SHIFTS:
        alt_iters = 1
    if not ENABLE_ROBUST_FITTING:
        robust = False
        irls_iters = 1

    m = len(stim_times)
    n = len(time)

    # Precompute unshifted kernels
    base_cols = []
    for p, st in enumerate(stim_times):
        td = float(tau_d_vec[p])
        k = iglusnfr_kernel(time - st, tau_r, td)
        base_cols.append(k)
    base_cols = np.column_stack(base_cols) if base_cols else np.zeros((n, 0))
    y_z = y[zoom_mask]

    deltas = np.zeros(m, float)
    a = np.zeros(m, float)

    for _ in range(alt_iters):
        # Build design matrix from precomputed kernels using interpolation
        if m:
            X = np.column_stack([
                np.interp(time - deltas[p], time, base_cols[:, p],
                          left=0.0, right=0.0)
                for p in range(m)
            ])
        else:
            X = np.zeros((n, 0))

        a, yhat = nnls_huber(
            y, X, time, zoom_mask,
            robust=robust, delta=delta, irls_iters=irls_iters
        )

        if not ENABLE_CONTINUOUS_SHIFTS or SHIFT_MODE != "continuous":
            continue

        # Residual for current shifts
        if m:
            X_full_shifted = np.column_stack([
                np.interp(time - deltas[p], time, base_cols[:, p],
                          left=0.0, right=0.0)
                for p in range(m)
            ])
        else:
            X_full_shifted = np.zeros((n, 0))
        X_shift = X_full_shifted[zoom_mask, :]
        r_loc = y_z - X_shift @ a

        for p in range(m):
            d0 = max(deltas[p], SHIFT_MIN_MS/1000.0)
            grid = np.arange(SHIFT_MIN_MS/1000.0, delta_max_s + 1e-12, delta_step_s)
            # local refinement: include current d0 if off-grid
            if not np.isclose(grid, d0).any():
                grid = np.sort(np.r_[grid, d0])

            xw_p = X_shift[:, p]
            best_val, best_d = np.inf, d0
            best_kw = xw_p.copy()

            for d in grid:
                kw_full = np.interp(time - d, time, base_cols[:, p],
                                    left=0.0, right=0.0)
                kw = kw_full[zoom_mask]
                rw = r_loc + a[p] * xw_p - a[p] * kw
                val = np.dot(rw, rw)
                if val < best_val:
                    best_val, best_d, best_kw = val, d, kw

            # Update current column and residual with the best shift
            r_loc = r_loc + a[p] * xw_p - a[p] * best_kw
            X_shift[:, p] = best_kw
            deltas[p] = best_d

    # Final design and amplitude fit with updated shifts
    if m:
        X = np.column_stack([
            np.interp(time - deltas[p], time, base_cols[:, p],
                      left=0.0, right=0.0)
            for p in range(m)
        ])
    else:
        X = np.zeros((n, 0))

    a, yhat = nnls_huber(
        y, X, time, zoom_mask,
        robust=ROBUST_LOSS, delta=HUBER_DELTA, irls_iters=IRLS_ITERS
    )
    return a, deltas, X, yhat

# --------------------------
# Segmented non-overlap backward fitter (new)
# --------------------------
def fit_amplitudes_no_overlap_backward(
    y, time, stim_times, tau_r, tau_d_vec, *,
    pre_zoom=0.15, post_zoom=0.60,
    robust=True, huber_delta=5.5, irls_iters=6,
    allow_shift=True, delta_max_s=0.001, delta_step_s=0.00025
):
    """Fit pulse amplitudes sequentially from last to first without allowing
    any single-pulse fit window to extend past the NEXT stimulus.

    For pulse p (0-index) with time st_p:
        window = [st_p - pre_zoom, min(st_{p+1}, st_p + post_zoom))
    Last pulse uses st_last + post_zoom for its upper bound.
    Later pulses are subtracted before fitting earlier ones (backward residual fitting).
    Returns (a, deltas, X, yhat) similar to fit_amplitudes_and_shifts.
    """
    n_pulses = len(stim_times)
    a = np.zeros(n_pulses, float)
    deltas = np.zeros(n_pulses, float)
    residual = y.copy()
    # Store per-pulse reconstructed components for optional design reconstruction
    comps = []  # list of (p, comp_vector)
    for p in reversed(range(n_pulses)):
        st = stim_times[p]
        next_st = stim_times[p+1] if p < n_pulses - 1 else None
        # Determine effective post window bound
        if next_st is not None:
            max_end = min(st + post_zoom, next_st)
        else:
            max_end = st + post_zoom
        eff_post = max(0.0, max_end - st)
        if eff_post < 1e-6:
            a[p] = 0.0
            deltas[p] = 0.0
            comps.append((p, np.zeros_like(y)))
            continue
        a_p, d_p = _fit_single_pulse_amp_consistent(
            residual, time, st, tau_r, tau_d_vec[p],
            pre_zoom=pre_zoom, post_zoom=eff_post,
            robust=robust, huber_delta=huber_delta, irls_iters=irls_iters,
            allow_shift=allow_shift, delta_max_s=delta_max_s, delta_step_s=delta_step_s
        )
        a[p] = a_p
        deltas[p] = d_p
        k_full = iglusnfr_kernel(time - (st + d_p), tau_r, tau_d_vec[p])
        comp = a_p * k_full
        residual = residual - comp  # subtract contribution for earlier fits
        comps.append((p, comp))
    # Reconstruct yhat (sum in original order)
    comps_sorted = sorted(comps, key=lambda x: x[0])
    yhat = np.sum([c for _, c in comps_sorted], axis=0)
    # Build design matrix with shifts applied (columns correspond to each pulse kernel)
    X = np.column_stack([iglusnfr_kernel(time - (stim_times[p] + deltas[p]), tau_r, tau_d_vec[p]) for p in range(n_pulses)]) if n_pulses else np.zeros((len(time),0))
    return a, deltas, X, yhat

# --------------------------
# Fast kinetics estimation from average trace
# --------------------------
def estimate_kinetics_from_average(time, y_avg, stim_times, baseline_mask, 
                                   tau_d0_fixed=None, tau_d_end_fixed=None):
    progress_print("Fast kinetics estimation from average trace...")

    # Grids (ms) -> seconds
    tau_r_grid = np.array(KIN_TAUR_GRID_MS, float) / 1000.0
    tau_d0_grid = np.array(KIN_TAUD0_GRID_MS, float) / 1000.0
    slope_grid = np.array(KIN_SLOPE_GRID_MS, float) / 1000.0

    zmask, _, _ = time_zoom_mask(time, stim_times[0], isi_s, n_pulses, pre_zoom, post_zoom)

    def obj_for(tau_r, tau_d_vec):
        try:
            X = make_design_with_shifts(time, stim_times, tau_r, tau_d_vec, np.zeros(n_pulses))
            a = np.maximum(0.0, nnls(X[zmask, :], y_avg[zmask])[0])
            r = y_avg - X @ a
            return np.dot(r[zmask], r[zmask]) / max(1, zmask.sum())
        except Exception:
            return 1e9

    best_val = np.inf
    best_params = (0.002, 0.006, 0.002)

    if tau_d0_fixed is not None and tau_d_end_fixed is not None and np.isfinite(tau_d0_fixed) and np.isfinite(tau_d_end_fixed):
        # Constrain tau_d progression to hit the requested endpoints exactly
        # Filter tau_r grid to be strictly below both taus (with margin)
        max_tau_r = min(tau_d0_fixed, tau_d_end_fixed) - TAUD_MIN_MARGIN
        tau_r_grid2 = tau_r_grid[tau_r_grid < max_tau_r]
        if tau_r_grid2.size == 0:
            # fallback: pick a single tau_r slightly below min endpoint
            tau_r_grid2 = np.array([max(1e-4, max_tau_r*0.5)])
        tau_d_vec_fixed = np.linspace(float(tau_d0_fixed), float(tau_d_end_fixed), n_pulses)
        total = len(tau_r_grid2)
        progress_print(f"Fast search with fixed τd endpoints (n={total} tau_r)")
        for tau_r in tau_r_grid2:
            # Enforce physical constraint τd > τr + margin; penalize invalid
            if np.any(tau_d_vec_fixed <= tau_r + TAUD_MIN_MARGIN):
                val = 1e9
            else:
                val = obj_for(tau_r, tau_d_vec_fixed)
            if val < best_val:
                best_val = val
                # store tau_d0_fit and implied slope
                slope = (tau_d_vec_fixed[-1] - tau_d_vec_fixed[0]) / max(1, (n_pulses-1))
                best_params = (tau_r, tau_d_vec_fixed[0], slope)
        tau_r_fit, tau_d0_fit, slope_fit = best_params
        tau_d_vec = np.linspace(tau_d0_fit, tau_d0_fit + slope_fit*(n_pulses-1), n_pulses)
        progress_print(f"Fast kinetics result (fixed τd ends): τr={tau_r_fit*1e3:.1f}ms, τd0={tau_d0_fit*1e3:.1f}ms, τdN={tau_d_vec[-1]*1e3:.1f}ms")
        progress_print(f"Tau_d progression: {[f'{td*1e3:.1f}ms' for td in tau_d_vec]}")
        return tau_r_fit, tau_d0_fit, slope_fit, tau_d_vec

    # Unconstrained grid (original behaviour)
    total_combinations = len(tau_r_grid) * len(tau_d0_grid) * len(slope_grid)
    progress_print(f"Fast grid search: {total_combinations} combinations")

    def fast_obj(tau_r, tau_d0, slope):
        tau_d_vec = np.array([max(tau_r + TAUD_MIN_MARGIN, tau_d0 + slope*(p-1))
                             for p in range(1, n_pulses+1)])
        return obj_for(tau_r, tau_d_vec)

    for tau_r in tau_r_grid:
        for tau_d0 in tau_d0_grid:
            for slope in slope_grid:
                val = fast_obj(tau_r, tau_d0, slope)
                if val < best_val:
                    best_val = val
                    best_params = (tau_r, tau_d0, slope)

    tau_r_fit, tau_d0_fit, slope_fit = best_params
    tau_d_vec = np.array([max(tau_r_fit + TAUD_MIN_MARGIN, tau_d0_fit + slope_fit*(p-1))
                         for p in range(1, n_pulses+1)])

    progress_print(f"Fast kinetics result: τr={tau_r_fit*1e3:.1f}ms, τd0={tau_d0_fit*1e3:.1f}ms, slope={slope_fit*1e3:.3f}ms/pulse")
    progress_print(f"Tau_d progression: {[f'{td*1e3:.1f}ms' for td in tau_d_vec]}")

    return tau_r_fit, tau_d0_fit, slope_fit, tau_d_vec

def fit_kinetics_pooled(time, Y_all, stim_times, baseline_mask, 
                        tau_d0_fixed=None, tau_d_end_fixed=None):
    if ENABLE_FAST_KINETICS:
        y_avg = np.nanmean(Y_all, axis=1)
        return estimate_kinetics_from_average(time, y_avg, stim_times, baseline_mask,
                                              tau_d0_fixed=tau_d0_fixed, tau_d_end_fixed=tau_d_end_fixed)
    progress_print("Fast kinetics disabled - using default values")
    tau_r_default = 0.004
    tau_d0_default = 0.015
    slope_default = 0.0
    tau_d_vec = np.full(n_pulses, tau_d0_default)
    return tau_r_default, tau_d0_default, slope_default, tau_d_vec

# --------------------------
# Plotting
# --------------------------
def build_os_reconstruction(time, stim_times, tau_r_fit, tau_d_vec, deltas, a_vec, oversample_factor=10):
    dt = float(np.median(np.diff(time)))
    zmask, z0, z1 = time_zoom_mask(time, stim_times[0], isi_s, n_pulses, pre_zoom, post_zoom)
    t_zoom = time[zmask]
    step = dt / int(max(1, oversample_factor))
    t_os = np.arange(t_zoom[0], t_zoom[-1] + 1e-12, step)
    y_os = np.zeros_like(t_os)
    for p, st in enumerate(stim_times):
        td = float(tau_d_vec[p]); sh = float(deltas[p])
        y_os += a_vec[p] * iglusnfr_kernel(t_os - (st + sh), tau_r_fit, td)
    return t_os, y_os, t_zoom, zmask, z0, z1

def plot_trace_and_ppr(time, raw_series, y_sg, stim_times, tau_r_fit, tau_d0_fit,
                        t_os, y_os, peak_win_ms, avg_N_points, n_pulses,
                        ppr_nnls_center=None, ppr_band=None,
                        title_prefix="", avg_mode=False, n_trials=1,
                        tau_d_vec=None, deltas=None, a_vec=None,
                        train_mean_norm: bool = False,
                        rand_amps=None, fail_threshold=None, success_threshold=None,
                        baseline_mask=None, tau_r_for_null=None, tau_d0_for_null=None,
                        show_raw=True, show_sg=True, show_nnls=True, has_nnls=True,
                        thr_label_override: str = None):
    # Ensure computed data cannot be hidden
    if raw_series is not None:
        show_raw = True
    if y_sg is not None:
        show_sg = True
    if has_nnls and (t_os is not None) and (y_os is not None):
        show_nnls = True
    amp_raw = windowed_max(time, raw_series, stim_times, peak_win_ms, avg_N_points, peak_search_pre_ms)
    amp_sg  = windowed_max(time, y_sg,      stim_times, peak_win_ms, avg_N_points, peak_search_pre_ms)

    ppr_raw = normalize_amplitudes(amp_raw)
    ppr_sg  = normalize_amplitudes(amp_sg)

    # If we intend to overlay baseline null fits on the main plot, extend left bound to full baseline range
    if SHOW_NULL_FITS_ON_MAIN and baseline_mask is not None and not avg_mode:
        # Construct a custom zoom mask spanning from earliest baseline point through original zoom end
        base_idx = np.flatnonzero(baseline_mask)
        if base_idx.size:
            left_ext = time[base_idx[0]]
        else:
            left_ext = stim_times[0] - pre_zoom
        right_ext = stim_times[0] + post_zoom + isi_s * (n_pulses - 1)
        zmask = (time >= left_ext) & (time <= right_ext)
        z0, z1 = left_ext, right_ext
    else:
        zmask, z0, z1 = time_zoom_mask(time, stim_times[0], isi_s, n_pulses, pre_zoom, post_zoom)
    t_zoom = time[zmask]

    # Always allocate a histogram panel for trial plots so null distribution visibility is consistent.
    if not avg_mode:
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 5, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :4])
        ax_hist = fig.add_subplot(gs[1, 4])
    else:
        # Average trace: keep simpler layout (no dedicated histogram needed typically)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
        ax_hist = None
    if show_raw and raw_series is not None:
        ax1.plot(time[zmask], raw_series[zmask], linewidth=1.2, label=("Average raw" if avg_mode else "Raw (baseline-subtracted)"))
    if show_sg and y_sg is not None:
        ax1.plot(time[zmask], y_sg[zmask], linewidth=1.6, label=("Average SG(9,2)" if avg_mode else "Savitzky–Golay (9,2)"))
    if show_nnls and has_nnls and (t_os is not None) and (y_os is not None):
        ax1.plot(t_os, y_os, linewidth=1.8, label=("Average robust NNLS + shifts" if avg_mode else "Robust NNLS + shifts"))


    # Overlay baseline null fits directly on main axis if requested
    if SHOW_NULL_FITS_ON_MAIN and baseline_mask is not None and not avg_mode and tau_r_for_null is not None and tau_d0_for_null is not None:
        try:
            starts_null = _enumerate_null_candidates(time, baseline_mask, stim_times[0], F0_WINDOW_S, pre_zoom, post_zoom)
            MAX_BASELINE_OVERLAYS = 80
            if len(starts_null) > MAX_BASELINE_OVERLAYS:
                # Thin purely for visual clarity
                idx_sel = np.linspace(0, len(starts_null)-1, MAX_BASELINE_OVERLAYS).round().astype(int)
                starts_null = [starts_null[i] for i in idx_sel]
            first_drawn = False
            for st in starts_null:
                avail_post = min(post_zoom, stim_times[0] - st - 1e-6, stim_times[0] - st)
                eff_post = max(NULL_MIN_POST_ZOOM_S, min(post_zoom, avail_post))
                a_hat, d_hat = _fit_single_pulse_amp_consistent(
                    raw_series, time, st, tau_r_for_null, tau_d0_for_null,
                    pre_zoom=pre_zoom, post_zoom=eff_post,
                    robust=ROBUST_LOSS, huber_delta=HUBER_DELTA, irls_iters=IRLS_ITERS,
                    allow_shift=ENABLE_CONTINUOUS_SHIFTS, delta_max_s=DELTA_MAX_MS/1000.0, delta_step_s=DELTA_STEP_MS/1000.0
                )
                t_fit = time[(time >= st - pre_zoom) & (time <= st + eff_post)]
                k_fit = iglusnfr_kernel(t_fit - (st + d_hat), tau_r_for_null, tau_d0_for_null)
                ax1.plot(t_fit, a_hat * k_fit, color='red', alpha=1.0, lw=1.0,
                         label=('Baseline null fits' if not first_drawn else None))
                first_drawn = True
        except Exception as ex:
            if SHOW_PROGRESS:
                progress_print(f"[null-main-overlay] failed: {ex}")

    # Calculate NNLS-OS peak markers first
    pt, pv = [], []
    if has_nnls and t_os is not None and y_os is not None and show_nnls:
        for st in stim_times:
            tp, _ = pick_peak_on_series(t_os, y_os, st, peak_win_ms, peak_search_pre_ms)
            idx = np.searchsorted(t_os, tp)
            halfN = avg_N_points//2
            i0 = max(0, idx - halfN); i1 = min(len(t_os)-1, idx + (avg_N_points-1-halfN))
            pt.append(tp); pv.append(np.mean(y_os[i0:i1+1]))

    # Compute residual-corrected amplitudes without plotting individual model peaks
    if tau_d_vec is not None and deltas is not None and a_vec is not None:
        t_full_start = min(st + deltas[i] for i, st in enumerate(stim_times) if z0 <= st <= z1)
        t_full_end = max(st + deltas[i] + 0.15 for i, st in enumerate(stim_times) if z0 <= st <= z1)
        t_full = np.linspace(t_full_start, t_full_end, 1000)
        amp_corrected = np.array(pv.copy()) if pv else np.zeros(n_pulses)
        individual_traces = []
        for p, st in enumerate(stim_times):
            if z0 <= st <= z1:
                td = float(tau_d_vec[p])
                sh = float(deltas[p])
                amp = float(a_vec[p])
                t_start = st + sh
                y_event_individual = np.zeros_like(t_full)
                mask = t_full >= t_start
                y_event_individual[mask] = amp * iglusnfr_kernel(t_full[mask] - t_start, tau_r_fit, td)
                individual_traces.append((t_full, y_event_individual))
                if p > 0 and p < len(amp_corrected):
                    tp = pt[p] if p < len(pt) else t_start + tau_r_fit
                    peak_idx = np.searchsorted(t_full, tp)
                    prev_y = individual_traces[p-1][1]
                    if 0 <= peak_idx < len(prev_y):
                        amp_corrected[p] = amp_corrected[p] - prev_y[peak_idx]
    else:
        amp_corrected = np.array(pv.copy()) if pv else np.zeros(n_pulses)

    for st in stim_times:
        if z0 <= st <= z1:
            ax1.axvline(st, linestyle=":", linewidth=1.0)

    ppr_corrected = normalize_amplitudes(amp_corrected) if len(amp_corrected) else np.zeros(n_pulses)

    ttl = title_prefix or (f"Average of {n_trials} trials" if avg_mode else f"Single trial")
    ax1.set_title(f"{ttl} — Raw vs SG vs NNLS(+δ)\nτr={tau_r_fit*1e3:.1f} ms, τd0={tau_d0_fit*1e3:.1f} ms")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("ΔF/F0" if USE_DF_OVER_F0 else "ΔF (baseline-subtracted)")
    ax1.legend(loc="upper right")

    x = np.arange(1, n_pulses+1)
    if show_raw:
        ax2.plot(x, ppr_raw,  marker="o", label="Raw (windowed max)")
    if show_sg:
        ax2.plot(x, ppr_sg,   marker="o", label="SG(9,2) (windowed max)")
    if show_nnls and has_nnls:
        ax2.plot(x, ppr_corrected, marker="s", color="darkred", label="NNLS residual-corrected", linewidth=2)

    if ppr_band is not None:
        lo, hi, epsf = ppr_band
        mask = np.isfinite(lo) & np.isfinite(hi)
        if mask.any():
            ax2.fill_between(x[mask], lo[mask], hi[mask], alpha=0.22, label=f"ε-band (×{epsf:.2f})")

    ax2.set_xticks(x); ax2.set_xlim(0.7, n_pulses+0.3)
    def safe_series_max(arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0.0
        if np.all(~np.isfinite(arr)):
            return 0.0
        with np.errstate(all='ignore'):
            try:
                return float(np.nanmax(arr))
            except Exception:
                return 0.0
    # If first pulse amplitude (denominator) is 0 or NaN, replace PPR curves with NaNs
    def sanitize_ppr(p):
        if p is None:
            return p
        p = np.asarray(p, float)
        if p.size == 0:
            return p
        if not np.isfinite(p[0]) or p[0] == 0:
            return np.full_like(p, np.nan)
        return p
    ppr_raw = sanitize_ppr(ppr_raw)
    ppr_sg = sanitize_ppr(ppr_sg)
    ppr_corrected = sanitize_ppr(ppr_corrected)
    if ppr_band is not None:
        lo, hi, epsf = ppr_band
        lo = sanitize_ppr(lo)
        hi = sanitize_ppr(hi)
        ppr_band = (lo, hi, epsf)
    series_for_ylim = []
    if show_raw: series_for_ylim.append(ppr_raw)
    if show_sg: series_for_ylim.append(ppr_sg)
    if show_nnls and has_nnls: series_for_ylim.append(ppr_corrected)
    if ppr_band is not None: series_for_ylim.append(ppr_band[1])
    if not series_for_ylim:
        series_for_ylim = [np.array([0,1])]
    ylim_top = max(safe_series_max(s) for s in series_for_ylim)
    ax2.set_ylim(0, max(1.05, 1.1*ylim_top))
    if train_mean_norm:
        ax2.set_ylabel("Amplitude / train mean")
    else:
        ax2.set_ylabel("PPR (An/A1 ratio)")
    ax2.set_xlabel("Pulse #")
    ax2.legend(loc="best")

    if ax_hist is not None:
        if rand_amps is not None and np.size(rand_amps):
            # Clean finite data
            data = np.asarray(rand_amps, float)
            data = data[np.isfinite(data)]

            # Freedman–Diaconis binning (robust), clamped to [10, 60]
            def _fd_bins(x):
                x = np.asarray(x, float)
                n = x.size
                if n < 2:
                    return 10
                q75, q25 = np.percentile(x, [75, 25])
                iqr = float(q75 - q25)
                if iqr <= 0:
                    return int(max(10, min(60, np.sqrt(n))))
                h = 2.0 * iqr * (n ** (-1/3))
                if h <= 0:
                    return int(max(10, min(60, np.sqrt(n))))
                k = int(np.ceil((np.nanmax(x) - np.nanmin(x)) / h))
                return int(max(10, min(60, k if np.isfinite(k) and k > 0 else 10)))

            bins = _fd_bins(data)

            # Draw histogram with subtle styling
            counts, edges, patches = ax_hist.hist(
                data, bins=bins, color="0.80", edgecolor="0.35", linewidth=0.6
            )

            # Zero line for reference
            ax_hist.axvline(0.0, color="0.2", alpha=0.25, linewidth=0.8)

            # Highlight region >= threshold
            thr = fail_threshold if (fail_threshold is not None and np.isfinite(fail_threshold)) else None
            if thr is not None:
                ax_hist.axvspan(thr, edges[-1], facecolor="tab:red", alpha=0.08, zorder=0)

            # Recolor bars by threshold
            if patches is not None and len(edges) >= 2 and thr is not None:
                for i, p in enumerate(patches):
                    mid = 0.5 * (edges[i] + edges[i+1])
                    if mid >= thr:
                        p.set_facecolor("#f4a3a3")
                        p.set_edgecolor("#cc4444")
                    else:
                        p.set_facecolor("#c9d4e8")
                        p.set_edgecolor("#4f6aa3")

            # Stats box
            with np.errstate(all='ignore'):
                mu = float(np.nanmean(data)) if data.size else np.nan
                sd = float(np.nanstd(data)) if data.size else np.nan
                med = float(np.nanmedian(data)) if data.size else np.nan
                mad = float(np.nanmedian(np.abs(data - med))) if data.size else np.nan
                sigma_hat = 1.4826 * mad if np.isfinite(mad) else np.nan
            stats_txt = f"μ={mu:.3g}  σ={sd:.3g}\nmedian={med:.3g}  σMAD={sigma_hat:.3g}"
            ax_hist.text(
                0.02, 0.98, stats_txt, transform=ax_hist.transAxes,
                ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.8', alpha=0.85)
            )

            # Y-limits and grid
            ymax = counts.max() * 1.15 if counts.size else 1.0
            ax_hist.set_ylim(0, ymax)
            ax_hist.grid(axis='y', color='0.85', linestyle='-', linewidth=0.5)

            # Primary threshold (rule-dependent)
            if thr is not None:
                thr_label = thr_label_override or "thr"
                ax_hist.axvline(thr, color="red", linestyle="--", linewidth=1.2)
                ax_hist.text(thr, 0.92 * ymax, f"{thr_label}={thr:.3g}",
                             color="red", ha="right", va="top")

            # Secondary success threshold (e.g. 2x primary threshold)
            if success_threshold is not None and np.isfinite(success_threshold):
                ax_hist.axvline(success_threshold, color="orange", linestyle=":", linewidth=1.2)
                ax_hist.text(success_threshold, 0.78 * ymax, f"2x={success_threshold:.3g}",
                             color="orange", ha="right", va="top")

            # Rug plot (subsampled if large)
            if data.size:
                step = max(1, int(data.size // 150))
                rug = np.sort(data)[::step]
                ax_hist.vlines(rug, 0, 0.02 * ymax, color='0.2', alpha=0.15, linewidth=0.5)

            ax_hist.set_xlabel("Null amplitude")
            ax_hist.set_ylabel("Count")
        else:
            ax_hist.set_axis_off()
            ax_hist.text(0.5, 0.5, "(no null amps)", transform=ax_hist.transAxes,
                         ha="center", va="center", fontsize=9, color="0.4")

    # Horizontal threshold lines over first 3 events on the TRACE axis (amplitude units)
    try:
        if fail_threshold is not None and np.isfinite(fail_threshold):
            thr_label = thr_label_override or "thr"
            show_events = min(3, len(stim_times))
            for i in range(show_events):
                st = stim_times[i]
                # span a local window around each stim for visibility
                left = st - 0.002
                right = st + 0.010
                ax1.hlines(fail_threshold, left, right, colors='red', linestyles='dotted', linewidth=1.0)
            ax1.text(stim_times[0], fail_threshold, thr_label, color='red', ha='left', va='bottom')
        if success_threshold is not None and np.isfinite(success_threshold):
            show_events = min(3, len(stim_times))
            for i in range(show_events):
                st = stim_times[i]
                left = st - 0.002
                right = st + 0.010
                ax1.hlines(success_threshold, left, right, colors='orange', linestyles='dotted', linewidth=1.0)
            ax1.text(stim_times[0], success_threshold, '2x thr', color='orange', ha='left', va='bottom')
    except Exception:
        pass

    plt.tight_layout()

    return fig, np.array(pv), amp_corrected, ppr_corrected

# --------------------------
# I/O & batch
# --------------------------
def extract_bouton_name(path: str) -> str:
    name = os.path.basename(path)
    m = re.search(r"bouton(\d+)", name, re.IGNORECASE)
    return f"bouton{m.group(1)}" if m else ""

def format_fiber_id(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = re.sub(r"(?i)_bouton.*", "", base)
    btn = extract_bouton_name(path)
    return f"{base}_{btn}" if btn else base

def compute_metrics_for_file(xlsx_path: str,
                             train_start_override: Optional[float] = None,
                             amp_method: Optional[str] = None,
                             fail_method: Optional[str] = None):
    progress_print(f"Loading data from {os.path.basename(xlsx_path)}")
    # Skip obvious batch output workbooks (avoid feeding our own summary back in)
    base_lower = os.path.basename(xlsx_path).lower()
    if base_lower.startswith("batch_measure_complex_output") or base_lower.endswith("_metrics.xlsx"):
        progress_print("  Detected summary workbook – skipping file")
        return []
    # Unified resolution
    train_start_local = resolve_train_start(xlsx_path, explicit=train_start_override)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_index)
    N = df.shape[1]
    if N < 3: raise ValueError(f"Expected ≥3 columns in {xlsx_path}")
    time_raw = pd.to_numeric(df.iloc[:, N-1], errors="coerce").to_numpy(float)
    valid_t  = np.isfinite(time_raw)
    time     = time_raw[valid_t]

    # Extract trial columns (all but last time column); coerce to numeric.
    trial_df = df.iloc[:, :N-2].apply(pd.to_numeric, errors="coerce")
    # Drop columns that are entirely NaN OR have fewer than 3 finite points (empty/removed trials)
    good_cols = []
    for c in trial_df.columns:
        col_vals = pd.to_numeric(trial_df[c], errors="coerce")
        if col_vals.notna().sum() >= 3:
            good_cols.append(c)
    if len(good_cols) != trial_df.shape[1]:
        removed = [c for c in trial_df.columns if c not in good_cols]
        progress_print(f"  Ignoring empty/invalid trial columns: {removed}")
    trial_df = trial_df[good_cols]
    if trial_df.shape[1] == 0:
        progress_print("  No valid trial columns remain after filtering; skipping file")
        return []
    data_raw = trial_df.to_numpy(float)
    data_raw = data_raw[valid_t, :]

    # Define stimulation times early so bleaching correction can exclude the train window
    stim_times = train_start_local + isi_s * np.arange(n_pulses)

    progress_print("Preprocessing data (interpolation, optional bleaching, baseline correction)")
    _data = np.zeros_like(data_raw)
    for j in range(_data.shape[1]):
        col = fill_nans_timewise(data_raw[:, j], time)
        if ENABLE_BLEACH_CORRECTION:
            col = apply_bleach_correction(time, col, train_start_local, stim_times)
        _data[:, j] = col

    # Optional early interrupt for bleaching debug
    if BLEACH_INTERRUPT:
        try:
            os.makedirs(BATCH_EXPORT_DIR, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(xlsx_path))[0]
            parent_tag = os.path.basename(os.path.dirname(xlsx_path)) or "root"
            outp = os.path.join(BATCH_EXPORT_DIR, f"debug_bleach_{parent_tag}_{base_name}.npz")
            np.savez_compressed(outp, time=time, data_raw=data_raw, data_corrected=_data, stim_times=stim_times, train_start=train_start_local)
            progress_print(f"Saved bleach debug snapshot: {outp}")
        except Exception as e:
            progress_print(f"Failed to save bleach debug snapshot: {e}")
        progress_print("BLEACH_INTERRUPT enabled: skipping further processing for this file")
        return []


    # Baseline mask: FULL pre-train region (all samples strictly before train start)
    baseline_mask = time < train_start_local
    # Fallback: if no pre-train samples (unlikely), take earliest 10%
    if baseline_mask.sum() < 5:
        n10 = max(1, int(0.1 * len(time)))
        baseline_mask = np.zeros_like(time, bool)
        baseline_mask[:n10] = True

    progress_print("Computing robust baseline correction")
    F0_all = np.zeros(_data.shape[1])
    for j in range(_data.shape[1]):
        baseline_data = _data[baseline_mask, j]
        baseline_data = baseline_data[np.isfinite(baseline_data)]
        F0_all[j] = np.nanmedian(baseline_data) if len(baseline_data) > 0 else 0.0

    if USE_DF_OVER_F0:
        safe_F0 = F0_all.copy()
        safe_F0[np.abs(safe_F0) < F0_EPS] = np.nan  # mark unreliable baselines
        Y_all = (_data - F0_all) / safe_F0
    else:
        Y_all = _data - F0_all

    n_trials = Y_all.shape[1]
    progress_print(f"Processing {n_trials} trials with {n_pulses} pulses")

    amp_method_u = (amp_method or AMP_MEASUREMENT_METHOD or "NNLS").upper()
    fail_method_u = (fail_method or FAILURE_MEASUREMENT_METHOD or "NNLS").upper()
    SAVGOL_ONLY = (amp_method_u == "SAVGOL" and fail_method_u == "SAVGOL")

    # Enforce visibility of active measurement (cannot hide exported method)
    global SHOW_TRACE_RAW, SHOW_TRACE_SAVGOL, SHOW_TRACE_NNLS
    try:
        if amp_method_u == 'RAW' and not SHOW_TRACE_RAW:
            SHOW_TRACE_RAW = True
        if amp_method_u == 'SAVGOL' and not SHOW_TRACE_SAVGOL:
            SHOW_TRACE_SAVGOL = True
        if amp_method_u == 'NNLS' or fail_method_u == 'NNLS':
            # Ensure NNLS visible if either role needs it
            SHOW_TRACE_NNLS = True
            SHOW_TRACE_NNLS = True
    except Exception:
        pass

    # If SAVGOL-only but NNLS visibility requested, compute NNLS too
    if SAVGOL_ONLY and SHOW_TRACE_NNLS:
        progress_print("SAVGOL-only requested, but NNLS trace visible -> computing NNLS as well")
        SAVGOL_ONLY = False

    # Pre-compute SG-smoothed per-trial traces (needed for both paths)
    Y_sg_all = np.zeros_like(Y_all)
    for j in range(n_trials):
        Y_sg_all[:, j] = sg_smooth(Y_all[:, j], sg_window, sg_poly)

    if SAVGOL_ONLY:
        progress_print("SAVGOL mode: skipping kinetics & NNLS fitting; using SG windowed maxima only")
        # Average trace amplitudes
        y_avg = np.nanmean(Y_all, axis=1)
        y_sg_avg = sg_smooth(y_avg, sg_window, sg_poly)
        amp_raw_avg = windowed_max(time, y_avg,    stim_times, peak_win_ms, avg_N_points, peak_search_pre_ms)
        amp_sg_avg  = windowed_max(time, y_sg_avg, stim_times, peak_win_ms, avg_N_points, peak_search_pre_ms)
        rows = []
        def row_for(method, amps, level, trial_num, extra=None):
            row = {
                "file": os.path.basename(xlsx_path),
                "bouton": extract_bouton_name(xlsx_path),
                "n_trials": int(n_trials),
                "method": method, "level": level, "trial": int(trial_num),
                "noise_std": np.nan,
            }
            for i in range(n_pulses):
                row[f"amp_{i+1}"] = float(amps[i]) if i < len(amps) else np.nan
                # PPR relative to first pulse
                if i == 0 or not np.isfinite(amps[0]) or amps[0] == 0:
                    row[f"ppr_{i+1}"] = 1.0 if i == 0 and np.isfinite(amps[0]) and amps[0] != 0 else np.nan
                else:
                    row[f"ppr_{i+1}"] = float(amps[i] / amps[0])
            if extra:
                row.update(extra)
            return row
        # Average rows
        rows.append(row_for("Raw-windowedMax", amp_raw_avg, "average-trace", 0))
        rows.append(row_for("SG-windowedMax",  amp_sg_avg,  "average-trace", 0))
        # Optional average plot (raw + SG only)
        if ENABLE_AVERAGE_PLOTS and (SHOW_PLOTS_DURING_BATCH or SAVE_PLOTS):
            zmask_plot, _, _ = time_zoom_mask(time, stim_times[0], isi_s, n_pulses, pre_zoom, post_zoom)
            # Provide dummy NNLS arrays as None
            fig, _, _, _ = plot_trace_and_ppr(
                time, y_avg, y_sg_avg, stim_times,
                tau_r_fit=0.0, tau_d0_fit=0.0,
                t_os=None, y_os=None,
                peak_win_ms=peak_win_ms, avg_N_points=avg_N_points, n_pulses=n_pulses,
                ppr_nnls_center=None, ppr_band=None,
                title_prefix=f"{os.path.basename(xlsx_path)} — Average (SG only)",
                avg_mode=True, n_trials=n_trials,
                tau_d_vec=None, deltas=None, a_vec=None,
                train_mean_norm=False,
                rand_amps=None, fail_threshold=None, success_threshold=None,
                show_raw=SHOW_TRACE_RAW, show_sg=SHOW_TRACE_SAVGOL, show_nnls=False, has_nnls=False
            )
            if SAVE_PLOTS:
                os.makedirs(os.path.join(BATCH_EXPORT_DIR, PLOTS_SUBDIR), exist_ok=True)
                base_name = os.path.splitext(os.path.basename(xlsx_path))[0]
                parent_tag = os.path.basename(os.path.dirname(xlsx_path)) or "root"
                out_png = os.path.join(BATCH_EXPORT_DIR, PLOTS_SUBDIR,
                                       f"{parent_tag}_{base_name}_average_savgol.png")
                fig.savefig(out_png, dpi=150)
            if SHOW_PLOTS_DURING_BATCH and PLOT_MODE != 'none':
                _show_now(fig, pause=PAUSE_PLOTS_DURING_BATCH)
            else:
                plt.close(fig)
        # Trial rows with null amplitude sampling from SG trace only
        for t in range(n_trials):
            if SHOW_PROGRESS and t % max(1, n_trials//5) == 0:
                progress_print(f"Trial {t+1}/{n_trials}")
            y_t = Y_all[:, t]
            y_sg_t = Y_sg_all[:, t]
            # Null sampling times (deterministic, limited)
            null_starts = compute_null_sim_times_simple(time, stim_times[0], F0_WINDOW_S, PEAK_SEARCH_POST_S, NULL_SIM_MAX_POINTS)
            # Keep only starts whose window stays before the real train start
            max_span = peak_win_ms / 1000.0
            null_starts = [st for st in null_starts if st + max_span <= stim_times[0]]
            null_amps = windowed_max(time, y_sg_t, null_starts, peak_win_ms, avg_N_points, peak_search_pre_ms) if len(null_starts) else np.array([])
            # SAVGOL: SD-based threshold => mean + N * std
            thr_max, pval_fun = baseline_threshold_and_pval(null_amps, NULL_FAIL_THRESHOLD_PARAM, mode='sd')
            amp_sg_t = windowed_max(time, y_sg_t, stim_times, peak_win_ms, avg_N_points, peak_search_pre_ms)
            row_sg = row_for("SG-windowedMax", amp_sg_t, "trial", t+1, extra={"thr_max_amp1": thr_max, "pval_amp1": pval_fun(amp_sg_t[0]) if amp_sg_t.size else np.nan, "noise_std": float(np.nanstd(null_amps)) if null_amps.size else np.nan})
            rows.append(row_sg)
            if ENABLE_PER_TRIAL_PLOTS and (SHOW_PLOTS_DURING_BATCH or SAVE_PLOTS):
                fig_t, _, _, _ = plot_trace_and_ppr(
                    time, y_t, y_sg_t, stim_times,
                    tau_r_fit=0.0, tau_d0_fit=0.0,
                    t_os=None, y_os=None,
                    peak_win_ms=peak_win_ms, avg_N_points=avg_N_points, n_pulses=n_pulses,
                    ppr_nnls_center=None, ppr_band=None,
                    title_prefix=f"{os.path.basename(xlsx_path)} — Trial {t+1}/{n_trials} (SG only)",
                    avg_mode=False, n_trials=n_trials,
                    tau_d_vec=None, deltas=None, a_vec=None,
                    train_mean_norm=False,
                    rand_amps=null_amps, fail_threshold=thr_max, success_threshold=None,
                    baseline_mask=baseline_mask, tau_r_for_null=None, tau_d0_for_null=None,
                    show_raw=SHOW_TRACE_RAW, show_sg=SHOW_TRACE_SAVGOL, show_nnls=False, has_nnls=False,
                    thr_label_override=_format_fail_threshold_label(NULL_FAIL_THRESHOLD_PARAM, mode='sd')
                )
                if SAVE_PLOTS:
                    os.makedirs(os.path.join(BATCH_EXPORT_DIR, PLOTS_SUBDIR), exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(xlsx_path))[0]
                    parent_tag = os.path.basename(os.path.dirname(xlsx_path)) or "root"
                    out_png = os.path.join(BATCH_EXPORT_DIR, PLOTS_SUBDIR,
                                           f"{parent_tag}_{base_name}_trial{t+1}_savgol.png")
                    fig_t.savefig(out_png, dpi=150)
                if SHOW_PLOTS_DURING_BATCH and PLOT_MODE != 'none':
                    _show_now(fig_t, pause=PAUSE_PLOTS_DURING_BATCH)
                else:
                    plt.close(fig_t)
        progress_print("SAVGOL mode complete for file")
        return rows

    # Pooled kinetics from average trace
    tau_r_fit, tau_d0_fit, slope_fit, tau_d_vec = fit_kinetics_pooled(
        time, Y_all, stim_times, baseline_mask
    )

    # Optional: enforce non-decreasing τd across pulses
    if ENFORCE_NONDECREASING_TAUD:
        tau_d_vec = np.maximum.accumulate(tau_d_vec)

    # === Test override: force all τd to a fixed value ===
    if FORCE_TAUD_MS is not None:
        try:
            tau_d_const_s = float(FORCE_TAUD_MS) / 1000.0
            tau_d_vec = np.full(n_pulses, tau_d_const_s)
            tau_d0_fit = tau_d_const_s
            slope_fit = 0.0
            progress_print(f"Forcing all τd to {FORCE_TAUD_MS:.1f} ms for testing")
        except Exception:
            pass

    # Animation disabled; null baseline fits are overlaid directly on per-trial plots when SHOW_NULL_FITS_ON_MAIN is True.

    progress_print("Fitting average trace (NNLS + shifts)")
    y_avg = np.nanmean(Y_all, axis=1)
    zmask, _, _ = time_zoom_mask(time, train_start_local, isi_s, n_pulses, pre_zoom, post_zoom)

    # NNLS/template-matched on average
    if ENABLE_SEGMENTED_NO_OVERLAP:
        a_avg, deltas_avg, X_avg, yhat_avg = fit_amplitudes_no_overlap_backward(
            y_avg, time, stim_times, tau_r_fit, tau_d_vec,
            pre_zoom=pre_zoom, post_zoom=post_zoom,
            robust=ENABLE_ROBUST_FITTING and ROBUST_LOSS, huber_delta=HUBER_DELTA, irls_iters=IRLS_ITERS,
            allow_shift=ENABLE_CONTINUOUS_SHIFTS,
            delta_max_s=DELTA_MAX_MS/1000.0, delta_step_s=DELTA_STEP_MS/1000.0
        )
    else:
        a_avg, deltas_avg, X_avg, yhat_avg = fit_amplitudes_and_shifts(
            y_avg, time, stim_times, tau_r_fit, tau_d_vec, baseline_mask, zmask,
            delta_max_s=DELTA_MAX_MS/1000.0, delta_step_s=DELTA_STEP_MS/1000.0,
            robust=ENABLE_ROBUST_FITTING and ROBUST_LOSS, delta=HUBER_DELTA, irls_iters=IRLS_ITERS,
            alt_iters=4
        )
    y_sg_avg = sg_smooth(y_avg, sg_window, sg_poly)
    ppr_nnls_center = normalize_amplitudes(a_avg)


    # Plot average
    if ENABLE_AVERAGE_PLOTS and (SHOW_PLOTS_DURING_BATCH or SAVE_PLOTS):
        progress_print("Generating average trace plot")
        t_os_avg, y_os_avg, _, _, _, _ = build_os_reconstruction(
            time, stim_times, tau_r_fit, tau_d_vec, deltas_avg, a_avg, oversample_factor=10
        )
        fig, _amp_model_avg, amp_corrected_avg, ppr_corrected_avg = plot_trace_and_ppr(
            time, y_avg, y_sg_avg, stim_times, tau_r_fit, tau_d0_fit,
            t_os_avg, y_os_avg, peak_win_ms, avg_N_points, n_pulses,
            ppr_nnls_center=ppr_nnls_center, ppr_band=None,
            title_prefix=f"{os.path.basename(xlsx_path)} — Average of {n_trials} trials",
            avg_mode=True, n_trials=n_trials,
            tau_d_vec=tau_d_vec, deltas=deltas_avg, a_vec=a_avg,
            
            train_mean_norm=False,
            rand_amps=None, fail_threshold=None, success_threshold=None,
            show_raw=SHOW_TRACE_RAW, show_sg=SHOW_TRACE_SAVGOL, show_nnls=SHOW_TRACE_NNLS, has_nnls=True
        )
        if SAVE_PLOTS:
            os.makedirs(os.path.join(BATCH_EXPORT_DIR, PLOTS_SUBDIR), exist_ok=True)
            base_name = os.path.splitext(os.path.basename(xlsx_path))[0]
            parent_tag = os.path.basename(os.path.dirname(xlsx_path)) or "root"
            out_png = os.path.join(BATCH_EXPORT_DIR, PLOTS_SUBDIR,
                                   f"{parent_tag}_{base_name}_average.png")
            fig.savefig(out_png, dpi=150)
        if SHOW_PLOTS_DURING_BATCH and PLOT_MODE != 'none':
            _show_now(fig, pause=PAUSE_PLOTS_DURING_BATCH)
        else:
            plt.close(fig)

    # Per-trial metrics & plots
    progress_print("Computing per-trial metrics")
    rows = []
    def row_for(method, amps, pprs, level, trial_num, noise_std=np.nan, extra=None):
        row = {
            "file": os.path.basename(xlsx_path),
            "bouton": extract_bouton_name(xlsx_path),
            "n_trials": int(n_trials),
            "method": method, "level": level, "trial": int(trial_num),
            "noise_std": float(noise_std),
        }
        for i in range(n_pulses):
            row[f"amp_{i+1}"] = float(amps[i]) if i < len(amps) else np.nan
            row[f"ppr_{i+1}"] = float(pprs[i]) if i < len(pprs) else np.nan
        if extra:
            row.update(extra)
        return row

    amp_raw_avg = windowed_max(time, y_avg,    stim_times, peak_win_ms, avg_N_points, peak_search_pre_ms)
    amp_sg_avg  = windowed_max(time, y_sg_avg, stim_times, peak_win_ms, avg_N_points, peak_search_pre_ms)
    rows.append(row_for("Raw-windowedMax", amp_raw_avg,  amp_raw_avg/max(amp_raw_avg[0],1e-12), "average-trace", 0))
    rows.append(row_for("SG-windowedMax",  amp_sg_avg,   amp_sg_avg/max(amp_sg_avg[0],1e-12),  "average-trace", 0))
    rows.append(row_for("RobustNNLS-coeff", a_avg,       ppr_nnls_center,                      "average-trace", 0))
    if 'amp_corrected_avg' in locals() and 'ppr_corrected_avg' in locals():
        rows.append(row_for("RobustNNLS-corrected", amp_corrected_avg, ppr_corrected_avg, "average-trace", 0))

    # Mark NNLS average row for reference (not used by export selection)
    for r in rows:
        if r.get("method") == "RobustNNLS-coeff" and r.get("level") == "average-trace":
            r["_nnls_ref"] = True

    # Individual trials
    for t in range(n_trials):
        if SHOW_PROGRESS and t % max(1, n_trials//5) == 0:
            progress_print(f"Processing trial {t+1}/{n_trials}")

        y_t = Y_all[:, t]
        y_sg_t = sg_smooth(y_t, sg_window, sg_poly)

        # NNLS/template-matched per trial
        if ENABLE_SEGMENTED_NO_OVERLAP:
            a_t, deltas_t, X_t, yhat_t = fit_amplitudes_no_overlap_backward(
                y_t, time, stim_times, tau_r_fit, tau_d_vec,
                pre_zoom=pre_zoom, post_zoom=post_zoom,
                robust=ENABLE_ROBUST_FITTING and ROBUST_LOSS, huber_delta=HUBER_DELTA, irls_iters=IRLS_ITERS,
                allow_shift=ENABLE_CONTINUOUS_SHIFTS,
                delta_max_s=DELTA_MAX_MS/1000.0, delta_step_s=DELTA_STEP_MS/1000.0
            )
        else:
            a_t, deltas_t, X_t, yhat_t = fit_amplitudes_and_shifts(
                y_t, time, stim_times, tau_r_fit, tau_d_vec, baseline_mask, zmask,
                delta_max_s=DELTA_MAX_MS/1000.0, delta_step_s=DELTA_STEP_MS/1000.0,
                robust=ENABLE_ROBUST_FITTING and ROBUST_LOSS, delta=HUBER_DELTA, irls_iters=IRLS_ITERS,
                alt_iters=4
            )

        # Compute raw/SG amplitudes for this trial (used for rows and p-values)
        amp_raw_t = windowed_max(time, y_t,    stim_times, peak_win_ms, avg_N_points, peak_search_pre_ms)
        amp_sg_t  = windowed_max(time, y_sg_t, stim_times, peak_win_ms, avg_N_points, peak_search_pre_ms)

        # === Failure thresholds based on selected failure method ===
        per_pulse_thr = {}
        per_pulse_pval = {}
        null_amps_fail = np.array([])
        label_mode = 'mad' if fail_method_u == 'NNLS' else 'sd'
        if fail_method_u == 'NNLS':
            # Consistent null & thresholds for pulse 1 (aligned with amp1 estimator)
            null_amps = sample_null_amplitudes_consistent(
                y_t, time, baseline_mask,
                tau_r_fit, tau_d0_fit,
                train_start=train_start_local, f0_window_s=F0_WINDOW_S,
                pre_zoom=pre_zoom, post_zoom=post_zoom,
                robust=ENABLE_ROBUST_FITTING and ROBUST_LOSS, huber_delta=HUBER_DELTA, irls_iters=IRLS_ITERS,
                allow_shift=ENABLE_CONTINUOUS_SHIFTS,
                delta_max_s=DELTA_MAX_MS/1000.0, delta_step_s=DELTA_STEP_MS/1000.0,
                n_samples=1000, seed=10_000 + t
            )
            thr_max1, pval_fun1 = baseline_threshold_and_pval(null_amps, NULL_FAIL_THRESHOLD_PARAM, mode='mad')
            # p-value evaluated against amplitude from AMP method
            amp_for_pval = a_t if amp_method_u == 'NNLS' else (amp_sg_t if amp_method_u == 'SAVGOL' else amp_raw_t)
            p_emp1 = pval_fun1(amp_for_pval[0]) if np.size(amp_for_pval) else np.nan
            per_pulse_thr = {1: thr_max1}
            per_pulse_pval = {1: p_emp1}
            # Additional pulses 2 & 3 from matched tau_d
            max_extra_pulse = min(3, n_pulses)
            for pulse_idx in range(2, max_extra_pulse+1):
                if a_t.size < pulse_idx:  # safety
                    continue
                try:
                    tau_d_for_p = tau_d_vec[pulse_idx-1] if (tau_d_vec is not None and len(tau_d_vec) >= pulse_idx) else tau_d0_fit
                    null_amps_p = sample_null_amplitudes_consistent(
                        y_t, time, baseline_mask,
                        tau_r_fit, tau_d_for_p,
                        train_start=train_start_local, f0_window_s=F0_WINDOW_S,
                        pre_zoom=pre_zoom, post_zoom=post_zoom,
                        robust=ENABLE_ROBUST_FITTING and ROBUST_LOSS, huber_delta=HUBER_DELTA, irls_iters=IRLS_ITERS,
                        allow_shift=ENABLE_CONTINUOUS_SHIFTS,
                        delta_max_s=DELTA_MAX_MS/1000.0, delta_step_s=DELTA_STEP_MS/1000.0,
                        n_samples=1000, seed=20_000 + 500*pulse_idx + t
                    )
                    thr_p, pval_fun_p = baseline_threshold_and_pval(null_amps_p, NULL_FAIL_THRESHOLD_PARAM, mode='mad')
                    per_pulse_thr[pulse_idx] = thr_p
                    amp_val = amp_for_pval[pulse_idx-1] if np.size(amp_for_pval) >= pulse_idx else np.nan
                    per_pulse_pval[pulse_idx] = pval_fun_p(amp_val) if np.isfinite(amp_val) else np.nan
                except Exception:
                    per_pulse_thr[pulse_idx] = np.nan
                    per_pulse_pval[pulse_idx] = np.nan
            null_amps_fail = null_amps
        else:
            # SAVGOL: SD-based threshold from SG null windows; use same threshold for pulses 1-3
            null_starts = compute_null_sim_times_simple(time, stim_times[0], F0_WINDOW_S, PEAK_SEARCH_POST_S, NULL_SIM_MAX_POINTS)
            max_span = peak_win_ms / 1000.0
            null_starts = [st for st in null_starts if st + max_span <= stim_times[0]]
            null_amps_sg = windowed_max(time, y_sg_t, null_starts, peak_win_ms, avg_N_points, peak_search_pre_ms) if len(null_starts) else np.array([])
            thr_sg, pval_fun_sg = baseline_threshold_and_pval(null_amps_sg, NULL_FAIL_THRESHOLD_PARAM, mode='sd')
            for pulse_idx in range(1, min(3, n_pulses)+1):
                per_pulse_thr[pulse_idx] = thr_sg
            amp_for_pval = a_t if amp_method_u == 'NNLS' else (amp_sg_t if amp_method_u == 'SAVGOL' else amp_raw_t)
            if np.size(amp_for_pval):
                per_pulse_pval[1] = pval_fun_sg(amp_for_pval[0])
                if np.size(amp_for_pval) >= 2:
                    per_pulse_pval[2] = pval_fun_sg(amp_for_pval[1])
                if np.size(amp_for_pval) >= 3:
                    per_pulse_pval[3] = pval_fun_sg(amp_for_pval[2])
            null_amps_fail = null_amps_sg

        # noise_std based on chosen failure null
        noise_std_t = float(np.nanstd(null_amps_fail)) if np.size(null_amps_fail) else np.nan

        ppr_raw_t  = normalize_amplitudes(amp_raw_t)
        ppr_sg_t   = normalize_amplitudes(amp_sg_t)
        ppr_nnls_t = normalize_amplitudes(a_t)
        rows.append(row_for("Raw-windowedMax",   amp_raw_t,  ppr_raw_t,  "trial", t+1))
        rows.append(row_for("SG-windowedMax",    amp_sg_t,   ppr_sg_t,   "trial", t+1))
        # Common extras with per-pulse thresholds
        extras_coeff = {
            "thr_max_amp1": per_pulse_thr.get(1, np.nan),
            "pval_amp1": per_pulse_pval.get(1, np.nan)
        }
        if 2 in per_pulse_thr:
            extras_coeff["thr_max_amp2"] = per_pulse_thr[2]; extras_coeff["pval_amp2"] = per_pulse_pval[2]
        if 3 in per_pulse_thr:
            extras_coeff["thr_max_amp3"] = per_pulse_thr[3]; extras_coeff["pval_amp3"] = per_pulse_pval[3]
        rows.append(row_for("RobustNNLS-coeff",  a_t,        ppr_nnls_t, "trial", t+1,
                    noise_std=noise_std_t,
                    extra=extras_coeff))

        # Residual-corrected NNLS
        amp_model_t = windowed_max(time, yhat_t, stim_times, peak_win_ms, avg_N_points, peak_search_pre_ms)
        amp_corrected_t = amp_model_t.copy()
        if tau_d_vec is not None and deltas_t is not None and a_t is not None:
            zmask_t, z0, z1 = time_zoom_mask(time, train_start_local, isi_s, n_pulses, pre_zoom, post_zoom)
            t_full_start = min(st + deltas_t[i] for i, st in enumerate(stim_times) if z0 <= st <= z1)
            t_full_end = max(st + deltas_t[i] + 0.15 for i, st in enumerate(stim_times) if z0 <= st <= z1)
            t_full = np.linspace(t_full_start, t_full_end, 1000)
            individual_traces = []
            for p, st in enumerate(stim_times):
                if z0 <= st <= z1:
                    td = float(tau_d_vec[p]); sh = float(deltas_t[p]); amp = float(a_t[p])
                    t_start = st + sh
                    y_event_individual = np.zeros_like(t_full)
                    mask = t_full >= t_start
                    y_event_individual[mask] = amp * iglusnfr_kernel(t_full[mask] - t_start, tau_r_fit, td)
                    individual_traces.append((t_full, y_event_individual))
                    if p > 0:
                        tp, _ = pick_peak_on_series(time, yhat_t, st, peak_win_ms, peak_search_pre_ms)
                        peak_idx = np.searchsorted(t_full, tp)
                        if 0 <= peak_idx < len(t_full):
                            prev_y_event = individual_traces[p-1][1]
                            amp_corrected_t[p] = amp_corrected_t[p] - prev_y_event[peak_idx]
    ppr_corrected_t = normalize_amplitudes(amp_corrected_t)
    extras_corr = {
        "thr_max_amp1": per_pulse_thr.get(1, np.nan),
        "pval_amp1": per_pulse_pval.get(1, np.nan)
    }
    if 2 in per_pulse_thr:
        extras_corr["thr_max_amp2"] = per_pulse_thr[2]; extras_corr["pval_amp2"] = per_pulse_pval[2]
    if 3 in per_pulse_thr:
        extras_corr["thr_max_amp3"] = per_pulse_thr[3]; extras_corr["pval_amp3"] = per_pulse_pval[3]
    rows.append(row_for("RobustNNLS-corrected", amp_corrected_t, ppr_corrected_t, "trial", t+1,
                noise_std=noise_std_t,
                extra=extras_corr))

    # Ensure the exported amplitude method row carries failure thresholds
    target_method = None
    if amp_method_u == 'SAVGOL':
        target_method = 'SG-windowedMax'
    elif amp_method_u == 'RAW':
        target_method = 'Raw-windowedMax'
    # For NNLS, the corrected row already has extras
    if target_method is not None:
        for r in reversed(rows):
            if r.get("method") == target_method and r.get("level") == "trial" and r.get("trial") == t+1:
                r["thr_max_amp1"] = per_pulse_thr.get(1, np.nan)
                r["pval_amp1"] = per_pulse_pval.get(1, np.nan)
                if 2 in per_pulse_thr:
                    r["thr_max_amp2"] = per_pulse_thr[2]; r["pval_amp2"] = per_pulse_pval.get(2, np.nan)
                if 3 in per_pulse_thr:
                    r["thr_max_amp3"] = per_pulse_thr[3]; r["pval_amp3"] = per_pulse_pval.get(3, np.nan)
                r["fail_amp_source"] = fail_method_u
                break


    if ENABLE_PER_TRIAL_PLOTS and (SHOW_PLOTS_DURING_BATCH or SAVE_PLOTS):
            t_os, y_os, t_zoom, _, _, _ = build_os_reconstruction(
                time, stim_times, tau_r_fit, tau_d_vec, deltas_t, a_t, oversample_factor=10
            )

            fig, _, _, _ = plot_trace_and_ppr(
                time, y_t, y_sg_t, stim_times, tau_r_fit, tau_d0_fit,
                t_os, y_os, peak_win_ms, avg_N_points, n_pulses,
                ppr_nnls_center=ppr_nnls_t,
                ppr_band=None,
                title_prefix=f"{os.path.basename(xlsx_path)} — Trial {t+1}/{n_trials}",
                avg_mode=False, n_trials=n_trials,
                tau_d_vec=tau_d_vec, deltas=deltas_t, a_vec=a_t,
                
                train_mean_norm=False,
                rand_amps=null_amps_fail,
                fail_threshold=per_pulse_thr.get(1, np.nan),
                success_threshold=None,
                baseline_mask=baseline_mask,
                tau_r_for_null=tau_r_fit,
                tau_d0_for_null=tau_d0_fit,
                show_raw=SHOW_TRACE_RAW, show_sg=SHOW_TRACE_SAVGOL, show_nnls=SHOW_TRACE_NNLS, has_nnls=True,
                thr_label_override=_format_fail_threshold_label(NULL_FAIL_THRESHOLD_PARAM, mode=label_mode)
            )

            if SAVE_PLOTS:
                os.makedirs(os.path.join(BATCH_EXPORT_DIR, PLOTS_SUBDIR), exist_ok=True)
                base_name = os.path.splitext(os.path.basename(xlsx_path))[0]
                parent_tag = os.path.basename(os.path.dirname(xlsx_path)) or "root"
                out_png = os.path.join(BATCH_EXPORT_DIR, PLOTS_SUBDIR,
                                       f"{parent_tag}_{base_name}_trial{t+1}.png")
                fig.savefig(out_png, dpi=150)
            if SHOW_PLOTS_DURING_BATCH and PLOT_MODE != 'none':
                _show_now(fig, pause=PAUSE_PLOTS_DURING_BATCH)
            else:
                plt.close(fig)

    progress_print("Finished processing all trials")
    return rows


def run_batch_export_gui():
    if tk is None:
        progress_print("tkinter not available; GUI mode disabled")
        return
    files = sorted(glob.glob(os.path.join(BATCH_EXPORT_DIR, "*.xlsx")))
    if not files:
        progress_print(f"No .xlsx files found in {BATCH_EXPORT_DIR}")
        return
    all_rows = []

    root = tk.Tk()
    root.title("Batch Navigator")
    idx = tk.IntVar(root, value=0)

    lbl_file = tk.Label(root)
    lbl_file.pack(anchor="w", padx=5, pady=5)

    var_fast = tk.BooleanVar(value=ENABLE_FAST_KINETICS)
    var_shift = tk.BooleanVar(value=ENABLE_CONTINUOUS_SHIFTS)
    var_robust = tk.BooleanVar(value=ENABLE_ROBUST_FITTING)
    var_ar = tk.BooleanVar(value=False)
    var_trial = tk.BooleanVar(value=ENABLE_PER_TRIAL_PLOTS)
    var_avg = tk.BooleanVar(value=ENABLE_AVERAGE_PLOTS)

    for text, var in [
        ("Fast kinetics", var_fast),
        ("Continuous shifts", var_shift),
        ("Robust fitting", var_robust),
        ("Per-trial plots", var_trial),
        ("Average plots", var_avg),
    ]:
        tk.Checkbutton(root, text=text, variable=var).pack(anchor="w")

    frame_sg = tk.Frame(root)
    frame_sg.pack(anchor="w", padx=5, pady=5)
    tk.Label(frame_sg, text="SG window").grid(row=0, column=0)
    entry_window = tk.Entry(frame_sg, width=5)
    entry_window.insert(0, str(sg_window))
    entry_window.grid(row=0, column=1)
    tk.Label(frame_sg, text="SG poly").grid(row=1, column=0)
    entry_poly = tk.Entry(frame_sg, width=5)
    entry_poly.insert(0, str(sg_poly))
    entry_poly.grid(row=1, column=1)

    lbl_status = tk.Label(root)
    lbl_status.pack(anchor="w", padx=5, pady=5)

    def update_file_label():
        i = idx.get()
        lbl_file.config(text=f"File {i+1}/{len(files)}: {os.path.basename(files[i])}")

    def process_current():
        global ENABLE_FAST_KINETICS, ENABLE_CONTINUOUS_SHIFTS, ENABLE_ROBUST_FITTING, ENABLE_PER_TRIAL_PLOTS, ENABLE_AVERAGE_PLOTS, sg_window, sg_poly
        ENABLE_FAST_KINETICS = var_fast.get()
        ENABLE_CONTINUOUS_SHIFTS = var_shift.get()
        ENABLE_ROBUST_FITTING = var_robust.get()
        ENABLE_PER_TRIAL_PLOTS = var_trial.get()
        ENABLE_AVERAGE_PLOTS = var_avg.get()
        try:
            sg_window = int(entry_window.get())
            sg_poly = int(entry_poly.get())
        except Exception:
            pass
        fp = files[idx.get()]
        progress_print(f"Processing: {os.path.basename(fp)}")
        rows = compute_metrics_for_file(fp,
                                       amp_method=AMP_MEASUREMENT_METHOD,
                                       fail_method=FAILURE_MEASUREMENT_METHOD)
        all_rows.extend(rows)
        progress_print(f"Completed: {os.path.basename(fp)} ({len(rows)} rows)")
        lbl_status.config(text=f"Processed {os.path.basename(fp)}")

    def next_file():
        i = idx.get()
        if i < len(files) - 1:
            idx.set(i + 1)
            update_file_label()

    def prev_file():
        i = idx.get()
        if i > 0:
            idx.set(i - 1)
            update_file_label()

    frame_btn = tk.Frame(root)
    frame_btn.pack(pady=5)
    tk.Button(frame_btn, text="Prev", command=prev_file).grid(row=0, column=0)
    tk.Button(frame_btn, text="Process", command=process_current).grid(row=0, column=1)
    tk.Button(frame_btn, text="Next", command=next_file).grid(row=0, column=2)

    def on_close():
        if all_rows:
            progress_print("Building output dataframe...")
            cols = [
                "file", "bouton", "n_trials", "level", "trial", "method",
                *[f"amp_{i+1}" for i in range(n_pulses)],
                *[f"ppr_{i+1}" for i in range(n_pulses)]
            ]
            df_out = pd.DataFrame(all_rows)
            for c in cols:
                if c not in df_out.columns:
                    df_out[c] = np.nan
            df_out = df_out[cols]
            out_path = os.path.join(BATCH_EXPORT_DIR, "amplitude_ppr_summary.csv")
            os.makedirs(BATCH_EXPORT_DIR, exist_ok=True)
            df_out.to_csv(out_path, index=False)
            progress_print(f"Saved CSV: {out_path} ({len(df_out)} rows)")
        root.destroy()

    tk.Button(frame_btn, text="Quit", command=on_close).grid(row=0, column=3)
    root.protocol("WM_DELETE_WINDOW", on_close)
    update_file_label()
    root.mainloop()

def run_batch_export(gui: bool = False, amp_method: str = None, fail_method: str = None):
    """Run batch export using global defaults.

    Parameters
    ----------
    gui : bool
        If True and tkinter is available, launch interactive GUI.
    amp_method : str, optional
        Amplitude method ("NNLS", "RAW", "SAVGOL").
    fail_method : str, optional
        Failure/threshold method ("NNLS", "SAVGOL").
    """
    if gui:
        if tk is None:
            progress_print("tkinter not available; running without GUI")
        else:
            return run_batch_export_gui()
    progress_print("Starting batch export...")
    meas_amp = (amp_method or AMP_MEASUREMENT_METHOD or "NNLS").upper()
    meas_fail = (fail_method or FAILURE_MEASUREMENT_METHOD or "NNLS").upper()
    try:
        batch_measure_complex(
            DEFAULT_BATCH_INPUT_DIRS,
            out_file=DEFAULT_BATCH_OUTPUT_FILE,
            amp_method=meas_amp,
            fail_method=meas_fail,
            max_files=BATCH_FILE_LIMIT,
            train_start_overrides=TRAIN_START_OVERRIDE_MAP,
        )
        progress_print(f"Saved Excel: {DEFAULT_BATCH_OUTPUT_FILE}")
    except Exception as e:
        progress_print(f"Batch export failed: {e}")


def batch_measure_complex(paths,
                         out_file="ppr_results.xlsx",
                         amp_method: str = None,
                         fail_method: str = None,
                         max_files=None,
                         train_start_overrides: Optional[Dict[str, float]] = None):
    """Run batch processing on directories and export a multi-tab Excel workbook.

    For every ``.xlsx`` file in each input directory the function runs
    :func:`compute_metrics_for_file`, extracts amplitudes and paired-pulse ratios
    from the chosen method's average trace and estimates per-pulse failure rates
    from the individual trials.  Results for each directory are written to a
    separate sheet with one row per fibre and a final row containing the
    across-fibre averages.

    Parameters
    ----------
    paths : str or list[str]
        Directory path or list of paths to process.
    out_file : str
        Output Excel filename.
    amp_method : {"NNLS", "RAW", "SAVGOL"}, optional
        Which method supplies amplitudes in the export. ``NNLS`` uses "RobustNNLS-corrected",
        ``RAW`` uses "Raw-windowedMax", ``SAVGOL`` uses "SG-windowedMax".
    fail_method : {"NNLS", "SAVGOL"}, optional
        Which method supplies null/thresholds for failure classification.
    max_files : int, optional
        Process at most this many files per directory. Set to ``None`` to
        process all files.
    """

    method_map = {
        "NNLS": "RobustNNLS-corrected",
        "RAW": "Raw-windowedMax",
        "SAVGOL": "SG-windowedMax",
    }
    amp_key = (amp_method or AMP_MEASUREMENT_METHOD or "NNLS").upper()
    fail_key = (fail_method or FAILURE_MEASUREMENT_METHOD or "NNLS").upper()
    method_key = method_map.get(amp_key)
    if method_key is None:
        raise ValueError(f"Unknown amp_method '{amp_key}'")

    if isinstance(paths, str):
        paths = [paths]

    out_file_abs = os.path.abspath(out_file)  # NEW: for exclusion test

    wrote_any = False
    per_trial_rows = []  # For secondary Excel: individual AMP1 outcomes
    with pd.ExcelWriter(out_file) as writer:
        # Pre-create a hidden placeholder sheet so that even if nothing gets written
        # the workbook remains valid (will be removed/replaced if real sheets added)
        try:
            pd.DataFrame({"placeholder": [1]}).to_excel(writer, sheet_name="_init", index=False)
        except Exception:
            pass
        for path in paths:
            files_all = sorted(glob.glob(os.path.join(path, "*.xlsx")))
            # EXCLUDE the output workbook itself and Excel temp lock files
            files = [
                f for f in files_all
                if os.path.abspath(f) != out_file_abs
                and not os.path.basename(f).startswith("~$")
            ]
            if max_files is not None:
                files = files[:max_files]
            if not files:
                continue

            rows = []
            for fp in files:
                # Derive override value (pass explicit only; resolver re-applies precedence with global map)
                override_val = None
                if train_start_overrides:
                    parent_dir = os.path.normcase(os.path.normpath(os.path.dirname(fp)))
                    best_key=None; best_len=-1
                    for k,v in train_start_overrides.items():
                        nk = os.path.normcase(os.path.normpath(k))
                        if parent_dir == nk or parent_dir.startswith(nk + os.sep) or parent_dir.endswith(nk):
                            L=len(nk)
                            if L>best_len:
                                best_len=L; best_key=k; override_val=v
                metrics = compute_metrics_for_file(fp, train_start_override=override_val,
                                                 amp_method=amp_key, fail_method=fail_key)
                # If compute_metrics_for_file returned nothing (e.g., BLEACH_INTERRUPT),
                # insert a placeholder row so the folder is not skipped entirely.
                if not metrics:
                    placeholder = {"measurement": amp_key, "ID": format_fiber_id(fp)}
                    for i in range(1, n_pulses+1):
                        placeholder[f"AMP{i}"] = np.nan
                    for i in range(2, n_pulses+1):
                        placeholder[f"PPR{i}/1"] = np.nan
                    for i in range(1, min(3, n_pulses)+1):
                        placeholder[f"%Fail{i}"] = np.nan
                    rows.append(placeholder)
                    # continue to next file without creating df_metrics
                    continue
                df_metrics = pd.DataFrame(metrics)
                avg_sel = df_metrics[
                    (df_metrics.get("method") == method_key)
                    & (df_metrics.get("level") == "average-trace")
                ]
                trial_sel = df_metrics[
                    (df_metrics.get("method") == method_key)
                    & (df_metrics.get("level") == "trial")
                ]

                # Collect per-trial AMP1 classification data (one row per trial)
                try:
                    if not trial_sel.empty and 'amp_1' in trial_sel.columns:
                        folder_name = os.path.basename(os.path.normpath(path))
                        base_file = os.path.splitext(os.path.basename(fp))[0]
                        for _, rtrial in trial_sel.iterrows():
                            a_val = pd.to_numeric(rtrial.get('amp_1'), errors='coerce')
                            thr_val = pd.to_numeric(rtrial.get('thr_max_amp1'), errors='coerce') if 'thr_max_amp1' in trial_sel.columns else np.nan
                            if not np.isfinite(a_val):
                                status = 'NA'
                            else:
                                status = 'success' if (np.isfinite(thr_val) and a_val > thr_val) else (
                                         'failure' if np.isfinite(thr_val) else 'NA')
                            per_trial_rows.append({
                                'AMP1': float(a_val) if np.isfinite(a_val) else np.nan,
                                'status': status,
                                'file': base_file,
                                'folder': folder_name,
                                'trial': int(rtrial.get('trial')) if 'trial' in rtrial else np.nan,
                            })
                except Exception:
                    pass
                if avg_sel.empty:
                    continue
                row = {"measurement": amp_key, "ID": format_fiber_id(fp)}
                for i in range(1, n_pulses+1):
                    row[f"AMP{i}"] = avg_sel.iloc[0].get(f"amp_{i}", np.nan)
                amp1 = row.get("AMP1")
                for i in range(2, n_pulses+1):
                    key = f"PPR{i}/1"
                    row[key] = (
                        row[f"AMP{i}"] / amp1
                        if (amp1 is not None and not np.isnan(amp1) and amp1 != 0)
                        else np.nan
                    )
                # New failure definition: each of pulses 1-3 uses its own baseline-derived threshold thr_max_amp{i}.
                # If a per-pulse threshold is missing, leave %Fail{i} as NaN (no fallback classification).
                for i in range(1, min(3, n_pulses)+1):
                    amp_vals = pd.to_numeric(trial_sel.get(f"amp_{i}"), errors="coerce")
                    thr_col = f"thr_max_amp{i}"
                    if thr_col in trial_sel.columns:
                        thr_vals = pd.to_numeric(trial_sel.get(thr_col), errors="coerce")
                        valid = (~amp_vals.isna()) & (~thr_vals.isna())
                        row[f"%Fail{i}"] = (float(np.mean(amp_vals[valid] <= thr_vals[valid])) * 100.0) if valid.any() else np.nan
                        if SHOW_PROGRESS and valid.any():
                            try:
                                n_valid = int(valid.sum()); n_fail = int((amp_vals[valid] <= thr_vals[valid]).sum())
                                progress_print(f"    [%Fail debug] {format_fiber_id(fp)} pulse{i}: fails={n_fail}/{n_valid} (thr{i})")
                            except Exception:
                                pass
                    else:
                        row[f"%Fail{i}"] = np.nan
                rows.append(row)
            if not rows:
                continue
            df_out = pd.DataFrame(rows)
            # Build column ordering without adding an average summary row. Move the first column ('measurement') to the end.
            ordered_cols_core = [f"AMP{i}" for i in range(1, n_pulses+1)] \
                + [f"PPR{i}/1" for i in range(2, n_pulses+1)] \
                + [f"%Fail{i}" for i in range(1, min(3, n_pulses)+1)]
            # Ensure columns exist even if empty
            for c in ["measurement", "ID", *ordered_cols_core]:
                if c not in df_out.columns:
                    df_out[c] = np.nan
            # New order: ID first, then metrics, finally measurement at end
            new_order = ["ID", *ordered_cols_core, "measurement"]
            df_out = df_out[new_order]
            sheet_name = os.path.basename(os.path.normpath(path))[:31]
            df_out.to_excel(writer, sheet_name=sheet_name, index=False)
            wrote_any = True
        # Cleanup placeholder if we wrote at least one real sheet
        if wrote_any:
            try:
                del writer.book["_init"]
            except Exception:
                pass
        else:
            try:
                placeholder_df = pd.DataFrame({"info": ["No data files processed"]})
                placeholder_df.to_excel(writer, sheet_name="Summary", index=False)
            except Exception:
                pass

    # Write secondary per-trial AMP1 Excel if any rows collected
    if per_trial_rows:
        try:
            per_trial_df = pd.DataFrame(per_trial_rows)
            sec_path = os.path.splitext(out_file)[0] + "_trials.xlsx"
            with pd.ExcelWriter(sec_path) as w2:
                per_trial_df.to_excel(w2, sheet_name='Trials', index=False)
            progress_print(f"Wrote per-trial AMP1 Excel: {sec_path} ({len(per_trial_df)} rows)")
        except Exception as ex:
            progress_print(f"Failed writing per-trial AMP1 Excel: {ex}")


# --------------------------
# Baseline-only quick preview (no fitting) for a single file
# --------------------------
def baseline_preview(xlsx_path: str, train_start_override: Optional[float] = None,
                     show=True, save=False, save_dir: Optional[str] = None):
    """Plot only the baseline-corrected raw & SG-smoothed average trace for a single file.

    Steps:
      1. Load workbook (first sheet by sheet_index)
      2. Interpolate NaNs, optional bleach correction (if enabled globally)
      3. Baseline subtraction / ΔF/F0 normalization (depending on USE_DF_OVER_F0)
      4. Compute average across trials; plot raw average vs SG smoothing with stim markers.
    """
    try:
        progress_print(f"Baseline-only preview: {os.path.basename(xlsx_path)}")
        # Resolve train start override
        if train_start_override is None:
            for k, v in TRAIN_START_OVERRIDE_MAP.items():
                if xlsx_path.startswith(k) or xlsx_path.endswith(k):
                    train_start_local = float(v); break
            else:
                train_start_local = train_start_s
        else:
            train_start_local = float(train_start_override)
        df = pd.read_excel(xlsx_path, sheet_name=sheet_index)
        N = df.shape[1]
        if N < 3:
            raise ValueError("Expected ≥3 columns (trials + time)")
        time_raw = pd.to_numeric(df.iloc[:, N-1], errors="coerce").to_numpy(float)
        valid_t = np.isfinite(time_raw)
        time = time_raw[valid_t]
        data_raw = df.iloc[:, :N-2].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        data_raw = data_raw[valid_t, :]
        stim_times = train_start_local + isi_s * np.arange(n_pulses)
        # Interpolate & optional bleach correction
        data_corr = np.zeros_like(data_raw)
        for j in range(data_raw.shape[1]):
            col = fill_nans_timewise(data_raw[:, j], time)
            if ENABLE_BLEACH_CORRECTION:
                col = apply_bleach_correction(time, col, train_start_local, stim_times)
            data_corr[:, j] = col
        baseline_mask_local = (time >= (train_start_local - F0_WINDOW_S)) & (time < train_start_local)
        if not np.any(baseline_mask_local):
            n10 = max(1, int(0.1*len(time)))
            baseline_mask_local = np.zeros_like(time, bool); baseline_mask_local[:n10] = True
        F0 = np.array([np.nanmedian(col[baseline_mask_local]) for col in data_corr.T])
        if USE_DF_OVER_F0:
            safe_F0 = F0.copy(); safe_F0[np.abs(safe_F0) < F0_EPS] = np.nan
            Y = (data_corr.T - F0[:, None]) / safe_F0[:, None]
        else:
            Y = data_corr.T - F0[:, None]
        y_avg = np.nanmean(Y, axis=0)
        y_sg = sg_smooth(y_avg, sg_window, sg_poly)
        zmask, z0, z1 = time_zoom_mask(time, train_start_local, isi_s, n_pulses, pre_zoom, post_zoom)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time[zmask], y_avg[zmask], label='Average raw', lw=1.3, color='tab:gray')
        ax.plot(time[zmask], y_sg[zmask], label=f'SG({sg_window},{sg_poly})', lw=1.6, color='tab:blue')
        for st in stim_times:
            if z0 <= st <= z1:
                ax.axvline(st, ls=':', color='k', lw=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ΔF/F0' if USE_DF_OVER_F0 else 'ΔF (baseline-subtracted)')
        ax.set_title(f'Baseline preview: {os.path.basename(xlsx_path)}')
        ax.legend(loc='best')
        plt.tight_layout()
        if save:
            target_dir = save_dir or os.path.join(BATCH_EXPORT_DIR, PLOTS_SUBDIR)
            os.makedirs(target_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(xlsx_path))[0]
            parent_tag = os.path.basename(os.path.dirname(xlsx_path)) or 'root'
            out_png = os.path.join(target_dir, f"baseline_{parent_tag}_{base_name}.png")
            fig.savefig(out_png, dpi=150)
            progress_print(f"Saved baseline preview: {out_png}")
        if show and not save:
            _show_now(fig, pause=0.05)
        elif show and save:
            try:
                fig.show()
            except Exception:
                pass
        if not show:
            plt.close(fig)
    except Exception as e:
        progress_print(f"Baseline preview failed: {e}")


def process_single_file(xlsx_path: str, amp_method="NNLS", fail_method="NNLS",
                         train_start_override: Optional[float] = None,
                         out_excel: Optional[str] = None,
                         train_start_overrides: Optional[Dict[str, float]] = None):
    """Process a single workbook and optionally write a one-row Excel summary.

    Returns the summary row dict (or None if failure / interruption).
    Mirrors the logic used inside batch_measure_complex for consistency.
    """
    try:
        metrics = compute_metrics_for_file(
            xlsx_path,
            train_start_override=train_start_override,
            amp_method=amp_method,
            fail_method=fail_method
        )
        if not metrics:
            progress_print("No metrics returned (possibly interrupted or BLEACH_INTERRUPT).")
            return None
        method_map = {
            "NNLS": "RobustNNLS-corrected",
            "RAW": "Raw-windowedMax",
            "SAVGOL": "SG-windowedMax",
        }
        method_key = method_map.get(str(amp_method).upper())
        if method_key is None:
            progress_print(f"Unknown amp_method '{amp_method}'")
            return None
        df = pd.DataFrame(metrics)
        avg_sel = df[(df.get("method") == method_key) & (df.get("level") == "average-trace")]
        if avg_sel.empty:
            progress_print(f"No average-trace row for method {method_key}")
            return None
        row = {"measurement": amp_method, "ID": format_fiber_id(xlsx_path)}
        for i in range(1, n_pulses+1):
            row[f"AMP{i}"] = avg_sel.iloc[0].get(f"amp_{i}", np.nan)
        amp1 = row.get("AMP1")
        for i in range(2, n_pulses+1):
            key = f"PPR{i}/1"
            row[key] = (
                row[f"AMP{i}"] / amp1
                if (amp1 is not None and not np.isnan(amp1) and amp1 != 0)
                else np.nan
            )
        trial_sel = df[(df.get("method") == method_key) & (df.get("level") == "trial")]
        # Per-pulse thresholds (1-3) usage; if threshold missing, leave NaN
        for i in range(1, min(3, n_pulses)+1):
            amp_vals = pd.to_numeric(trial_sel.get(f"amp_{i}"), errors="coerce")
            thr_col = f"thr_max_amp{i}"
            if thr_col in trial_sel.columns:
                thr_vals = pd.to_numeric(trial_sel.get(thr_col), errors="coerce")
                valid = (~amp_vals.isna()) & (~thr_vals.isna())
                row[f"%Fail{i}"] = (float(np.mean(amp_vals[valid] <= thr_vals[valid])) * 100.0) if valid.any() else np.nan
                if SHOW_PROGRESS and valid.any():
                    try:
                        n_valid = int(valid.sum()); n_fail = int((amp_vals[valid] <= thr_vals[valid]).sum())
                        progress_print(f"  [Single %Fail debug] pulse{i}: fails={n_fail}/{n_valid} (thr{i})")
                    except Exception:
                        pass
            else:
                row[f"%Fail{i}"] = np.nan
        progress_print("Single-file summary (key metrics):")
        for k in sorted(row.keys()):
            if k.startswith(("AMP", "PPR", "%Fail")):
                progress_print(f"  {k}: {row[k]}")
        if out_excel:
            df_out = pd.DataFrame([row])
            with pd.ExcelWriter(out_excel) as w:
                df_out.to_excel(w, sheet_name="SingleFile", index=False)
            progress_print(f"Wrote single-file Excel: {out_excel}")
        return row
    except Exception as e:
        progress_print(f"Single file processing failed: {e}")
        return None


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch or single-file pulse analysis")
    parser.add_argument("--single", help="Process only this .xlsx file (skip batch)")
    parser.add_argument("--amp-method", default=AMP_MEASUREMENT_METHOD,
                        help="Amplitude method (NNLS, RAW, SAVGOL)")
    parser.add_argument("--fail-method", default=FAILURE_MEASUREMENT_METHOD,
                        help="Failure method (NNLS, SAVGOL)")
    parser.add_argument("--out", help="Optional output Excel for single-file summary")
    parser.add_argument("--train-start", type=float,
                        help="Override train_start (seconds) for single file")
    parser.add_argument("--no-show", action="store_true", help="Disable showing interactive plots")
    parser.add_argument("--save-plots", action="store_true", help="Save plots (PNG) during this run")
    parser.add_argument("--no-save", action="store_true", help="Do not save plots even if SAVE_PLOTS was True in file")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline preview (no fitting)")
    parser.add_argument("--plot-mode", choices=['replace','keep','none'], help="Figure display mode: replace, keep, none")
    args, unknown = parser.parse_known_args()

    # Runtime flag overrides
    if args.no_show:
        SHOW_PLOTS_DURING_BATCH = False
    if args.save_plots:
        SAVE_PLOTS = True
    if args.no_save:
        SAVE_PLOTS = False
    if getattr(args, 'plot_mode', None):
        PLOT_MODE = args.plot_mode
        if PLOT_MODE == 'none':
            SHOW_PLOTS_DURING_BATCH = False

    if args.single:
        progress_print(f"Running single-file mode on {args.single}")
        if args.baseline_only:
            baseline_preview(args.single, train_start_override=args.train_start,
                             show=SHOW_PLOTS_DURING_BATCH, save=SAVE_PLOTS)
        else:
            process_single_file(
                args.single,
                amp_method=args.amp_method,
                fail_method=args.fail_method,
                train_start_override=args.train_start,
                out_excel=args.out
            )
        if KEEP_FIGS_OPEN_ON_FINISH and SHOW_PLOTS_DURING_BATCH and not SAVE_PLOTS:
            try: plt.ioff()
            except Exception: pass
            print("Processing complete. Close figures to finish.")
            try: plt.show()
            except Exception: pass
    else:
        if RUN_BATCH_EXPORT:
            try:
                run_batch_export(gui=USE_GUI,
                                 amp_method=args.amp_method,
                                 fail_method=args.fail_method)
            except KeyboardInterrupt:
                progress_print("Batch export aborted by user")
            except Exception as e:
                progress_print(f"Batch export failed: {e}")
            finally:
                if KEEP_FIGS_OPEN_ON_FINISH and SHOW_PLOTS_DURING_BATCH and not SAVE_PLOTS:
                    try: plt.ioff()
                    except Exception: pass
                    print("Batch complete. Close figures to finish.")
                    try:
                        plt.show()
                    except Exception:
                        pass
