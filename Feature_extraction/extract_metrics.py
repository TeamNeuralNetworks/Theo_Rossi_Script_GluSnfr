"""Clear, streamlined iGluSnFR train analysis

This module provides a compact, publication‑friendly API that is
mathematically equivalent to the main pipeline. It extracts per‑pulse
amplitudes and PPRs from iGluSnFR trains with minimal moving parts and a
simple options dictionary for configuration and plotting.

Key steps performed (mirrors batch_measure_complex):
  1) Interpolate NaNs and (optionally) correct slow bleaching
  2) Baseline to ΔF/F0 using the median over the full pre‑train interval
  3) Estimate kinetics from the average trace (rise τr; per‑pulse decay τd)
  4) Robust NNLS per‑pulse amplitudes with micro‑shifts (backward, no overlap)
  5) Per‑pulse amplitudes via local averaged max around each stimulus
  6) Null amplitudes and a MAD‑based threshold/p‑value for A1

Only the pieces necessary for this workflow are implemented here, using the
same formulas and defaults as the original pipeline for equivalence.
"""

from typing import Optional, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls

try:
    from smoothing import (
            fill_nans_timewise,
            sg_smooth,
            iglusnfr_kernel,
            time_zoom_mask,
            windowed_max,
            pick_peak_on_series,
            compute_no_signal_mask,
    )
except Exception:
    # Fallback: allow importing when current working dir is this subfolder
    import os, sys
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from smoothing import (
        fill_nans_timewise,
        sg_smooth,
        iglusnfr_kernel,
        time_zoom_mask,
        windowed_max,
        pick_peak_on_series,
        compute_no_signal_mask,
    )


"""
Default parameters consolidated into a single dictionary for clarity.
These values match the original pipeline and are applied consistently.
"""
DEFAULTS = {
    # Smoothing
    'sg_window': 9,
    'sg_poly': 2,
    # Train fit/plot window
    'pre_zoom_s': 0.15,
    'post_zoom_s': 0.60,
    # Peak window (ms)
    'peak_window_ms': 25.0,
    'peak_avg_points': 5,
    'pre_peak_ms': 0.0,
    # Baseline and null sampling
    'f0_window_s': 0.4,
    'null_sim_max_points': 1000,
    'null_min_post_zoom_s': 0.05,
    'null_N': 3.0,  # MAD rule multiplier
    # Kinetics grids (ms)
    'kin_taur_grid_ms': [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
    'kin_taud0_grid_ms': [1.6, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0],
    'kin_slope_grid_ms': [0.0, 0.25, 0.5, 1.0, 2.0],
    # Robust NNLS + micro‑shift
    'huber_delta': 5.5,
    'irls_iters': 6,
    'delta_max_s': 2.0 / 1000.0,
    'delta_step_s': 0.25 / 1000.0,
    'shift_min_s': 0.05 / 1000.0,
    # ΔF/F0 baseline safety
    'f0_eps': 1e-9,
    # Bleach correction
    'bleach_huber_delta': 3.0,
    'bleach_tau_range_factor': (0.25, 4.0),
    'bleach_n_tau': 25,
    # Event model (kernel) used for per-pulse fitting
    # Supported: 'double_exp' (difference-of-exponentials), 'cooperative'
    'event_model': 'cooperative',
    # Cooperative exponent n
    'coop_n': 2.0,
    # Measurement and thresholds
    # measurement: which amplitude series to use for p-values/classification
    #   'NNLS' | 'SAVGOL' | 'RAW'
    'measurement': 'NNLS',
    # threshold_mode: 'auto' selects 'MAD' for NNLS and 'SD' for SAVGOL;
    # can be forced to 'mad' or 'sd'
    'threshold_mode': 'auto',
    # failure classification method: 'NNLS' | 'SAVGOL' | 'RAW'
    # default None means "use measurement"
    'fail_method': None,
    # Toggle per-pulse micro-shifts during fitting and null sampling
    'allow_shift': True,
    # Auto-select event model from multi-trial average
    'auto_event_model': True,
    # Per-pulse behavior: 'vary_shape' allows τ (or shape) to vary across pulses;
    # 'amplitude_only' locks shape across pulses and fits only amplitudes with auto offset.
    'per_pulse_mode': 'vary_shape',  # 'vary_shape' | 'amplitude_only'
}

# Selected kernel (set inside extract_metrics based on options; default is iglusnfr_kernel)
_KERNEL_FUN = iglusnfr_kernel


# -------------------------
# Small utilities
# -------------------------

def _nnls_irls_singlecol(y: np.ndarray, k: np.ndarray, *, robust: bool, huber_delta: float, iters: int) -> float:
    """Single‑column NNLS with optional Huber IRLS (amplitude ≥ 0)."""
    a = max(0.0, nnls(k[:, None], y)[0][0])
    if not robust:
        return a
    for _ in range(max(1, int(iters))):
        r = y - a * k
        absr = np.abs(r)
        w = np.where(absr <= huber_delta, 1.0, huber_delta / np.maximum(absr, 1e-12))
        Wsqrt = np.sqrt(w)
        kw = k * Wsqrt
        yw = y * Wsqrt
        a = max(0.0, nnls(kw[:, None], yw)[0][0])
    return float(a)


def _fit_single_pulse_amp(
    y: np.ndarray,
    t: np.ndarray,
    stim_time: float,
    tau_r_s: float,
    tau_d_s: float,
    *,
    pre_zoom_s: float,
    post_zoom_s: float,
    robust: bool = True,
    huber_delta: float,
    irls_iters: int,
    allow_shift: bool = True,
    delta_max_s: float,
    delta_step_s: float,
    shift_min_s: float,
) -> Tuple[float, float]:
    """Estimate a single‑pulse amplitude at `stim_time` with optional micro‑shift.

    Returns (amplitude, best_shift_s). Amplitude is read by robust NNLS against
    the kernel segment in the local window [stim − pre, stim + post].
    """
    local_mask = (t >= (stim_time - pre_zoom_s)) & (t <= (stim_time + post_zoom_s))
    if not np.any(local_mask):
        return 0.0, shift_min_s
    y_seg = y[local_mask]
    best_a, best_d = 0.0, shift_min_s
    shifts = (np.arange(shift_min_s, delta_max_s + 1e-12, delta_step_s)
              if allow_shift else np.array([shift_min_s]))
    for d in shifts:
        k_full = _KERNEL_FUN(t - (stim_time + d), tau_r_s, tau_d_s)
        k_loc = k_full[local_mask]
        if k_loc.size < 3 or np.all(k_loc == 0):
            continue
        a_loc = _nnls_irls_singlecol(y_seg, k_loc, robust=robust, huber_delta=huber_delta, iters=irls_iters)
        if a_loc > best_a:
            best_a, best_d = a_loc, d
    return float(best_a), float(best_d)


def fit_amplitudes_no_overlap_backward(
    y: np.ndarray,
    t: np.ndarray,
    stim_times: np.ndarray,
    tau_r_s: float,
    tau_d_vec_s: np.ndarray,
    *,
    pre_zoom_s: float,
    post_zoom_s: float,
    robust: bool = True,
    huber_delta: float,
    irls_iters: int,
    allow_shift: bool = True,
    delta_max_s: float,
    delta_step_s: float,
    shift_min_s: float,
):
    """Backward, non‑overlap per‑pulse fitting with micro‑shifts.

    For pulse p, fit in [t_p − pre, min(t_{p+1}, t_p + post)] and subtract the
    reconstructed component before moving to the previous pulse. Returns
    (amplitudes, shifts, design, reconstruction).
    """
    n = len(stim_times)
    a = np.zeros(n, float)
    d = np.zeros(n, float)
    residual = y.copy()
    components = []
    for p in reversed(range(n)):
        st = float(stim_times[p])
        next_st = float(stim_times[p+1]) if p < n - 1 else None
        max_end = min(st + post_zoom_s, next_st) if next_st is not None else (st + post_zoom_s)
        avail_post = max(0.0, max_end - st)
        if avail_post < 1e-6:
            components.append((p, np.zeros_like(y)))
            continue
        a_p, d_p = _fit_single_pulse_amp(
            residual, t, st, tau_r_s, float(tau_d_vec_s[p]),
            pre_zoom_s=pre_zoom_s, post_zoom_s=avail_post,
            robust=robust, huber_delta=huber_delta, irls_iters=irls_iters,
            allow_shift=allow_shift, delta_max_s=delta_max_s, delta_step_s=delta_step_s,
            shift_min_s=shift_min_s,
        )
        a[p] = a_p; d[p] = d_p
        k = _KERNEL_FUN(t - (st + d_p), tau_r_s, float(tau_d_vec_s[p]))
        comp = a_p * k
        residual = residual - comp
        components.append((p, comp))
    components.sort(key=lambda x: x[0])
    yhat = np.sum([c for _, c in components], axis=0) if components else np.zeros_like(y)
    X = np.column_stack([
        _KERNEL_FUN(t - (float(stim_times[p]) + d[p]), tau_r_s, float(tau_d_vec_s[p]))
        for p in range(n)
    ]) if n else np.zeros((t.size, 0))
    return a, d, X, yhat


def compute_localmax_corrected_amps(
    t: np.ndarray,
    y: np.ndarray,
    stim_times: np.ndarray,
    win_ms: float,
    n_avg: int,
    pre_ms: float,
    a_vec: np.ndarray,
    d_vec: np.ndarray,
    tau_r_s: float,
    tau_d_vec_s: np.ndarray,
):
    """Local averaged max around each stimulus, minus previous event residual."""
    if y is None or t.size == 0 or np.size(y) == 0:
        return np.zeros(len(stim_times), float)
    amps = []
    for p, st in enumerate(stim_times):
        v = windowed_max(t, y, [st], win_ms, int(n_avg), pre_ms)
        amp_p = float(v[0]) if np.size(v) else 0.0
        tp, _ = pick_peak_on_series(t, y, st, win_ms, pre_ms)
        if (
            p > 0 and a_vec is not None and d_vec is not None and tau_d_vec_s is not None
            and len(a_vec) > (p - 1) and len(d_vec) > (p - 1) and len(tau_d_vec_s) > (p - 1)
        ):
            st_prev = float(stim_times[p - 1])
            td_prev = float(tau_d_vec_s[p - 1])
            sh_prev = float(d_vec[p - 1])
            a_prev = float(a_vec[p - 1])
            prev = a_prev * _KERNEL_FUN(t - (st_prev + sh_prev), tau_r_s, td_prev)
            prev_at_tp = float(np.interp(tp, t, prev)) if np.isfinite(tp) else 0.0
            amp_p -= prev_at_tp
        amps.append(amp_p)
    return np.asarray(amps, float)


def sample_null_amplitudes_consistent(
    y: np.ndarray,
    t: np.ndarray,
    baseline_mask: np.ndarray,
    tau_r_s: float,
    tau_d_s: float,
    *,
    train_start: float,
    f0_window_s: float,
    pre_zoom_s: float,
    post_zoom_s: float,
    robust: bool = True,
    huber_delta: float,
    irls_iters: int,
    allow_shift: bool = True,
    delta_max_s: float,
    delta_step_s: float,
    null_min_post_zoom_s: float,
    null_sim_max_points: int,
    peak_window_ms: float,
    peak_avg_points: int,
    pre_peak_ms: float,
    shift_min_s: float,
    n_samples: int = 1000,
    seed: int = 0,
):
    """Null distribution for A1 using the same single‑pulse estimator.

    Simulated start times lie in the final f0_window_s before train_start and
    are thinned to at most null_sim_max_points.
    """
    idx = np.flatnonzero(baseline_mask)
    if idx.size < 10:
        return np.array([])
    baseline_start = t[idx[0]]
    baseline_end = t[idx[-1]]
    null_start = max(baseline_start, train_start - f0_window_s)
    null_end = min(baseline_end, train_start)
    st_min = null_start + pre_zoom_s
    st_max = null_end - null_min_post_zoom_s
    if st_max <= st_min:
        return np.array([])
    cand_mask = (t >= st_min) & (t <= st_max)
    starts_full = t[cand_mask]
    if starts_full.size == 0:
        return np.array([])
    limit = int(min(int(null_sim_max_points), int(n_samples)))
    if starts_full.size > limit:
        idx = np.linspace(0, starts_full.size - 1, limit).round().astype(int)
        starts = starts_full[idx]
    else:
        starts = starts_full

    amps = []
    for st in starts:
        avail_post = min(post_zoom_s, train_start - st - 1e-6, null_end - st)
        if avail_post < null_min_post_zoom_s:
            continue
        a_hat, d_hat = _fit_single_pulse_amp(
            y, t, float(st), tau_r_s, tau_d_s,
            pre_zoom_s=pre_zoom_s, post_zoom_s=avail_post,
            robust=robust, huber_delta=huber_delta, irls_iters=irls_iters,
            allow_shift=allow_shift, delta_max_s=delta_max_s, delta_step_s=delta_step_s,
            shift_min_s=shift_min_s,
        )
        t_fit = t[(t >= st - pre_zoom_s) & (t <= st + avail_post)]
        if t_fit.size:
            k_fit = _KERNEL_FUN(t_fit - (st + d_hat), tau_r_s, tau_d_s)
            y_evt = a_hat * k_fit
            val = windowed_max(t_fit, y_evt, [st], peak_window_ms, peak_avg_points, pre_peak_ms)
            amps.append(float(val[0]) if np.size(val) else 0.0)
    return np.asarray(amps, float)


def baseline_threshold_and_pval(null_amps: np.ndarray, N: float, mode: str = "mad"):
    """Return (threshold, pval_fn) under a single rule.

    mode:
      - 'sd'  => mean(null)   + N * std(null)
      - 'mad' => median(null) + N * (1.4826 * MAD(null))
    pval_fn(x) = Pr(null >= x) with +1 smoothing.
    """
    if null_amps is None or np.size(null_amps) == 0 or not np.isfinite(N):
        return np.nan, (lambda x: np.nan)
    a = np.asarray(null_amps, float)
    mode_u = (mode or '').strip().lower()
    if mode_u == 'sd':
        mu = float(np.nanmean(a))
        sd = float(np.nanstd(a))
        thr = float(mu + float(N) * sd) if np.isfinite(sd) else np.nan
    else:
        med = float(np.nanmedian(a))
        mad = float(np.nanmedian(np.abs(a - med)))
        sigma_hat = 1.4826 * mad
        thr = float(med + float(N) * sigma_hat) if np.isfinite(sigma_hat) else np.nan
    def pval(x):
        return float((np.sum(a >= x) + 1) / (a.size + 1))
    return thr, pval


# -------------------------
# Bleach correction (robust mono‑exp)
# -------------------------

def _fit_monoexp_robust(
    t: np.ndarray,
    y: np.ndarray,
    mask_fit: np.ndarray,
    *,
    n_iter: int = 4,
    huber_delta: float,
    tau_range_factor,
    n_tau: int,
):
    """Robust fit of A + B exp(−t/τ) on masked samples; returns (trend, tau_best)."""
    tt = np.asarray(t, float)
    yy = np.asarray(y, float)
    m = mask_fit & np.isfinite(yy)
    if m.sum() < 10:
        return np.zeros_like(yy), np.nan
    t_fit = tt[m]
    y_fit = yy[m]
    dur = tt[-1] - tt[0]
    tmin = max(1e-6, tau_range_factor[0] * dur)
    tmax = max(tmin * 1.01, tau_range_factor[1] * dur)
    tau_grid = np.geomspace(tmin, tmax, int(n_tau))
    best = (np.inf, None, None)
    for tau in tau_grid:
        e = np.exp(-t_fit / tau)
        X = np.column_stack([np.ones_like(e), e])
        w = np.ones_like(e)
        for _ in range(max(1, int(n_iter))):
            w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
            Wsqrt = np.sqrt(w)
            Xw = X * Wsqrt[:, None]
            yw = y_fit * Wsqrt
            coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
            resid = y_fit - (coef[0] + coef[1] * e)
            absr = np.abs(resid)
            w = np.where(absr <= huber_delta, 1.0, huber_delta / np.maximum(absr, 1e-12))
        r2 = float(np.sum(resid**2))
        if r2 < best[0]:
            best = (r2, coef, tau)
    if best[1] is None:
        return np.zeros_like(yy), np.nan
    A, B = best[1]
    tau_best = float(best[2])
    trend = A + B * np.exp(-tt / tau_best)
    return trend, tau_best


def apply_bleach_correction(
    t: np.ndarray,
    y: np.ndarray,
    train_start_s: float,
    stim_times: np.ndarray,
    *,
    post_zoom_s: float,
    peak_window_ms: float,
    pre_peak_ms: float,
    huber_delta: float,
    tau_range_factor,
    n_tau: int,
) -> np.ndarray:
    """Correct slow bleaching by subtracting a robust mono‑exp trend.

    Fit uses pre‑train baseline and (if available) post‑train quiet region
    detected via compute_no_signal_mask to avoid stimulus‑evoked epochs.
    """
    try:
        # Fit on regions without evoked signal
        mask_quiet = compute_no_signal_mask(
            t, y, stim_times, post_zoom_s, peak_win_ms=peak_window_ms, peak_search_pre_ms=pre_peak_ms
        )
        trend, tau = _fit_monoexp_robust(
            t, y, mask_quiet, n_iter=4,
            huber_delta=huber_delta, tau_range_factor=tau_range_factor, n_tau=n_tau
        )
        corrected = y - trend + float(np.nanmedian(y[t < train_start_s]))
        return corrected
    except Exception:
        return y.copy()


# -------------------------
# Kinetics (fast grid on average)
# -------------------------

def estimate_kinetics_from_average(
    t: np.ndarray,
    y_avg: np.ndarray,
    stim_times: np.ndarray,
    *,
    taur_grid_ms,
    taud0_grid_ms,
    slope_grid_ms,
    pre_zoom_s: float,
    post_zoom_s: float,
) -> Tuple[float, float, float, np.ndarray]:
    """Grid search τr, τd0, slope on the average trace (zoomed window)."""
    tau_r_grid = np.array(taur_grid_ms, float) / 1000.0
    tau_d0_grid = np.array(taud0_grid_ms, float) / 1000.0
    slope_grid = np.array(slope_grid_ms, float) / 1000.0
    isi_guess = float(stim_times[1] - stim_times[0]) if len(stim_times) > 1 else 0.05
    zmask, _, _ = time_zoom_mask(t, float(stim_times[0]), isi_guess, len(stim_times), pre_zoom_s, post_zoom_s)

    def obj_for(tau_r, tau_d_vec):
        X = np.column_stack([_KERNEL_FUN(t - st, tau_r, td) for st, td in zip(stim_times, tau_d_vec)])
        a = np.maximum(0.0, nnls(X[zmask, :], y_avg[zmask])[0]) if X.size else np.zeros(len(stim_times))
        r = y_avg - X @ a
        return float(np.dot(r[zmask], r[zmask]) / max(1, zmask.sum()))

    best = (np.inf, 0.002, 0.006, 0.0)
    for tau_r in tau_r_grid:
        for tau_d0 in tau_d0_grid:
            for slope in slope_grid:
                tau_d_vec = tau_d0 + slope * np.arange(len(stim_times))
                if np.any(tau_d_vec <= tau_r + 0.0002):
                    val = 1e9
                else:
                    val = obj_for(tau_r, tau_d_vec)
                if val < best[0]:
                    best = (val, tau_r, tau_d0, slope)
    _, tau_r_fit, tau_d0_fit, slope_fit = best
    tau_d_vec = tau_d0_fit + slope_fit * np.arange(len(stim_times))
    # Always enforce non‑decreasing τd across pulses (simpler and more stable)
    tau_d_vec = np.maximum.accumulate(tau_d_vec)
    return float(tau_r_fit), float(tau_d0_fit), float(slope_fit), np.asarray(tau_d_vec, float)


# -------------------------
# Public API
# -------------------------

def extract_metrics(
    time: np.ndarray,
    trials: np.ndarray,
    train_start: float,
    isi: float,
    n_pulses: int,
    options: Optional[Dict] = None,
) -> Dict:
    """Extract per‑pulse metrics with a minimal, explicit interface.

    Options dictionary keys (all optional):
      - bleach: bool (default True) — apply mono‑exp bleaching correction
      - normalize_dff: bool (default True) — work in ΔF/F0
      - sg_window: int (default 9)
      - sg_poly: int (default 2)
      - peak_window_ms: float (default 25.0)
      - peak_avg_points: int (default 5)
      - pre_peak_ms: float (default 0.0)
      - null_N: float (default 3.0) — threshold multiplier
      - measurement: {'NNLS'|'SAVGOL'|'RAW'} (default 'NNLS') — which amplitudes are
        used for p‑values/classification
      - threshold_mode: {'auto'|'mad'|'sd'} (default 'auto') — auto = MAD for NNLS,
        SD for SAVGOL
      - share_thr_1to3: bool (default True) — reuse A1 threshold for pulses 2 and 3
      - allow_shift: bool (default True) — enable per‑pulse micro‑shifts
      - event_model: {'double_exp'|'cooperative'} (default 'cooperative') — template used
        for NNLS fitting and residual subtraction; 'cooperative' uses a Hill‑like rise*exp decay
      - coop_n: float (default 2.0) — cooperative exponent for the cooperative model
      - plot: dict with keys
          - enabled: bool (default False)
          - traces: list of {'raw','savgol','nnls'} (default ['nnls'])
          - show_decay: bool (default True)
          - trials: bool (default False) — also plot each trial with its fit
          - baseline: bool (default False) — for each trial, plot a two-panel
          figure with baseline fits + noise histogram and the train; forces
          trials=True when enabled
    """
    # Parse options (merge into a single config dict)
    opts = options.copy() if isinstance(options, dict) else {}
    plot_opts = opts.get('plot', {}) if isinstance(opts.get('plot', {}), dict) else {}
    want_plot = bool(plot_opts.get('enabled', False))
    traces = list(plot_opts.get('traces', ['nnls']))
    show_decay = bool(plot_opts.get('show_decay', True))
    plot_trials = bool(plot_opts.get('trials', False))
    baseline_figs = bool(plot_opts.get('baseline', False))
    if baseline_figs:
        plot_trials = True  # baseline panel requires per-trial figures
    cfg = {**DEFAULTS, **{k: v for k, v in opts.items() if k != 'plot'}}
    # Optional auto-calibration of event model from multi-trial data (run after preprocessing)
    do_bleach = bool(cfg.get('bleach', True))
    use_dff = bool(cfg.get('normalize_dff', True))
    sgW = int(cfg['sg_window']); sgP = int(cfg['sg_poly'])
    win_ms = float(cfg['peak_window_ms']); n_avg = int(cfg['peak_avg_points']); pre_ms = float(cfg['pre_peak_ms'])
    null_N = float(cfg['null_N'])
    meas = str(cfg.get('measurement', 'NNLS')).strip().upper()
    failm = str((cfg.get('fail_method') or meas)).strip().upper()
    thr_mode = str(cfg.get('threshold_mode', 'auto')).strip().lower()
    allow_shift = bool(cfg.get('allow_shift', True))

    # Shapes & schedule
    t = np.asarray(time, float).reshape(-1)
    Y = np.asarray(trials, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.shape[0] != t.size:
        raise ValueError("trials must have same number of samples as time")
    stim_times = float(train_start) + float(isi) * np.arange(int(n_pulses))

    # Preprocess trials: interpolate NaNs, optional bleach, then ΔF/F0 baseline
    baseline_mask = (t < float(train_start))
    # Fallback: if too few baseline points, use earliest 10% of the trace
    if baseline_mask.sum() < 5:
        n10 = max(1, int(0.1 * len(t)))
        baseline_mask = np.zeros_like(t, dtype=bool)
        baseline_mask[:n10] = True
    Yc = np.zeros_like(Y)
    for j in range(Y.shape[1]):
        yj = fill_nans_timewise(Y[:, j], t)
        if do_bleach:
            yj = apply_bleach_correction(
                t, yj, float(train_start), stim_times,
                post_zoom_s=cfg['post_zoom_s'], peak_window_ms=cfg['peak_window_ms'], pre_peak_ms=cfg['pre_peak_ms'],
                huber_delta=cfg['bleach_huber_delta'], tau_range_factor=cfg['bleach_tau_range_factor'], n_tau=cfg['bleach_n_tau']
            )
        Yc[:, j] = yj
    # F0 per trial over full pre‑train baseline
    F0 = np.zeros(Yc.shape[1])
    for j in range(Yc.shape[1]):
        b = Yc[baseline_mask, j]
        b = b[np.isfinite(b)]
        F0[j] = np.nanmedian(b) if b.size else 0.0
    if use_dff:
        safe_F0 = np.where(np.abs(F0) < cfg['f0_eps'], np.nan, F0)
        Yd_norm = (Yc - F0) / safe_F0
        # If a trial has invalid F0 (nan/zero) and becomes all-NaN, fall back to subtract-only for that trial
        bad_cols = ~np.isfinite(Yd_norm).any(axis=0)
        Yd = Yd_norm.copy()
        if np.any(bad_cols):
            Yd[:, bad_cols] = (Yc[:, bad_cols] - F0[bad_cols])
    else:
        Yd = Yc - F0

    # Sanitize: fill any remaining NaNs per trial to avoid failures in NNLS/curve_fit
    for j in range(Yd.shape[1]):
        if not np.all(np.isfinite(Yd[:, j])):
            Yd[:, j] = fill_nans_timewise(Yd[:, j], t)

    # Configure kernel function for the chosen event model (τ‑varying or fixed template)
    event_model = str(cfg.get('event_model', 'cooperative')).strip().lower()
    coop_n_default = float(cfg.get('coop_n', 2.0))
    em_settings = cfg.get('event_model_settings', {})
    if em_settings is None:
        em_settings = {}
    if not isinstance(em_settings, dict):
        raise ValueError("event_model_settings must be a dict of parameter overrides")

    def _build_kernel_from_library(name: str):
        """Return (kernel_fun, spec) using Model_Calibration.event_models."""
        try:
            try:
                from Model_Calibration.event_models import get_event_model
            except Exception:
                from event_models import get_event_model  # type: ignore
            spec = get_event_model(name)
        except Exception as e:
            raise ValueError(f"Unknown library model '{name}': {e}")
        def _kernel_from_params(params_fixed: Dict[str, float]):
            # returns kernel(dt_s, tau_r, tau_d) ignoring tau_r/tau_d (fixed template)
            def kfun(dt_s: np.ndarray, tau_r: float, tau_d: float) -> np.ndarray:
                dt_s = np.asarray(dt_s, float)
                dt_ms = dt_s * 1000.0
                pars = []
                for p in spec['params']:
                    if p == 'amp':
                        pars.append(1.0)
                    elif p == 't_peak':
                        pars.append(0.0)
                    else:
                        pars.append(params_fixed[p])
                y = spec['func'](dt_ms, *pars)
                tp = np.maximum(dt_s, 0.0)
                support = tp <= 0.2
                area = np.trapz(y[support], dt_s[support]) if np.any(support) else 1.0
                return y / max(area, 1e-12)
            return kfun
        def _kernel_tau_varying(mapper):
            # mapper builds full param vector from tau_r,tau_d
            def kfun(dt_s: np.ndarray, tau_r: float, tau_d: float) -> np.ndarray:
                dt_s = np.asarray(dt_s, float)
                dt_ms = dt_s * 1000.0
                pars = mapper(float(tau_r), float(tau_d))
                y = spec['func'](dt_ms, *pars)
                tp = np.maximum(dt_s, 0.0)
                support = tp <= 0.2
                area = np.trapz(y[support], dt_s[support]) if np.any(support) else 1.0
                return y / max(area, 1e-12)
            return kfun
        return spec, _kernel_from_params, _kernel_tau_varying

    global _KERNEL_FUN
    # τ‑varying supported directly
    varying_supported = {'double_exp', 'cooperative', 'bilinear'}
    if event_model in varying_supported:
        spec, _, make_var = _build_kernel_from_library(event_model if event_model != 'double_exp' else 'double_exp')
        # Validate settings
        if em_settings:
            allowed = set(spec['params']) - {'amp', 't_peak'}
            # For τ‑varying cooperative allow only n_coop override; for others no overrides
            if event_model == 'cooperative':
                extra_keys = set(em_settings.keys()) - {'n_coop'}
                if extra_keys:
                    raise ValueError(f"Unsupported event_model_settings for cooperative: {sorted(extra_keys)}")
                n_coop = float(em_settings.get('n_coop', coop_n_default))
                def _map(tau_r, tau_d):
                    return [1.0, float(tau_r), float(tau_d), float(n_coop), 0.0]
                _KERNEL_FUN = make_var(_map)
            elif event_model == 'double_exp':
                if em_settings:
                    raise ValueError("double_exp in τ‑varying mode does not accept event_model_settings")
                def _map(tau_r, tau_d):
                    return [1.0, float(tau_r), float(tau_d), 0.0]
                _KERNEL_FUN = make_var(_map)
            elif event_model == 'bilinear':
                if em_settings:
                    raise ValueError("bilinear in τ‑varying mode does not accept event_model_settings")
                # bilinear expects t_rise(ms), t_decay(ms)
                def _map(tau_r, tau_d):
                    return [1.0, float(tau_r) * 1000.0, float(tau_d) * 1000.0, 0.0]
                _KERNEL_FUN = make_var(_map)
        else:
            if event_model == 'cooperative':
                def _map(tau_r, tau_d):
                    return [1.0, float(tau_r), float(tau_d), float(coop_n_default), 0.0]
                _KERNEL_FUN = make_var(_map)
            elif event_model == 'double_exp':
                def _map(tau_r, tau_d):
                    return [1.0, float(tau_r), float(tau_d), 0.0]
                _KERNEL_FUN = make_var(_map)
            else:  # bilinear
                def _map(tau_r, tau_d):
                    return [1.0, float(tau_r) * 1000.0, float(tau_d) * 1000.0, 0.0]
                _KERNEL_FUN = make_var(_map)
    else:
        # Fixed-template path via 'library:<name>' or direct library name
        lib_name = event_model
        if lib_name.startswith('library:'):
            lib_name = lib_name.split(':', 1)[1].strip().lower()
        # Build kernel from fitted or user-provided shape params
        spec, make_fixed, _ = _build_kernel_from_library(lib_name)
        # Validate overrides against spec
        allowed = set(spec['params']) - {'amp', 't_peak'}
        unknown = set(em_settings.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown event_model_settings for '{lib_name}': {sorted(unknown)}. Allowed keys: {sorted(allowed)}")
        # Fit missing params on the average event
        try:
            # Fit in 0..30 ms relative to train start
            t_ms = (t - float(train_start)) * 1000.0
            m0 = (t_ms >= 0.0) & (t_ms <= 30.0)
            tf = t_ms[m0]
            yf = np.nanmean(Yd, axis=1)[m0]
            p0 = spec['p0_func'](yf, tf)
            from scipy.optimize import curve_fit as _cf
            popt, _ = _cf(spec['func'], tf, yf, p0=p0, bounds=spec['bounds'], maxfev=3000)
        except Exception:
            popt = p0  # fallback to initializer
        # Build fixed param dict
        params_fixed: Dict[str, float] = {}
        for key, val in zip(spec['params'], popt):
            params_fixed[key] = float(val)
        # Apply user overrides
        for k, v in em_settings.items():
            params_fixed[k] = float(v)
        # Remove amplitude and t_peak (we enforce amp=1, t_peak=0 in kernel)
        params_fixed.pop('amp', None)
        params_fixed.pop('t_peak', None)
        _KERNEL_FUN = make_fixed(params_fixed)
        progress_print(f"[info] Using fixed-template matching for model '{lib_name}' (no per-pulse decay progression).", show=True)

    # Optional: run auto-selection now that Yd is preprocessed and finite
    if bool(cfg.get('auto_event_model', True)):
        try:
            try:
                from Model_Calibration.auto_model_settings import auto_select_event_model_settings
            except Exception:
                from auto_model_settings import auto_select_event_model_settings  # type: ignore
            auto = auto_select_event_model_settings(
                t, Yd, train_start=float(train_start), isi=float(isi), n_pulses=int(n_pulses),
                candidates=("double_exp","cooperative"), window_ms=(0.0, 30.0)
            )
            rec = auto.get('options', {})
            for k in ('event_model','coop_n'):
                if k in rec:
                    cfg[k] = rec[k]
        except Exception:
            pass

    # Average trace and kinetics
    y_avg = np.nanmean(Yd, axis=1)
    tau_r, tau_d0, slope, tau_d_vec = estimate_kinetics_from_average(
        t, y_avg, stim_times,
        taur_grid_ms=cfg['kin_taur_grid_ms'], taud0_grid_ms=cfg['kin_taud0_grid_ms'],
        slope_grid_ms=cfg['kin_slope_grid_ms'], pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s']
    )

    # Fit average trace (backward, no overlap) and measure amplitudes
    a_avg, d_avg, X_avg, yhat_avg = fit_amplitudes_no_overlap_backward(
        y_avg, t, stim_times, tau_r, tau_d_vec,
        pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
        robust=True, huber_delta=cfg['huber_delta'], irls_iters=cfg['irls_iters'],
        allow_shift=allow_shift, delta_max_s=cfg['delta_max_s'], delta_step_s=cfg['delta_step_s'],
        shift_min_s=cfg['shift_min_s'],
    )
    y_sg_avg = sg_smooth(y_avg, sgW, sgP) if 'savgol' in traces else None

    # If user requests amplitude-only per-pulse behavior, lock shape across pulses
    if str(cfg.get('per_pulse_mode', 'vary_shape')).lower() == 'amplitude_only':
        tau_d_vec = np.full_like(tau_d_vec, float(tau_d0))
        progress_print("[info] Using amplitude-only per-pulse fitting (shape locked; auto offset enabled).", show=True)

    amp_nnls_avg = compute_localmax_corrected_amps(
        t, yhat_avg, stim_times, win_ms, n_avg, pre_ms, a_avg, d_avg, tau_r, tau_d_vec
    )
    def _norm(a):
        a = np.asarray(a, float)
        d = a[0] if a.size else np.nan
        return a / d if np.isfinite(d) and abs(d) > 1e-12 else a * np.nan
    ppr_nnls_avg = _norm(amp_nnls_avg)
    amp_raw_avg = compute_localmax_corrected_amps(t, y_avg, stim_times, win_ms, n_avg, pre_ms, a_avg, d_avg, tau_r, tau_d_vec)
    amp_sg_avg = compute_localmax_corrected_amps(t, y_sg_avg if y_sg_avg is not None else y_avg, stim_times, win_ms, n_avg, pre_ms, a_avg, d_avg, tau_r, tau_d_vec)

    # Per‑trial metrics and null thresholds (MAD rule) for A1
    per_trial: List[Dict] = []
    thr_list: List[float] = []
    pval_list: List[float] = []
    figures_trials = []  # optional per-trial figures
    for j in range(Yd.shape[1]):
        yj = Yd[:, j]
        # Always compute SG-smoothed series; may be used for SAVGOL-based thresholds
        yj_sg = sg_smooth(yj, sgW, sgP)
        a_t, d_t, X_t, yhat_t = fit_amplitudes_no_overlap_backward(
            yj, t, stim_times, tau_r, tau_d_vec,
            pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
            robust=True, huber_delta=cfg['huber_delta'], irls_iters=cfg['irls_iters'],
            allow_shift=allow_shift, delta_max_s=cfg['delta_max_s'], delta_step_s=cfg['delta_step_s'],
            shift_min_s=cfg['shift_min_s'],
        )
        amp_raw = compute_localmax_corrected_amps(t, yj, stim_times, win_ms, n_avg, pre_ms, a_t, d_t, tau_r, tau_d_vec)
        amp_sg = compute_localmax_corrected_amps(t, yj_sg, stim_times, win_ms, n_avg, pre_ms, a_t, d_t, tau_r, tau_d_vec)
        amp_nn = compute_localmax_corrected_amps(t, yhat_t, stim_times, win_ms, n_avg, pre_ms, a_t, d_t, tau_r, tau_d_vec)

        # Choose threshold rule and null amplitude strategy (auto follows fail_method)
        eff_mode = ('sd' if failm == 'SAVGOL' or failm == 'RAW' else 'mad') if thr_mode == 'auto' else thr_mode
        if eff_mode not in ('mad', 'sd'):
            eff_mode = 'mad'

        if failm == 'SAVGOL' and eff_mode == 'sd':
            # Compute null amplitudes directly on SG baseline via windowed maxima
            idx = np.flatnonzero(baseline_mask)
            if idx.size >= 10:
                baseline_start = t[idx[0]]; baseline_end = t[idx[-1]]
                null_start = max(baseline_start, float(train_start) - cfg['f0_window_s'])
                null_end = min(baseline_end, float(train_start))
                st_min = null_start + cfg['pre_zoom_s']
                st_max = null_end - cfg['null_min_post_zoom_s']
                cand = (t >= st_min) & (t <= st_max)
                starts = t[cand]
                if starts.size > int(cfg['null_sim_max_points']):
                    ii = np.linspace(0, starts.size - 1, int(cfg['null_sim_max_points'])).round().astype(int)
                    starts = starts[ii]
                null_amps = windowed_max(t, yj_sg, list(starts), win_ms, n_avg, pre_ms) if starts.size else np.array([])
            else:
                null_amps = np.array([])
        elif failm == 'RAW' and eff_mode == 'sd':
            # SD rule on RAW baseline windowed maxima
            idx = np.flatnonzero(baseline_mask)
            if idx.size >= 10:
                baseline_start = t[idx[0]]; baseline_end = t[idx[-1]]
                null_start = max(baseline_start, float(train_start) - cfg['f0_window_s'])
                null_end = min(baseline_end, float(train_start))
                st_min = null_start + cfg['pre_zoom_s']
                st_max = null_end - cfg['null_min_post_zoom_s']
                cand = (t >= st_min) & (t <= st_max)
                starts = t[cand]
                if starts.size > int(cfg['null_sim_max_points']):
                    ii = np.linspace(0, starts.size - 1, int(cfg['null_sim_max_points'])).round().astype(int)
                    starts = starts[ii]
                null_amps = windowed_max(t, yj, list(starts), win_ms, n_avg, pre_ms) if starts.size else np.array([])
            else:
                null_amps = np.array([])
        else:
            # NNLS-consistent null on pre-train window using the single-pulse estimator
            null_amps = sample_null_amplitudes_consistent(
                yj, t, baseline_mask, tau_r, tau_d0,
                train_start=float(train_start), f0_window_s=cfg['f0_window_s'],
                pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                robust=True, huber_delta=cfg['huber_delta'], irls_iters=cfg['irls_iters'],
                allow_shift=allow_shift, delta_max_s=cfg['delta_max_s'], delta_step_s=cfg['delta_step_s'],
                null_min_post_zoom_s=cfg['null_min_post_zoom_s'], null_sim_max_points=cfg['null_sim_max_points'],
                peak_window_ms=cfg['peak_window_ms'], peak_avg_points=cfg['peak_avg_points'], pre_peak_ms=cfg['pre_peak_ms'],
                shift_min_s=cfg['shift_min_s'], n_samples=1000, seed=10_000 + j,
            )

        thr1, pfun = baseline_threshold_and_pval(null_amps, null_N, mode=eff_mode)

        # Pick amplitude series for p-values/classification
        if meas == 'SAVGOL':
            a_for_p = amp_sg
        elif meas == 'RAW':
            a_for_p = amp_raw
        else:
            a_for_p = amp_nn

        a1 = float(a_for_p[0]) if a_for_p.size else np.nan
        p1 = float(pfun(a1)) if np.isfinite(a1) else np.nan
        thr_list.append(thr1); pval_list.append(p1)
        # Compute p-values for pulses 2 and 3 using the same threshold as A1
        p2 = float(pfun(a_for_p[1])) if (a_for_p.size >= 2 and np.isfinite(a_for_p[1])) else np.nan
        p3 = float(pfun(a_for_p[2])) if (a_for_p.size >= 3 and np.isfinite(a_for_p[2])) else np.nan

        per_trial.append({
            'amp_raw': amp_raw,
            'amp_savgol': amp_sg,
            'amp_nnls': amp_nn,
            'ppr_raw': _norm(amp_raw),
            'ppr_savgol': _norm(amp_sg),
            'ppr_nnls': _norm(amp_nn),
            'a_coeff': a_t,
            'delta_s': d_t,
            'y_proc': yj,
            'yhat': yhat_t,
            'thr_shared': thr1,
            'pval_amp1': p1,
            'pval_amp2': p2,
            'pval_amp3': p3,
        })

        # Optional: per-trial plot
        if want_plot and plot_trials:
            zmask_t, z0, z1 = time_zoom_mask(
                t, float(train_start), float(isi), int(n_pulses), cfg['pre_zoom_s'], cfg['post_zoom_s']
            )
            tz = t[zmask_t]

            if baseline_figs:
                # Two-panel figure: (1) train window; (2) baseline window + null fits + histogram
                fig_t, (ax_train, ax_base) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [2.2, 1.6]})
                # Train panel
                for st in stim_times:
                    ax_train.axvline(st, color='k', linestyle=':', linewidth=0.8, alpha=0.6)
                if 'raw' in traces:
                    ax_train.plot(tz, yj[zmask_t], label='raw', color='0.6')
                if 'savgol' in traces and yj_sg is not None:
                    ax_train.plot(tz, yj_sg[zmask_t], label='savgol', color='tab:green')
                if 'nnls' in traces:
                    ax_train.plot(tz, yhat_t[zmask_t], label='nnls model', color='tab:blue')
                if show_decay and 'nnls' in traces and np.size(a_t):
                    for p in range(1, int(n_pulses)):
                        st_prev = float(stim_times[p - 1])
                        td_prev = float(tau_d_vec[p - 1])
                        sh_prev = float(d_t[p - 1]) if len(d_t) > (p - 1) else 0.0
                        amp_prev = float(a_t[p - 1]) if len(a_t) > (p - 1) else 0.0
                        k_prev = _KERNEL_FUN(tz - (st_prev + sh_prev), tau_r, td_prev)
                        ax_train.plot(tz, amp_prev * k_prev, color='tab:orange', linestyle='--', linewidth=1.0, alpha=0.85)
                # Failure threshold line (thin red dotted)
                if np.isfinite(thr1):
                    ax_train.axhline(thr1, color='red', linestyle=':', linewidth=0.8)
                ax_train.set_xlim(z0, z1)
                ax_train.set_xlabel('Time (s)')
                ax_train.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
                ax_train.legend(loc='upper right', frameon=False)
                ax_train.set_title(f'Trial {j+1}: train window')

                # Baseline panel (pre-train)
                base_mask = (t < float(train_start))
                tb = t[base_mask]
                if tb.size:
                    if 'raw' in traces:
                        ax_base.plot(tb, yj[base_mask], color='0.4', linewidth=1.0, label='baseline')
                    # Overlay a subset of null-fit events in red
                    baseline_start = tb[0]; baseline_end = tb[-1]
                    null_start = max(baseline_start, float(train_start) - cfg['f0_window_s'])
                    null_end = min(baseline_end, float(train_start))
                    st_min = null_start + cfg['pre_zoom_s']
                    st_max = null_end - cfg['null_min_post_zoom_s']
                    cand_mask = (t >= st_min) & (t <= st_max)
                    starts_full = t[cand_mask]
                    if starts_full.size:
                        # Evaluate amplitudes for all candidates, then draw the strongest few for visibility
                        events = []  # (amp, start, shift)
                        for stcand in starts_full:
                            avail_post = min(cfg['post_zoom_s'], float(train_start) - stcand - 1e-6, null_end - stcand)
                            if avail_post < cfg['null_min_post_zoom_s']:
                                continue
                            a_hat_b, d_hat_b = _fit_single_pulse_amp(
                                yj, t, float(stcand), tau_r, tau_d0,
                                pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=avail_post,
                                robust=True, huber_delta=cfg['huber_delta'], irls_iters=cfg['irls_iters'],
                                allow_shift=True, delta_max_s=cfg['delta_max_s'], delta_step_s=cfg['delta_step_s'],
                                shift_min_s=cfg['shift_min_s'],
                            )
                            events.append((float(a_hat_b), float(stcand), float(d_hat_b)))
                        events = [e for e in events if np.isfinite(e[0]) and e[0] > 0]
                        events.sort(key=lambda e: e[0], reverse=True)
                        draw_n = min(20, len(events))
                        for a_hat_b, stcand, d_hat_b in events[:draw_n]:
                            y_evt_b = a_hat_b * _KERNEL_FUN(tb - (stcand + d_hat_b), tau_r, tau_d0)
                            ax_base.plot(tb, y_evt_b, color='red', alpha=0.5, linewidth=1.0)
                    # Inset histogram of null amplitudes with threshold
                    try:
                        ax_in = ax_base.inset_axes([0.65, 0.55, 0.33, 0.4])
                        data = np.asarray(null_amps, float)
                        if data.size:
                            ax_in.hist(data[~np.isnan(data)], bins='fd', color='#c9d4e8', edgecolor='#4f6aa3')
                            if np.isfinite(thr1):
                                ax_in.axvline(thr1, color='red', linestyle='--', linewidth=0.9)
                        ax_in.set_title('noise', fontsize=8)
                        ax_in.tick_params(labelsize=7)
                    except Exception:
                        pass
                ax_base.set_xlabel('Time (s)')
                ax_base.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
                ax_base.set_title('Baseline window + null-fit events')

                # Match Y limits across panels for direct visual comparison
                try:
                    ymin = min(ax_train.get_ylim()[0], ax_base.get_ylim()[0])
                    ymax = max(ax_train.get_ylim()[1], ax_base.get_ylim()[1])
                    ax_train.set_ylim(ymin, ymax)
                    ax_base.set_ylim(ymin, ymax)
                except Exception:
                    pass
            else:
                # Single-panel per-trial figure (train window only)
                fig_t, ax_t = plt.subplots(figsize=(10, 4))
                for st in stim_times:
                    ax_t.axvline(st, color='k', linestyle=':', linewidth=0.8, alpha=0.6)
                if 'raw' in traces:
                    ax_t.plot(tz, yj[zmask_t], label='raw', color='0.6')
                if 'savgol' in traces and yj_sg is not None:
                    ax_t.plot(tz, yj_sg[zmask_t], label='savgol', color='tab:green')
                if 'nnls' in traces:
                    ax_t.plot(tz, yhat_t[zmask_t], label='nnls model', color='tab:blue')
                if show_decay and 'nnls' in traces and np.size(a_t):
                    for p in range(1, int(n_pulses)):
                        st_prev = float(stim_times[p - 1])
                        td_prev = float(tau_d_vec[p - 1])
                        sh_prev = float(d_t[p - 1]) if len(d_t) > (p - 1) else 0.0
                        amp_prev = float(a_t[p - 1]) if len(a_t) > (p - 1) else 0.0
                        k_prev = iglusnfr_kernel(tz - (st_prev + sh_prev), tau_r, td_prev)
                        ax_t.plot(tz, amp_prev * k_prev, color='tab:orange', linestyle='--', linewidth=1.0, alpha=0.85)
                if np.isfinite(thr1):
                    ax_t.axhline(thr1, color='red', linestyle=':', linewidth=0.8)
                ax_t.set_xlim(z0, z1)
                ax_t.set_xlabel('Time (s)')
                ax_t.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
                ax_t.legend(loc='upper right', frameon=False)
                ax_t.set_title(f'Trial {j+1} (selected overlays)')

            try:
                plt.show(block=False); plt.pause(0.01)
            except Exception:
                pass
            figures_trials.append(fig_t)

    # Optional: average plot with a left event-fit panel (0–30 ms) + right main plot
    figure = None
    if want_plot:
        figure = plt.figure(figsize=(12, 5))
        gs = figure.add_gridspec(1, 2, width_ratios=[1, 4], wspace=0.15)
        # Left: aggregated event + model fit (0..30 ms)
        axL = figure.add_subplot(gs[0, 0])
        try:
            t_ms = (t - float(train_start)) * 1000.0
            isi_ms = float(isi) * 1000.0
            min_x = -3.0
            max_x = isi_ms
            m0 = (t_ms >= min_x) & (t_ms < max_x)
            axL.plot(t_ms[m0], y_avg[m0], color='k', lw=1.5, label='Average')
            # Overlay best-fit library model matching current kernel choice
            try:
                try:
                    from Model_Calibration.event_models import get_event_model
                except Exception:
                    from event_models import get_event_model  # type: ignore
                spec = get_event_model(event_model)
                tf = t_ms[m0]; yf = y_avg[m0]
                p0 = spec['p0_func'](yf, tf)
                try:
                    from scipy.optimize import curve_fit as _cf
                    popt, _ = _cf(spec['func'], tf, yf, p0=p0, bounds=spec['bounds'], maxfev=3000)
                except Exception:
                    popt = p0
                yhat_ev = spec['func'](tf, *popt)
                axL.plot(tf, yhat_ev, color='crimson', ls='--', lw=1.8, label=event_model)
                try:
                    txt = ", ".join(f"{n}={v:.3g}" for n, v in zip(spec['params'], popt))
                    axL.text(0.02, 0.02, txt, transform=axL.transAxes, fontsize=7,
                             va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                except Exception:
                    pass
            except Exception:
                pass
            axL.axvline(0.0, color='k', ls=':', alpha=0.5, lw=0.8)
            axL.set_xlim(min_x, max_x)
            axL.set_xlabel('Time (ms)')
            axL.set_ylabel('ΔF/F0' if USE_DF_OVER_F0 else 'ΔF')
            axL.set_title('Event fit (0–30 ms)', fontsize=10)
            try:
                axL.set_aspect('equal', adjustable='box')
            except Exception:
                pass
        except Exception:
            pass

        # Right: main average plot
        ax = figure.add_subplot(gs[0, 1])
        zmask, z0, z1 = time_zoom_mask(t, float(train_start), float(isi), int(n_pulses), cfg['pre_zoom_s'], cfg['post_zoom_s'])
        tz = t[zmask]
        for st in stim_times:
            ax.axvline(st, color='k', linestyle=':', linewidth=0.8, alpha=0.6)
        if 'raw' in traces:
            ax.plot(tz, y_avg[zmask], label='raw', color='0.6')
        if 'savgol' in traces and y_sg_avg is not None:
            ax.plot(tz, y_sg_avg[zmask], label='savgol', color='tab:green')
        if 'nnls' in traces:
            ax.plot(tz, yhat_avg[zmask], label='nnls model', color='tab:blue')
        if show_decay and 'nnls' in traces and np.size(a_avg):
            for p in range(1, int(n_pulses)):
                st_prev = float(stim_times[p - 1])
                td_prev = float(tau_d_vec[p - 1])
                sh_prev = float(d_avg[p - 1])
                amp_prev = float(a_avg[p - 1])
                k_prev = _KERNEL_FUN(tz - (st_prev + sh_prev), tau_r, td_prev)
                ax.plot(tz, amp_prev * k_prev, color='tab:orange', linestyle='--', linewidth=1.0, alpha=0.85)
        ax.set_xlim(z0, z1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
        ax.legend(loc='upper right', frameon=False)
        ax.set_title('Average trace (selected overlays)')
        try:
            plt.show(block=False); plt.pause(0.05)
        except Exception:
            pass

    return {
        'tau_r_s': float(tau_r),
        'tau_d_s': np.asarray(tau_d_vec, float),
        'stim_times_s': np.asarray(stim_times, float),
        'average': {
            'amp_raw': np.asarray(amp_raw_avg, float),
            'amp_savgol': np.asarray(amp_sg_avg, float),
            'amp_nnls': np.asarray(amp_nnls_avg, float),
            'ppr_nnls': np.asarray(ppr_nnls_avg, float),
            'y_avg': np.asarray(y_avg, float),
            'yhat_avg': np.asarray(yhat_avg, float),
        },
        'per_trial': per_trial,
        'time_s': np.asarray(t, float),
        'threshold_amp1': np.asarray(thr_list, float),
        'pval_amp1': np.asarray(pval_list, float),
        'figure': figure,
        'figure_event_model': None,
        'figures_trials': figures_trials,
    }


# ----------------------------------------------
# Convenience: export multiple folders to Excel
# ----------------------------------------------
def export_folders_to_excel(paths,
                            out_file,
                            *,
                            train_start: float,
                            isi: float,
                            n_pulses: int,
                            options: Optional[Dict] = None):
    """Process all .xlsx files in each folder and write a multi-sheet Excel.

    - One sheet per input folder (sheet named after the folder's basename)
    - Each sheet: one row per file with AMP1..AMPn, PPR2/1.., optional %Fail1..3

    Options follow extract_metrics; 'measurement' selects the amplitude series
    used for export; 'fail_method' controls failure classification.
    """
    import os, glob, zipfile
    import pandas as pd

    if isinstance(paths, str):
        paths = [paths]

    def _is_valid_xlsx(path: str) -> bool:
        try:
            with zipfile.ZipFile(path) as z:
                return '[Content_Types].xml' in z.namelist()
        except Exception:
            return False

    cfg = {} if options is None else dict(options)
    meas = str(cfg.get('measurement', 'NNLS')).strip().upper()
    failm = str((cfg.get('fail_method') or meas)).strip().upper()

    with pd.ExcelWriter(out_file) as writer:
        wrote_any = False
        for folder in paths:
            files = [p for p in glob.glob(os.path.join(folder, "*.xlsx")) if _is_valid_xlsx(p)]
            if not files:
                continue
            rows = []
            for fp in files:
                try:
                    df = pd.read_excel(fp, sheet_name=0)
                    t_raw = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
                    X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
                    ok = np.isfinite(t_raw)
                    t = t_raw[ok]; trials = X[ok, :]
                    res = extract_metrics(t, trials, train_start=train_start, isi=isi, n_pulses=n_pulses, options=cfg)

                    # Choose amplitude series per measurement
                    if meas == 'SAVGOL':
                        amp_avg = res['average']['amp_savgol']
                    elif meas == 'RAW':
                        amp_avg = res['average']['amp_raw']
                    else:
                        amp_avg = res['average']['amp_nnls']

                    # Build row
                    base = os.path.splitext(os.path.basename(fp))[0]
                    row = {'ID': base}
                    amp_avg = [float(x) if x is not None else float('nan') for x in amp_avg]
                    for i, v in enumerate(amp_avg, start=1):
                        row[f"AMP{i}"] = v
                    a1 = row.get('AMP1')
                    for i in range(2, int(n_pulses) + 1):
                        row[f"PPR{i}/1"] = (row.get(f"AMP{i}") / a1) if (a1 not in (None, 0) and pd.notna(a1)) else float('nan')

                    # Failure % for pulses 1..3 using fail_method amplitudes and shared thresholds if present
                    # Collect per-trial from res['per_trial']
                    def _amp_vec_per_trial(key):
                        vals = []
                        for r in res['per_trial']:
                            arr = r.get(key)
                            vals.append(arr if arr is not None else [])
                        return vals

                    if failm == 'SAVGOL':
                        per_amp = _amp_vec_per_trial('amp_savgol')
                    elif failm == 'RAW':
                        per_amp = _amp_vec_per_trial('amp_raw')
                    else:
                        per_amp = _amp_vec_per_trial('amp_nnls')

                    # Use per-trial shared threshold and compare first 3 pulses
                    n_fail = [0, 0, 0]
                    n_valid = [0, 0, 0]
                    for idx_trial, r in enumerate(res['per_trial']):
                        thr = r.get('thr_shared')
                        amps = per_amp[idx_trial]
                        if thr is None or not (isinstance(amps, (list, tuple, np.ndarray))):
                            continue
                        for k in range(3):
                            if len(amps) > k:
                                ak = float(amps[k])
                                if np.isfinite(ak) and np.isfinite(thr):
                                    n_valid[k] += 1
                                    if ak <= thr:
                                        n_fail[k] += 1
                    for k in range(3):
                        row[f"%Fail{k+1}"] = round((n_fail[k] / n_valid[k]) * 100.0, 2) if n_valid[k] else float('nan')

                    rows.append(row)
                except Exception:
                    continue

            if rows:
                import pandas as _pd
                df_out = _pd.DataFrame(rows)
                # Ensure ordered columns
                cols = ['ID'] + [f"AMP{i}" for i in range(1, int(n_pulses) + 1)] + [f"PPR{i}/1" for i in range(2, int(n_pulses) + 1)] + [f"%Fail{i}" for i in range(1, 4)]
                for c in cols:
                    if c not in df_out.columns:
                        df_out[c] = float('nan')
                df_out = df_out[cols]
                sheet = os.path.basename(os.path.normpath(folder))[:31]
                df_out.to_excel(writer, sheet_name=sheet, index=False)
                wrote_any = True

        if not wrote_any:
            # Placeholder sheet
            import pandas as _pd
            _pd.DataFrame({"info": ["No valid data found"]}).to_excel(writer, sheet_name="Summary", index=False)
