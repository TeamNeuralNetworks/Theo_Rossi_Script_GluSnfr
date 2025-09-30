"""Clear, streamlined iGluSnFR train analysis

This module provides a compact, publication‑friendly API that is
mathematically equivalent to the main pipeline. It extracts per‑pulse
amplitudes and PPRs from iGluSnFR trains with minimal moving parts and a
simple options dictionary for configuration and plotting.

Key steps performed (mirrors batch_measure_complex):
  1) Interpolate NaNs and (optionally) correct slow bleaching
  2) Baseline to ΔF/F0 using the median over the full pre‑train interval
  3) Estimate kinetics from the average trace (rise τr; per‑pulse decay τd)
  4) Robust NNLS per‑pulse amplitudes with micro‑shifts (forward, no overlap)
  5) Per‑pulse amplitudes via local averaged max around each stimulus
  6) Null amplitudes and a MAD‑based threshold/p‑value for A1

Only the pieces necessary for this workflow are implemented here, using the
same formulas and defaults as the original pipeline for equivalence.
"""

from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls

try:
    from smoothing import (
            progress_print,
            fill_nans_timewise,
            sg_smooth,
            iglusnfr_kernel,
            time_zoom_mask,
            windowed_max,
            pick_peak_on_series,
            compute_no_signal_mask,
            build_median_recut_waveform,
    )
except Exception:
    # Fallback: allow importing when current working dir is this subfolder
    import os, sys
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from smoothing import (
        progress_print,
        fill_nans_timewise,
        sg_smooth,
        iglusnfr_kernel,
        time_zoom_mask,
        windowed_max,
        pick_peak_on_series,
        compute_no_signal_mask,
        build_median_recut_waveform,
    )

try:
    from Model_Calibration.simple_curve_fit import fit_average_event
except Exception:  # pragma: no cover - allow running from subfolder
    from simple_curve_fit import fit_average_event  # type: ignore


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
    # Extend decay grids so τd can grow well beyond 10 ms when trains slow down
    'kin_taud0_grid_ms': [1.6, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0, 12.5, 15.0, 18.0, 22.0, 28.0, 35.0, 45.0, 60.0],
    'kin_slope_grid_ms': [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0],
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
    # Supported (varying): 'double_exp' (rise+decay), 'cooperative', 'bilinear'
    # Default is a rise+decay kernel (difference of exponentials)
    'event_model': 'double_exp',
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
    # Kinetics source and progression controls
    #  - fit_source: 'global' | 'average' | 'individual'
    #    * global: fit a single event template from all trials (recut median)
    #    * average: fit kinetics on the multi-trial average trace
    #    * individual: fit kinetics per trial then aggregate (median)
    'fit_source': 'global',
    # Decay progression across train (applies to all fit_source modes)
    #  - 'fixed': apply one τd to the whole train (median of estimates)
    #  - 'free_monotonic': interpolate between first and last τd, non-decreasing
    #  - 'linear': non-negative slope linear regression across pulses
    'decay_progression_mode': 'linear',
    'anchor_first_tau': False,  # If True, anchor first event's tau as minimum (fastest decay)
    'anchor_final_tau': True,  # If True, anchor last event's tau as maximum (most reliable, no following events)
    # NNLS weight control
    'nnls_weight_mode': 'uniform',  # 'uniform', 'linear', 'exponential', 'savgol'
    'nnls_weight_tau_s': None,  # Time constant for exponential or slope for linear (auto if None)
    'fit_diagnostic_plot': False,  # Display weight and decay progression diagnostics
}

# Recut options: oversample factor and projection ('mean'|'median'|'std')
DEFAULTS.update({
    'recut_oversample': 1,
    'recut_projection': 'median',
    'recut_peak_recenter': 0,
})

# Selected kernel (set inside extract_metrics based on options; default is iglusnfr_kernel)
_KERNEL_FUN = iglusnfr_kernel


# -------------------------
# Small utilities
# -------------------------

def _calculate_nnls_weights(
    t: np.ndarray,
    stim_times: np.ndarray,
    isi: float,
    weight_mode: str = 'uniform',
    weight_tau_s: Optional[float] = None,
    y_ref: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculate weights for NNLS fitting based on weight_mode.
    
    Args:
        t: Time array
        stim_times: Stimulus times array  
        isi: Inter-stimulus interval in seconds
        weight_mode: 'uniform', 'linear', 'exponential', or 'savgol'
        weight_tau_s: Time constant for exponential decay or linear slope (in seconds)
        y_ref: Optional reference trace used for Savitzky-Golay weights (must
            match ``t`` in shape when ``weight_mode`` is ``'savgol'``)
        
    Returns:
        weights: Array same length as t with weights for each timepoint
    """
    weights = np.ones_like(t)
    
    if weight_mode == 'uniform':
        return weights
    
    elif weight_mode == 'linear':
        tau = weight_tau_s if weight_tau_s is not None else isi
        for i, stim_t in enumerate(stim_times):
            # Find next stimulus time (or end of trace)
            if i < len(stim_times) - 1:
                next_stim_t = stim_times[i + 1]
            else:
                # For last stimulus, use same interval as previous
                if len(stim_times) > 1:
                    next_stim_t = stim_t + (stim_times[-1] - stim_times[-2])
                else:
                    next_stim_t = stim_t + isi
            
            # Create linear decay from 1 to 0 over the interval
            mask = (t >= stim_t) & (t < next_stim_t)
            if np.any(mask):
                t_rel = t[mask] - stim_t
                duration = next_stim_t - stim_t
                # Linear decay with controllable slope
                slope_factor = tau / duration  # tau controls how steep the decay is
                decay = 1.0 - (t_rel / duration) * slope_factor
                weights[mask] = np.maximum(decay, 0.0)  # Clip at 0
                
    elif weight_mode == 'exponential':
        tau = weight_tau_s if weight_tau_s is not None else 0.010  # Default 10ms
        for i, stim_t in enumerate(stim_times):
            # Find next stimulus time (or end of trace) 
            if i < len(stim_times) - 1:
                next_stim_t = stim_times[i + 1]
            else:
                if len(stim_times) > 1:
                    next_stim_t = stim_t + (stim_times[-1] - stim_times[-2])
                else:
                    next_stim_t = stim_t + isi
            
            # Create exponential decay
            mask = (t >= stim_t) & (t < next_stim_t)
            if np.any(mask):
                t_rel = t[mask] - stim_t
                weights[mask] = np.exp(-t_rel / tau)
    
    elif weight_mode == 'savgol':
        if y_ref is None:
            raise ValueError("Savgol weight_mode requires a reference trace (y_ref)")
        ref = np.asarray(y_ref, float)
        if ref.shape != t.shape:
            raise ValueError("Reference trace for savgol weights must match time array shape")
        ref = np.where(np.isfinite(ref), ref, 0.0)
        ref = np.abs(ref)
        ref = ref - np.nanmin(ref)
        max_ref = np.nanmax(ref)
        if not np.isfinite(max_ref) or max_ref <= 0:
            return weights
        norm = ref / max_ref
        weights = 0.1 + 0.9 * norm
        return weights

    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    return weights


def _trim_spines(ax):
    """Hide top/right spines for a cleaner look (safe no-op on failure)."""
    try:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    except Exception:
        pass


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


def _nnls_weighted(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted NNLS solver.
    
    Args:
        X: Design matrix (n_timepoints, n_pulses)
        y: Target vector (n_timepoints,)
        weights: Weight vector (n_timepoints,)
        
    Returns:
        amplitudes: Non-negative amplitudes (n_pulses,)
    """
    if X.size == 0:
        return np.zeros(X.shape[1] if X.ndim > 1 else 0)
    
    # Apply weights by scaling both X and y
    Wsqrt = np.sqrt(weights)
    X_weighted = X * Wsqrt[:, None]
    y_weighted = y * Wsqrt
    
    return np.maximum(0.0, nnls(X_weighted, y_weighted)[0])


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
    event_t0_s: float = 0.0,
) -> Tuple[float, float]:
    """Estimate amplitude at ``stim_time`` with optional micro-shift.

    Assumes baseline has been corrected; residual from earlier events should be
    subtracted before calling. Returns ``(amplitude, best_shift_s)``.
    """
    anchor = float(stim_time) + float(event_t0_s)
    local_mask = (t >= (anchor - pre_zoom_s)) & (t <= (anchor + post_zoom_s))
    if not np.any(local_mask):
        return 0.0, 0.0
    y_seg = y[local_mask]
    best_a, best_d = 0.0, 0.0
    if allow_shift:
        start = max(float(shift_min_s), 0.0)
        if delta_max_s > 0 and start <= delta_max_s + 1e-12:
            pos_shifts = np.arange(start, delta_max_s + 1e-12, delta_step_s)
            shifts = np.concatenate(([0.0], pos_shifts)) if pos_shifts.size else np.array([0.0])
        else:
            shifts = np.array([0.0])
    else:
        shifts = np.array([0.0])
    shifts = np.unique(shifts.astype(float))
    for d in shifts:
        k_full = _KERNEL_FUN(t - (anchor + d), tau_r_s, tau_d_s)
        k_loc = k_full[local_mask]
        if k_loc.size < 3 or np.all(k_loc == 0):
            continue
        a_loc = _nnls_irls_singlecol(y_seg, k_loc, robust=robust, huber_delta=huber_delta, iters=irls_iters)
        if a_loc > best_a:
            best_a, best_d = a_loc, d
    return float(best_a), float(best_d)


def fit_amplitudes_no_overlap_forward(
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
    event_t0_s: float = 0.0,
):
    """Forward, non‑overlap per‑pulse fitting with micro‑shifts.
    
    Returns (amplitudes, shifts, design, reconstruction, components_list).
    """
    n = len(stim_times)
    a = np.zeros(n, float)
    d = np.zeros(n, float)
    residual = y.copy()
    
    event_t0_s = float(event_t0_s)

    for p in range(n):
        st = float(stim_times[p])
        anchor = st + event_t0_s
        if p < n - 1:
            next_anchor = float(stim_times[p + 1]) + event_t0_s
        else:
            next_anchor = None
        max_end = min(anchor + post_zoom_s, next_anchor) if next_anchor is not None else (anchor + post_zoom_s)
        avail_post = max(0.0, max_end - anchor)
        
        if avail_post < 1e-6:
            continue
            
        a_p, d_p = _fit_single_pulse_amp(
            residual, t, st, tau_r_s, float(tau_d_vec_s[p]),
            pre_zoom_s=pre_zoom_s, post_zoom_s=avail_post,
            robust=robust, huber_delta=huber_delta, irls_iters=irls_iters,
            allow_shift=allow_shift, delta_max_s=delta_max_s, delta_step_s=delta_step_s,
            shift_min_s=shift_min_s, event_t0_s=event_t0_s,
        )
        
        a[p] = a_p
        d[p] = d_p
        
        # Subtract this component from residual for next iteration
        k = _KERNEL_FUN(t - (anchor + d_p), tau_r_s, float(tau_d_vec_s[p]))
        comp = a_p * k
        residual = residual - comp
    
    # NOW reconstruct all components using the fitted parameters
    # These are the actual components as they appear in the final signal
    components = []
    for p in range(n):
        if a[p] > 0:  # Only create component if amplitude is non-zero
            anchor_p = float(stim_times[p]) + event_t0_s
            k = _KERNEL_FUN(t - (anchor_p + d[p]), tau_r_s, float(tau_d_vec_s[p]))
            components.append(a[p] * k)
        else:
            components.append(np.zeros_like(y))
    
    # The reconstruction is the sum of all components
    yhat = np.sum(components, axis=0) if components else np.zeros_like(y)
    
    # Build design matrix for reference
    X = (
        np.column_stack([
            _KERNEL_FUN(
                t - (float(stim_times[p]) + event_t0_s + d[p]),
                tau_r_s,
                float(tau_d_vec_s[p]),
            )
            for p in range(n)
        ])
        if n
        else np.zeros((t.size, 0))
    )
    
    return a, d, X, yhat, components

def compute_localmax_corrected_amps(
    t: np.ndarray,
    y: np.ndarray,
    stim_times: np.ndarray,
    win_ms: float,
    n_avg: int,
    pre_ms: float,
    d_vec: np.ndarray,
    tau_r_s: float,
    tau_d_vec_s: np.ndarray,
    *,
    event_t0_s: float = 0.0,
):
    """Local averaged max around each stimulus, removing earlier events.

    Events are processed sequentially from start to finish. After measuring the
    peak for pulse ``p`` the corresponding kernel scaled by that peak is
    subtracted from ``y`` so that later pulses are unaffected by the earlier
    ones.
    """
    if y is None or t.size == 0 or np.size(y) == 0:
        return np.zeros(len(stim_times), float)
    y_resid = y.copy()
    amps = []
    event_t0_s = float(event_t0_s)
    for p, st in enumerate(stim_times):
        center = float(st) + event_t0_s
        v = windowed_max(t, y_resid, [center], win_ms, int(n_avg), pre_ms)
        amp_p = float(v[0]) if np.size(v) else 0.0
        amps.append(amp_p)
        if (
            d_vec is not None
            and tau_d_vec_s is not None
            and len(d_vec) > p
            and len(tau_d_vec_s) > p
        ):
            k = _KERNEL_FUN(
                t - (float(st) + event_t0_s + float(d_vec[p])),
                tau_r_s,
                float(tau_d_vec_s[p]),
            )
            y_resid = y_resid - amp_p * k
    return np.asarray(amps, float)


def compute_peak_corrected_from_components(
    t: np.ndarray,
    y_meas: np.ndarray,
    stim_times: np.ndarray,
    components: Optional[List[np.ndarray]],
    *,
    win_ms: float,
    pre_ms: float,
) -> np.ndarray:
    """Return per‑pulse amplitudes corrected for prior‑pulse residuals.

    For each pulse ``p`` we locate the peak time on the provided measurement
    series ``y_meas`` and subtract the sum of NNLS component contributions from
    all earlier pulses evaluated at that peak time. This matches the visual
    interpretation “red dot minus white triangle”.

    Parameters
    ----------
    t : 1D np.ndarray
        Time vector
    y_meas : 1D np.ndarray
        Measurement series used to locate peaks (e.g. yhat, SavGol, or raw)
    stim_times : 1D np.ndarray
        Stimulus times in seconds
    components : list[np.ndarray] | None
        NNLS per‑pulse components (length = n_pulses). May be None (returns
        uncorrected peaks in that case).
    win_ms, pre_ms : float
        Peak search window and pre‑search lead, in milliseconds
    """
    t = np.asarray(t, float)
    y = np.asarray(y_meas, float)
    n = int(len(stim_times))
    if t.size == 0 or y.size == 0 or n == 0:
        return np.zeros(n, float)

    # Build cumulative baseline of previous pulses only: B_p(t) = sum_{k<p} comp[k](t)
    baselines = None
    if isinstance(components, list) and components and all(hasattr(c, "__len__") for c in components):
        cum = np.zeros_like(y)
        baselines = []
        for p in range(n):
            baselines.append(cum.copy())
            if p < len(components):
                cum = cum + np.asarray(components[p], float)

    amps_corr = np.zeros(n, float)
    for p, st in enumerate(stim_times):
        tp, vp = pick_peak_on_series(t, y, float(st), win_ms, pre_ms)
        idx = int(np.argmin(np.abs(t - tp))) if t.size else 0
        base_prev = float(baselines[p][idx]) if baselines is not None else 0.0
        amps_corr[p] = float(vp) - base_prev
    return amps_corr

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
    seed: int = 42,
    event_t0_s: float = 0.0,
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
    # Cover the entire f0 window; exclude starts whose peak window would
    # overlap the train onset.
    st_min = null_start
    st_max = null_end - (peak_window_ms / 1000.0)
    if st_max <= st_min:
        return np.array([])
    cand_mask = (t >= st_min) & (t <= st_max)
    starts_full = t[cand_mask]
    if starts_full.size == 0:
        return np.array([])
    limit = int(min(int(null_sim_max_points), int(n_samples)))
    rng = np.random.default_rng(seed)
    if starts_full.size > limit:
        idx_sel = rng.choice(starts_full.size, size=limit, replace=False)
        idx_sel.sort()
        starts = starts_full[idx_sel]
    else:
        starts = starts_full

    amps = []
    event_t0_s = float(event_t0_s)
    for st in starts:
        anchor = float(st) + event_t0_s
        avail_post = min(post_zoom_s, train_start - anchor - 1e-6, null_end - st)
        if avail_post < null_min_post_zoom_s:
            continue
        a_hat, d_hat = _fit_single_pulse_amp(
            y, t, float(st), tau_r_s, tau_d_s,
            pre_zoom_s=pre_zoom_s, post_zoom_s=avail_post,
            robust=robust, huber_delta=huber_delta, irls_iters=irls_iters,
            allow_shift=allow_shift, delta_max_s=delta_max_s, delta_step_s=delta_step_s,
            shift_min_s=shift_min_s, event_t0_s=event_t0_s,
        )
        t_fit = t[(t >= anchor - pre_zoom_s) & (t <= anchor + avail_post)]
        if t_fit.size:
            k_fit = _KERNEL_FUN(t_fit - (anchor + d_hat), tau_r_s, tau_d_s)
            y_evt = a_hat * k_fit
            val = windowed_max(
                t_fit,
                y_evt,
                [float(st) + event_t0_s],
                peak_window_ms,
                peak_avg_points,
                pre_peak_ms,
            )
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
        baseline_sel = y[(t < train_start_s) & np.isfinite(y)]
        if baseline_sel.size == 0:
            baseline_sel = y[np.isfinite(y)]
        baseline_ref = float(np.nanmedian(baseline_sel)) if baseline_sel.size else 0.0
        corrected = y - trend + baseline_ref
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
    isi: float = 0.05,
    weight_mode: str = 'uniform',
    weight_tau_s: Optional[float] = None,
    sg_window: int = DEFAULTS['sg_window'],
    sg_poly: int = DEFAULTS['sg_poly'],
    event_t0_s: float = 0.0,
) -> Tuple[float, float, float, np.ndarray]:
    """Grid search τr, τd0, slope on the average trace (zoomed window)."""
    tau_r_grid = np.array(taur_grid_ms, float) / 1000.0
    tau_d0_grid = np.array(taud0_grid_ms, float) / 1000.0
    slope_grid = np.array(slope_grid_ms, float) / 1000.0
    isi_guess = float(stim_times[1] - stim_times[0]) if len(stim_times) > 1 else isi
    anchor0 = float(stim_times[0]) + float(event_t0_s)
    zmask, _, _ = time_zoom_mask(t, anchor0, isi_guess, len(stim_times), pre_zoom_s, post_zoom_s)

    # Calculate weights for NNLS fitting
    if weight_mode == 'savgol':
        y_weight_ref = sg_smooth(fill_nans_timewise(y_avg, t), int(sg_window), int(sg_poly))
    else:
        y_weight_ref = None
    weights = _calculate_nnls_weights(
        t,
        stim_times,
        isi_guess,
        weight_mode,
        weight_tau_s,
        y_ref=y_weight_ref,
    )

    def obj_for(tau_r, tau_d_vec):
        X = np.column_stack([
            _KERNEL_FUN(t - (float(st) + float(event_t0_s)), tau_r, td)
            for st, td in zip(stim_times, tau_d_vec)
        ])
        if X.size == 0:
            a = np.zeros(len(stim_times))
        else:
            # Use weighted NNLS
            a = _nnls_weighted(X[zmask, :], y_avg[zmask], weights[zmask])
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
        SD for SAVGOL. Failure rates compare pulses 1–3 against a single baseline
        threshold computed from pulse 1; pulses 2/3 amplitudes subtract residual
        pre‑stim currents before comparison
      - allow_shift: bool (default True) — enable per‑pulse micro‑shifts
      - recut_peak_recenter: int | tuple | None (default 0) — number of samples
        permitted for peak realignment before averaging; 0/None keeps stimulus
        alignment
      - event_model: {'double_exp'|'cooperative'} (default 'double_exp') — template used
        for NNLS fitting and residual subtraction; 'cooperative' uses a Hill‑like rise*exp decay
      - plot: dict with keys
          - enabled: bool (default False)
          - traces: list of {'raw','savgol','nnls'} (default ['nnls'])
          - show_decay: bool (default True)
          - trials: bool (default False) — also plot each trial with its fit
          - baseline: bool (default False) — for each trial, plot a two-panel
          figure with baseline fits + noise histogram and the train; forces
          trials=True when enabled
          - residuals: bool (default False) — add residual diagnostics panels
            comparing pre-train baseline noise vs. residuals after subtracting
            the selected measurement model (NNLS or SavGol)
    """
    # Parse options (merge into a single config dict)
    opts = options.copy() if isinstance(options, dict) else {}
    plot_opts = opts.get('plot', {}) if isinstance(opts.get('plot', {}), dict) else {}
    want_plot = bool(plot_opts.get('enabled', False))
    traces = list(plot_opts.get('traces', ['nnls']))
    show_decay = bool(plot_opts.get('show_decay', True))
    plot_trials = bool(plot_opts.get('trials', False))
    baseline_figs = bool(plot_opts.get('baseline', False))
    plot_residuals = bool(plot_opts.get('residuals', False))
    plot_peaks_details = bool(plot_opts.get('plot_peaks_details', False))
    if baseline_figs:
        plot_trials = True  # baseline panel requires per-trial figures
    cfg = {**DEFAULTS, **{k: v for k, v in opts.items() if k != 'plot'}}
    if 'nnls_show_weights' in opts and 'fit_diagnostic_plot' not in opts:
        try:
            cfg['fit_diagnostic_plot'] = bool(opts.get('nnls_show_weights', False))
        except Exception:
            cfg['fit_diagnostic_plot'] = bool(cfg.get('fit_diagnostic_plot', False))
    interpolated_settings: List[Dict[str, Any]] = []
    global_fit_params: Dict[str, float] = {}
    explicit_em_settings = {}
    if isinstance(opts.get('event_model_settings'), dict):
        # Preserve user-provided overrides so later auto-fits do not clobber them
        explicit_em_settings = dict(opts['event_model_settings'])
    cfg_em_settings = cfg.get('event_model_settings')
    if isinstance(cfg_em_settings, dict):
        # Use a shallow copy to avoid mutating the caller's dict in-place
        cfg['event_model_settings'] = dict(cfg_em_settings)
    else:
        cfg['event_model_settings'] = {}
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
    peak_recenter = cfg.get('recut_peak_recenter', 0)

    # Shapes & schedule
    t = np.asarray(time, float).reshape(-1)
    Y = np.asarray(trials, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.shape[0] != t.size:
        raise ValueError("trials must have same number of samples as time")
    stim_times = float(train_start) + float(isi) * np.arange(int(n_pulses))

    # ---- ISI-aware guards and defaults ----
    isi = float(isi)
    isi_ms = isi * 1000.0

    # Peak window: auto if not user-overridden or too wide for ISI
    if ('peak_window_ms' not in cfg) or (float(cfg['peak_window_ms']) >= isi_ms):
        cfg['peak_window_ms'] = max(6.0, min(12.0, 0.45 * isi_ms))
    if ('peak_avg_points' not in cfg) or (int(cfg['peak_avg_points']) > 5):
        cfg['peak_avg_points'] = 3
    if 'pre_peak_ms' not in cfg:
        cfg['pre_peak_ms'] = 0.0

    # Micro-shift: cap to a fraction of ISI
    cfg['delta_max_s'] = min(float(cfg.get('delta_max_s', 0.002)), 0.25 * isi)

    # Null-fit single-pulse needs some post window before next stim
    cfg['null_min_post_zoom_s'] = min(float(cfg.get('null_min_post_zoom_s', 0.05)), 0.5 * isi)

    # Recut event: keep 3 ms guard before next stimulus, never exceed 20 ms
    _POST_EVT_MS = max(6.0, min(20.0, isi_ms - 3.0))

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
    # Default must match DEFAULTS['event_model'] for consistency
    event_model = str(cfg.get('event_model', DEFAULTS.get('event_model', 'double_exp'))).strip().lower()
    em_settings = cfg.get('event_model_settings', {}) or {}
    if not isinstance(em_settings, dict):
        raise ValueError("event_model_settings must be a dict of parameter overrides")
    coop_n_default = float(em_settings.get('n_coop', 2.0))

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
            # Amplitude is peak-scaled as defined by the model spec (no area normalization).
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
                return y
            return kfun
        def _kernel_tau_varying(mapper):
            # mapper builds full param vector from tau_r,tau_d; no extra normalization
            def kfun(dt_s: np.ndarray, tau_r: float, tau_d: float) -> np.ndarray:
                dt_s = np.asarray(dt_s, float)
                dt_ms = dt_s * 1000.0
                pars = mapper(float(tau_r), float(tau_d))
                y = spec['func'](dt_ms, *pars)
                return y
            return kfun
        return spec, _kernel_from_params, _kernel_tau_varying

    global _KERNEL_FUN
    # Helper to set kernel from current cfg and return effective (event_model, n_coop|None)
    def _apply_event_model_from_cfg() -> Tuple[str, Optional[float]]:
        nonlocal event_model, coop_n_default, em_settings
        # τ‑varying supported directly
        varying_supported = {'double_exp', 'cooperative', 'bilinear'}
        evm = str(cfg.get('event_model', event_model)).strip().lower()
        em_settings = cfg.get('event_model_settings', {}) or {}
        if not isinstance(em_settings, dict):
            raise ValueError("event_model_settings must be a dict of parameter overrides")
        if evm.startswith('library:'):
            _lib = evm.split(':', 1)[1].strip().lower()
            if _lib in varying_supported:
                evm = _lib
        if evm in varying_supported:
            spec, _, make_var = _build_kernel_from_library(evm if evm != 'double_exp' else 'double_exp')
            if evm == 'cooperative':
                extra_keys = set(em_settings.keys()) - {'n_coop'}
                if extra_keys:
                    raise ValueError(f"Unsupported event_model_settings for cooperative: {sorted(extra_keys)}")
                coop_n_default = float(em_settings.get('n_coop', coop_n_default))
                def _map(tau_r, tau_d):
                    return [1.0, float(tau_r), float(tau_d), float(coop_n_default), 0.0]
                _KERNEL_FUN = make_var(_map)
                progress_print(f"[model] Using event model 'cooperative' (τ‑varying), n_coop={coop_n_default}")
                event_model = evm
                return evm, coop_n_default
            if em_settings:
                raise ValueError(f"{evm} in τ‑varying mode does not accept event_model_settings")
            if evm == 'double_exp':
                def _map(tau_r, tau_d):
                    return [1.0, float(tau_r), float(tau_d), 0.0]
                _KERNEL_FUN = make_var(_map)
                progress_print("[model] Using event model 'double_exp' (τ‑varying)")
                event_model = evm
                return evm, None
            # bilinear
            def _map(tau_r, tau_d):
                return [1.0, float(tau_r) * 1000.0, float(tau_d) * 1000.0, 0.0]
            _KERNEL_FUN = make_var(_map)
            progress_print("[model] Using event model 'bilinear' (τ‑varying)")
            event_model = evm
            return evm, None
        else:
            # Fixed template
            lib_name = evm
            if lib_name.startswith('library:'):
                lib_name = lib_name.split(':', 1)[1].strip().lower()
            spec, make_fixed, _ = _build_kernel_from_library(lib_name)
            allowed = set(spec['params']) - {'amp', 't_peak'}
            unknown = set(em_settings.keys()) - allowed
            if unknown:
                raise ValueError(f"Unknown event_model_settings for '{lib_name}': {sorted(unknown)}. Allowed keys: {sorted(allowed)}")
            # Fit missing params on the average event (placeholder; uses y_avg later if needed)
            # For fixed-template, kernel ignores tau_r/tau_d and uses fitted params
            _KERNEL_FUN = make_fixed({k: float(v) for k, v in em_settings.items() if k in allowed})
            progress_print(f"[model] Using fixed-template '{lib_name}' (amplitude-only per pulse).")
            event_model = lib_name
            return lib_name, None

    # First application with initial cfg
    ev_model_name, n_coop_effective = _apply_event_model_from_cfg()
    varying_supported_names = {"double_exp", "cooperative", "bilinear"}
    is_varying_model = ev_model_name in varying_supported_names

    # No auto model selection: honor explicit event_model; otherwise use default.

    # Process NNLS weight configuration
    weight_mode = str(cfg.get('nnls_weight_mode', 'uniform')).strip().lower()
    weight_tau_s = cfg.get('nnls_weight_tau_s', None)
    if weight_tau_s is not None:
        weight_tau_s = float(weight_tau_s)
    elif weight_mode == 'linear':
        weight_tau_s = float(isi)  # Default linear decay uses ISI
    elif weight_mode == 'exponential':
        if cfg.get('fit_source', 'global') == 'global':
            # Will be set later from tau_d when available
            weight_tau_s = 0.010  # Default 10ms until tau_d is estimated
        else:
            weight_tau_s = 0.010  # Default 10ms for other modes
    elif weight_mode == 'savgol':
        weight_tau_s = None

    # Average trace and kinetics
    y_avg = np.nanmean(Yd, axis=1)
    # Optional: store recut snippets for plotting/diagnostics
    recut_snippets = None
    recut_t_rel = None
    recut_avg = None

    # Helper: estimate base kinetics from recut average of all trials/events
    def _estimate_from_recut_average():
        try:
            # Honor explicit top-level option 'recut_snippets'.
            need_snips = bool(
                cfg.get('plot', {}).get('enabled', False)
                or cfg.get('recut_snippets', False)
            )
            if need_snips:
                t_rel, avg, snippets = build_median_recut_waveform(
                    t, Yd, stim_times, pre_ms=5.0, post_ms=_POST_EVT_MS,
                    peak_win_ms=cfg['peak_window_ms'], peak_search_pre_ms=cfg['pre_peak_ms'],
                    oversample=int(cfg.get('recut_oversample', 1)),
                    projection=str(cfg.get('recut_projection', 'median')).lower(),
                    peak_recenter=peak_recenter,
                    return_snippets=True,
                )

            else:
                t_rel, avg = build_median_recut_waveform(
                    t, Yd, stim_times, pre_ms=5.0, post_ms=_POST_EVT_MS,
                    peak_win_ms=cfg['peak_window_ms'], peak_search_pre_ms=cfg['pre_peak_ms'],
                    oversample=int(cfg.get('recut_oversample', 1)),
                    projection=str(cfg.get('recut_projection', 'median')).lower(),
                    peak_recenter=peak_recenter,
                )
                snippets = None
            if t_rel is None or avg is None:
                raise ValueError('recut_average unavailable')
            # Capture snippets and recut outputs for outer scope plotting if returned
            nonlocal recut_snippets, recut_t_rel, recut_avg
            recut_t_rel = t_rel
            recut_avg  = avg
            recut_snippets = snippets
            if 'snippets' in locals():
                recut_snippets = snippets
            # Grid search on (tau_r, tau_d0) using current kernel
            tau_r_grid = np.array(cfg['kin_taur_grid_ms'], float) / 1000.0
            tau_d0_grid = np.array(cfg['kin_taud0_grid_ms'], float) / 1000.0
            best = (np.inf, 0.002, 0.015)
            dt = float(np.median(np.diff(t_rel)))
            dt_s = dt
            # Use _KERNEL_FUN for current model (cooperative uses n_coop)
            for tr in tau_r_grid:
                for td in tau_d0_grid:
                    k = _KERNEL_FUN(t_rel, tr, td)
                    denom = float(np.sum(k**2))
                    amp = float(np.sum(avg * k)) / denom if denom > 0 else 1.0
                    fit = amp * k
                    err = float(np.nanmean((avg - fit) ** 2))
                    if err < best[0]:
                        best = (err, tr, td)
            _, tau_r_b, tau_d0_b = best
            # default slope 0, tau_d_vec constant; progression handled below
            return float(tau_r_b), float(tau_d0_b)
        except Exception:
            return None, None

    # Choose kinetics according to fit_source
    fit_source = str(cfg.get('fit_source', 'global')).lower()
    # Normalize decay mode and accept a few synonyms/typos
    raw_mode = str(cfg.get('decay_progression_mode', 'linear')).strip().lower()
    _mode_map = {
        'linear_anchored': 'linear',
        'monotonic': 'free_monotonic',
        'free': 'free_monotonic',
        'freed_monotonic': 'free_monotonic',
    }
    dec_mode = _mode_map.get(raw_mode, raw_mode)
    if dec_mode not in {'fixed', 'free_monotonic', 'linear'}:
        try:
            progress_print(f"[warn] Unknown decay_progression_mode='{raw_mode}', falling back to 'linear'. Allowed: fixed|free_monotonic|linear")
        except Exception:
            pass
        dec_mode = 'linear'

    # Offset (seconds) between stimulus time and actual event onset
    event_t0_s = 0.0

    def _estimate_single_event_tau(event_idx, tau_r_local, tau_d0_fallback):
        """Fit tau_d for a single event on the average trace.

        Fits only the decay after the event peak, focusing on the post-event window
        to minimize contamination from subsequent events in the train.
        Returns the fast decay estimate, amplitude, and optional extra parameter
        estimates when available.
        """
        try:
            st = stim_times[event_idx] + event_t0_s
            # Use a focused window: from event onset to next event (or post_zoom_s if last event)
            if event_idx < len(stim_times) - 1:
                # Not the last event: fit up to the next event
                next_st = stim_times[event_idx + 1] + event_t0_s
                zmask_evt = (t >= st) & (t < next_st)
            else:
                # Last event: use full post_zoom window
                zmask_evt = (t >= st) & (t <= (st + cfg['post_zoom_s']))

            tf = t[zmask_evt]; yf = y_avg[zmask_evt]

            if tf.size < 3:
                # Not enough points to fit
                return float(tau_d0_fallback), 1.0, None

            tau_d_grid_ms = np.array(cfg['kin_taud0_grid_ms'], float)
            tau_d_grid = tau_d_grid_ms / 1000.0
            best = (np.inf, tau_d0_fallback, 1.0)

            for td in tau_d_grid:
                k = _KERNEL_FUN(tf - st, tau_r_local, td)
                denom = float(np.sum(k**2))
                amp = float(np.sum(yf * k)) / denom if denom > 0 else 1.0
                fit = amp * k
                err = float(np.nanmean((yf - fit) ** 2))
                if err < best[0]:
                    best = (err, td, amp)

            tau_evt = float(best[1]); amp_evt = float(best[2])

            extra_params = None
            # Attempt a richer per-event fit when progression is not fixed
            if dec_mode in ('linear', 'free_monotonic'):
                try:
                    local_t_ms = (tf - st) * 1000.0
                    if local_t_ms.size and np.isfinite(local_t_ms[0]):
                        local_t_ms = local_t_ms - float(local_t_ms[0])
                    fit_res = fit_average_event(
                        local_t_ms,
                        yf,
                        event_model,
                        window_ms=(0.0, float(local_t_ms[-1]) if local_t_ms.size else 50.0),
                    )
                    if fit_res is not None:
                        params_dict = fit_res[0]
                        if isinstance(params_dict, dict):
                            extra_params = {
                                k: float(v)
                                for k, v in params_dict.items()
                                if k not in {'amp', 't_peak', '_recut'}
                            }
                except Exception:
                    extra_params = None

            return tau_evt, amp_evt, extra_params
        except Exception:
            return float(tau_d0_fallback), 1.0, None

    def _fit_all_events_on_average(tau_r_local, tau_d0_fallback):
        """Fit tau_d for each event individually on the average trace."""
        tau_vec = []
        amp_vec = []
        param_series: Dict[str, List[float]] = {}
        for i in range(n_pulses):
            tau_i, amp_i, extra = _estimate_single_event_tau(i, tau_r_local, tau_d0_fallback)
            tau_vec.append(tau_i)
            amp_vec.append(amp_i)

            filtered_params: Dict[str, float] = {}
            if isinstance(extra, dict):
                for key, val in extra.items():
                    if key in {'amp', 't_peak', '_recut'}:
                        continue
                    try:
                        filtered_params[key] = float(val)
                    except Exception:
                        filtered_params[key] = np.nan

            all_keys = set(param_series.keys()) | set(filtered_params.keys())
            for key in all_keys:
                param_series.setdefault(key, [])
                series = param_series[key]
                while len(series) < i:
                    series.append(np.nan)
                if key in filtered_params:
                    val = filtered_params[key]
                    series.append(float(val) if np.isfinite(val) else np.nan)
                else:
                    series.append(np.nan)

        tau_array = np.array(tau_vec, float)
        amp_array = np.array(amp_vec, float)
        for key, values in list(param_series.items()):
            while len(values) < n_pulses:
                values.append(np.nan)
            param_series[key] = np.array(values, float)
        try:
            tau_ms_str = ", ".join(f"{v*1000:.1f}" for v in tau_array)
            progress_print(f"[per-event fit] Individual τd (ms): [{tau_ms_str}]")
        except Exception:
            pass
        return tau_array, amp_array, param_series

    def _robust_linear_fit(x, y, max_iter=10, huber_delta=2.0):
        """Robust linear regression using IRLS with Huber weights.

        Args:
            x: Independent variable (pulse indices)
            y: Dependent variable (tau values)
            max_iter: Maximum IRLS iterations
            huber_delta: Huber threshold for outlier detection

        Returns:
            (a, b): intercept and slope
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        if len(x) < 2:
            return float(y[0]) if len(y) > 0 else 0.0, 0.0

        # Initial OLS fit
        A = np.column_stack([np.ones_like(x), x])
        try:
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b = float(coef[0]), float(coef[1])
        except Exception:
            return float(y[0]), 0.0

        # IRLS iterations
        weights = np.ones_like(y)
        for iter_num in range(max_iter):
            # Compute residuals
            yfit = a + b * x
            residuals = y - yfit

            # Robust scale estimate (MAD)
            mad = float(np.nanmedian(np.abs(residuals - np.nanmedian(residuals))))
            if mad < 1e-10:
                break
            scale = 1.4826 * mad  # Scale factor for normal distribution

            # Huber weights
            normalized_resid = np.abs(residuals) / scale
            weights = np.where(
                normalized_resid <= huber_delta,
                1.0,
                huber_delta / normalized_resid
            )

            # Log outliers on first iteration
            if iter_num == 0:
                outlier_mask = weights < 0.95
                if np.any(outlier_mask):
                    outlier_indices = np.where(outlier_mask)[0]
                    try:
                        outlier_str = ", ".join(f"event {int(x[i])+1}(τ={y[i]*1000:.1f}ms,w={weights[i]:.2f})"
                                               for i in outlier_indices)
                        progress_print(f"[robust fit] Detected outliers: {outlier_str}")
                    except Exception:
                        pass

            # Weighted least squares
            W = np.diag(weights)
            try:
                coef = np.linalg.lstsq(W @ A, W @ y, rcond=None)[0]
                a_new, b_new = float(coef[0]), float(coef[1])

                # Check convergence
                if abs(a_new - a) < 1e-6 and abs(b_new - b) < 1e-6:
                    break
                a, b = a_new, b_new
            except Exception:
                break

        return a, b

    def _monotonic_regression(y_in):
        """Apply isotonic regression to ensure non-decreasing values."""
        from scipy.interpolate import PchipInterpolator
        y = np.asarray(y_in, float)
        # Ensure monotonically non-decreasing by cumulative maximum
        y_mono = np.maximum.accumulate(y)
        # Smooth with PCHIP interpolation while preserving monotonicity
        x = np.arange(len(y))
        try:
            # PCHIP preserves monotonicity if input is monotonic
            interp = PchipInterpolator(x, y_mono)
            y_smooth = interp(x)
            # Force monotonic again in case of numerical issues
            y_smooth = np.maximum.accumulate(y_smooth)
            return y_smooth
        except Exception:
            return y_mono

    def _apply_progression(tau_r_in, tau_d_vec_in, tau_d0_in, source_method):
        """Apply decay progression rules based on mode and fit_source.

        Args:
            tau_r_in: Rise time constant
            tau_d_vec_in: Per-event decay time constants (raw)
            tau_d0_in: Fallback single decay constant
            source_method: 'global', 'average', or 'individual'
        """
        y = np.asarray(tau_d_vec_in, float)

        # Validate inputs
        if y.size != n_pulses:
            try:
                progress_print(f"[warning] tau_d_vec has wrong size {y.size}, expected {n_pulses}. Using fallback.")
            except Exception:
                pass
            y = np.full(n_pulses, float(tau_d0_in) if np.isfinite(tau_d0_in) else 0.010)

        # Replace any NaN/Inf with tau_d0
        if not np.all(np.isfinite(y)):
            try:
                progress_print(f"[warning] tau_d_vec contains NaN/Inf. Replacing with tau_d0={tau_d0_in*1000:.1f}ms")
            except Exception:
                pass
            fallback_tau = float(tau_d0_in) if np.isfinite(tau_d0_in) else 0.010
            y[~np.isfinite(y)] = fallback_tau

        # If all values are invalid, use fallback
        if not np.isfinite(y).any():
            y = np.full(n_pulses, float(tau_d0_in) if np.isfinite(tau_d0_in) else 0.010)

        anchor_first = bool(cfg.get('anchor_first_tau', False))
        anchor_final = bool(cfg.get('anchor_final_tau', True))

        if dec_mode == 'fixed':
            # Use single tau for all events
            td_med = float(np.nanmedian(y)) if np.isfinite(y).any() else float(tau_d0_in)
            return tau_r_in, np.full(n_pulses, td_med)

        elif dec_mode == 'linear':
            # Linear regression with non-negative slope
            x = np.arange(n_pulses, dtype=float)

            if anchor_first and anchor_final and n_pulses > 2:
                # Both anchors: force line through first and last points
                tau_first = float(y[0])
                tau_final = float(y[-1])
                # Constrained line: passes through (0, tau_first) and (n-1, tau_final)
                b = max(0.0, (tau_final - tau_first) / max(1, n_pulses - 1))
                a = tau_first
                yfit = a + b * x
                yfit[0] = tau_first
                yfit[-1] = tau_final
            elif anchor_final and n_pulses > 1:
                # Anchor last tau only: force line to pass through last point
                tau_final = float(y[-1])
                # Robust fit to first n-1 points
                a, b = _robust_linear_fit(x[:-1], y[:-1])
                b = max(0.0, b)  # Ensure non-negative slope
                a = tau_final - b * (n_pulses - 1)  # Adjust to pass through final point
                # Ensure intercept is positive (tau cannot be negative)
                if a < 1e-4:  # 0.1 ms minimum
                    a = 1e-4
                    # Recalculate slope to pass through final point
                    b = (tau_final - a) / max(1, n_pulses - 1)
                yfit = a + b * x
                yfit[-1] = tau_final
                # Ensure all values are positive
                yfit = np.maximum(yfit, 1e-4)
            elif anchor_first and n_pulses > 1:
                # Anchor first tau only: force line to pass through first point
                tau_first = float(y[0])
                # Robust fit to points 2-N
                a, b = _robust_linear_fit(x[1:], y[1:])
                b = max(0.0, b)  # Ensure non-negative slope
                a = tau_first  # Force through first point: a + b*0 = tau_first
                yfit = a + b * x
                yfit[0] = tau_first
            else:
                # Standard robust linear regression without anchors
                a, b = _robust_linear_fit(x, y)
                b = max(0.0, b)  # Ensure non-negative slope
                yfit = a + b * x

            # Ensure monotonic (can only get slower)
            yfit = np.maximum.accumulate(yfit)
            return tau_r_in, yfit

        else:  # 'free_monotonic'
            if anchor_first and anchor_final and n_pulses > 2:
                # Both anchors: interpolate between first and last with monotonic constraint
                tau_first = float(y[0])
                tau_final = float(y[-1])
                # Ensure all values between first and final
                y_clipped = np.clip(y, tau_first, tau_final)
                y_clipped[0] = tau_first
                y_clipped[-1] = tau_final
                # Apply monotonic regression
                yfit = _monotonic_regression(y_clipped)
                yfit[0] = tau_first
                yfit[-1] = tau_final
            elif anchor_final and n_pulses > 1:
                # Anchor final tau only
                tau_final = float(y[-1])
                y_clipped = np.minimum(y, tau_final)
                yfit = _monotonic_regression(y_clipped)
                yfit[-1] = tau_final
            elif anchor_first and n_pulses > 1:
                # Anchor first tau only
                tau_first = float(y[0])
                y_clipped = np.maximum(y, tau_first)
                yfit = _monotonic_regression(y_clipped)
                yfit[0] = tau_first
            else:
                # Apply monotonic regression without anchors
                yfit = _monotonic_regression(y)
            return tau_r_in, yfit

        # Final validation: ensure output is finite and monotonic
        if not np.all(np.isfinite(yfit)):
            try:
                progress_print(f"[warning] _apply_progression produced NaN/Inf. Using fallback.")
            except Exception:
                pass
            fallback_tau = float(tau_d0_in) if np.isfinite(tau_d0_in) else 0.010
            yfit = np.full(n_pulses, fallback_tau)

        # Ensure all tau values are positive (minimum 0.1 ms)
        yfit = np.maximum(yfit, 1e-4)

        # Ensure final monotonic constraint (safety)
        yfit = np.maximum.accumulate(yfit)

        return tau_r_in, yfit

    # Variables for optional display overlays and logging
    tau_last_display = None
    amp_last_display = None
    tau_d_vec_raw = None  # Initial per-event estimates before any constraints
    tau_d_vec_constrained = None  # After clipping/anchoring (for global mode)
    per_event_param_map: Dict[str, np.ndarray] = {}

    if fit_source == 'global':
        # Match the demo: recut + average all events then fit via curve_fit
        need_snips = bool(
            cfg.get('plot', {}).get('enabled', False)
            or cfg.get('recut_snippets', False)
        )
        res = fit_average_event(
            t, Yd, event_model, stim_times,
            oversample=int(cfg['recut_oversample']),
            projection=str(cfg['recut_projection']).lower(),
            peak_recenter=peak_recenter,
            return_snippets=need_snips,
        )

        fitted = None
        if res is not None:
            fitted, t_avg_evt, y_avg_evt = res

            # Reject clearly invalid global fits for tight ISI
            td_try = float(fitted.get('tau_decay', fitted.get('tau_decay_fast', np.inf)))
            if (not np.isfinite(td_try)) or (td_try > 1.2 * isi):
                fitted = None  # force fallback

            # Pull recut outputs attached by helper (if any)
            if isinstance(fitted, dict) and ('_recut' in fitted):
                try:
                    t_rel_s, avg_s, snippets_s = fitted.pop('_recut')
                    recut_t_rel, recut_avg, recut_snippets = t_rel_s, avg_s, snippets_s
                except Exception:
                    pass

        if fitted is None:
            # ---- Fallback path: recut/grid on average ----
            tr_b, td0_b = _estimate_from_recut_average()
            if tr_b is None:
                tr_b, td0_b, slope, tau_d_vec0 = estimate_kinetics_from_average(
                    t, y_avg, stim_times,
                    taur_grid_ms=cfg['kin_taur_grid_ms'], taud0_grid_ms=cfg['kin_taud0_grid_ms'],
                    slope_grid_ms=cfg['kin_slope_grid_ms'], pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                    isi=isi, weight_mode=cfg['nnls_weight_mode'], weight_tau_s=weight_tau_s,
                    sg_window=cfg['sg_window'], sg_poly=cfg['sg_poly'], event_t0_s=event_t0_s,
                )
            tau_r = float(tr_b); tau_d0 = float(td0_b)

            # Prefer recut average for left panel if available
            global_fit_params = {
                'tau_rise': float(tau_r),
                'tau_decay': float(tau_d0),
            }
            # Use the recut average if available; otherwise fall back to the
            # train-aligned average series.
            if (recut_t_rel is not None) and (recut_avg is not None):
                try:
                    t_avg_evt = np.asarray(recut_t_rel, float) * 1000.0
                    y_avg_evt = np.asarray(recut_avg, float)
                except Exception:
                    t_avg_evt = (t - float(train_start)) * 1000.0
                    y_avg_evt = y_avg
            else:
                t_avg_evt = (t - float(train_start)) * 1000.0
                y_avg_evt = y_avg
            event_t0_s = 0.0  # no peak shift available in fallback
        else:
            # ---- Use parameters from fitted dict ----
            fitted, t_avg_evt, y_avg_evt = res
            # If the fit helper attached recut outputs to the params dict, capture them
            try:
                if isinstance(fitted, dict) and '_recut' in fitted:
                    t_rel_s, avg_s, snippets_s = fitted.pop('_recut')
                    recut_t_rel = t_rel_s
                    recut_avg = avg_s
                    recut_snippets = snippets_s
            except Exception:
                pass
            try:
                global_fit_params = {
                    k: float(v)
                    for k, v in fitted.items()
                    if k != 'amp' and k != 't_peak' and k != '_recut'
                }
            except Exception:
                global_fit_params = {}
            tau_r = float(fitted.get('tau_rise', np.nan))
            tau_d0 = float(fitted.get('tau_decay', np.nan))
            if not np.isfinite(tau_d0):
                tau_d0 = float(fitted.get('tau_decay_fast', np.nan))
                if not np.isfinite(tau_d0):
                    tau_d0 = float(fitted.get('tau_decay_slow', np.nan))
            if not np.isfinite(tau_r):
                tau_r = 0.002
            if not np.isfinite(tau_d0):
                tau_d0 = 0.010
            event_t0_s = float(fitted.get('t_peak', 0.0)) / 1000.0

            # Carry over cooperative n if present (unless user forced a value)
            cfg.setdefault('event_model_settings', {})
            if event_model == 'cooperative' and ('n_coop' in fitted) and ('n_coop' not in explicit_em_settings):
                cfg['event_model_settings']['n_coop'] = float(fitted['n_coop'])
            elif event_model not in varying_supported_names:
                for k, v in fitted.items():
                    if k not in ('amp', 't_peak') and np.isfinite(v) and (k not in explicit_em_settings):
                        cfg['event_model_settings'][k] = float(v)

            # Optional override
            tau_decay_override = cfg['event_model_settings'].get('tau_decay')
            if tau_decay_override is not None:
                try:
                    tau_decay_override = float(tau_decay_override)
                    if np.isfinite(tau_decay_override):
                        tau_d0 = tau_decay_override
                except Exception:
                    pass

            ev_model_name, n_coop_effective = _apply_event_model_from_cfg()
            is_varying_model = ev_model_name in varying_supported_names
            progress_print(f"[fit][global] curve_fit τr={tau_r*1000:.2f}ms τd={tau_d0*1000:.2f}ms model={event_model}")


        # Global fit_source: anchor global tau to middle event, then constrain per-event fits
        if dec_mode == 'fixed':
            # Fixed mode: use single global tau for all events
            tau_d_vec0 = np.full(n_pulses, float(tau_d0))
            tau_d_vec_raw = np.asarray(tau_d_vec0, float)
            tau_d_vec_constrained = None  # No constraints applied in fixed mode
        elif dec_mode in ('linear', 'free_monotonic'):
            # Fit tau for each event on average trace
            tau_per_evt, amp_per_evt, param_map = _fit_all_events_on_average(tau_r, tau_d0)
            per_event_param_map = {k: np.asarray(v, float) for k, v in param_map.items()}
            tau_d_vec_raw = np.asarray(tau_per_evt, float)

            # Anchor global tau at middle event
            mid_idx = n_pulses // 2
            tau_anchor = float(tau_d0)  # Global tau from recut template

            # Apply constraints: force middle event to global tau, then clip others
            tau_constrained = np.copy(tau_per_evt)
            tau_constrained[mid_idx] = tau_anchor  # Force middle event to global tau

            # Events before mid cannot be slower than anchor
            for i in range(mid_idx):
                if tau_constrained[i] > tau_anchor:
                    tau_constrained[i] = tau_anchor

            # Events after mid cannot be faster than anchor
            for i in range(mid_idx + 1, n_pulses):
                if tau_constrained[i] < tau_anchor:
                    tau_constrained[i] = tau_anchor

            tau_d_vec_constrained = np.asarray(tau_constrained, float)
            tau_d_vec0 = tau_constrained
            tau_last_display = float(tau_per_evt[-1])
            amp_last_display = float(amp_per_evt[-1])
        else:
            tau_d_vec0 = np.full(n_pulses, float(tau_d0))
            tau_d_vec_raw = np.asarray(tau_d_vec0, float)
            tau_d_vec_constrained = None

        tau_r, tau_d_vec = _apply_progression(tau_r, tau_d_vec0, tau_d0, 'global')
        if is_varying_model:
            progress_print(f"[fit] source=global | τd0={tau_d_vec[0]*1000:.2f}ms → τdN={tau_d_vec[-1]*1000:.2f}ms | mode={dec_mode}")
    elif fit_source == 'individual':
        # Individual fit_source: fit each trial independently
        tau_rs = []
        tau_d_mat = []
        for j in range(Yd.shape[1]):
            try:
                trj, td0j, slopej, tdvecj = estimate_kinetics_from_average(
                    t, Yd[:, j], stim_times,
                    taur_grid_ms=cfg['kin_taur_grid_ms'], taud0_grid_ms=cfg['kin_taud0_grid_ms'],
                    slope_grid_ms=cfg['kin_slope_grid_ms'], pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                    isi=isi, weight_mode=weight_mode, weight_tau_s=weight_tau_s,
                    sg_window=cfg['sg_window'], sg_poly=cfg['sg_poly'], event_t0_s=event_t0_s,
                )
                tau_rs.append(trj)
                tau_d_mat.append(tdvecj)
            except Exception:
                continue
        tau_r = float(np.nanmedian(tau_rs)) if tau_rs else 0.002
        if tau_d_mat:
            tau_d_vec0 = np.nanmedian(np.vstack(tau_d_mat), axis=0)
        else:
            tau_d_vec0 = np.full(n_pulses, 0.010)
        tau_d0 = float(tau_d_vec0[0])
        tau_d_vec_raw = np.asarray(tau_d_vec0, float)
        per_event_param_map = {}
        tau_r, tau_d_vec = _apply_progression(tau_r, tau_d_vec0, tau_d0, 'individual')
        if is_varying_model:
            progress_print(f"[fit] source=individual | τd0={tau_d_vec[0]*1000:.2f}ms → τdN={tau_d_vec[-1]*1000:.2f}ms | mode={dec_mode}")

    else:  # 'average'
        # Average fit_source: fit each event on the average trace
        if dec_mode == 'fixed':
            # For fixed mode, use grid search for single tau
            tau_r, tau_d0, slope, tau_d_vec0 = estimate_kinetics_from_average(
                t, y_avg, stim_times,
                taur_grid_ms=cfg['kin_taur_grid_ms'], taud0_grid_ms=cfg['kin_taud0_grid_ms'],
                slope_grid_ms=cfg['kin_slope_grid_ms'], pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                isi=isi, weight_mode=weight_mode, weight_tau_s=weight_tau_s,
                sg_window=cfg['sg_window'], sg_poly=cfg['sg_poly'], event_t0_s=event_t0_s,
            )
            tau_d_vec_raw = np.asarray(tau_d_vec0, float)
        elif dec_mode in ('linear', 'free_monotonic'):
            # Fit tau_r globally first
            tau_r, tau_d0, slope, _ = estimate_kinetics_from_average(
                t, y_avg, stim_times,
                taur_grid_ms=cfg['kin_taur_grid_ms'], taud0_grid_ms=cfg['kin_taud0_grid_ms'],
                slope_grid_ms=cfg['kin_slope_grid_ms'], pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                isi=isi, weight_mode=weight_mode, weight_tau_s=weight_tau_s,
                sg_window=cfg['sg_window'], sg_poly=cfg['sg_poly'], event_t0_s=event_t0_s,
            )
            # Now fit each individual event on the average trace
            tau_per_evt, amp_per_evt, param_map = _fit_all_events_on_average(tau_r, tau_d0)
            per_event_param_map = {k: np.asarray(v, float) for k, v in param_map.items()}
            tau_d_vec0 = tau_per_evt
            tau_d_vec_raw = np.asarray(tau_per_evt, float)
            tau_last_display = float(tau_per_evt[-1])
            amp_last_display = float(amp_per_evt[-1])
        else:
            tau_r, tau_d0, slope, tau_d_vec0 = estimate_kinetics_from_average(
                t, y_avg, stim_times,
                taur_grid_ms=cfg['kin_taur_grid_ms'], taud0_grid_ms=cfg['kin_taud0_grid_ms'],
                slope_grid_ms=cfg['kin_slope_grid_ms'], pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
                isi=isi, weight_mode=weight_mode, weight_tau_s=weight_tau_s,
                sg_window=cfg['sg_window'], sg_poly=cfg['sg_poly'], event_t0_s=event_t0_s,
            )
            tau_d_vec_raw = np.asarray(tau_d_vec0, float)

        tau_r, tau_d_vec = _apply_progression(tau_r, tau_d_vec0, tau_d0, 'average')
        if is_varying_model:
            progress_print(f"[fit] source=average | τd0={tau_d_vec[0]*1000:.2f}ms → τdN={tau_d_vec[-1]*1000:.2f}ms | mode={dec_mode}")

    # Build interpolated-setting summaries for diagnostics and downstream use
    anchor_first_cfg = bool(cfg.get('anchor_first_tau', False))
    anchor_final_cfg = bool(cfg.get('anchor_final_tau', True))
    anchor_applicable = dec_mode in ('linear', 'free_monotonic')
    mid_index = int(min(max(0, n_pulses // 2), max(0, n_pulses - 1))) if n_pulses > 0 else 0
    anchor_template = {
        'first': anchor_applicable and anchor_first_cfg and n_pulses > 0,
        'final': anchor_applicable and anchor_final_cfg and n_pulses > 0,
        'mid': anchor_applicable and (fit_source == 'global') and n_pulses > 0,
        'mid_index': mid_index,
    }
    progression_label_map = {
        'fixed': 'fixed (median)',
        'linear': 'linear regression',
        'free_monotonic': 'monotonic spline',
    }
    progression_label = progression_label_map.get(dec_mode, dec_mode)
    if anchor_applicable:
        if anchor_first_cfg and anchor_final_cfg:
            progression_label += ' (anchored first & final)'
        elif anchor_final_cfg:
            progression_label += ' (anchored to final)'
        elif anchor_first_cfg:
            progression_label += ' (anchored to first)'

    interpolated_settings = []
    if n_pulses > 0 and np.size(tau_d_vec):
        def _format_display(name: str, override: Optional[str] = None) -> str:
            if override is not None:
                return override
            if not name:
                return name
            if name.startswith('tau'):
                return 'τ' + name[3:]
            if name.startswith('t_'):
                return 't' + name[2:]
            return name

        def _register_setting(
            name: str,
            final_values,
            *,
            raw=None,
            constrained=None,
            label: Optional[str] = None,
            unit: str = 'ms',
            scale: float = 1000.0,
            note: Optional[str] = None,
            direction_note: Optional[str] = None,
            derivation: Optional[str] = None,
        ) -> None:
            arr_final = np.asarray(final_values, float)
            if arr_final.ndim == 0:
                arr_final = np.full(n_pulses, float(arr_final))
            if arr_final.size != n_pulses or not np.any(np.isfinite(arr_final)):
                return

            arr_raw = None
            if raw is not None:
                arr_raw = np.asarray(raw, float)
                if arr_raw.ndim == 0:
                    arr_raw = np.full(n_pulses, float(arr_raw))
                if arr_raw.size != n_pulses or not np.any(np.isfinite(arr_raw)):
                    arr_raw = None

            arr_constrained = None
            if constrained is not None:
                arr_constrained = np.asarray(constrained, float)
                if arr_constrained.ndim == 0:
                    arr_constrained = np.full(n_pulses, float(arr_constrained))
                if arr_constrained.size != n_pulses or not np.any(np.isfinite(arr_constrained)):
                    arr_constrained = None

            entry: Dict[str, Any] = {
                'name': name,
                'display': label or _format_display(name),
                'unit': unit,
                'scale': float(scale),
                'raw': arr_raw,
                'constrained': arr_constrained,
                'final': arr_final,
                'note': note,
                'direction_note': direction_note,
                'fit_source': fit_source,
                'decay_progression_mode': dec_mode,
                'progression_label': progression_label,
                'derivation': derivation or 'direct',
            }

            anchors: List[Dict[str, Any]] = []
            if anchor_template['first']:
                anchors.append({'kind': 'first', 'index': 0, 'value': float(arr_final[0])})
            if anchor_template['mid'] and 0 <= anchor_template['mid_index'] < n_pulses:
                idx = int(anchor_template['mid_index'])
                anchors.append({'kind': 'mid', 'index': idx, 'value': float(arr_final[idx])})
            if anchor_template['final']:
                anchors.append({'kind': 'final', 'index': n_pulses - 1, 'value': float(arr_final[-1])})
            entry['anchors'] = anchors
            interpolated_settings.append(entry)

        def _progress_metric_series(
            raw_series,
            base_value,
            *,
            min_value: float = 1e-4,
            clip: Optional[Tuple[Optional[float], Optional[float]]] = None,
        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            if raw_series is None:
                return None, None, None
            arr_raw = np.asarray(raw_series, float)
            if arr_raw.size != n_pulses or not np.any(np.isfinite(arr_raw)):
                return None, None, None

            finite_vals = arr_raw[np.isfinite(arr_raw)]
            anchor_val = None
            if base_value is not None and np.isfinite(base_value):
                anchor_val = float(base_value)

            fallback_candidates = []
            if anchor_val is not None:
                fallback_candidates.append(anchor_val)
            if finite_vals.size:
                fallback_candidates.append(float(np.nanmedian(finite_vals)))
                fallback_candidates.append(float(finite_vals[0]))

            fallback_val = None
            for candidate in fallback_candidates:
                if np.isfinite(candidate):
                    fallback_val = float(candidate)
                    break
            if fallback_val is None:
                fallback_val = 0.010

            if min_value is not None:
                fallback_val = max(float(min_value), float(fallback_val))

            arr_clean = arr_raw.copy()
            arr_clean[~np.isfinite(arr_clean)] = fallback_val
            if min_value is not None:
                arr_clean = np.maximum(arr_clean, float(min_value))

            arr_constrained = arr_clean.copy()
            if fit_source == 'global' and anchor_template['mid'] and anchor_val is not None:
                idx = anchor_template['mid_index']
                if 0 <= idx < n_pulses:
                    arr_constrained[idx] = anchor_val
            if min_value is not None:
                arr_constrained = np.maximum(arr_constrained, float(min_value))

            _, progressed = _apply_progression(tau_r, arr_constrained, fallback_val, fit_source)
            progressed = np.asarray(progressed, float)

            if clip is not None:
                lo, hi = clip
                if lo is not None:
                    progressed = np.maximum(progressed, float(lo))
                    arr_constrained = np.maximum(arr_constrained, float(lo))
                    arr_clean = np.maximum(arr_clean, float(lo))
                if hi is not None:
                    progressed = np.minimum(progressed, float(hi))
                    arr_constrained = np.minimum(arr_constrained, float(hi))
                    arr_clean = np.minimum(arr_clean, float(hi))

            return progressed, arr_clean, arr_constrained

        if np.isfinite(tau_r):
            global_fit_params.setdefault('tau_rise', float(tau_r))
        if np.isfinite(tau_d0):
            if 'tau_decay_fast' not in global_fit_params and 'tau_decay' not in global_fit_params:
                global_fit_params.setdefault('tau_decay', float(tau_d0))

        fast_key = 'tau_decay_fast' if 'tau_decay_fast' in global_fit_params else 'tau_decay'
        fast_label = _format_display(fast_key)
        _register_setting(
            fast_key,
            tau_d_vec,
            raw=tau_d_vec_raw,
            constrained=tau_d_vec_constrained,
            label=fast_label,
            unit='ms',
            scale=1000.0,
            note='fast component (directly fitted progression)',
            derivation='fitted_progression',
        )

        fast_global = None
        if fast_key in global_fit_params and np.isfinite(global_fit_params[fast_key]):
            fast_global = float(global_fit_params[fast_key])
        elif 'tau_decay' in global_fit_params and np.isfinite(global_fit_params['tau_decay']):
            fast_global = float(global_fit_params['tau_decay'])
        elif np.isfinite(tau_d0):
            fast_global = float(tau_d0)

        slow_seq = None
        slow_seq_raw = None
        slow_seq_constrained = None
        slow_from_fit = False
        slow_base = global_fit_params.get('tau_decay_slow')
        if dec_mode in ('linear', 'free_monotonic'):
            slow_seq, slow_seq_raw, slow_seq_constrained = _progress_metric_series(
                per_event_param_map.get('tau_decay_slow'),
                slow_base,
            )
            if slow_seq is not None:
                slow_from_fit = True
                _register_setting(
                    'tau_decay_slow',
                    slow_seq,
                    raw=slow_seq_raw,
                    constrained=slow_seq_constrained,
                    label=_format_display('tau_decay_slow'),
                    unit='ms',
                    scale=1000.0,
                    note='slower component (directly fitted progression)',
                    derivation='fitted_progression',
                )

        if (
            slow_seq is None
            and slow_base is not None
            and np.isfinite(slow_base)
            and fast_global is not None
            and fast_global > 0
        ):
            ratio_slow = float(slow_base) / float(fast_global)
            if np.isfinite(ratio_slow) and ratio_slow > 0:
                slow_seq = tau_d_vec * ratio_slow
                slow_seq_raw = tau_d_vec_raw * ratio_slow if tau_d_vec_raw is not None else None
                slow_seq_constrained = (
                    tau_d_vec_constrained * ratio_slow if tau_d_vec_constrained is not None else None
                )
                _register_setting(
                    'tau_decay_slow',
                    slow_seq,
                    raw=slow_seq_raw,
                    constrained=slow_seq_constrained,
                    label=_format_display('tau_decay_slow'),
                    unit='ms',
                    scale=1000.0,
                    note='slower component (scaled by global τ_slow/τ_fast)',
                    derivation='scaled_from_fast_progression',
                )

        if slow_seq is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_final = np.divide(
                    slow_seq,
                    tau_d_vec,
                    out=np.full_like(slow_seq, np.nan),
                    where=np.isfinite(tau_d_vec) & (np.abs(tau_d_vec) > 0),
                )
            ratio_raw = None
            if slow_seq_raw is not None and tau_d_vec_raw is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_raw = np.divide(
                        slow_seq_raw,
                        tau_d_vec_raw,
                        out=np.full_like(slow_seq_raw, np.nan),
                        where=np.isfinite(tau_d_vec_raw) & (np.abs(tau_d_vec_raw) > 0),
                    )
            ratio_constrained = None
            if slow_seq_constrained is not None and tau_d_vec_constrained is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_constrained = np.divide(
                        slow_seq_constrained,
                        tau_d_vec_constrained,
                        out=np.full_like(slow_seq_constrained, np.nan),
                        where=np.isfinite(tau_d_vec_constrained) & (np.abs(tau_d_vec_constrained) > 0),
                    )
            if np.any(np.isfinite(ratio_final)):
                _register_setting(
                    'tau_ratio_slow_fast',
                    ratio_final,
                    raw=ratio_raw,
                    constrained=ratio_constrained,
                    label=_format_display('', 'τ_slow/τ_fast'),
                    unit='',
                    scale=1.0,
                    note='slow/fast decay ratio (from fitted slow & fast progressions)' if slow_from_fit else 'slow/fast decay ratio (global constant)',
                    derivation='derived_from_fitted_components' if slow_from_fit else 'global_ratio_constant',
                )

        rise_base = global_fit_params.get('tau_rise', float(tau_r) if np.isfinite(tau_r) else np.nan)
        rise_seq = None
        rise_raw = None
        rise_constrained = None
        rise_from_fit = False
        if dec_mode in ('linear', 'free_monotonic'):
            rise_seq, rise_raw, rise_constrained = _progress_metric_series(
                per_event_param_map.get('tau_rise'),
                rise_base,
            )
            if rise_seq is not None:
                rise_from_fit = True
                _register_setting(
                    'tau_rise',
                    rise_seq,
                    raw=rise_raw,
                    constrained=rise_constrained,
                    label=_format_display('tau_rise'),
                    unit='ms',
                    scale=1000.0,
                    direction_note='lower = faster rise',
                    note='rise component (directly fitted progression)',
                    derivation='fitted_progression',
                )

        if (
            rise_seq is None
            and rise_base is not None
            and np.isfinite(rise_base)
            and fast_global is not None
            and fast_global > 0
        ):
            ratio_rise = float(rise_base) / float(fast_global)
            if np.isfinite(ratio_rise) and ratio_rise > 0:
                rise_seq = tau_d_vec * ratio_rise
                rise_raw = tau_d_vec_raw * ratio_rise if tau_d_vec_raw is not None else None
                rise_constrained = (
                    tau_d_vec_constrained * ratio_rise if tau_d_vec_constrained is not None else None
                )
                _register_setting(
                    'tau_rise',
                    rise_seq,
                    raw=rise_raw,
                    constrained=rise_constrained,
                    label=_format_display('tau_rise'),
                    unit='ms',
                    scale=1000.0,
                    direction_note='lower = faster rise',
                    note='derived from fast decay using global τ_rise/τ_fast ratio',
                    derivation='scaled_from_fast_progression',
                )

        frac_fast = global_fit_params.get('frac_fast')
        frac_seq = None
        frac_raw = None
        frac_constrained = None
        frac_from_fit = False
        if dec_mode in ('linear', 'free_monotonic'):
            frac_seq, frac_raw, frac_constrained = _progress_metric_series(
                per_event_param_map.get('frac_fast'),
                frac_fast,
                min_value=1e-6,
                clip=(0.0, 1.0),
            )
            if frac_seq is not None:
                frac_from_fit = True
                _register_setting(
                    'frac_fast',
                    frac_seq,
                    raw=frac_raw,
                    constrained=frac_constrained,
                    label=_format_display('frac_fast'),
                    unit='',
                    scale=1.0,
                    note='fast component weight (directly fitted progression)',
                    derivation='fitted_progression',
                )

        if not frac_from_fit and frac_fast is not None and np.isfinite(frac_fast):
            frac_arr = np.full(n_pulses, float(frac_fast), float)
            _register_setting(
                'frac_fast',
                frac_arr,
                raw=frac_arr.copy(),
                constrained=None,
                label=_format_display('frac_fast'),
                unit='',
                scale=1.0,
                note='fast component weight (global constant)',
                derivation='global_constant',
            )

    # Update exponential weight tau if using global fit_source and auto tau
    if (weight_mode == 'exponential' and cfg.get('nnls_weight_tau_s', None) is None
        and cfg.get('fit_source', 'global') == 'global'):
        weight_tau_s = float(tau_d_vec[0])  # Use estimated tau_d

    # Validate tau_d_vec and apply fallbacks if needed
    tau_d_vec = np.asarray(tau_d_vec, float)
    if not np.all(np.isfinite(tau_d_vec)):
        bad_indices = np.where(~np.isfinite(tau_d_vec))[0]
        try:
            progress_print(f"[warning] tau_d_vec contains NaN/Inf at indices {bad_indices.tolist()}. Applying fallback.")
        except Exception:
            pass
        # Replace bad values with tau_d0 or nearest valid value
        if np.isfinite(tau_d0):
            tau_d_vec[~np.isfinite(tau_d_vec)] = tau_d0
        else:
            # Last resort: use 10ms default
            tau_d_vec[~np.isfinite(tau_d_vec)] = 0.010
        # Ensure monotonic after fixing
        tau_d_vec = np.maximum.accumulate(tau_d_vec)

    # Validate tau_r
    if not np.isfinite(tau_r):
        try:
            progress_print(f"[warning] tau_r is NaN/Inf. Using fallback 0.002s.")
        except Exception:
            pass
        tau_r = 0.002

    # Print the τd vector to be used for constrained refitting
    try:
        td_ms_list = ", ".join(f"{v*1000:.2f}" for v in tau_d_vec.tolist())
        progress_print(f"[constrain] τd vector (ms) prior to NNLS refit: [{td_ms_list}]")
        progress_print("[constrain] Refitting average and trials with τd fixed per pulse to this vector (amp+jitter only).")
    except Exception:
        pass

    # Debug: Check which array has NaN/Inf
    if not np.all(np.isfinite(y_avg)):
        nan_count = np.sum(~np.isfinite(y_avg))
        nan_indices = np.where(~np.isfinite(y_avg))[0]
        progress_print(f"[debug] y_avg contains {nan_count} NaN/Inf values at indices: {nan_indices[:10].tolist()}{'...' if len(nan_indices) > 10 else ''}")
    if not np.all(np.isfinite(t)):
        nan_count = np.sum(~np.isfinite(t))
        nan_indices = np.where(~np.isfinite(t))[0]
        progress_print(f"[debug] t contains {nan_count} NaN/Inf values at indices: {nan_indices[:10].tolist()}{'...' if len(nan_indices) > 10 else ''}")
    if not np.all(np.isfinite(stim_times)):
        nan_count = np.sum(~np.isfinite(stim_times))
        nan_indices = np.where(~np.isfinite(stim_times))[0]
        progress_print(f"[debug] stim_times contains {nan_count} NaN/Inf values at indices: {nan_indices.tolist()}")

    # Fit average trace (forward, no overlap) and measure amplitudes
    a_avg, d_avg, X_avg, yhat_avg, comp_avg = fit_amplitudes_no_overlap_forward(
        y_avg, t, stim_times, tau_r, tau_d_vec,
        pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
        robust=True, huber_delta=cfg['huber_delta'], irls_iters=cfg['irls_iters'],
        allow_shift=allow_shift, delta_max_s=cfg['delta_max_s'], delta_step_s=cfg['delta_step_s'],
        shift_min_s=cfg['shift_min_s'], event_t0_s=event_t0_s,
    )
    need_avg_sg = (
        ('savgol' in traces)
        or (meas == 'SAVGOL')
        or (failm == 'SAVGOL')
        or (weight_mode == 'savgol')
    )
    if need_avg_sg:
        y_sg_avg = sg_smooth(fill_nans_timewise(y_avg, t), sgW, sgP)
    else:
        y_sg_avg = None

    amp_nnls_avg = compute_localmax_corrected_amps(
        t, yhat_avg, stim_times, win_ms, n_avg, pre_ms, d_avg, tau_r, tau_d_vec,
        event_t0_s=event_t0_s,
    )
    amp_nnls_corr_avg = compute_peak_corrected_from_components(
        t, yhat_avg, stim_times, comp_avg, win_ms=win_ms, pre_ms=pre_ms
    )
    def _norm(a):
        a = np.asarray(a, float)
        d = a[0] if a.size else np.nan
        return a / d if np.isfinite(d) and abs(d) > 1e-12 else a * np.nan
    ppr_nnls_avg = _norm(amp_nnls_avg)
    ppr_nnls_corr_avg = _norm(amp_nnls_corr_avg)
    amp_raw_avg = compute_localmax_corrected_amps(
        t, y_avg, stim_times, win_ms, n_avg, pre_ms, d_avg, tau_r, tau_d_vec,
        event_t0_s=event_t0_s,
    )
    amp_raw_corr_avg = compute_peak_corrected_from_components(
        t, y_avg, stim_times, comp_avg, win_ms=win_ms, pre_ms=pre_ms
    )
    amp_sg_avg = compute_localmax_corrected_amps(
        t,
        y_sg_avg if y_sg_avg is not None else y_avg,
        stim_times,
        win_ms,
        n_avg,
        pre_ms,
        d_avg,
        tau_r,
        tau_d_vec,
        event_t0_s=event_t0_s,
    )
    amp_sg_corr_avg = compute_peak_corrected_from_components(
        t, (y_sg_avg if y_sg_avg is not None else y_avg), stim_times, comp_avg, win_ms=win_ms, pre_ms=pre_ms
    )

    # Optional: average plot with a left event-fit panel (0-50 ms) + right main plot
    figure = None
    fit_diag_figure = None
    if want_plot:
        # If residual diagnostics requested, allocate an extra bottom row
        if plot_residuals:
            figure = plt.figure(figsize=(12, 9.2))
            gs = figure.add_gridspec(2, 2, height_ratios=[2.4, 1.4], width_ratios=[1.5, 4], wspace=0.15, hspace=0.28)
        else:
            figure = plt.figure(figsize=(12, 5))
            gs = figure.add_gridspec(1, 2, width_ratios=[1.5, 4], wspace=0.15)

        resid_avg = None
        model_avg_for_resid = None
        if plot_residuals:
            if meas == 'SAVGOL' and (y_sg_avg is not None):
                model_avg_for_resid = y_sg_avg
            else:
                model_avg_for_resid = yhat_avg
            if model_avg_for_resid is not None:
                try:
                    resid_avg = np.asarray(y_avg - model_avg_for_resid, float)
                except Exception:
                    resid_avg = None
        # Left: aggregated event + model fit (−3..next stim−guard)
        axL = figure.add_subplot(gs[0, 0])
        _trim_spines(axL)
        try:
            t_ms_evt = t_avg_evt if 't_avg_evt' in locals() else (t - float(train_start)) * 1000.0
            y_evt = y_avg_evt if 'y_avg_evt' in locals() else y_avg
            isi_ms = float(isi) * 1000.0
            # Display window: from -3 ms before stim to just before next stim
            # Use a 3 ms guard before the next stimulus to avoid overlap
            min_x = -3.0
            max_x = max(isi_ms - 3.0, 0.0)
            m0 = (t_ms_evt >= min_x) & (t_ms_evt < max_x)
            axL.plot(t_ms_evt[m0], y_evt[m0], color='k', lw=1.5, label='Average')
            # Overlay best-fit library model matching current kernel choice
            try:
                try:
                    from Model_Calibration.event_models import get_event_model
                except Exception:
                    from event_models import get_event_model  # type: ignore
                # Resolve the effective model name
                _name = event_model
                if _name.startswith('library:'):
                    _name = _name.split(':', 1)[1].strip().lower()
                spec = get_event_model(_name)
                tf = t_ms_evt[m0]; yf = y_evt[m0]
                # Build overlay params to reflect the model actually used:
                popt = None
                if _name in {'double_exp','cooperative','bilinear'}:
                    # Use the kinetics selected for this run (tau_r, tau_d0)
                    # Respect any fitted t_peak so the overlay shifts correctly
                    t_peak_ms = 0.0
                    try:
                        if 'fitted' in locals() and fitted is not None:
                            t_peak_ms = float(fitted.get('t_peak', 0.0))
                    except Exception:
                        t_peak_ms = 0.0
                    if _name == 'double_exp':
                        # [amp, tau_rise(s), tau_decay(s), t_peak(ms)]
                        pars = [1.0, float(tau_r), float(tau_d0), t_peak_ms]
                    elif _name == 'cooperative':
                        ems = cfg.get('event_model_settings', {}) or {}
                        n_used = float(ems.get('n_coop', 2.0))
                        # [amp, tau_rise(s), tau_decay(s), n_coop, t_peak(ms)]
                        pars = [1.0, float(tau_r), float(tau_d0), n_used, t_peak_ms]
                    else:  # bilinear expects ms values for rise/decay durations
                        pars = [1.0, float(tau_r)*1000.0, float(tau_d0)*1000.0, t_peak_ms]
                    yshape = spec['func'](tf, *pars)
                    denom = float(np.sum(yshape**2)) if np.isfinite(yshape).any() else 0.0
                    amp_ls = float(np.sum(yf*yshape))/denom if denom > 0 else 1.0
                    pars[0] = amp_ls
                    popt = pars
                    yhat_ev = spec['func'](tf, *popt)
                else:
                    # For fixed-template models, DO NOT refit here: honor
                    # the effective event_model_settings that were applied to
                    # the kernel earlier. Build the parameter vector in the
                    # order expected by the spec and only solve a linear LS
                    # for amplitude so the overlay matches scale.
                    params = []
                    t_peak_ms = 0.0
                    try:
                        if 'fitted' in locals() and fitted is not None:
                            t_peak_ms = float(fitted.get('t_peak', 0.0))
                    except Exception:
                        t_peak_ms = 0.0
                    ems = cfg.get('event_model_settings', {}) or {}
                    for name in spec['params']:
                        if name == 'amp':
                            params.append(1.0)
                        elif name == 't_peak':
                            params.append(t_peak_ms)
                        else:
                            # Prefer values from the recent global fit if available;
                            # then explicit user overrides in event_model_settings;
                            # then tau_r/tau_d0 mapping; finally default p0.
                            if 'fitted' in locals() and isinstance(fitted, dict) and name in fitted and np.isfinite(fitted.get(name, np.nan)):
                                params.append(float(fitted[name]))
                            elif name in ems and np.isfinite(ems.get(name, np.nan)):
                                params.append(float(ems[name]))
                            elif name == 'tau_decay':
                                params.append(float(tau_d0))  # seconds
                            elif name == 'tau_rise':
                                params.append(float(tau_r))
                            else:
                                # If unknown, fall back to p0 for stability
                                p0 = spec['p0_func'](yf, tf)
                                idx = spec['params'].index(name)
                                params.append(float(p0[idx]))
                    # Compute LS amplitude against the fixed shape
                    yshape = spec['func'](tf, *params)
                    denom = float(np.sum(yshape**2)) if np.isfinite(yshape).any() else 0.0
                    amp_ls = float(np.sum(yf * yshape)) / denom if denom > 0 else 1.0
                    params[0] = amp_ls
                    popt = params
                    yhat_ev = spec['func'](tf, *popt)
                axL.plot(tf, yhat_ev, color='crimson', ls='--', lw=1.8, label=_name)
                try:
                    # Omit 't_peak' and stack vertically; include amp at top for context
                    pairs = [(n, v) for n, v in zip(spec['params'], popt)]
                    pairs = [(n, v) for n, v in pairs if n != 't_peak']
                    txt = "\n".join(f"{n}={v:.3g}" for n, v in pairs)
                    axL.text(0.98, 0.98, txt, transform=axL.transAxes, fontsize=8,
                             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except Exception:
                    pass
            except Exception:
                pass
            axL.axvline(0.0, color='k', ls=':', alpha=0.5, lw=0.8)
            axL.set_xlim(min_x, max_x)
            # Autoscale y with a small margin to avoid a squashed panel
            try:
                y_slice = y_evt[m0]
                ymins = np.nanmin(y_slice) if np.size(y_slice) else 0.0
                ymaxs = np.nanmax(y_slice) if np.size(y_slice) else 1.0
                if 'yhat_ev' in locals():
                    ymins = min(ymins, float(np.nanmin(yhat_ev)))
                    ymaxs = max(ymaxs, float(np.nanmax(yhat_ev)))
                span = max(1e-6, ymaxs - ymins)
                pad = 0.1 * span
                axL.set_ylim(ymins - pad, ymaxs + pad)
            except Exception:
                pass
            axL.set_xlabel('Time (ms)')
            axL.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
            try:
                axL.set_title(f'Event fit (−3.0–{max_x:.1f} ms)', fontsize=10)
            except Exception:
                axL.set_title('Event fit', fontsize=10)
        except Exception:
            pass

        # Right: main average plot
        ax = figure.add_subplot(gs[0, 1])
        _trim_spines(ax)
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
        if plot_residuals and resid_avg is not None:
            try:
                resid_offset = -1
                ax.plot(
                    tz,
                    resid_avg[zmask] + resid_offset,
                    label='residual (offset)',
                    color='tab:purple',
                    linewidth=1.2,
                    alpha=0.9,
                )
                ax.axhline(resid_offset, color='tab:purple', linestyle=':', linewidth=0.8, alpha=0.7)
            except Exception:
                pass
        # Replace the orange line plotting section with:
        # Replace the orange line plotting with cumulative reconstruction:
        if show_decay and 'nnls' in traces and comp_avg is not None:
            cumulative = np.zeros_like(y_avg)
            for p in range(len(stim_times)):
                if p >= len(comp_avg):
                    continue
                # Add this component to the cumulative sum
                cumulative = cumulative + comp_avg[p]

                # Plot the cumulative reconstruction up to this point
                # This shows the "baseline" including all decay from previous pulses
                ax.plot(
                    tz,
                    cumulative[zmask],
                    color='tab:orange',
                    linestyle='--',
                    linewidth=1.0,
                    alpha=0.6 - p * 0.04,  # Fade with each pulse
                )
        # If anchored (linear/monotonic), show the last-event pre-refit fit as an additional red overlay
        try:
            if dec_mode in ('linear','free_monotonic') and (tau_last_display is not None) and (amp_last_display is not None):
                last_st = float(stim_times[-1]) + event_t0_s
                k_last = _KERNEL_FUN(tz - last_st, tau_r, float(tau_last_display))
                ax.plot(tz, float(amp_last_display) * k_last, color='crimson', linestyle='--', linewidth=1.4, alpha=0.9, label='last fit (pre-refit)')
        except Exception:
            pass

        # Optional: overlay peak markers and residual-at-peak triangles
        if plot_peaks_details:
            # Choose which series defines the "measurement" trace for peak picking
            if meas == 'SAVGOL' and (y_sg_avg is not None):
                y_for_peaks = y_sg_avg
            elif meas == 'RAW':
                y_for_peaks = y_avg
            else:
                y_for_peaks = yhat_avg

            # Build cumulative baseline from previous pulses only using NNLS components
            # For pulse p, baseline_prev[p, :] = sum_{k < p} comp_avg[k]
            baseline_prev_only = []
            if comp_avg is not None:
                cum = np.zeros_like(y_avg)
                for p in range(len(stim_times)):
                    baseline_prev_only.append(cum.copy())
                    if p < len(comp_avg):
                        cum = cum + comp_avg[p]

            peak_ts: list = []
            peak_vals: list = []
            resid_vals: list = []
            for p, st in enumerate(stim_times):
                tp, vp = pick_peak_on_series(t, y_for_peaks, float(st), win_ms, pre_ms)
                peak_ts.append(float(tp))
                peak_vals.append(float(vp))
                # Residual-under-peak = baseline from prior pulses at that time
                try:
                    i0 = int(np.argmin(np.abs(t - tp)))
                    base_prev = baseline_prev_only[p][i0] if baseline_prev_only else 0.0
                    resid_vals.append(float(base_prev))
                except Exception:
                    resid_vals.append(np.nan)

            # Red circles at peaks (on the chosen measurement trace)
            try:
                ax.scatter(peak_ts, peak_vals, s=70, color='red', edgecolors='white', linewidths=0.9, zorder=6, label='peaks')
            except Exception:
                pass
            # Down-pointing triangles for residual at peak time
            try:
                # Triangles sit on the orange dashed baseline (previous pulses only)
                ax.scatter(peak_ts, resid_vals, s=60, marker='v', facecolors='white', edgecolors='tab:red', linewidths=1.0, zorder=5, label='residual at peak')
            except Exception:
                pass
        ax.set_xlim(z0, z1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
        ax.legend(loc='upper right', frameon=False)
        ax.set_title('Average trace (selected overlays)')

        # Residual diagnostics panel (average): place directly under the main
        # average panel and show a small inset histogram (no separate figure).
        if plot_residuals and (resid_avg is not None) and (model_avg_for_resid is not None):
            try:
                # Bottom-right: residual trace aligned with the top-right panel
                axR = figure.add_subplot(gs[1, 1], sharex=ax)
                axR.plot(tz, resid_avg[zmask], color='tab:purple', lw=1.2, label='residual (avg − model)')
                axR.axvline(float(train_start), color='k', ls=':', lw=0.8, alpha=0.6)
                axR.set_xlim(z0, z1)
                axR.set_xlabel('Time (s)')
                axR.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
                axR.set_title('Residuals (average)')
                axR.legend(loc='upper right', frameon=False, fontsize=8)
                _trim_spines(axR)

                # Inset histogram of residuals in the zoom window with fixed bins and Gaussian fit
                try:
                    ax_in = axR.inset_axes([0.70, 0.55, 0.28, 0.4])
                    rv = np.asarray(resid_avg[zmask], float)
                    rv = rv[np.isfinite(rv)]
                    if rv.size:
                        bin_w = (0.01 if use_dff else 10.0)
                        lo = float(np.nanmin(rv))
                        hi = float(np.nanmax(rv))
                        if not np.isfinite(lo):
                            lo = 0.0
                        if not np.isfinite(hi) or hi <= lo:
                            hi = lo + bin_w
                        edges = np.arange(lo, hi + bin_w, bin_w)
                        ax_in.hist(rv, bins=edges, color='#d8c7e8', edgecolor='#6b4fa3')
                        # Gaussian fit overlay across full inset range
                        try:
                            mu = float(np.nanmean(rv))
                            sigma = float(np.nanstd(rv))
                        except Exception:
                            mu, sigma = float('nan'), float('nan')
                        if np.isfinite(sigma) and sigma > 0:
                            x0, x1 = ax_in.get_xlim()
                            x = np.linspace(x0, x1, 400)
                            pdf = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                            N = rv.size
                            y = N * bin_w * pdf
                            ax_in.plot(x, y, color='#26457a', linewidth=1.4, label='fit')
                    ax_in.set_title('residual', fontsize=8)
                    ax_in.tick_params(labelsize=7)
                except Exception:
                    pass

                # Add an empty placeholder under the left event-fit panel to
                # keep the grid balanced.
                try:
                    ax_placeholder = figure.add_subplot(gs[1, 0])
                    ax_placeholder.axis('off')
                except Exception:
                    pass
            except Exception:
                pass

        # Show the complete average figure immediately
        try:
            plt.show(block=False); plt.pause(0.05)
        except Exception:
            pass

        # Fit diagnostic figure: show immediately after average figure
        if cfg.get('fit_diagnostic_plot', False) and interpolated_settings:
            try:
                # Layout with uniform setting panels and a dedicated legend panel
                # at the top-left of the settings grid. Weight kernel remains
                # full-width on the first row.
                import math
                # Order settings (put fast decay first if present)
                settings_ordered = list(interpolated_settings)
                for i, info in enumerate(interpolated_settings):
                    name_i = str(info.get('name', '')).lower()
                    if name_i in {'tau_decay', 'tau_decay_fast'} and i != 0:
                        settings_ordered.insert(0, settings_ordered.pop(i))
                        break

                total_panels = 1 + len(settings_ordered)  # +1 for legend
                n_cols = min(3, total_panels) if total_panels > 0 else 1
                n_rows = int(math.ceil(total_panels / float(n_cols)))

                fig_width = 5.5 + 3.2 * n_cols
                fit_diag_figure = plt.figure(figsize=(fig_width, 6.8))
                height_ratios = [1.2] + [1.0] * n_rows
                gs_fd = fit_diag_figure.add_gridspec(
                    1 + n_rows, n_cols, height_ratios=height_ratios, hspace=0.28, wspace=0.22
                )

                # Weight kernel axis (full width)
                ax_w = fit_diag_figure.add_subplot(gs_fd[0, :])

                # Weight kernel panel
                if weight_mode == 'savgol' and y_sg_avg is None:
                    y_vis = sg_smooth(fill_nans_timewise(y_avg, t), sgW, sgP)
                else:
                    y_vis = y_sg_avg
                weights = _calculate_nnls_weights(
                    t,
                    stim_times,
                    isi,
                    weight_mode,
                    weight_tau_s,
                    y_ref=y_vis,
                )
                ax_w.plot(t, weights, color='#1f77b4', linewidth=2.0, label=f'{weight_mode} weight')
                for i, st in enumerate(stim_times):
                    label = 'stimulus' if i == 0 else None
                    ax_w.axvline(st, color='red', linestyle='--', alpha=0.6, linewidth=0.9, label=label)
                ax_w.set_xlabel('Time (s)')
                ax_w.set_ylabel('Weight')
                if weight_tau_s is not None:
                    ax_w.set_title(f'Weight kernel (τ={weight_tau_s*1000:.1f} ms)')
                else:
                    ax_w.set_title('Weight kernel')
                # No per-axis legend; a global legend is rendered separately
                ax_w.grid(True, alpha=0.2)

                anchor_styles = {
                    'first': {'color': 'green', 'vline': '-.', 'hline': ':', 'marker': 's', 'size': 250},
                    'mid': {'color': '#1f77b4', 'vline': '--', 'hline': ':', 'marker': 'D', 'size': 200},
                    'final': {'color': 'purple', 'vline': '-.', 'hline': ':', 'marker': '*', 'size': 300},
                }
                anchor_names = {
                    'first': 'first event',
                    'mid': 'mid event',
                    'final': 'final event',
                }

                def _plot_setting_axis(ax, info: Dict[str, Any]) -> None:
                    _trim_spines(ax)
                    pulses = np.arange(1, np.asarray(info['final']).size + 1, dtype=float)
                    scale = float(info.get('scale', 1.0))
                    label = str(info.get('display', info.get('name', 'setting')))
                    unit = str(info.get('unit', ''))
                    final_vals = np.asarray(info['final'], float) * scale

                    raw_vals = info.get('raw')
                    if raw_vals is not None and np.any(np.isfinite(raw_vals)):
                        ax.plot(
                            pulses,
                            np.asarray(raw_vals, float) * scale,
                            marker='o',
                            markersize=8,
                            color='#d62728',
                            linestyle=':',
                            linewidth=1.5,
                            alpha=0.8,
                            zorder=3,
                        )

                    constrained_vals = info.get('constrained')
                    if constrained_vals is not None and np.any(np.isfinite(constrained_vals)):
                        ax.scatter(
                            pulses,
                            np.asarray(constrained_vals, float) * scale,
                            marker='x',
                            s=80,
                            color='#ff7f0e',
                            linewidths=2,
                            alpha=0.8,
                            zorder=4,
                        )

                    ax.plot(
                        pulses,
                        final_vals,
                        marker='s',
                        markersize=6,
                        color='#2ca02c',
                        linewidth=2.5,
                        alpha=0.9,
                        zorder=5,
                    )

                    for anchor in info.get('anchors', []):
                        kind = anchor.get('kind')
                        style = anchor_styles.get(kind)
                        if style is None:
                            continue
                        idx = int(anchor.get('index', 0))
                        if idx < 0 or idx >= len(pulses):
                            continue
                        value = float(anchor.get('value', np.nan))
                        if not np.isfinite(value):
                            continue
                        y_disp = value * scale
                        x_pos = pulses[idx]
                        ax.axvline(
                            x_pos,
                            color=style['color'],
                            linestyle=style['vline'],
                            alpha=0.6,
                            linewidth=2.0,
                        )
                        ax.axhline(
                            y_disp,
                            color=style['color'],
                            linestyle=style['hline'],
                            alpha=0.4,
                            linewidth=1.5,
                        )
                        ax.scatter(
                            [x_pos],
                            [y_disp],
                            marker=style['marker'],
                            s=style['size'],
                            color=style['color'],
                            alpha=0.7,
                            zorder=10,
                            edgecolors='black',
                            linewidths=1.5,
                        )

                    ax.set_xlabel('Pulse #')
                    ylabel = label if not unit else f"{label} ({unit})"
                    ax.set_ylabel(ylabel)
                    # Simplify titles: show only the parameter label
                    ax.set_title(label)
                    ax.grid(True, alpha=0.3)

                # Legend panel at top-left of settings grid
                from matplotlib.lines import Line2D
                ax_leg = fit_diag_figure.add_subplot(gs_fd[1, 0])
                ax_leg.axis('off')
                handles = [
                    Line2D([0], [0], marker='o', color='#d62728', linestyle=':', linewidth=1.5,
                           markersize=8, label='initial per-event', markerfacecolor='#d62728'),
                    Line2D([0], [0], marker='x', color='#ff7f0e', linestyle='None',
                           markersize=8, markeredgewidth=2, label='constrained'),
                    Line2D([0], [0], marker='s', color='#2ca02c', linestyle='-', linewidth=2.5,
                           markersize=6, label='final progression'),
                    Line2D([0], [0], marker='s', color='green', linestyle='-', linewidth=2.0,
                           markersize=8, label='anchor: first'),
                    Line2D([0], [0], marker='D', color='#1f77b4', linestyle='--', linewidth=2.0,
                           markersize=8, label='anchor: mid'),
                    Line2D([0], [0], marker='*', color='purple', linestyle='-.', linewidth=2.0,
                           markersize=12, label='anchor: final'),
                ]
                ax_leg.legend(handles=handles, loc='center', ncol=3, frameon=True, fontsize=9, framealpha=0.9, title='Legend')

                # Fill the remaining slots row-major, skipping the legend cell
                idx = 0
                for r in range(1, 1 + n_rows):
                    for c in range(n_cols):
                        if r == 1 and c == 0:
                            continue  # legend cell already filled
                        if idx >= len(settings_ordered):
                            break
                        ax_s = fit_diag_figure.add_subplot(gs_fd[r, c])
                        _plot_setting_axis(ax_s, settings_ordered[idx])
                        idx += 1

                fit_diag_figure.tight_layout()
                try:
                    plt.show(block=False); plt.pause(0.05)
                except Exception:
                    pass
            except Exception as e:
                print(f"[warning] Failed to render fit diagnostics: {e}")

    # Per‑trial metrics and null thresholds (MAD rule) for A1
    per_trial: List[Dict] = []
    thr_list: List[float] = []
    pval_list: List[float] = []
    figures_trials = []  # optional per-trial figures
    for j in range(Yd.shape[1]):
        yj = Yd[:, j]
        # Always compute SG-smoothed series; may be used for SAVGOL-based thresholds
        yj_sg = sg_smooth(yj, sgW, sgP)
        a_t, d_t, X_t, yhat_t, comp_t = fit_amplitudes_no_overlap_forward(
            yj, t, stim_times, tau_r, tau_d_vec,
            pre_zoom_s=cfg['pre_zoom_s'], post_zoom_s=cfg['post_zoom_s'],
            robust=True, huber_delta=cfg['huber_delta'], irls_iters=cfg['irls_iters'],
            allow_shift=allow_shift, delta_max_s=cfg['delta_max_s'], delta_step_s=cfg['delta_step_s'],
            shift_min_s=cfg['shift_min_s'], event_t0_s=event_t0_s,
        )
        amp_raw = compute_localmax_corrected_amps(
            t, yj, stim_times, win_ms, n_avg, pre_ms, d_t, tau_r, tau_d_vec,
            event_t0_s=event_t0_s,
        )
        amp_raw_corr = compute_peak_corrected_from_components(
            t, yj, stim_times, comp_t, win_ms=win_ms, pre_ms=pre_ms
        )
        amp_sg = compute_localmax_corrected_amps(
            t, yj_sg, stim_times, win_ms, n_avg, pre_ms, d_t, tau_r, tau_d_vec,
            event_t0_s=event_t0_s,
        )
        amp_sg_corr = compute_peak_corrected_from_components(
            t, yj_sg, stim_times, comp_t, win_ms=win_ms, pre_ms=pre_ms
        )
        amp_nn = compute_localmax_corrected_amps(
            t, yhat_t, stim_times, win_ms, n_avg, pre_ms, d_t, tau_r, tau_d_vec,
            event_t0_s=event_t0_s,
        )
        amp_nn_corr = compute_peak_corrected_from_components(
            t, yhat_t, stim_times, comp_t, win_ms=win_ms, pre_ms=pre_ms
        )

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
                # Candidate starts cover the full f0 window and exclude any
                # start whose measurement window would touch the train.
                st_min = null_start
                st_max = null_end - (cfg['peak_window_ms'] / 1000.0)
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
                st_min = null_start
                st_max = null_end - (cfg['peak_window_ms'] / 1000.0)
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
                shift_min_s=cfg['shift_min_s'], n_samples=1000,
                seed=cfg.get('seed', 42), event_t0_s=event_t0_s,
            )

        thr1, pfun = baseline_threshold_and_pval(null_amps, null_N, mode=eff_mode)

        # Pick amplitude series for p-values/classification
        if meas == 'SAVGOL':
            a_for_p = amp_sg_corr
        elif meas == 'RAW':
            a_for_p = amp_raw_corr
        else:
            a_for_p = amp_nn_corr

        a1 = float(a_for_p[0]) if a_for_p.size else np.nan
        p1 = float(pfun(a1)) if np.isfinite(a1) else np.nan
        thr_list.append(thr1); pval_list.append(p1)
        # Compute p-values for pulses 2 and 3 using the same baseline threshold as A1
        p2 = float(pfun(a_for_p[1])) if (a_for_p.size >= 2 and np.isfinite(a_for_p[1])) else np.nan
        p3 = float(pfun(a_for_p[2])) if (a_for_p.size >= 3 and np.isfinite(a_for_p[2])) else np.nan

        per_trial.append({
            'amp_raw': amp_raw,
            'amp_raw_corr': amp_raw_corr,
            'amp_savgol': amp_sg,
            'amp_savgol_corr': amp_sg_corr,
            'amp_nnls': amp_nn,
            'amp_nnls_corr': amp_nn_corr,
            'ppr_raw': _norm(amp_raw),
            'ppr_savgol': _norm(amp_sg),
            'ppr_nnls': _norm(amp_nn),
            'ppr_raw_corr': _norm(amp_raw_corr),
            'ppr_savgol_corr': _norm(amp_sg_corr),
            'ppr_nnls_corr': _norm(amp_nn_corr),
            'a_coeff': a_t,
            'delta_s': d_t,
            'y_proc': yj,
            'yhat': yhat_t,
            'components': comp_t,
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

            # Layout: main train panel on top; optional residuals directly
            # underneath; optional baseline at the bottom. Residuals share the
            # time axis with the train; baseline is not time-aligned.
            want_resid_row = bool(plot_residuals)
            want_base_row = bool(baseline_figs)
            if want_resid_row or want_base_row:
                n_rows = 1 + int(want_resid_row) + int(want_base_row)
                if n_rows == 3:
                    ratios = [2.4, 1.3, 1.6]
                elif want_resid_row and not want_base_row:
                    ratios = [2.4, 1.3]
                else:
                    ratios = [2.4, 1.6]
                fig_t = plt.figure(figsize=(11, 9.0))
                gs_t = fig_t.add_gridspec(n_rows, 1, height_ratios=ratios, hspace=0.25)
                ax_train = fig_t.add_subplot(gs_t[0, 0])
                row = 1
                ax_resid = None
                ax_base = None
                if want_resid_row:
                    ax_resid = fig_t.add_subplot(gs_t[row, 0], sharex=ax_train)
                    row += 1
                if want_base_row:
                    ax_base = fig_t.add_subplot(gs_t[row, 0])  # no sharex (not time-aligned)
            else:
                fig_t, ax_train = plt.subplots(figsize=(10, 4.8))
                ax_resid = None
                ax_base = None

            # Train panel
            for st in stim_times:
                ax_train.axvline(st, color='k', linestyle=':', linewidth=0.8, alpha=0.6)
            if 'raw' in traces:
                ax_train.plot(tz, yj[zmask_t], label='raw', color='0.6')
            if 'savgol' in traces and yj_sg is not None:
                ax_train.plot(tz, yj_sg[zmask_t], label='savgol', color='tab:green')
            if 'nnls' in traces:
                ax_train.plot(tz, yhat_t[zmask_t], label='nnls model', color='tab:blue')
            if show_decay and 'nnls' in traces and comp_t is not None:
                cumulative_t = np.zeros_like(yj)
                for p in range(len(stim_times)):
                    if p >= len(comp_t):
                        continue
                    cumulative_t = cumulative_t + comp_t[p]
                    ax_train.plot(
                        tz,
                        cumulative_t[zmask_t],
                        color='tab:orange',
                        linestyle='--',
                        linewidth=1.0,
                        alpha=0.6 - p * 0.04,
                    )
            # Peak markers and residual-at-peak triangles for this trial
            if plot_peaks_details:
                if meas == 'SAVGOL' and (yj_sg is not None):
                    y_for_peaks_t = yj_sg
                elif meas == 'RAW':
                    y_for_peaks_t = yj
                else:
                    y_for_peaks_t = yhat_t

                # Precompute per-pulse cumulative baseline from previous pulses (NNLS comps)
                baseline_prev_only_t = []
                if comp_t is not None:
                    cum_t = np.zeros_like(yj)
                    for pp in range(len(stim_times)):
                        baseline_prev_only_t.append(cum_t.copy())
                        if pp < len(comp_t):
                            cum_t = cum_t + comp_t[pp]

                peak_ts_t, peak_vals_t, resid_vals_t = [], [], []
                for pp, stp in enumerate(stim_times):
                    tp, vp = pick_peak_on_series(t, y_for_peaks_t, float(stp), win_ms, pre_ms)
                    peak_ts_t.append(float(tp))
                    peak_vals_t.append(float(vp))
                    try:
                        i0 = int(np.argmin(np.abs(t - tp)))
                        base_prev = baseline_prev_only_t[pp][i0] if baseline_prev_only_t else 0.0
                        resid_vals_t.append(float(base_prev))
                    except Exception:
                        resid_vals_t.append(np.nan)
                try:
                    ax_train.scatter(peak_ts_t, peak_vals_t, s=60, color='red', edgecolors='white', linewidths=0.9, zorder=6)
                except Exception:
                    pass
                try:
                    ax_train.scatter(peak_ts_t, resid_vals_t, s=55, marker='v', facecolors='white', edgecolors='tab:red', linewidths=1.0, zorder=5)
                except Exception:
                    pass
            # Failure threshold line (thin red dotted)
            if np.isfinite(thr1):
                ax_train.axhline(thr1, color='red', linestyle=':', linewidth=0.8)
            ax_train.set_xlim(z0, z1)
            ax_train.set_xlabel('Time (s)')
            ax_train.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
            ax_train.legend(loc='upper right', frameon=False)
            ax_train.set_title(f'Trial {j+1}: train window')
            _trim_spines(ax_train)

            # Residuals panel directly underneath the main panel
            if ax_resid is not None:
                try:
                    model_t = (yj_sg if (meas == 'SAVGOL' and yj_sg is not None) else yhat_t)
                    resid_t = yj - model_t
                    ax_resid.plot(tz, resid_t[zmask_t], color='tab:purple', lw=1.2, label='residual (trial − model)')
                    ax_resid.axvline(float(train_start), color='k', ls=':', lw=0.8, alpha=0.6)
                    ax_resid.set_xlim(z0, z1)
                    ax_resid.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
                    ax_resid.set_title('Residuals in train window')
                    ax_resid.legend(loc='upper right', frameon=False, fontsize=8)
                    # Inset histogram of residuals (zoom window) with fixed bins and Gaussian fit
                    try:
                        ax_in_r = ax_resid.inset_axes([0.65, 0.55, 0.33, 0.4])
                        rdata = np.asarray(resid_t[zmask_t], float)
                        rdata = rdata[np.isfinite(rdata)]
                        if rdata.size:
                            bin_w = (0.01 if use_dff else 10.0)
                            lo = float(np.nanmin(rdata))
                            hi = float(np.nanmax(rdata))
                            if not np.isfinite(lo):
                                lo = 0.0
                            if not np.isfinite(hi) or hi <= lo:
                                hi = lo + bin_w
                            edges = np.arange(lo, hi + bin_w, bin_w)
                            ax_in_r.hist(rdata, bins=edges, color='#d8c7e8', edgecolor='#6b4fa3')
                            # Fit Gaussian to residuals and overlay across full inset range
                            try:
                                mu = float(np.nanmean(rdata))
                                sigma = float(np.nanstd(rdata))
                            except Exception:
                                mu, sigma = float('nan'), float('nan')
                            if np.isfinite(sigma) and sigma > 0:
                                x0, x1 = ax_in_r.get_xlim()
                                x = np.linspace(x0, x1, 400)
                                pdf = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                                N = rdata.size
                                y = N * bin_w * pdf
                                ax_in_r.plot(x, y, color='#26457a', linewidth=1.4, label='fit')
                        ax_in_r.set_title('residual', fontsize=8)
                        ax_in_r.tick_params(labelsize=7)
                    except Exception:
                        pass
                except Exception:
                    pass
                _trim_spines(ax_resid)

            # Baseline panel (pre-train) at the bottom (if requested)
            if ax_base is not None:
                base_mask = (t < float(train_start))
                tb = t[base_mask]
                use_savgol_baseline = False
                if tb.size:
                    # Highlight the f0 baseline window used for null sampling
                    try:
                        baseline_start = tb[0]
                        baseline_end = tb[-1]
                        null_start = max(baseline_start, float(train_start) - cfg['f0_window_s'])
                        # Shade from null_start up to first stimulus time
                        ax_base.axvspan(null_start, float(train_start), facecolor='#ffd500', alpha=0.18, zorder=0.1)
                        # Vertical dotted line at first stimulus time
                        ax_base.axvline(float(train_start), color='k', ls=':', lw=0.8, alpha=0.6)
                        # Ensure the x-range reaches the dotted line
                        ax_base.set_xlim(baseline_start, float(train_start))
                    except Exception:
                        pass
                    use_savgol_baseline = (
                        (meas == 'NNLS')
                        and (failm != meas)
                        and (yj_sg is not None)
                    )
                    if 'raw' in traces:
                        ax_base.plot(tb, yj[base_mask], color='0.4', linewidth=1.0, label='baseline')
                    if use_savgol_baseline:
                        ax_base.plot(
                            tb,
                            yj_sg[base_mask],
                            color='red',
                            linewidth=1.1,
                            label='savgol',
                        )
                    else:
                        # Overlay a subset of null-fit events in red
                        baseline_start = tb[0]; baseline_end = tb[-1]
                        null_start = max(baseline_start, float(train_start) - cfg['f0_window_s'])
                        null_end = min(baseline_end, float(train_start))
                        st_min = null_start + cfg['pre_zoom_s']
                        st_max = null_end - cfg['null_min_post_zoom_s']
                        cand_mask = (t >= st_min) & (t <= st_max)
                        starts_full = t[cand_mask]
                        if starts_full.size:
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
                                    shift_min_s=cfg['shift_min_s'], event_t0_s=event_t0_s,
                                )
                                events.append((float(a_hat_b), float(stcand), float(d_hat_b)))
                            events = [e for e in events if np.isfinite(e[0]) and e[0] > 0]
                            events.sort(key=lambda e: e[0], reverse=True)
                            draw_n = min(20, len(events))
                            for a_hat_b, stcand, d_hat_b in events[:draw_n]:
                                anchor_b = float(stcand) + event_t0_s
                                y_evt_b = a_hat_b * _KERNEL_FUN(tb - (anchor_b + d_hat_b), tau_r, tau_d0)
                                ax_base.plot(tb, y_evt_b, color='red', alpha=0.5, linewidth=1.0)
                    # Inset histogram of null amplitudes with threshold and fitted noise curve
                    try:
                        ax_in = ax_base.inset_axes([0.65, 0.55, 0.33, 0.4])
                        data = np.asarray(null_amps, float)
                        data = data[np.isfinite(data)]
                        if data.size:
                            # Fixed bin width based on measurement scale
                            bin_w = (0.01 if use_dff else 10.0)
                            lo = 0.0  # null amplitudes are non‑negative
                            hi = float(np.nanmax(data)) if np.isfinite(np.nanmax(data)) else 0.0
                            if hi <= lo:
                                hi = lo + bin_w
                            edges = np.arange(lo, hi + bin_w, bin_w)
                            # Draw histogram and capture bin counts
                            n_hist, bins_hist, _ = ax_in.hist(
                                data, bins=edges, color='#c9d4e8', edgecolor='#4f6aa3'
                            )
                            # Choose noise model overlay based on baseline measurement method:
                            #   - NNLS => half‑normal (cut Gaussian, non‑negative)
                            #   - RAW/SAVGOL => Gaussian
                            if failm == 'NNLS':
                                # Zero‑inflated half‑normal: mass at 0 for the clipped negative side
                                # plus a continuous half‑normal on x>0.
                                N = data.size
                                zero_tol = max(1e-12, 0.5 * bin_w)
                                is_zero = data <= zero_tol
                                n_zero = int(np.sum(is_zero))
                                p0 = float(n_zero) / float(N) if N else 0.0
                                pos = data[~is_zero]
                                # MLE for half‑normal sigma on strictly positive samples
                                try:
                                    sigma = float(np.sqrt(np.mean(np.square(np.clip(pos, 0.0, None))))) if pos.size else float('nan')
                                except Exception:
                                    sigma = float('nan')
                                x0, x1 = ax_in.get_xlim()
                                x = np.linspace(x0, x1, 400)
                                if np.isfinite(sigma) and sigma > 0:
                                    x_clip = np.clip(x, 0.0, None)
                                    pdf = (np.sqrt(2.0) / (sigma * np.sqrt(np.pi))) * np.exp(-(x_clip**2) / (2.0 * sigma * sigma))
                                    pdf = np.where(x >= 0.0, pdf, 0.0)
                                    y = (N * (1.0 - p0) * bin_w) * pdf  # scale continuous part
                                    ax_in.plot(x, y, color='#26457a', linewidth=1.4, label='fit')
                                # Axis scaling: ignore the 0-bin height; use next highest bin
                                try:
                                    max_other = float(np.nanmax(n_hist[1:])) if n_hist.size > 1 else 0.0
                                except Exception:
                                    max_other = 0.0
                                ymax = max(1.0, max_other * 1.15)
                                ax_in.set_ylim(0.0, ymax)
                                # Annotate the zero-bin count instead of scaling to it
                                if n_zero > 0:
                                    ax_in.text(
                                        0.02 * (x1 - x0) + x0,
                                        ymax * 0.92,
                                        f"0-bin: {n_zero}",
                                        fontsize=7,
                                        ha='left', va='top', color='#26457a',
                                        bbox=dict(boxstyle='round,pad=0.12', facecolor='white', alpha=0.7, lw=0)
                                    )
                            else:
                                # Gaussian fit
                                try:
                                    mu = float(np.nanmean(data))
                                    sig = float(np.nanstd(data))
                                except Exception:
                                    mu, sig = float('nan'), float('nan')
                                if np.isfinite(sig) and sig > 0:
                                    x0, x1 = ax_in.get_xlim()
                                    x = np.linspace(x0, x1, 400)
                                    pdf = (1.0 / (np.sqrt(2.0 * np.pi) * sig)) * np.exp(-0.5 * ((x - mu) / sig) ** 2)
                                    N = data.size
                                    y = N * bin_w * pdf
                                    ax_in.plot(x, y, color='#26457a', linewidth=1.4, label='fit')
                            if np.isfinite(thr1):
                                ax_in.axvline(thr1, color='red', linestyle='--', linewidth=0.9)
                        ax_in.set_title('noise', fontsize=8)
                        ax_in.tick_params(labelsize=7)
                    except Exception:
                        pass
                ax_base.set_xlabel('Time (s)')
                ax_base.set_ylabel('ΔF/F0' if use_dff else 'ΔF')
                if use_savgol_baseline:
                    ax_base.set_title('Baseline window (savgol)')
                else:
                    ax_base.set_title('Baseline window + null-fit events')
                _trim_spines(ax_base)

            # Match Y limits across comparable panels (exclude residuals)
            try:
                ylims = [ax_train.get_ylim()]
                if ax_base is not None:
                    ylims.append(ax_base.get_ylim())
                ymin = min([y[0] for y in ylims])
                ymax = max([y[1] for y in ylims])
                ax_train.set_ylim(ymin, ymax)
                if ax_base is not None:
                    ax_base.set_ylim(ymin, ymax)
            except Exception:
                pass

            # Show trial figures immediately after they are created
            try:
                plt.show(block=False); plt.pause(0.01)
            except Exception:
                pass
            figures_trials.append(fig_t)

    return {
        'tau_r_s': float(tau_r),
        'tau_d_s': np.asarray(tau_d_vec, float),
        'stim_times_s': np.asarray(stim_times, float),
        'event_model': {
            'name': ev_model_name,
            'n_coop': (float(n_coop_effective) if n_coop_effective is not None else None),
        },
        'interpolated_settings': interpolated_settings,
        'average': {
            'amp_raw': np.asarray(amp_raw_avg, float),
            'amp_savgol': np.asarray(amp_sg_avg, float),
            'amp_nnls': np.asarray(amp_nnls_avg, float),
            'amp_raw_corr': np.asarray(amp_raw_corr_avg, float),
            'amp_savgol_corr': np.asarray(amp_sg_corr_avg, float),
            'amp_nnls_corr': np.asarray(amp_nnls_corr_avg, float),
            'ppr_nnls': np.asarray(ppr_nnls_avg, float),
            'ppr_nnls_corr': np.asarray(ppr_nnls_corr_avg, float),
            'y_avg': np.asarray(y_avg, float),
            'yhat_avg': np.asarray(yhat_avg, float),
        },
        'recut_snippets': recut_snippets,
        'recut_t_rel': np.asarray(recut_t_rel, float) if recut_t_rel is not None else None,
        'recut_avg': np.asarray(recut_avg, float) if recut_avg is not None else None,
        'per_trial': per_trial,
        'time_s': np.asarray(t, float),
        'threshold_amp1': np.asarray(thr_list, float),
        'pval_amp1': np.asarray(pval_list, float),
        'figure': figure,
        'figure_fit_diagnostic': fit_diag_figure,
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
                    t = t_raw[ok] ; trials = X[ok, :]
                    
                    # add time offset of 2ms
                    t += 0.002

                    res = extract_metrics(t, trials, train_start=train_start, isi=isi, n_pulses=n_pulses, options=cfg)

                    # Choose corrected amplitude series per measurement when available
                    if meas == 'SAVGOL':
                        amp_avg = res['average'].get('amp_savgol_corr', res['average']['amp_savgol'])
                    elif meas == 'RAW':
                        amp_avg = res['average'].get('amp_raw_corr', res['average']['amp_raw'])
                    else:
                        amp_avg = res['average'].get('amp_nnls_corr', res['average']['amp_nnls'])

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
                        per_amp = _amp_vec_per_trial('amp_savgol_corr')
                    elif failm == 'RAW':
                        per_amp = _amp_vec_per_trial('amp_raw_corr')
                    else:
                        per_amp = _amp_vec_per_trial('amp_nnls_corr')

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
