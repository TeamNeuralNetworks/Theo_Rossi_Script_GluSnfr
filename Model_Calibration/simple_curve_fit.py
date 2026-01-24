"""Shared utilities for fitting an average iGluSnFR event with simple models.

Provides a single helper so both the demo script and the extract_metrics
pipeline use the exact same curve_fit logic when estimating global kinetics.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit


def fit_average_event(
    time,
    data,
    model_name: str,
    stim_times_s: Optional[Sequence[float]] = None,
    *,
    window_ms: Tuple[float, float] = (0.0, 50.0),
    maxfev: int = 3000,
    pre_ms: float = 5.0,
    post_ms: float = 50.0,
    oversample: int = 1,
    projection: str = 'mean',
    peak_recenter=0,
    return_snippets: bool = False,
    onset_method: str = 'inflection',
    onset_baseline_threshold: float = 0.15,
    early_events_only: int = 0,
    fixed_tau_slow: Optional[float] = None,
) -> Optional[Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
    """Recut trials, average, and fit an event model.

    Parameters
    ----------
    time : array-like
        Time vector (seconds if `stim_times_s` provided, otherwise milliseconds).
    data : array-like
        If `stim_times_s` is provided, shape should be `(T, N)` for `T` time
        points and `N` trials. Otherwise, `data` is the already averaged
        waveform aligned to `time`.
    model_name : str
        Name of the event model defined in module event_models.
    stim_times_s : sequence, optional
        Stimulus times in seconds. When given, `time` is interpreted in seconds
        and trials are recut around each stimulus before averaging (mean across
        all recut snippets).
    window_ms : tuple, optional
        (start, end) window in milliseconds used for fitting, by default
        (0, 50).
    maxfev : int, optional
        Maximum function evaluations passed to `curve_fit`.
    pre_ms, post_ms : float, optional
        Window around each stimulus used when recutting trials.
    peak_recenter : int | tuple | None, optional
        Maximum number of samples permitted when shifting snippet peaks prior to
        averaging; pass 0/None to keep stimulus-aligned windows.
    onset_method : str, optional
        Method for onset detection to exclude contaminated baseline:
        - 'inflection': Find inflection point (minimum derivative) and exclude points before it
        - 'baseline_threshold': Exclude all points below baseline + threshold * (peak - baseline)
        - 'none': No onset masking
        Default is 'inflection'.
    onset_baseline_threshold : float, optional
        Threshold factor for baseline_threshold method. Points below
        baseline + threshold * (peak - baseline) are excluded. Default is 0.15
        (15% of peak amplitude above baseline).
    early_events_only : int, optional
        If > 0, use only the first N events for computing the recut average.
        This helps capture fast kinetics uncontaminated by slow summation from
        later events. Default is 0 (use all events).
    fixed_tau_slow : float, optional
        If provided (in seconds), fixes tau_decay_slow to this value during
        curve_fit. This allows fitting tau_fast from early events while using
        a pre-estimated tau_slow (e.g., from post-train decay). Default is None.

    Returns
    -------
    tuple | None
        `(params, t_ms, y_avg)` where `params` is a mapping of parameter
        names to fitted values and `t_ms`/`y_avg` are the averaged waveform.
        `None` if fitting fails.
    """
    try:
        if stim_times_s is not None:
            try:
                from smoothing import build_median_recut_waveform
            except Exception:  # pragma: no cover - allow running from subfolder
                import os, sys
                REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
                if REPO_ROOT not in sys.path:
                    sys.path.insert(0, REPO_ROOT)
                from smoothing import build_median_recut_waveform  # type: ignore
            res = build_median_recut_waveform(
                time,
                data,
                stim_times_s,
                pre_ms=pre_ms,
                post_ms=post_ms,
                peak_win_ms=25.0,
                peak_search_pre_ms=0.0,
                oversample=int(oversample),
                projection=str(projection).lower(),
                peak_recenter=peak_recenter,
                return_snippets=True,  # Always get snippets to support early_events_only
            )
            # Unpack - snippets always returned now
            t_rel_s, avg, snippets = res
            if t_rel_s is None or avg is None:
                return None
            
            # If early_events_only > 0, recompute average using only first N events
            # Snippets are ordered: [stim1_trial1, stim1_trial2, ..., stim2_trial1, ...]
            if early_events_only > 0 and snippets is not None and len(snippets) > 0:
                n_stims = len(stim_times_s)
                n_trials = len(snippets) // n_stims if n_stims > 0 else 0
                if n_trials > 0:
                    n_early = min(early_events_only, n_stims)
                    early_snip_indices = []
                    for ev_idx in range(n_early):
                        for tr_idx in range(n_trials):
                            early_snip_indices.append(ev_idx * n_trials + tr_idx)
                    early_snippets = [snippets[i] for i in early_snip_indices if i < len(snippets)]
                    if early_snippets:
                        S_early = np.array(early_snippets, dtype=float)
                        avg = np.nanmean(S_early, axis=0)
                        try:
                            from smoothing import progress_print
                            progress_print(f"[fit_average_event] Using only first {n_early} events ({len(early_snippets)} snippets) for fast kinetics fit")
                        except Exception:
                            pass
            
            # Store for return if caller wanted snippets
            if not return_snippets:
                snippets = None
            
            t_ms = np.asarray(t_rel_s, float) * 1000.0
            y_avg = np.asarray(avg, float)
        else:
            t_ms = np.asarray(time, float)
            y_avg = np.asarray(data, float)

        try:
            from .event_models import get_event_model
        except Exception:  # pragma: no cover - fallback when run from repo root
            from event_models import get_event_model  # type: ignore
        spec = get_event_model(model_name)
        mask = (t_ms >= window_ms[0]) & (t_ms <= window_ms[1])
        if not np.any(mask):
            return None
        tf = t_ms[mask]
        yf = y_avg[mask]

        # Onset detection to exclude contaminated baseline at high frequencies
        onset_idx = 0
        method = str(onset_method).lower()

        if method == 'inflection' and len(yf) > 5:
            # Find inflection point (minimum derivative) - where signal starts rising
            dy = np.gradient(yf, tf)
            # Smooth derivative to reduce noise
            from scipy.ndimage import uniform_filter1d
            dy_smooth = uniform_filter1d(dy, size=min(5, len(dy)))
            # Find first local minimum in derivative (where rise starts)
            # Look in first 30% of window where onset should occur
            search_len = max(5, int(len(dy_smooth) * 0.3))
            onset_idx = int(np.argmin(dy_smooth[:search_len]))
            # Validate: only exclude if there's actually a dip/notch (dy is negative)
            if dy_smooth[onset_idx] >= 0:
                onset_idx = 0  # No contamination detected, use full window

        elif method == 'baseline_threshold' and len(yf) > 5:
            # Exclude ALL points below baseline + threshold * (peak - baseline)
            # This is more aggressive than inflection detection
            # Estimate baseline from first few points (up to 20% of window or first 5 points)
            baseline_len = max(3, min(5, int(len(yf) * 0.2)))
            baseline = float(np.median(yf[:baseline_len]))
            # Find peak
            peak = float(np.nanmax(yf))
            # Compute threshold level
            threshold_level = baseline + onset_baseline_threshold * (peak - baseline)
            # Find first point that crosses threshold
            above_threshold = yf >= threshold_level
            if np.any(above_threshold):
                onset_idx = int(np.argmax(above_threshold))
            else:
                onset_idx = 0  # All points below threshold, keep everything
            # Onset detection debug messages (commented out - enable if needed for debugging)
            # try:
            #     from smoothing import progress_print
            #     progress_print(f"[fit_average_event] Baseline: {baseline:.3f}, Peak: {peak:.3f}, Threshold: {threshold_level:.3f}")
            # except Exception:
            #     pass

        # Apply onset masking
        if onset_idx > 0:
            tf = tf[onset_idx:]
            yf = yf[onset_idx:]

        p0 = spec['p0_func'](yf, tf)
        # Robust seeding for two_step_binding: quick coarse grid search
        # to avoid local minima/stagnation at defaults.
        try:
            if spec.get('name', '').lower() == 'two_step_binding':
                # Unpack current guesses safely
                amp0, tb0, tc0, td0, tp0 = [float(x) for x in p0]
                # Build small grids around typical ranges; clip to bounds
                lb, ub = spec['bounds']
                def _clip(v, lo, hi):
                    return float(min(max(v, lo), hi))
                tb_grid = np.array([
                    _clip(x, lb[1], ub[1]) for x in (0.0006, 0.0010, 0.0016, 0.0025, 0.0040)
                ])
                tc_grid = np.array([
                    _clip(x, lb[2], ub[2]) for x in (0.004, 0.006, 0.008, 0.012, 0.016)
                ])
                td_grid = np.array([
                    _clip(x, lb[3], ub[3]) for x in (0.020, 0.030, 0.040, 0.060, 0.080)
                ])
                tp_grid = np.array([
                    _clip(x, lb[4], ub[4]) for x in (tp0 - 2.0, tp0 - 1.0, tp0, tp0 + 1.0, tp0 + 2.0)
                ])
                best = (np.inf, amp0, tb0, tc0, td0, tp0)
                for tb in tb_grid:
                    for tc in tc_grid:
                        # Enforce sequential regime: tc >= tb
                        if tc <= tb:
                            continue
                        for td in td_grid:
                            for tp in tp_grid:
                                pars = [1.0, tb, tc, td, tp]
                                yshape = spec['func'](tf, *pars)
                                denom = float(np.sum(yshape ** 2))
                                if denom <= 0 or not np.isfinite(denom):
                                    continue
                                amp = float(np.sum(yf * yshape)) / denom
                                r = yf - amp * yshape
                                sse = float(np.dot(r, r))
                                if sse < best[0]:
                                    best = (sse, amp, tb, tc, td, tp)
                _, amp_b, tb_b, tc_b, td_b, tp_b = best
                p0 = [float(amp_b), float(tb_b), float(tc_b), float(td_b), float(tp_b)]
        except Exception:
            # If anything goes wrong, keep original p0
            pass
        
        # Robust seeding for iGluSnFR: grid search over key parameters
        # Critical for high-frequency trains where the short fitting window makes
        # curve_fit sensitive to initial conditions
        try:
            if spec.get('name', '').lower() == 'iglusnfr':
                # For iGluSnFR: amp, tau_rise, tau_decay_fast, tau_decay_slow, frac_fast, t_peak
                amp0, tr0, tdf0, tds0, ff0, tp0 = [float(x) for x in p0]
                lb, ub = spec['bounds']
                
                def _clip(v, lo, hi):
                    return float(min(max(v, lo), hi))
                
                # Grids for key parameters with NON-OVERLAPPING ranges
                # tau_rise: 0.5-3ms, tau_fast: 1.5-7ms (tighter), tau_slow: 15-150ms
                tau_rise_grid = np.array([_clip(x, lb[1], ub[1]) for x in (0.0005, 0.001, 0.0015, 0.002, 0.003)])
                tau_fast_grid = np.array([_clip(x, lb[2], ub[2]) for x in (0.0015, 0.002, 0.003, 0.004, 0.005, 0.007)])
                tau_slow_grid = np.array([_clip(x, lb[3], ub[3]) for x in (0.015, 0.025, 0.040, 0.060, 0.100, 0.150)])
                frac_fast_grid = np.array([_clip(x, lb[4], ub[4]) for x in (0.4, 0.55, 0.7, 0.85)])
                
                # Find peak location in data
                peak_idx = np.argmax(yf)
                tp_data = float(tf[peak_idx])
                tp_grid = np.array([_clip(x, lb[5], ub[5]) for x in (tp_data - 1.0, tp_data, tp_data + 1.0)])
                
                best = (np.inf, amp0, tr0, tdf0, tds0, ff0, tp0)
                for tr in tau_rise_grid:
                    for tdf in tau_fast_grid:
                        # Enforce tau_rise < tau_fast
                        if tr >= tdf:
                            continue
                        for tds in tau_slow_grid:
                            # Enforce tau_fast < tau_slow (with gap at 10ms boundary)
                            if tdf >= tds:
                                continue
                            for ff in frac_fast_grid:
                                for tp in tp_grid:
                                    pars = [1.0, tr, tdf, tds, ff, tp]
                                    yshape = spec['func'](tf, *pars)
                                    denom = float(np.sum(yshape ** 2))
                                    if denom <= 0 or not np.isfinite(denom):
                                        continue
                                    amp = float(np.sum(yf * yshape)) / denom
                                    if amp <= 0:
                                        continue
                                    r = yf - amp * yshape
                                    # Weight residuals: emphasize post-peak fit (t > t_peak)
                                    weights = np.ones_like(r)
                                    weights[tf > tp] = 2.0  # Double weight for decay portion
                                    sse = float(np.sum(weights * r ** 2))
                                    if sse < best[0]:
                                        best = (sse, amp, tr, tdf, tds, ff, tp)
                
                if best[0] < np.inf:
                    _, amp_b, tr_b, tdf_b, tds_b, ff_b, tp_b = best
                    p0 = [float(amp_b), float(tr_b), float(tdf_b), float(tds_b), float(ff_b), float(tp_b)]
                    
                    # Dynamically constrain bounds based on grid search results
                    # This prevents curve_fit from diverging to edge solutions
                    lb_list = list(spec['bounds'][0])
                    ub_list = list(spec['bounds'][1])
                    
                    # Constrain tau_rise upper bound to < tau_fast (prevents role swapping)
                    tau_rise_max = min(ub_list[1], tdf_b * 0.9)
                    ub_list[1] = float(tau_rise_max)
                    
                    # Keep tau_fast bounds as specified (don't constrain from grid search)
                    # This allows curve_fit to find the optimal tau_fast within the full range
                    
                    spec['bounds'] = (lb_list, ub_list)
                    
                    try:
                        from smoothing import progress_print
                        progress_print(f"[fit_average_event] iGluSnFR grid search: tau_r={tr_b*1000:.1f}ms, tau_fast={tdf_b*1000:.1f}ms, tau_slow={tds_b*1000:.1f}ms, frac_fast={ff_b:.2f}")
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Robust seeding for iGluSnFR TRI-EXPONENTIAL: grid search over key parameters
        try:
            if spec.get('name', '').lower() == 'iglusnfr_tri':
                # For iGluSnFR_tri: amp, tau_rise, tau_decay_fast, tau_decay_slow, tau_decay_superslow, frac_fast, frac_slow, t_peak
                amp0, tr0, tdf0, tds0, tdss0, ff0, fs0, tp0 = [float(x) for x in p0]
                lb, ub = spec['bounds']
                
                def _clip(v, lo, hi):
                    return float(min(max(v, lo), hi))
                
                # Grids for key parameters with NON-OVERLAPPING ranges
                # tau_rise: 0.5-4ms, tau_fast: 1-10ms, tau_slow: 10-25ms, tau_superslow: 25-100ms
                tau_rise_grid = np.array([_clip(x, lb[1], ub[1]) for x in (0.001, 0.002, 0.003, 0.004)])
                tau_fast_grid = np.array([_clip(x, lb[2], ub[2]) for x in (0.002, 0.003, 0.005, 0.007, 0.010)])
                tau_slow_grid = np.array([_clip(x, lb[3], ub[3]) for x in (0.012, 0.015, 0.020, 0.025)])
                # tau_superslow often fixed from post-train decay, use narrower grid
                tau_superslow_grid = np.array([_clip(x, lb[4], ub[4]) for x in (0.030, 0.045, 0.060)])
                frac_fast_grid = np.array([_clip(x, lb[5], ub[5]) for x in (0.5, 0.65, 0.8)])
                frac_slow_grid = np.array([_clip(x, lb[6], ub[6]) for x in (0.15, 0.25)])  # Intermediate fraction
                
                # Find peak location in data
                peak_idx = np.argmax(yf)
                tp_data = float(tf[peak_idx])
                tp_grid = np.array([_clip(x, lb[7], ub[7]) for x in (tp_data - 1.0, tp_data, tp_data + 1.0)])
                
                best = (np.inf, amp0, tr0, tdf0, tds0, tdss0, ff0, fs0, tp0)
                for tr in tau_rise_grid:
                    for tdf in tau_fast_grid:
                        # Enforce tau_rise < tau_fast
                        if tr >= tdf:
                            continue
                        for tds in tau_slow_grid:
                            # Enforce tau_fast < tau_slow
                            if tdf >= tds:
                                continue
                            for tdss in tau_superslow_grid:
                                # Enforce tau_slow < tau_superslow
                                if tds >= tdss:
                                    continue
                                for ff in frac_fast_grid:
                                    for fs in frac_slow_grid:
                                        # Ensure frac_superslow = 1 - ff - fs > 0
                                        if ff + fs >= 0.95:
                                            continue
                                        for tp in tp_grid:
                                            pars = [1.0, tr, tdf, tds, tdss, ff, fs, tp]
                                            yshape = spec['func'](tf, *pars)
                                            denom = float(np.sum(yshape ** 2))
                                            if denom <= 0 or not np.isfinite(denom):
                                                continue
                                            amp = float(np.sum(yf * yshape)) / denom
                                            if amp <= 0:
                                                continue
                                            r = yf - amp * yshape
                                            # Weight residuals: emphasize post-peak fit
                                            weights = np.ones_like(r)
                                            weights[tf > tp] = 2.0
                                            sse = float(np.sum(weights * r ** 2))
                                            if sse < best[0]:
                                                best = (sse, amp, tr, tdf, tds, tdss, ff, fs, tp)
                
                if best[0] < np.inf:
                    _, amp_b, tr_b, tdf_b, tds_b, tdss_b, ff_b, fs_b, tp_b = best
                    p0 = [float(amp_b), float(tr_b), float(tdf_b), float(tds_b), float(tdss_b), float(ff_b), float(fs_b), float(tp_b)]
                    
                    try:
                        from smoothing import progress_print
                        progress_print(f"[fit_average_event] iGluSnFR TRI grid search: tau_r={tr_b*1000:.1f}ms, tau_fast={tdf_b*1000:.1f}ms, tau_slow={tds_b*1000:.1f}ms, tau_superslow={tdss_b*1000:.1f}ms, frac_fast={ff_b:.2f}, frac_slow={fs_b:.2f}")
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Build sigma weights for curve_fit to emphasize the fast decay portion
        # The initial decay (0-5ms after peak) is where the fast component dominates
        # Lower sigma = higher weight in the least-squares objective
        sigma = None
        try:
            model_name = spec.get('name', '').lower()
            if model_name in ('iglusnfr', 'iglusnfr_tri'):
                peak_idx = int(np.argmax(yf))
                t_peak = float(tf[peak_idx])
                sigma = np.ones_like(yf)
                # High weight (low sigma) for initial decay: peak to peak+5ms
                fast_decay_mask = (tf >= t_peak) & (tf <= t_peak + 0.005)
                sigma[fast_decay_mask] = 0.3  # ~3x higher weight
                # Medium weight for mid decay: peak+5ms to peak+10ms
                mid_decay_mask = (tf > t_peak + 0.005) & (tf <= t_peak + 0.010)
                sigma[mid_decay_mask] = 0.6  # ~1.7x higher weight
                # Standard weight for late decay and rise
                
                # If fixed_tau_slow provided, lock tau_decay_slow bounds (or tau_decay_superslow for tri-exp)
                if fixed_tau_slow is not None:
                    lb_list = list(spec['bounds'][0])
                    ub_list = list(spec['bounds'][1])
                    tds_fixed = float(fixed_tau_slow)
                    
                    if model_name == 'iglusnfr_tri':
                        # For tri-exp: fixed_tau_slow -> tau_decay_superslow (index 4)
                        lb_list[4] = tds_fixed * 0.999
                        ub_list[4] = tds_fixed * 1.001
                        p0[4] = tds_fixed
                        try:
                            from smoothing import progress_print
                            progress_print(f"[fit_average_event] tau_superslow FIXED to {tds_fixed*1000:.1f}ms (from post-train decay)")
                        except Exception:
                            pass
                    else:
                        # For bi-exp: fixed_tau_slow -> tau_decay_slow (index 3)
                        lb_list[3] = tds_fixed * 0.999
                        ub_list[3] = tds_fixed * 1.001
                        p0[3] = tds_fixed
                        try:
                            from smoothing import progress_print
                            progress_print(f"[fit_average_event] tau_slow FIXED to {tds_fixed*1000:.1f}ms (from post-train decay)")
                        except Exception:
                            pass
                    
                    spec['bounds'] = (lb_list, ub_list)
        except Exception:
            sigma = None
        
        popt, _ = curve_fit(
            spec['func'], tf, yf, p0=p0, bounds=spec['bounds'], maxfev=maxfev,
            sigma=sigma, absolute_sigma=False
        )
        params = {name: float(val) for name, val in zip(spec['params'], popt)}
        # If caller requested recut snippets, attach them to the params dict
        # so callers that call this helper via the pipeline can access them
        # without changing the function's primary return signature.
        try:
            if return_snippets and 'snippets' in locals() and 't_rel_s' in locals() and 'avg' in locals():
                params['_recut'] = (t_rel_s, avg, snippets)
                try:
                    from smoothing import progress_print
                    progress_print(f"[fit_average_event] Attached {len(snippets)} snippets to params dict")
                except Exception:
                    pass
            elif return_snippets:
                try:
                    from smoothing import progress_print
                    progress_print(f"[fit_average_event] return_snippets={return_snippets}, but snippets not in locals")
                except Exception:
                    pass
        except Exception as e:
            try:
                from smoothing import progress_print
                progress_print(f"[fit_average_event] Failed to attach snippets: {e}")
            except Exception:
                pass
        return params, t_ms, y_avg
    except Exception as e:
        try:
            from smoothing import progress_print
            import traceback
            progress_print(f"[fit_average_event] FAILED with error: {type(e).__name__}: {e}")
            progress_print(f"[fit_average_event] Traceback: {traceback.format_exc()}")
        except Exception:
            pass
        return None
