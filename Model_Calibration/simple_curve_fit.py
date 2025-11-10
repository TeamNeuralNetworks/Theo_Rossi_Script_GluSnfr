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
                return_snippets=bool(return_snippets),
            )
            # Unpack depending on whether snippets were returned
            if return_snippets:
                t_rel_s, avg, snippets = res
            else:
                t_rel_s, avg = res
            if t_rel_s is None or avg is None:
                return None
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
        popt, _ = curve_fit(
            spec['func'], tf, yf, p0=p0, bounds=spec['bounds'], maxfev=maxfev
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
    except Exception:
        return None
