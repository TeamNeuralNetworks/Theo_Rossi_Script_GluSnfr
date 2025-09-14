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
    align_by_peak: bool = False,
) -> Optional[Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
    """Recut trials, average, and fit an event model.

    Parameters
    ----------
    time : array-like
        Time vector (seconds if ``stim_times_s`` provided, otherwise milliseconds).
    data : array-like
        If ``stim_times_s`` is provided, shape should be ``(T, N)`` for ``T`` time
        points and ``N`` trials. Otherwise, ``data`` is the already averaged
        waveform aligned to ``time``.
    model_name : str
        Name of the event model defined in :mod:`event_models`.
    stim_times_s : sequence, optional
        Stimulus times in seconds. When given, ``time`` is interpreted in seconds
        and trials are recut around each stimulus before averaging (mean across
        all recut snippets).
    window_ms : tuple, optional
        (start, end) window in milliseconds used for fitting, by default
        (0, 50).
    maxfev : int, optional
        Maximum function evaluations passed to ``curve_fit``.
    pre_ms, post_ms : float, optional
        Window around each stimulus used when recutting trials.
    align_by_peak : bool, optional
        If True, align each recut snippet by its local peak before averaging.
        Defaults to ``False`` so that time zero corresponds to the stimulus.

    Returns
    -------
    tuple | None
        ``(params, t_ms, y_avg)`` where ``params`` is a mapping of parameter
        names to fitted values and ``t_ms``/``y_avg`` are the averaged waveform.
        ``None`` if fitting fails.
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
            t_rel_s, avg = build_median_recut_waveform(
                time,
                data,
                stim_times_s,
                pre_ms=pre_ms,
                post_ms=post_ms,
                align_by_peak=align_by_peak,
                peak_win_ms=25.0,
                peak_search_pre_ms=0.0,
                stat="mean",
            )
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
        popt, _ = curve_fit(
            spec['func'], tf, yf, p0=p0, bounds=spec['bounds'], maxfev=maxfev
        )
        params = {name: float(val) for name, val in zip(spec['params'], popt)}
        return params, t_ms, y_avg
    except Exception:
        return None
