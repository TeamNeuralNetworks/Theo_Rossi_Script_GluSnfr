import time
import numpy as np
from scipy.signal import lfilter, savgol_filter
from scipy.linalg import toeplitz
from scipy.optimize import nnls, minimize, NonlinearConstraint


def progress_print(msg, show=True):
    if show:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")


def fill_nans_timewise(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    yf = y.copy()
    finite = np.isfinite(yf)
    if not np.any(finite):
        return np.zeros_like(yf)
    yf[~finite] = np.interp(t[~finite], t[finite], yf[finite])
    if not np.all(np.isfinite(yf)):
        first = np.argmax(finite)
        last = len(yf) - 1 - np.argmax(finite[::-1])
        yf[:first] = yf[first]
        yf[last + 1:] = yf[last]
        yf[~np.isfinite(yf)] = 0.0
    return yf


def sg_smooth(y, window, poly):
    w = int(window)
    if w % 2 == 0:
        w += 1
    wmin = poly + 2 if ((poly + 2) % 2 == 1) else poly + 3
    w = max(w, wmin)
    w = min(w, len(y) - 1 if (len(y) % 2 == 0) else len(y))
    if w < 3:
        return y.copy()
    return savgol_filter(y, w, poly)


def tv_denoise_1d(signal, lambda_tv=1.0, n_iter=100):
    """Basic 1D total variation denoising via dual formulation."""
    y = np.asarray(signal, float)
    x = y.copy()
    p = np.zeros(len(x) - 1)
    lam = 1.0 / max(float(lambda_tv), 1e-9)
    for _ in range(int(n_iter)):
        x = y - lam * np.r_[p[0], p[1:] - p[:-1], -p[-1]]
        grad = np.diff(x)
        p = np.clip(p + grad / lam, -1.0, 1.0)
    x = y - lam * np.r_[p[0], p[1:] - p[:-1], -p[-1]]
    return x


def wavelet_denoise(signal, wavelet="db4", level=None, threshold_factor=1.0):
    """Wavelet denoising with soft thresholding using PyWavelets."""
    x = np.asarray(signal, float)
    if level is None:
        level = pywt.dwt_max_level(len(x), wavelet)
    coeffs = pywt.wavedec(x, wavelet, level=level)
    if coeffs:
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if coeffs[-1].size else 0.0
        thresh = threshold_factor * sigma * np.sqrt(2 * np.log(len(x))) if len(x) else 0.0
        coeffs[1:] = [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]
    out = pywt.waverec(coeffs, wavelet)
    return out[:len(x)]


def robust_lowess(signal, time, frac=0.05, it=3):
    """Naive LOWESS implementation with robust reweighting."""
    y = np.asarray(signal, float)
    x = np.asarray(time, float)
    n = y.size
    span = max(1, int(frac * n))
    if span % 2 == 0:
        span += 1
    y_fit = y.copy()
    for _ in range(int(it)):
        resid = y - y_fit
        s = np.median(np.abs(resid)) or 1.0
        robust = (1 - (resid / (6 * s)) ** 2) ** 2
        robust[np.abs(resid) >= 6 * s] = 0.0
        for i in range(n):
            start = max(0, i - span // 2)
            end = min(n, i + span // 2 + 1)
            xw = x[start:end]
            yw = y[start:end]
            w = robust[start:end]
            dist = np.abs(xw - x[i]) / (span / 2)
            kernel = (1 - dist ** 3) ** 3
            w = w * kernel
            if np.sum(w) == 0:
                y_fit[i] = y[i]
            else:
                b = np.polyfit(xw, yw, 1, w=w)
                y_fit[i] = b[0] * x[i] + b[1]
    return y_fit


def bilateral_filter_1d(signal, spatial_sigma=5.0, range_sigma=None):
    """Deterministic 1D bilateral filter."""
    y = np.asarray(signal, float)
    n = y.size
    spatial_sigma = max(float(spatial_sigma), 1e-6)
    if range_sigma is None:
        range_sigma = np.std(y) * 0.5 if n else 0.0
    range_sigma = max(float(range_sigma), 1e-6)
    out = np.zeros_like(y)
    half = int(3 * spatial_sigma)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        idx = np.arange(start, end)
        spatial = np.exp(-0.5 * ((idx - i) / spatial_sigma) ** 2)
        intensity = np.exp(-0.5 * ((y[start:end] - y[i]) / range_sigma) ** 2)
        w = spatial * intensity
        w /= np.sum(w) if np.sum(w) else 1.0
        out[i] = np.sum(y[start:end] * w)
    return out


def iglusnfr_kernel(t, tau_r, tau_d):
    tp = np.maximum(t, 0.0)
    k = (1.0 - np.exp(-tp / tau_r)) * np.exp(-tp / tau_d)
    support = tp <= 0.2
    area = np.trapz(k[support], t[support]) if np.any(support) else 1.0
    return k / max(area, 1e-12)


def make_design_matrix(time, stim_times, tau_r, tau_d):
    return np.column_stack([
        iglusnfr_kernel(time - st, tau_r, tau_d) for st in stim_times
    ])


def make_design_matrix_jitter(time, stim_times, tau_r, tau_d, shift_grid):
    cols, pulse_idx, shift_idx = [], [], []
    for j, st in enumerate(stim_times):
        for k, d in enumerate(shift_grid):
            cols.append(iglusnfr_kernel(time - (st + d), tau_r, tau_d))
            pulse_idx.append(j)
            shift_idx.append(k)
    return np.column_stack(cols), np.array(pulse_idx), np.array(shift_idx)


def make_design_with_shifts(time, stim_times, tau_r, tau_d_vec, deltas):
    """Design matrix with per-pulse decay and micro-shifts."""
    cols = []
    for p, st in enumerate(stim_times):
        td = float(tau_d_vec[p])
        sh = float(deltas[p])
        cols.append(iglusnfr_kernel(time - (st + sh), tau_r, td))
    return np.column_stack(cols) if cols else np.zeros((len(time), 0))


def aryule(y, order=2):
    y = np.asarray(y, float)
    y = y - np.nanmean(y)
    y = y[np.isfinite(y)]
    n = len(y)
    if n <= order + 1:
        return np.zeros(order, float)
    r = np.array([np.dot(y[: n - k], y[k:]) / n for k in range(order + 1)], float)
    R = toeplitz(r[:-1])
    a = np.linalg.solve(R, r[1:])
    return a


def choose_ar_on_baseline(y, baseline_mask, max_p=4):
    yb = np.asarray(y[baseline_mask], float)
    yb = yb[np.isfinite(yb)]
    if yb.size < 20:
        return np.r_[1.0], 0
    best = (0, np.inf, np.r_[1.0])
    n = yb.size
    for p in range(1, max_p + 1):
        try:
            a = aryule(yb, order=p)
            A = np.r_[1.0, -a]
            e = lfilter(A, [1.0], yb)
            s2 = np.mean(e ** 2)
            k = p
            aicc = n * np.log(max(s2, 1e-12)) + 2 * (k + 1) + 2 * (k + 1) * (k + 2) / max(n - k - 2, 1)
            if aicc < best[1]:
                best = (p, aicc, A)
        except Exception:
            continue
    return best[2], best[0]


def whiten_pair(y, X, baseline_mask, order=2):
    try:
        a = aryule(y[baseline_mask], order=order)
    except Exception:
        a = np.zeros(order, float)
    A = np.r_[1.0, -a]
    y_w = lfilter(A, [1.0], y)
    X_w = np.vstack([lfilter(A, [1.0], X[:, j]) for j in range(X.shape[1])]).T
    ok = np.isfinite(y_w)
    for j in range(X_w.shape[1]):
        ok &= np.isfinite(X_w[:, j])
    return y_w[ok], X_w[ok, :], ok


def whiten_with_A(y, X, A):
    y_w = lfilter(A, [1.0], y)
    X_w = np.vstack([lfilter(A, [1.0], X[:, j]) for j in range(X.shape[1])]).T
    ok = np.isfinite(y_w)
    for j in range(X_w.shape[1]):
        ok &= np.isfinite(X_w[:, j])
    return y_w[ok], X_w[ok, :], ok


def nnls_fit(y, X, baseline_mask, order=2):
    y_w, X_w, ok = whiten_pair(y, X, baseline_mask, order=order)
    if X_w.shape[0] == 0:
        return np.zeros(X.shape[1]), np.zeros_like(y)
    col_norms = np.linalg.norm(X_w, axis=0) + 1e-12
    Xn = X_w / col_norms
    coef_n, _ = nnls(Xn, y_w)
    a_hat = coef_n / col_norms
    return a_hat, (X @ a_hat)


def ppr_epsilon_band_ratio(
    y,
    X,
    baseline_mask,
    a_ref,
    order=2,
    eps_factor=1.05,
    j_denom=0,
    maxiter=300,
):
    y_w, X_w, ok = whiten_pair(y, X, baseline_mask, order=order)
    m = X_w.shape[1]
    if y_w.size == 0 or m == 0:
        return (
            np.full(m, np.nan),
            np.full(m, np.nan),
            np.full(m, np.nan),
        )
    a0 = np.clip(np.asarray(a_ref, float), 0, None)
    res0 = y_w - X_w @ a0
    r2_best = float(res0 @ res0)
    r2_lim = r2_best * float(eps_factor)

    def cons_fun(a):
        d = X_w @ a - y_w
        return d @ d - r2_lim

    nlc = NonlinearConstraint(cons_fun, -np.inf, 0.0)
    bounds = [(0.0, None)] * m
    tiny = 1e-12

    def f_factory(j):
        ej = np.zeros(m)
        ej[j] = 1.0
        e1 = np.zeros(m)
        e1[j_denom] = 1.0

        def f(a):
            den = a[j_denom] + tiny
            return a[j] / den

        def g(a):
            den = a[j_denom] + tiny
            num = a[j]
            return ej / den - (num / (den * den)) * e1

        return f, g

    center = np.full(m, np.nan)
    lo = np.full(m, np.nan)
    hi = np.full(m, np.nan)
    den0 = max(a0[j_denom], tiny)
    center = a0 / den0
    center[0] = 1.0

    for j in range(m):
        f, g = f_factory(j)
        res_min = minimize(
            f,
            a0,
            jac=g,
            method="trust-constr",
            constraints=[nlc],
            bounds=bounds,
            options={"maxiter": maxiter, "gtol": 1e-10, "xtol": 1e-10},
        )
        lo[j] = f(res_min.x) if res_min.success else min(center[j], max(0.0, center[j] * 0.9))
        res_max = minimize(
            lambda a: -f(a),
            a0,
            jac=lambda a: -g(a),
            method="trust-constr",
            constraints=[nlc],
            bounds=bounds,
            options={"maxiter": maxiter, "gtol": 1e-10, "xtol": 1e-10},
        )
        hi[j] = f(res_max.x) if res_max.success else max(center[j], center[j] * 1.1)
        if not np.isfinite(lo[j]):
            lo[j] = center[j]
        if not np.isfinite(hi[j]):
            hi[j] = center[j]
        if lo[j] > hi[j]:
            lo[j], hi[j] = hi[j], lo[j]
    return center, lo, hi


def windowed_max(series_t, series_y, stims, win_ms, N, pre_ms=0.0):
    series_t = np.asarray(series_t, float)
    series_y = np.asarray(series_y, float)
    stims = np.atleast_1d(stims)
    vals = []
    if series_t.size < 2:
        return np.zeros(len(stims)) if np.isscalar(stims) else np.zeros_like(stims, dtype=float)
    dt = float(np.median(np.diff(series_t)))
    half_post = int(round((win_ms / 1000.0) / max(dt, 1e-9)))
    half_pre = int(round((pre_ms / 1000.0) / max(dt, 1e-9)))
    halfN = max(0, int(N) // 2)
    for st in stims:
        i_stim = int(np.searchsorted(series_t, st))
        i0 = max(0, i_stim - half_pre)
        i1 = min(len(series_t) - 1, i_stim + half_post)
        seg = series_y[i0 : i1 + 1]
        if seg.size == 0:
            vals.append(0.0)
            continue
        imax = i0 + np.argmax(seg)
        j0 = max(0, imax - halfN)
        j1 = min(len(series_y), imax + halfN + 1)
        vals.append(np.mean(series_y[j0:j1]))
    return np.array(vals)


def time_zoom_mask(time, train_start_s, isi_s, n_pulses, pre_zoom, post_zoom):
    z0 = max(time.min(), train_start_s - pre_zoom)
    z1 = min(time.max(), train_start_s + isi_s * (n_pulses - 1) + post_zoom)
    return (time >= z0) & (time <= z1), z0, z1


def pick_peak_on_series(t_series, y_series, st, win_ms, pre_ms=0.0):
    """Locate the peak around a stimulus within a search window.

    Notes
    -----
    Previously the center index was chosen with ``np.searchsorted`` which picks the
    first sample *greater than or equal to* ``st``.  When the nominal stimulus time
    falls *between* two sampled time points this biases the center one sample
    later, so when we subsequently re‑express the snippet relative to the chosen
    center, the waveform appears shifted *left* (earlier) by ~1 sample.  Using the
    nearest sample instead of the first >= sample removes this systematic
    half‑sample / one‑sample bias observed as a 1–2 point left shift in recut
    medians.
    """
    t_series = np.asarray(t_series, float)
    y_series = np.asarray(y_series, float)
    if t_series.size < 2:
        return st, 0.0
    dt = float(np.median(np.diff(t_series)))
    post = int(round((win_ms / 1000.0) / max(dt, 1e-12)))
    pre = int(round((pre_ms / 1000.0) / max(dt, 1e-12)))
    # Use nearest sample to stimulus time to avoid systematic +1 index bias
    i_center = int(np.argmin(np.abs(t_series - st)))
    i0 = max(0, i_center - pre)
    i1 = min(len(t_series) - 1, i_center + post)
    seg = y_series[i0 : i1 + 1]
    if seg.size == 0:
        return st, 0.0
    imax_local = int(np.argmax(seg))
    imax = i0 + imax_local
    return t_series[imax], y_series[imax]


def compute_no_signal_mask(
    time,
    y_series,
    stim_times,
    post_zoom_s,
    peak_win_ms: float = 25.0,
    peak_search_pre_ms: float = 2.0,
):
    """Mask regions without stimulus-evoked signal for AR model fitting.

    Includes samples before the first stimulus and after the last detected
    peak plus ``post_zoom_s``.  Peak locations are estimated around each
    stimulus using :func:`pick_peak_on_series`.
    """

    try:
        peak_times = []
        for st in np.atleast_1d(stim_times):
            tp, _ = pick_peak_on_series(
                time, y_series, st, peak_win_ms, peak_search_pre_ms
            )
            peak_times.append(tp)
        last_peak = np.nanmax(peak_times) if peak_times else np.nan
        pre_mask = time < stim_times[0]
        if np.isfinite(last_peak):
            post_mask = time >= (last_peak + post_zoom_s)
        else:
            post_mask = np.zeros_like(time, dtype=bool)
        mask = pre_mask | post_mask
        if not np.any(mask):
            mask = pre_mask
        return mask
    except Exception:
        # Fall back to using only the pre-stim baseline
        return time < stim_times[0]


def build_median_recut_waveform(
    time,
    Y_all,
    stim_times,
    pre_ms=2.0,
    post_ms=200.0,
    align_by_peak=True,
    peak_win_ms=25.0,
    peak_search_pre_ms=2.0,
    stat="median",
):
    """Aggregate waveform across all events after recutting around each stimulus.

    Parameters
    ----------
    stat : {'median', 'mean'}, optional
        Reduction to apply across recut snippets, by default ``'median'``.
    """
    time = np.asarray(time, float)
    if time.size < 2 or Y_all is None or np.size(Y_all) == 0:
        return None, None
    dt = float(np.median(np.diff(time)))
    pre_s = float(pre_ms) / 1000.0
    post_s = float(post_ms) / 1000.0
    t_rel = np.arange(-pre_s, post_s + 1e-12, dt)
    Y_all = np.atleast_2d(Y_all)
    snippets = []
    for st in np.atleast_1d(stim_times):
        for j in range(Y_all.shape[1]):
            y = np.asarray(Y_all[:, j], float)
            center = st
            if align_by_peak:
                try:
                    tp, _ = pick_peak_on_series(
                        time, y, st, peak_win_ms, peak_search_pre_ms
                    )
                    center = tp
                except Exception:
                    pass
            t_win = center + t_rel
            y_win = np.full_like(t_win, np.nan, dtype=float)
            m = (t_win >= time[0]) & (t_win <= time[-1])
            if np.any(m):
                y_win[m] = np.interp(t_win[m], time, y)
            snippets.append(y_win)
    if not snippets:
        return None, None
    S = np.vstack(snippets)
    if S.size == 0:
        return None, None
    if stat == "mean":
        wave = np.nanmean(S, axis=0)
    else:
        wave = np.nanmedian(S, axis=0)
    return t_rel, wave


def fit_template_decay(
    t_rel,
    med_wave,
    tau_r=0.002,
    tau_d_grid_ms=None,
    oversample=4,
):
    """Fit an iGluSnFR-like template with adjustable decay to a waveform."""
    if t_rel is None or med_wave is None:
        return np.nan, None
    t_rel = np.asarray(t_rel, float)
    med_wave = np.asarray(med_wave, float)
    if t_rel.size < 2 or med_wave.size != t_rel.size:
        return np.nan, None

    dt = float(np.median(np.diff(t_rel)))
    if dt <= 0:
        return np.nan, None

    if tau_d_grid_ms is None:
        tau_d_grid_ms = np.arange(2.0, 81.0, 1.0)

    dt_os = dt / float(oversample)
    t_os = np.arange(t_rel[0], t_rel[-1] + dt_os / 2, dt_os)
    wave_os = np.interp(t_os, t_rel, med_wave)

    best_err = np.inf
    best_tau = np.nan
    best_fit = None

    for td_ms in tau_d_grid_ms:
        td = td_ms / 1000.0
        templ = iglusnfr_kernel(t_os, tau_r, td)
        M = np.column_stack([np.ones_like(templ), templ])
        try:
            coef, _, _, _ = np.linalg.lstsq(M, wave_os, rcond=None)
        except Exception:
            continue
        C, A = coef
        fit_os = C + A * templ
        err = np.nanmean((wave_os - fit_os) ** 2)
        if err < best_err:
            best_err = err
            best_tau = td
            best_fit = fit_os

    if best_fit is None:
        return np.nan, None

    fit_curve = np.interp(t_rel, t_os, best_fit)
    return best_tau, fit_curve


def build_median_recut_figure(
    t_rel,
    med_wave,
    fit_curve=None,
    title="Median recut transient",
):
    """Create a figure showing the median recut waveform.

    Parameters
    ----------
    t_rel : array-like
        Time relative to stimulus (s) for the median waveform.
    med_wave : array-like
        Median waveform values.
    fit_curve : array-like or None
        Optional waveform to overlay (e.g., fitted template).
    title : str
        Figure title.
    """
    if t_rel is None or med_wave is None:
        return None

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_rel * 1000.0, med_wave, linewidth=1.6, label="Median")

    if fit_curve is not None:
        ax.plot(
            t_rel * 1000.0,
            fit_curve,
            linestyle=":",
            color="red",
            linewidth=1.2,
            label="Template fit",
        )
        ax.legend(loc="best")

    ax.axvline(0.0, color="k", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("ΔF (median)")
    if title:
        ax.set_title(title)
    return fig
