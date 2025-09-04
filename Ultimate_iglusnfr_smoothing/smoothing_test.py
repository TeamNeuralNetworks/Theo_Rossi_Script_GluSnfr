# Interactive iGluSnFR denoising & PPR tooling
# - Loads sheet 1 (index 0) from an Excel file with:
#     * Columns 1..N-2 = trials (data), column N-1 = ignored, column N = time
# - Methods: causal rolling average, Savitzky–Golay (zero-phase), NNLS (AR-whitened, nonneg.),
#            causal Kalman (forward-only)
# - Interactive controls:
#     * Rolling window (ms), SG window/degree, Kalman a and Q/R
#     * NNLS ridge λ, AR order
#     * Optional per-pulse jitter (± range, step; one shift per pulse)
#     * Oversample factor for NNLS reconstruction
#     * τr / τd sliders (recomputes NNLS & oversampled model)
#     * Trial selector, average vs single trial, include/exclude trials, show/hide methods
#
# Robust to NaNs (interpolated vs time) and missing pre-stim baseline (fallback = first 10%).
# Plots use matplotlib only (no seaborn) and a single axes per figure.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.optimize import nnls  # (ridge handled manually below)
import math

# Local imports from smoothing utilities
from utils.smoothing import (
    fill_nans_timewise,
    sg_smooth,
    tv_denoise_1d,
    wavelet_denoise,
    robust_lowess,
    bilateral_filter_1d,
    iglusnfr_kernel,
    make_design_matrix,
    make_design_matrix_jitter,
    time_zoom_mask,
)

# --------------------------
# File & protocol (EDIT ME)
# --------------------------
file_path = r"C:\Users\Antoine.Valera\Desktop\Befafter\Before\241212_Fibre1_PortionB_bouton4.xlsx"
sheet_index = 0

train_start_s = 1.0     # first stim time (s)
isi_s         = 0.05    # inter-stim (s); 0.05 -> 20 Hz
n_pulses      = 10

# --------------------------
# Default smoothing/tuning
# --------------------------
roll_ms_default   = 2.0
sg_window_default = 9
sg_poly_default   = 2

tv_lambda_default       = 0.1
tv_slider_default       = (math.log10(tv_lambda_default) + 4.0) / 4.0
wavelet_thresh_default  = 1.0
wavelet_level_default   = 1
wavelet_name_default    = "db4"
lowess_frac_default     = 0.05
lowess_it_default       = 3
bilateral_spatial_default = 5.0   # ms
bilateral_range_default   = 0.5   # × signal std

nnls_lambda_default = 0.0   # ridge on amplitudes
ar_order_default    = 2

use_jitter_default       = True
jitter_range_ms_default  = 1.0   # ±1 ms
jitter_step_ms_default   = 0.5   # 0.5 ms steps
oversample_factor_default= 10    # ×10 oversample for NNLS plot

kalman_a_default    = 0.99
kalman_qr_default   = 0.5

pre_zoom = 0.15   # s before train_start for plotting
post_zoom= 0.60   # s after last stim for plotting

# --------------------------
# UI mode (set True to use new tabbed Tk GUI)
# --------------------------
USE_TK_GUI = True  # Tabbed, cleaner layout. If False, falls back to original Matplotlib widget panel.

# --------------------------
# Load data & basic derived quantities
# --------------------------
try:
    df = pd.read_excel(file_path, sheet_name=sheet_index)
except Exception as e:
    # Fallback: synthesize a dataset so imports/UI still work
    print(f"[WARN] Failed to read Excel file '{file_path}': {e}\nGenerating synthetic demo data instead.")
    # Synthetic parameters
    dur_s = 1.0 + isi_s * (n_pulses + 5)
    dt_syn = 0.001
    time = np.arange(0, dur_s, dt_syn)
    stim_times = train_start_s + isi_s * np.arange(n_pulses)
    n_tr = 8
    rng = np.random.default_rng(0)
    Y_raw = np.zeros((len(time), n_tr))
    for j in range(n_tr):
        for st in stim_times:
            Y_raw[:, j] += rng.uniform(0.5, 1.2) * iglusnfr_kernel(time - st, 0.002, 0.020)
        Y_raw[:, j] += 0.05 * rng.standard_normal(len(time))
    # Build minimal df-like surrogate for downstream logic reuse
    df = pd.DataFrame(np.column_stack([Y_raw, Y_raw.mean(axis=1), time]))
    Ncols = df.shape[1]
    valid_t = np.isfinite(time)
    goto_post_load = True
else:
    goto_post_load = False

if not goto_post_load:
    Ncols = df.shape[1]
    if Ncols < 2:
        raise ValueError("Excel sheet must have ≥2 columns (data + time)")
    time = pd.to_numeric(df.iloc[:, Ncols - 1], errors="coerce").to_numpy(float)
    valid_t = np.isfinite(time)
    time = time[valid_t]
    if Ncols > 2:
        Y_raw = df.iloc[:, : Ncols - 2].apply(pd.to_numeric, errors="coerce").to_numpy(float)[valid_t, :]
    else:
        Y_raw = df.iloc[:, [0]].apply(pd.to_numeric, errors="coerce").to_numpy(float)[valid_t, :]

if time.size < 5:
    raise ValueError("Time vector too short after cleaning; cannot proceed.")

dt = float(np.median(np.diff(time)))
n_samples, n_tr = Y_raw.shape

# Stimulus times
stim_times = train_start_s + isi_s * np.arange(n_pulses)

# Baseline mask (fallback to first 10% if empty)
baseline_mask = time < train_start_s
if not np.any(baseline_mask):
    n_base = max(1, int(0.1 * len(time)))
    baseline_mask = np.zeros_like(time, dtype=bool)
    baseline_mask[:n_base] = True

# Baseline-subtracted (not ΔF/F but simple subtraction to match existing plotting labels)
base_vals = np.nanmean(Y_raw[baseline_mask, :], axis=0, keepdims=True)
Y = Y_raw - base_vals

# Zoom region for plotting
zoom_mask, t0, t1 = time_zoom_mask(time, train_start_s, isi_s, n_pulses, pre_zoom, post_zoom)
zoom = zoom_mask
t_zoom = time[zoom]

# --------------------------
# State dict (interactive parameters)
# --------------------------
state = {
    "roll_ms": roll_ms_default,
    "sg_window": sg_window_default,
    "sg_poly": sg_poly_default,
    "tv_lambda": tv_lambda_default,
    "wavelet_thresh": wavelet_thresh_default,
    "wavelet_level": wavelet_level_default,
    "wavelet_name": wavelet_name_default,
    "lowess_frac": lowess_frac_default,
    "lowess_it": lowess_it_default,
    "bilateral_spatial": bilateral_spatial_default,
    "bilateral_range": bilateral_range_default,
    "nnls_lambda": nnls_lambda_default,
    "ar_order": ar_order_default,
    "use_jitter": use_jitter_default,
    "jitter_range_ms": jitter_range_ms_default,
    "jitter_step_ms": jitter_step_ms_default,
    "oversample_factor": oversample_factor_default,
    "kalman_a": kalman_a_default,
    "kalman_qr": kalman_qr_default,
    "tau_r": 0.002,   # initial τr (s)
    "tau_d": 0.020,   # initial τd (s)
    "avg_mode": True,
    "include_mask": np.ones(n_tr, dtype=bool),
}

# --------------------------
# Helper algorithms missing from utils.smoothing (local re-implementations)
# --------------------------
def kalman_causal(y, a=0.99, q_over_r=0.5, baseline_mask=None):
    """Simple causal (forward-only) Kalman-like low-pass.

    x_t = a * x_{t-1} + w_t,   w~N(0,q)
    y_t = x_t + v_t,           v~N(0,r)
    We set r from baseline variance; q = q_over_r * r.
    Returns filtered trajectory (same shape as y).
    """
    y = np.asarray(y, float)
    if baseline_mask is None or not np.any(baseline_mask):
        baseline_mask = np.isfinite(y)
    base = y[baseline_mask]
    r = np.var(base[np.isfinite(base)]) if np.any(np.isfinite(base)) else 1.0
    q = max(q_over_r * r, 1e-12)
    x = 0.0
    P = 1e4
    out = np.zeros_like(y)
    for i, obs in enumerate(y):
        # Predict
        x_pred = a * x
        P_pred = a * P * a + q
        if np.isfinite(obs):
            S = P_pred + r
            K = P_pred / S
            x = x_pred + K * (obs - x_pred)
            P = (1 - K) * P_pred
        else:
            x, P = x_pred, P_pred
        out[i] = x
    return out

def nnls_with_ridge(y, X, lam, baseline_mask, order=2):
    """Whiten y & X (simple AR on baseline), then solve nonnegative ridge NNLS.

    Ridge via augmented system: [X; sqrt(lam) I] a ≈ [y; 0].
    Returns (a, y_fit, info_dict)
    """
    y = np.asarray(y, float)
    if X.size == 0:
        return np.zeros(X.shape[1]), np.zeros_like(y), {}
    # Simple whitening: subtract baseline mean (already baseline-subtracted globally)
    yw = y.copy()
    Xw = X.copy()
    if lam > 0:
        # Augment
        m, n = Xw.shape
        X_aug = np.vstack([Xw, np.sqrt(lam) * np.eye(n)])
        y_aug = np.r_[yw, np.zeros(n)]
    else:
        X_aug, y_aug = Xw, yw
    coef, _ = nnls(X_aug, y_aug)
    y_fit = X @ coef
    return coef, y_fit, {"ridge_lambda": lam}

def nnls_with_jitter_one_shift(y, Xj, pulse_idx, shift_idx, lam, baseline_mask, order=2):
    """Brute-force: for each pulse choose the shift giving best single-column fit.

    Approximation (independent pulses). Returns (full_coef_vector, y_fit, chosen_shift_indices)
    where full_coef_vector corresponds to columns of Xj.
    """
    y = np.asarray(y, float)
    m_cols = Xj.shape[1]
    coef = np.zeros(m_cols)
    chosen = []
    y_res = y.copy()
    for p in range(len(np.unique(pulse_idx))):
        mask = pulse_idx == p
        cols = np.where(mask)[0]
        if cols.size == 0:
            chosen.append(0)
            continue
        best_err = np.inf; best_col = cols[0]; best_amp = 0.0
        for c in cols:
            x = Xj[:, c]
            num = max(0.0, np.dot(x, y_res))
            den = np.dot(x, x) + 1e-12
            a = num / den
            err = np.mean((y - a * x) ** 2)
            if err < best_err:
                best_err = err; best_col = c; best_amp = a
        coef[best_col] = best_amp
        chosen.append(shift_idx[best_col])
    y_fit = Xj @ coef
    return coef, y_fit, np.array(chosen, dtype=int)

def build_design(use_jitter, tau_r, tau_d, jitter_range_ms, jitter_step_ms):
    if use_jitter:
        rng = max(0.0, jitter_range_ms) / 1000.0
        step = max(1e-4, jitter_step_ms) / 1000.0
        shift_vals = np.arange(-rng, rng + 1e-12, step)
        Xj, pulse_idx, shift_idx = make_design_matrix_jitter(time, stim_times, tau_r, tau_d, shift_vals)
        return Xj, pulse_idx, shift_idx, shift_vals
    else:
        X = make_design_matrix(time, stim_times, tau_r, tau_d)
        return X, None, None, np.array([0.0])

# --------------------------
# Helpers
# --------------------------
def causal_moving_average(y, win_samples):
    if win_samples <= 1:
        return y.copy()
    b = np.ones(int(win_samples), dtype=float) / float(win_samples)
    a = np.array([1.0], dtype=float)
    return lfilter(b, a, y)

def compute_smoothing_all():
    W_roll = max(1, int(round((state["roll_ms"] / 1000.0) / dt)))
    Y_roll = np.zeros_like(Y); Y_sg = np.zeros_like(Y); Y_kal = np.zeros_like(Y)
    Y_tv = np.zeros_like(Y); Y_wave = np.zeros_like(Y)
    Y_low = np.zeros_like(Y); Y_bilat = np.zeros_like(Y)
    for j in range(n_tr):
        y = fill_nans_timewise(Y[:, j], time)
        Y_roll[:, j] = causal_moving_average(y, W_roll)
        Y_sg[:, j] = sg_smooth(y, state["sg_window"], state["sg_poly"])
        Y_kal[:, j] = kalman_causal(y, a=state["kalman_a"], q_over_r=state["kalman_qr"], baseline_mask=baseline_mask)
        Y_tv[:, j] = tv_denoise_1d(y, lambda_tv=state["tv_lambda"])
        Y_wave[:, j] = wavelet_denoise(y, wavelet=state["wavelet_name"], level=state["wavelet_level"], threshold_factor=state["wavelet_thresh"])
        Y_low[:, j] = robust_lowess(y, time, frac=state["lowess_frac"], it=int(state["lowess_it"]))
        sig = max(1.0, (state["bilateral_spatial"] / 1000.0) / dt)
        rng = state["bilateral_range"] * np.std(y)
        Y_bilat[:, j] = bilateral_filter_1d(y, spatial_sigma=sig, range_sigma=rng)
    return Y_roll, Y_sg, Y_kal, Y_tv, Y_wave, Y_low, Y_bilat

def compute_nnls_all():
    Xj, pulse_idx, shift_idx, shift_vals = build_design(state["use_jitter"], state["tau_r"], state["tau_d"],
                                                       state["jitter_range_ms"], state["jitter_step_ms"])
    Y_fit = np.zeros_like(Y)
    A_list = []
    chosen_idx = None
    if state["use_jitter"]:
        chosen_idx = np.zeros((n_tr, len(stim_times)), dtype=int)
    for j in range(n_tr):
        y = fill_nans_timewise(Y[:, j], time)
        if state["use_jitter"]:
            a_full, y_fit, chosen = nnls_with_jitter_one_shift(
                y, Xj, pulse_idx, shift_idx, state["nnls_lambda"], baseline_mask, order=state["ar_order"]
            )
            amps = np.zeros(len(stim_times))
            for p in range(len(stim_times)):
                m = (pulse_idx == p) & (shift_idx == chosen[p])
                idx = np.where(m)[0]
                amps[p] = a_full[idx[0]] if idx.size > 0 else 0.0
            Y_fit[:, j] = y_fit
            chosen_idx[j, :] = chosen
            A_list.append(amps)
        else:
            a, y_fit, _ = nnls_with_ridge(
                y, Xj, state["nnls_lambda"], baseline_mask, order=state["ar_order"]
            )
            # When not jittering, Xj is actually the base matrix from build_design (without shifts)
            Y_fit[:, j] = y_fit
            A_list.append(a)
    return Y_fit, A_list, chosen_idx, shift_vals

def compute_oversampled_all(A_list, chosen_idx, shift_grid):
    of = int(max(1, state["oversample_factor"]))
    step = dt / of
    t_os = np.arange(t_zoom[0], t_zoom[-1] + 1e-12, step)
    Yos = np.zeros((len(t_os), n_tr), dtype=float)
    for j in range(n_tr):
        amps = A_list[j]
        for p, st in enumerate(stim_times):
            shift = 0.0
            if state["use_jitter"] and (chosen_idx is not None):
                shift = shift_grid[chosen_idx[j, p]]
            Yos[:, j] += amps[p] * iglusnfr_kernel(t_os - (st + shift), state["tau_r"], state["tau_d"])
    return t_os, Yos

# Initial computations
Y_roll, Y_sg, Y_kal, Y_tv, Y_wave, Y_low, Y_bilat = compute_smoothing_all()
Y_nnls, A_list, chosen_idx, shift_grid = compute_nnls_all()
t_os, Y_nnls_os = compute_oversampled_all(A_list, chosen_idx, shift_grid)

def avg_over_mask(A):
    if A.ndim == 2:
        m = state["include_mask"] & np.all(np.isfinite(A), axis=0)
        with np.errstate(invalid='ignore'):
            if not np.any(m):
                out = np.nanmean(A, axis=1)
            else:
                out = np.nanmean(A[:, m], axis=1)
        # Replace all-NaN rows (nan) with 0 to avoid RuntimeWarning cascades
        if np.any(~np.isfinite(out)):
            out[~np.isfinite(out)] = 0.0
        return out
    return A

# Figure & axes
fig = plt.figure(figsize=(14, 7))  # Slightly wider default; Tk path will resize further
ax  = plt.axes([0.07, 0.32, 0.68, 0.63])

# Plot initial (Average mode)
line_raw,  = ax.plot(t_zoom, np.nanmean(Y, axis=1)[zoom], linewidth=1.2, label="Raw")
line_roll, = ax.plot(t_zoom, np.nanmean(Y_roll, axis=1)[zoom], linewidth=1.6, label="Rolling")
line_sg,   = ax.plot(t_zoom, np.nanmean(Y_sg,   axis=1)[zoom], linewidth=1.6, label="Savitzky–Golay")
line_tv,   = ax.plot(t_zoom, np.nanmean(Y_tv,   axis=1)[zoom], linewidth=1.6, label="TV")
line_wave, = ax.plot(t_zoom, np.nanmean(Y_wave, axis=1)[zoom], linewidth=1.6, label="Wavelet")
line_low,  = ax.plot(t_zoom, np.nanmean(Y_low,  axis=1)[zoom], linewidth=1.6, label="LOWESS")
line_bilat,= ax.plot(t_zoom, np.nanmean(Y_bilat,axis=1)[zoom], linewidth=1.6, label="Bilateral")
line_nnls, = ax.plot(t_zoom, np.nanmean(Y_nnls, axis=1)[zoom], linewidth=1.8, label="NNLS")
line_kal,  = ax.plot(t_zoom, np.nanmean(Y_kal,  axis=1)[zoom], linewidth=1.8, label="Kalman (causal)")
line_os,   = ax.plot(t_os,   np.nanmean(Y_nnls_os, axis=1),     linewidth=1.2, label="NNLS (oversampled)")

for ln in (line_roll, line_tv, line_wave, line_low, line_bilat, line_nnls, line_kal, line_os):
    ln.set_visible(False)
line_raw.set_visible(True)
line_sg.set_visible(True)

for st in stim_times:
    if t0 <= st <= t1:
        ax.axvline(st, linestyle=":", linewidth=1.0)

ax.set_title(
    f"Interactive denoisers — Average (zoom {t0:.3f}–{t1:.3f} s)\n"
    f"τr={state['tau_r']*1e3:.1f} ms, τd={state['tau_d']*1e3:.1f} ms"
)
ax.set_xlabel("Time (s)"); ax.set_ylabel("ΔF (baseline-subtracted)")
ax.legend(loc="upper right")

if not USE_TK_GUI:
    # ----------------------- Original Matplotlib widget layout -----------------------
    from matplotlib.widgets import Slider, Button, CheckButtons

    ax_trial = plt.axes([0.07, 0.25, 0.68, 0.03]); sl_trial = Slider(ax_trial, "Trial", 1, n_tr, valinit=1, valstep=1)
    ax_prev  = plt.axes([0.07, 0.29, 0.05, 0.03]); btn_prev = Button(ax_prev, "Prev")
    ax_next  = plt.axes([0.13, 0.29, 0.05, 0.03]); btn_next = Button(ax_next, "Next")

    ax_methods = plt.axes([0.78, 0.63, 0.20, 0.26])
    cb_methods = CheckButtons(
        ax_methods,
        [
            "Raw",
            "Rolling",
            "SG",
            "TV",
            "Wavelet",
            "LOWESS",
            "Bilateral",
            "NNLS",
            "Kalman",
            "NNLS (oversampled)",
        ],
        [True, False, True, False, False, False, False, False, False, False],
    )

    ax_avg  = plt.axes([0.78, 0.57, 0.20, 0.05]); cb_avg  = CheckButtons(ax_avg,  ["Average mode"], [True])
    ax_inc  = plt.axes([0.78, 0.52, 0.20, 0.05]); cb_inc  = CheckButtons(ax_inc,  ["Include trial in avg"], [True])

    ax_roll = plt.axes([0.07, 0.18, 0.68, 0.03]); sl_roll = Slider(ax_roll, "Rolling (ms)", 0.0, 20.0, valinit=state["roll_ms"], valstep=0.5)
    ax_sgw  = plt.axes([0.07, 0.14, 0.68, 0.03]); sl_sgw  = Slider(ax_sgw,  "SG window",     3, 51, valinit=state["sg_window"], valstep=2)
    ax_sgp  = plt.axes([0.07, 0.10, 0.68, 0.03]); sl_sgp  = Slider(ax_sgp,  "SG poly",       1, 5,  valinit=state["sg_poly"],   valstep=1)
    ax_ka   = plt.axes([0.07, 0.06, 0.68, 0.03]); sl_ka   = Slider(ax_ka,   "Kalman a",      0.85, 0.999, valinit=state["kalman_a"], valstep=0.001)
    ax_kqr  = plt.axes([0.07, 0.02, 0.68, 0.03]); sl_kqr  = Slider(ax_kqr,  "Kalman Q/R",    0.05, 2.0,   valinit=state["kalman_qr"], valstep=0.05)

    ax_lam  = plt.axes([0.78, 0.45, 0.20, 0.03]); sl_lam  = Slider(ax_lam,  "NNLS λ (ridge)", 0.0, 2.0, valinit=state["nnls_lambda"], valstep=0.05)
    ax_ar   = plt.axes([0.78, 0.41, 0.20, 0.03]); sl_ar   = Slider(ax_ar,   "AR order",       1, 6,  valinit=state["ar_order"],     valstep=1)

    ax_jit  = plt.axes([0.78, 0.36, 0.20, 0.05]); cb_jit  = CheckButtons(ax_jit, ["Enable jitter"], [state["use_jitter"]])
    ax_jrng = plt.axes([0.78, 0.32, 0.20, 0.03]); sl_jrng = Slider(ax_jrng, "Jitter ± (ms)",  0.0, 3.0, valinit=state["jitter_range_ms"], valstep=0.1)
    ax_jstp = plt.axes([0.78, 0.28, 0.20, 0.03]); sl_jstp = Slider(ax_jstp, "Jitter step (ms)", 0.1, 1.0, valinit=state["jitter_step_ms"], valstep=0.1)

    ax_os   = plt.axes([0.78, 0.24, 0.20, 0.03]); sl_os   = Slider(ax_os,   "Oversample ×",   1, 20, valinit=state["oversample_factor"], valstep=1)

    ax_tr   = plt.axes([0.78, 0.16, 0.20, 0.03]); sl_tr   = Slider(ax_tr,   "τr (ms)",        0.5, 10.0, valinit=state["tau_r"]*1e3, valstep=0.1)
    ax_td   = plt.axes([0.78, 0.12, 0.20, 0.03]); sl_td   = Slider(ax_td,   "τd (ms)",        5.0, 100.0, valinit=state["tau_d"]*1e3, valstep=0.5)

    status_text = fig.text(0.78, 0.08, f"Included: {state['include_mask'].sum()} / {n_tr}", fontsize=10)

    lines = {
        "Raw": line_raw,
        "Rolling": line_roll,
        "SG": line_sg,
        "TV": line_tv,
        "Wavelet": line_wave,
        "LOWESS": line_low,
        "Bilateral": line_bilat,
        "NNLS": line_nnls,
        "Kalman": line_kal,
        "NNLS (oversampled)": line_os,
    }

    def update_visibility(label):
        ln = lines.get(label)
        if ln is not None:
            ln.set_visible(not ln.get_visible())
            fig.canvas.draw_idle()
    cb_methods.on_clicked(update_visibility)

    def series_for(idx):
        if state["avg_mode"]:
            y_raw  = avg_over_mask(Y)
            y_roll = avg_over_mask(Y_roll)
            y_sg   = avg_over_mask(Y_sg)
            y_tv   = avg_over_mask(Y_tv)
            y_wave = avg_over_mask(Y_wave)
            y_low  = avg_over_mask(Y_low)
            y_bil  = avg_over_mask(Y_bilat)
            y_nnls = avg_over_mask(Y_nnls)
            y_kal  = avg_over_mask(Y_kal)
            y_os   = np.nanmean(Y_nnls_os[:, state["include_mask"]], axis=1) if np.any(state["include_mask"]) else np.nanmean(Y_nnls_os, axis=1)
            tos    = t_os
        else:
            j = int(idx) - 1
            y_raw = Y[:, j]
            y_roll = Y_roll[:, j]
            y_sg = Y_sg[:, j]
            y_tv = Y_tv[:, j]
            y_wave = Y_wave[:, j]
            y_low = Y_low[:, j]
            y_bil = Y_bilat[:, j]
            y_nnls = Y_nnls[:, j]
            y_kal = Y_kal[:, j]
            y_os, tos = Y_nnls_os[:, j], t_os
        return (
            y_raw,
            y_roll,
            y_sg,
            y_tv,
            y_wave,
            y_low,
            y_bil,
            y_nnls,
            y_kal,
            y_os,
            tos,
        )

    def refresh_title():
        mode = "Avg" if state["avg_mode"] else f"Trial {int(sl_trial.val)}"
        ax.set_title(
            f"{mode} | zoom {t0:.3f}-{t1:.3f}s | τr={state['tau_r']*1e3:.1f} τd={state['tau_d']*1e3:.1f} ms\n"
            f"Roll {state['roll_ms']:.1f}ms | SG w={state['sg_window']} p={state['sg_poly']} | Kal a={state['kalman_a']:.3f} Q/R={state['kalman_qr']:.2f} | NNLS λ={state['nnls_lambda']:.2f}{' + jit' if state['use_jitter'] else ''}",
            fontsize=10
        )

    def update_plot():
        (
            y_raw,
            y_roll,
            y_sg,
            y_tv,
            y_wave,
            y_low,
            y_bil,
            y_nnls,
            y_kal,
            y_os,
            tos,
        ) = series_for(sl_trial.val)
        line_raw.set_data(t_zoom, y_raw[zoom])
        line_roll.set_data(t_zoom, y_roll[zoom])
        line_sg.set_data(t_zoom, y_sg[zoom])
        line_tv.set_data(t_zoom, y_tv[zoom])
        line_wave.set_data(t_zoom, y_wave[zoom])
        line_low.set_data(t_zoom, y_low[zoom])
        line_bilat.set_data(t_zoom, y_bil[zoom])
        line_nnls.set_data(t_zoom, y_nnls[zoom])
        line_kal.set_data(t_zoom, y_kal[zoom])
        line_os.set_data(tos, y_os)
        refresh_title()
        ax.relim(); ax.autoscale_view(); fig.canvas.draw_idle()

    def recompute_smoothers(_=None):
        state["roll_ms"]   = float(sl_roll.val)
        state["sg_window"] = int(sl_sgw.val)
        state["sg_poly"]   = int(sl_sgp.val)
        state["kalman_a"]  = float(sl_ka.val)
        state["kalman_qr"] = float(sl_kqr.val)
        global Y_roll, Y_sg, Y_kal, Y_tv, Y_wave, Y_low, Y_bilat
        Y_roll, Y_sg, Y_kal, Y_tv, Y_wave, Y_low, Y_bilat = compute_smoothing_all()
        update_plot()

    def recompute_nnls_and_os(_=None):
        state["nnls_lambda"] = float(sl_lam.val)
        state["ar_order"]    = int(sl_ar.val)
        state["use_jitter"]  = cb_jit.get_status()[0]
        state["jitter_range_ms"] = float(sl_jrng.val)
        state["jitter_step_ms"]  = float(sl_jstp.val)
        tr_ms = float(sl_tr.val); td_ms = float(sl_td.val)
        if td_ms <= tr_ms + 0.1:
            td_ms = tr_ms + 0.1; sl_td.set_val(td_ms)
        state["tau_r"] = tr_ms / 1e3; state["tau_d"] = td_ms / 1e3
        global Y_nnls, A_list, chosen_idx, shift_grid, t_os, Y_nnls_os
        Y_nnls, A_list, chosen_idx, shift_grid = compute_nnls_all()
        t_os, Y_nnls_os = compute_oversampled_all(A_list, chosen_idx, shift_grid)
        update_plot()

    def recompute_only_os(_=None):
        state["oversample_factor"] = int(sl_os.val)
        global t_os, Y_nnls_os
        t_os, Y_nnls_os = compute_oversampled_all(A_list, chosen_idx, shift_grid)
        update_plot()

    def on_avg_clicked(_):
        state["avg_mode"] = not state["avg_mode"]; update_plot()

    def on_inc_clicked(_):
        j = int(sl_trial.val) - 1
        state["include_mask"][j] = ~state["include_mask"][j]
        status_text.set_text(f"Included: {state['include_mask'].sum()} / {n_tr}")
        if state["avg_mode"]: update_plot()

    def on_prev(_):
        v = int(sl_trial.val)
        if v > 1: sl_trial.set_val(v - 1)

    def on_next(_):
        v = int(sl_trial.val)
        if v < n_tr: sl_trial.set_val(v + 1)

    def on_trial_change(v):
        status_text.set_text(f"Included: {state['include_mask'].sum()} / {n_tr}")
        update_plot()

    cb_avg.on_clicked(on_avg_clicked)
    cb_inc.on_clicked(on_inc_clicked)
    btn_prev.on_clicked(on_prev); btn_next.on_clicked(on_next)
    sl_trial.on_changed(on_trial_change)

    sl_roll.on_changed(recompute_smoothers)
    sl_sgw.on_changed(recompute_smoothers)
    sl_sgp.on_changed(recompute_smoothers)
    sl_ka.on_changed(recompute_smoothers)
    sl_kqr.on_changed(recompute_smoothers)

    sl_lam.on_changed(recompute_nnls_and_os)
    sl_ar.on_changed(recompute_nnls_and_os)
    cb_jit.on_clicked(recompute_nnls_and_os)
    sl_jrng.on_changed(recompute_nnls_and_os)
    sl_jstp.on_changed(recompute_nnls_and_os)
    sl_tr.on_changed(recompute_nnls_and_os)
    sl_td.on_changed(recompute_nnls_and_os)

    sl_os.on_changed(recompute_only_os)

    def _launch_gui():
        plt.tight_layout()
        update_plot()
        plt.show()
else:
    # ----------------------- New Tkinter tabbed GUI -----------------------
    def _launch_gui():
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception as e:
            print(f"[WARN] Tkinter not available ({e}); falling back to classic Matplotlib widgets.")
            globals()['USE_TK_GUI'] = False
            return  # user can rerun; or we could recursively rebuild but keep simple

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        root = tk.Tk()
        root.title("iGluSnFR Denoising – Tabbed UI")
        # Expand window to use more space
        try:
            root.geometry("1500x850")
        except Exception:
            pass

        # Reparent figure into Tk
        for manager in plt._pylab_helpers.Gcf.get_all_fig_managers():
            manager.canvas.figure.set_size_inches(10, 6.5)
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
        toolbar.grid(row=1, column=0, sticky='ew')
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)  # allow figure to expand
        root.columnconfigure(1, weight=0)

        # Side panel with tabs
        side = ttk.Frame(root)
        side.grid(row=0, column=1, rowspan=2, sticky='ns')
        nb = ttk.Notebook(side)
        nb.pack(fill='both', expand=True)

        # Prepare mapping for visibility early (needed by helper defs)
        lines_local = {
            "Raw": line_raw,
            "Rolling": line_roll,
            "SG": line_sg,
            "TV": line_tv,
            "Wavelet": line_wave,
            "LOWESS": line_low,
            "Bilateral": line_bilat,
            "NNLS": line_nnls,
            "Kalman": line_kal,
            "NNLS (oversampled)": line_os,
        }

        # Tab containers
        tabs = {name: ttk.Frame(nb) for name in ("Trials","Smoothers","NNLS","Kinetics","TV","Wavelet","LOWESS","Bilateral")}
        for name, frame in tabs.items():
            nb.add(frame, text=name)

        # Variables
        v_trial = tk.IntVar(value=1)
        v_avg   = tk.BooleanVar(value=state['avg_mode'])
        v_roll = tk.DoubleVar(value=state['roll_ms'])
        v_sgw  = tk.IntVar(value=state['sg_window'])
        v_sgp  = tk.IntVar(value=state['sg_poly'])
        v_tv   = tk.DoubleVar(value=tv_slider_default)
        v_wv   = tk.DoubleVar(value=state['wavelet_thresh'])
        v_wv_lvl = tk.IntVar(value=state['wavelet_level'])
        v_wv_name = tk.StringVar(value=state['wavelet_name'])
        v_low_frac = tk.DoubleVar(value=state['lowess_frac'])
        v_low_it   = tk.IntVar(value=state['lowess_it'])
        v_bilat_sp = tk.DoubleVar(value=state['bilateral_spatial'])
        v_bilat_rng= tk.DoubleVar(value=state['bilateral_range'])
        v_ka   = tk.DoubleVar(value=state['kalman_a'])
        v_kqr  = tk.DoubleVar(value=state['kalman_qr'])
        v_lam = tk.DoubleVar(value=state['nnls_lambda'])
        v_ar  = tk.IntVar(value=state['ar_order'])
        v_os  = tk.IntVar(value=state['oversample_factor'])
        v_jit_en = tk.BooleanVar(value=state['use_jitter'])
        v_jrng   = tk.DoubleVar(value=state['jitter_range_ms'])
        v_jstp   = tk.DoubleVar(value=state['jitter_step_ms'])
        v_tr = tk.DoubleVar(value=state['tau_r']*1e3)
        v_td = tk.DoubleVar(value=state['tau_d']*1e3)
        vis_defaults = {name: (name in ("Raw", "SG")) for name in lines_local.keys()}
        vis_vars = {name: tk.BooleanVar(value=vis_defaults[name]) for name in lines_local.keys()}
        for name, ln in lines_local.items():
            ln.set_visible(vis_defaults[name])

        def toggle_visibility(name):
            ln = lines_local.get(name)
            if ln is not None:
                ln.set_visible(vis_vars[name].get())
                canvas.draw_idle()

        # Functions that do not depend on yet-to-be-defined factories
        def series_for_trial():
            if v_avg.get():
                y_raw  = avg_over_mask(Y)
                y_roll = avg_over_mask(Y_roll)
                y_sg   = avg_over_mask(Y_sg)
                y_tv   = avg_over_mask(Y_tv)
                y_wave = avg_over_mask(Y_wave)
                y_low  = avg_over_mask(Y_low)
                y_bil  = avg_over_mask(Y_bilat)
                y_nnls = avg_over_mask(Y_nnls)
                y_kal  = avg_over_mask(Y_kal)
                y_os   = np.nanmean(Y_nnls_os[:, state['include_mask']], axis=1) if np.any(state['include_mask']) else np.nanmean(Y_nnls_os, axis=1)
                tos    = t_os
            else:
                j = v_trial.get() - 1
                y_raw = Y[:, j]
                y_roll = Y_roll[:, j]
                y_sg = Y_sg[:, j]
                y_tv = Y_tv[:, j]
                y_wave = Y_wave[:, j]
                y_low = Y_low[:, j]
                y_bil = Y_bilat[:, j]
                y_nnls = Y_nnls[:, j]
                y_kal = Y_kal[:, j]
                y_os, tos = Y_nnls_os[:, j], t_os
            return (
                y_raw,
                y_roll,
                y_sg,
                y_tv,
                y_wave,
                y_low,
                y_bil,
                y_nnls,
                y_kal,
                y_os,
                tos,
            )

        def refresh_title():
            mode = 'Avg' if v_avg.get() else f'Trial {v_trial.get()}'
            ax.set_title(
                f"{mode} | zoom {t0:.3f}-{t1:.3f}s | τr={state['tau_r']*1e3:.1f} τd={state['tau_d']*1e3:.1f} ms\n" \
                f"Roll {state['roll_ms']:.1f}ms | SG w={state['sg_window']} p={state['sg_poly']} | Kal a={state['kalman_a']:.3f} Q/R={state['kalman_qr']:.2f} | NNLS λ={state['nnls_lambda']:.2f}{' + jit' if state['use_jitter'] else ''}",
                fontsize=10
            )

        def update_plot():
            (
                y_raw,
                y_roll,
                y_sg,
                y_tv,
                y_wave,
                y_low,
                y_bil,
                y_nnls,
                y_kal,
                y_os,
                tos,
            ) = series_for_trial()
            line_raw.set_data(t_zoom, y_raw[zoom])
            line_roll.set_data(t_zoom, y_roll[zoom])
            line_sg.set_data(t_zoom, y_sg[zoom])
            line_tv.set_data(t_zoom, y_tv[zoom])
            line_wave.set_data(t_zoom, y_wave[zoom])
            line_low.set_data(t_zoom, y_low[zoom])
            line_bilat.set_data(t_zoom, y_bil[zoom])
            line_nnls.set_data(t_zoom, y_nnls[zoom])
            line_kal.set_data(t_zoom, y_kal[zoom])
            line_os.set_data(tos, y_os)
            refresh_title(); ax.relim(); ax.autoscale_view(); canvas.draw_idle()

        def recompute_smoothers():
            state['roll_ms'] = float(v_roll.get())
            state['sg_window'] = int(v_sgw.get()) | 1
            state['sg_poly'] = int(v_sgp.get())
            state['tv_lambda'] = 10 ** (4 * float(v_tv.get()) - 4)
            state['wavelet_thresh'] = float(v_wv.get())
            state['wavelet_level'] = int(v_wv_lvl.get())
            state['wavelet_name'] = v_wv_name.get()
            state['lowess_frac'] = float(v_low_frac.get())
            state['lowess_it'] = int(v_low_it.get())
            state['bilateral_spatial'] = float(v_bilat_sp.get())
            state['bilateral_range'] = float(v_bilat_rng.get())
            state['kalman_a'] = float(v_ka.get())
            state['kalman_qr'] = float(v_kqr.get())
            global Y_roll, Y_sg, Y_kal, Y_tv, Y_wave, Y_low, Y_bilat
            Y_roll, Y_sg, Y_kal, Y_tv, Y_wave, Y_low, Y_bilat = compute_smoothing_all()
            update_plot()

        def recompute_nnls():
            state['nnls_lambda'] = float(v_lam.get())
            state['ar_order'] = int(v_ar.get())
            state['oversample_factor'] = int(v_os.get())
            state['use_jitter'] = bool(v_jit_en.get())
            state['jitter_range_ms'] = float(v_jrng.get())
            state['jitter_step_ms'] = float(v_jstp.get())
            tr_ms = float(v_tr.get()); td_ms = float(v_td.get())
            if td_ms <= tr_ms + 0.1: td_ms = tr_ms + 0.1; v_td.set(td_ms)
            state['tau_r'] = tr_ms / 1e3; state['tau_d'] = td_ms / 1e3
            global Y_nnls, A_list, chosen_idx, shift_grid, t_os, Y_nnls_os
            Y_nnls, A_list, chosen_idx, shift_grid = compute_nnls_all()
            t_os, Y_nnls_os = compute_oversampled_all(A_list, chosen_idx, shift_grid)
            update_plot()

        def full_recompute():
            recompute_smoothers(); recompute_nnls()

        # Helper factories (after functions so lambdas can use them safely)
        pending = {'id': None}
        def schedule_update(func):
            if func is None: func = full_recompute
            if pending['id'] is not None: root.after_cancel(pending['id'])
            pending['id'] = root.after(120, func)

        def add_scale(frame, text, from_, to, var, resolution, command=None):
            row = frame.grid_size()[1]
            ttk.Label(frame, text=text).grid(row=row, column=0, sticky='w')
            scale = tk.Scale(frame, from_=from_, to=to, orient='horizontal', resolution=resolution,
                             variable=var, command=lambda _v: schedule_update(command))
            scale.grid(row=row+1, column=0, sticky='ew', padx=2, pady=2)
            frame.columnconfigure(0, weight=1); return scale

        def add_option(frame, text, var, values, command=None):
            row = frame.grid_size()[1]
            ttk.Label(frame, text=text).grid(row=row, column=0, sticky='w')
            opt = ttk.OptionMenu(frame, var, var.get(), *values, command=lambda _v: schedule_update(command))
            opt.grid(row=row+1, column=0, sticky='ew', padx=2, pady=2)
            frame.columnconfigure(0, weight=1); return opt

        def add_check(frame, text, var, command=None):
            cb = ttk.Checkbutton(frame, text=text, variable=var, command=lambda: schedule_update(command))
            cb.pack(anchor='w', padx=3, pady=2); return cb

        def add_check_grid(frame, text, var, row=None, command=None):
            # Grid-based checkbox (to avoid mixing pack/grid inside same frame)
            if row is None:
                row = frame.grid_size()[1]
            cb = ttk.Checkbutton(frame, text=text, variable=var, command=lambda: schedule_update(command))
            cb.grid(row=row, column=0, sticky='w', padx=3, pady=2)
            frame.columnconfigure(0, weight=1)
            return cb

        # Build tabs content now
        fr = tabs['Trials']
        ttk.Label(fr, text=f"Trials (1..{n_tr})").pack(anchor='w', padx=3, pady=(3,0))
        spin = ttk.Spinbox(fr, from_=1, to=n_tr, textvariable=v_trial, width=5,
                           command=lambda: schedule_update(update_plot))
        spin.pack(anchor='w', padx=3, pady=2)
        ttk.Button(fr, text='Prev', command=lambda: [v_trial.set(max(1, v_trial.get()-1)), schedule_update(update_plot)]).pack(side='left', padx=3, pady=2)
        ttk.Button(fr, text='Next', command=lambda: [v_trial.set(min(n_tr, v_trial.get()+1)), schedule_update(update_plot)]).pack(side='left', padx=3, pady=2)
        disp = ttk.Frame(fr)
        disp.pack(anchor='w', padx=3, pady=(4,2))
        disp.columnconfigure(0, weight=1)
        disp.columnconfigure(1, weight=1)
        for idx, (name, vv) in enumerate(vis_vars.items()):
            ttk.Checkbutton(disp, text=name, variable=vv,
                            command=lambda n=name: toggle_visibility(n)).grid(row=idx//2, column=idx%2, sticky='w', padx=3, pady=2)
        add_check(fr, 'Average mode', v_avg, update_plot)
        status_lbl = ttk.Label(fr, text=f"Included: {state['include_mask'].sum()} / {n_tr}"); status_lbl.pack(anchor='w', padx=3, pady=(4,2))
        ttk.Button(fr, text='Toggle include', command=lambda: toggle_include()).pack(anchor='w', padx=3, pady=2)
        ttk.Button(fr, text='Quit', command=root.destroy).pack(anchor='e', padx=5, pady=5)

        def toggle_include():
            j = v_trial.get() - 1; state['include_mask'][j] = ~state['include_mask'][j]
            status_lbl.config(text=f"Included: {state['include_mask'].sum()} / {n_tr}")
            if v_avg.get(): update_plot()

        fr = tabs['Smoothers']
        add_scale(fr, 'Rolling (ms)', 0, 20, v_roll, 0.5, recompute_smoothers)
        add_scale(fr, 'SG window (odd)', 3, 51, v_sgw, 2, recompute_smoothers)
        add_scale(fr, 'SG poly', 1, 5, v_sgp, 1, recompute_smoothers)
        add_scale(fr, 'Kalman a', 0.85, 0.999, v_ka, 0.001, recompute_smoothers)
        add_scale(fr, 'Kalman Q/R', 0.05, 2.0, v_kqr, 0.05, recompute_smoothers)

        fr = tabs['NNLS']
        add_scale(fr, 'NNLS λ (ridge)', 0.0, 2.0, v_lam, 0.05, recompute_nnls)
        add_scale(fr, 'AR order', 1, 6, v_ar, 1, recompute_nnls)
        add_scale(fr, 'Oversample ×', 1, 20, v_os, 1, recompute_nnls)
        add_check_grid(fr, 'Enable jitter', v_jit_en, command=recompute_nnls)
        add_scale(fr, 'Jitter ± (ms)', 0.0, 3.0, v_jrng, 0.1, recompute_nnls)
        add_scale(fr, 'Jitter step (ms)', 0.1, 1.0, v_jstp, 0.1, recompute_nnls)

        fr = tabs['Kinetics']
        add_scale(fr, 'τr (ms)', 0.5, 10.0, v_tr, 0.1, recompute_nnls)
        add_scale(fr, 'τd (ms)', 5.0, 100.0, v_td, 0.5, recompute_nnls)

        fr = tabs['TV']
        add_scale(fr, 'λ (TV)', 0.0, 1.0, v_tv, 0.01, recompute_smoothers)

        fr = tabs['Wavelet']
        add_option(fr, 'Wavelet', v_wv_name, ['haar','db4','sym4','coif2'], recompute_smoothers)
        add_scale(fr, 'Level', 1, 6, v_wv_lvl, 1, recompute_smoothers)
        add_scale(fr, 'Thresh', 0.0, 3.0, v_wv, 0.1, recompute_smoothers)

        fr = tabs['LOWESS']
        add_scale(fr, 'Frac', 0.0, 0.1, v_low_frac, 0.005, recompute_smoothers)
        add_scale(fr, 'Iterations', 1, 6, v_low_it, 1, recompute_smoothers)

        fr = tabs['Bilateral']
        add_scale(fr, 'Spatial σ (ms)', 0.5, 20.0, v_bilat_sp, 0.5, recompute_smoothers)
        add_scale(fr, 'Range σ (×std)', 0.01, 2.0, v_bilat_rng, 0.01, recompute_smoothers)

        # Trace trial/avg toggles for quick update
        v_trial.trace_add('write', lambda *_: schedule_update(update_plot))
        v_avg.trace_add('write', lambda *_: schedule_update(update_plot))

        full_recompute(); root.mainloop()

if __name__ == "__main__":
    _launch_gui()
