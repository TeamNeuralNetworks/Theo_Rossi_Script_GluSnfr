"""
In-trial evolution of repeated, noisy events (ΔF/F₀) with per-trial smoothing
-----------------------------------------------------------------------------

Goal
----
1) Compute ΔF/F₀ *per trial* using a pre-stimulus baseline (no cross-trial mixing).
2) Denoise each trial independently using a state-space Kalman–RTS smoother
   with automatic, per-trial smoothness selection via innovations likelihood.
3) Quantify how responses evolve *within trials* across a train of events:
   extract pulse-by-pulse amplitudes in a short window after each stimulus.
4) Visualize:
   - Top-left  : raw ΔF/F₀ (trials in light gray, across-trial mean in black)
   - Bottom-left: raw ΔF/F₀ heatmap (trials × time)
   - Top-right : *smoothed* ΔF/F₀ (trials + across-trial mean)
   - Bottom-right: trials × pulses heatmap of within-trial amplitudes

Notes
-----
- The Kalman–RTS smoother implements a local-level model:
      x_t   = x_{t-1} + w_t,   w_t ~ N(0, q)        # latent clean signal
      y_t   = x_t + v_t,       v_t ~ N(0, r)        # observed ΔF/F₀
  We fit the ratio λ = q/r per trial by maximizing the innovations likelihood,
  then run an RTS backward pass to obtain the smoothed trajectory.
- Everything is *per trial* by design; cross-trial averaging is only for
  display (black thick curves).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from scipy.ndimage import gaussian_filter1d  # optional, used for simple Gaussian smoothing
except Exception:
    gaussian_filter1d = None  # Fallback handled in function

# -----------------------------
# I/O
# -----------------------------
def load_xlsx_trials(path, sheet_index=0):
    df = pd.read_excel(path, sheet_name=sheet_index)
    N = df.shape[1]
    time = pd.to_numeric(df.iloc[:, N-1], errors="coerce").to_numpy(float)
    valid_t = np.isfinite(time)
    time = time[valid_t]

    # All trial columns except the "average" column (N-1) and the time column (N)
    if N >= 2 and (N - 2) > 0:
        data_raw = df.iloc[:, :N-2].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    else:
        # Fallback: if no explicit trial columns, use the average column as a single trial
        # (2nd-to-last column), if available.
        if N >= 2:
            data_raw = df.iloc[:, [N-2]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        else:
            # No usable columns; return empty arrays
            return np.array([]), np.empty((0, 0))
    data_raw = data_raw[valid_t, :]
    return time, data_raw

# Helper: safe row-wise nanmean without warnings
def _row_nanmean(a: np.ndarray):
    if a.size == 0:
        return np.array([])
    counts = np.sum(np.isfinite(a), axis=1)
    sums = np.nansum(a, axis=1)
    out = np.full(a.shape[0], np.nan)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out

# Helper: safe column-wise nanmean without warnings
def _col_nanmean(a: np.ndarray):
    if a.size == 0:
        return np.array([])
    counts = np.sum(np.isfinite(a), axis=0)
    sums = np.nansum(a, axis=0)
    out = np.full(a.shape[1], np.nan)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out

# -----------------------------
# ΔF/F0 per trial
# -----------------------------
def compute_dff(Y_all, time, train_start_s=1.0):
    baseline_mask = time < train_start_s
    # Fallback if no baseline samples exist (e.g., time starts after train_start_s)
    if not np.any(baseline_mask):
        n = max(1, int(0.1 * len(time)))  # use first 10% of samples as baseline
        baseline_mask = np.zeros_like(time, dtype=bool)
        baseline_mask[:n] = True
    F0 = np.nanmean(Y_all[baseline_mask, :], axis=0, keepdims=True)
    dff = (Y_all - F0) / np.maximum(F0, 1e-12)
    return dff

# -----------------------------
# Kalman filter + RTS smoother (local-level model), no SciPy required
# -----------------------------
def kalman_rts_smoother(y, q, r, m0=0.0, P0=1e4):
    """
    y: 1D array (NaNs allowed -> treated as missing obs)
    q: process noise var
    r: obs noise var
    m0, P0: prior mean/variance
    returns: x_smooth (same shape as y)
    """
    T = len(y)
    # Forward pass (Kalman filter)
    m = np.zeros(T)    # filtered mean
    P = np.zeros(T)    # filtered var
    a = np.zeros(T)    # predicted mean
    R = np.zeros(T)    # predicted var
    K = np.zeros(T)    # Kalman gain

    a0, R0 = m0, P0
    for t in range(T):
        # Predict
        a[t] = a0
        R[t] = R0 + q
        if np.isfinite(y[t]):
            # Update
            S = R[t] + r
            K[t] = R[t] / S
            m[t] = a[t] + K[t] * (y[t] - a[t])
            P[t] = (1.0 - K[t]) * R[t]
        else:
            # Missing observation: no update
            m[t] = a[t]
            P[t] = R[t]
        # Set up next step
        a0, R0 = m[t], P[t]

    # Backward pass (RTS smoother)
    x = np.zeros(T)
    x[-1] = m[-1]
    C = np.zeros(T-1)  # smoother gain
    for t in range(T-2, -1, -1):
        denom = P[t] + q
        C[t] = P[t] / denom if denom > 0 else 0.0
        x[t] = m[t] + C[t] * (x[t+1] - m[t])

    return x

def kalman_innovations_loglike(y, q, r, m0=0.0, P0=1e4):
    """
    Compute innovations log-likelihood for local-level model.
    Missing y (NaN) are skipped.
    """
    T = len(y)
    a, R = m0, P0
    ll = 0.0
    two_pi = 2.0 * np.pi
    for t in range(T):
        # Predict
        a_pred, R_pred = a, R + q
        if np.isfinite(y[t]):
            v = y[t] - a_pred          # innovation
            S = R_pred + r             # innovation variance
            ll += -0.5 * (np.log(S) + (v*v)/S + np.log(two_pi))
            # Update
            K = R_pred / S
            a = a_pred + K * v
            R = (1. - K) * R_pred
        else:
            # No update with missing
            a, R = a_pred, R_pred
    return ll

def smooth_trial_auto(y, r_hint=None, lam_grid=np.logspace(-4, 0, 9)):
    """
    Fit λ = q/r by maximizing innovations log-likelihood, then RTS smooth.
    r is set from baseline variance (if provided via r_hint) or y's median variance chunk.
    """
    # If r (obs noise) not provided, estimate from high-SNR-free region via MAD
    if r_hint is None or not np.isfinite(r_hint) or r_hint <= 0:
        # Robust estimate: median absolute deviation of 1st differences
        dy = np.diff(y[np.isfinite(y)])
        if len(dy) == 0:
            r = 1e-2
        else:
            mad = np.median(np.abs(dy - np.median(dy))) + 1e-12
            r = (1.4826 * mad)**2
    else:
        r = float(r_hint)

    # Grid search over λ
    best_ll, best_lam = -np.inf, lam_grid[0]
    for lam in lam_grid:
        q = max(lam * r, 1e-12)
        ll = kalman_innovations_loglike(y, q, r)
        if ll > best_ll:
            best_ll, best_lam = ll, lam

    q_star = max(best_lam * r, 1e-12)
    x_smooth = kalman_rts_smoother(y, q_star, r)
    return x_smooth, q_star, r, best_lam

def smooth_all_trials(dff, baseline_mask, lam_grid=np.logspace(-4, 0, 9)):
    """
    Smooth each trial independently.
    r is estimated from the *baseline* part of each trial to stay event-agnostic.
    """
    T, n_trials = dff.shape
    X = np.zeros_like(dff)
    lam_used = np.zeros(n_trials)
    for j in range(n_trials):
        y = dff[:, j]
        base = y[baseline_mask]
        r_hint = np.var(base[np.isfinite(base)]) if np.any(np.isfinite(base)) else None
        xs, q, r, lam = smooth_trial_auto(y, r_hint=r_hint, lam_grid=lam_grid)
        X[:, j] = xs
        lam_used[j] = lam
    return X, lam_used

# -----------------------------
# Pulse metrics (per trial)
# -----------------------------
def pulse_amplitudes(dff_like, time, stim_times, window_s=0.08):
    """
    Compute per-pulse amplitude as the max ΔF/F0 in [stim, stim+window].
    Returns A with shape (n_trials, n_pulses).
    """
    T, n_trials = dff_like.shape
    n_pulses = len(stim_times)
    A = np.full((n_trials, n_pulses), np.nan)
    for pi, st in enumerate(stim_times):
        t0 = np.searchsorted(time, st, side="left")
        t1 = np.searchsorted(time, st + window_s, side="right")
        t1 = min(t1, T)
        if t1 <= t0:
            continue
        seg = dff_like[t0:t1, :]
        A[:, pi] = np.nanmax(seg, axis=0)
    return A

# -----------------------------
# Event-by-event smoothing (windowed)
# -----------------------------
def smooth_events_kalman(dff, time, stim_times, pre_s=0.02, post_s=0.08,
                         lam_grid=np.logspace(-3, 1, 9)):
    """
    Smooth only short windows around each stimulus per trial using the Kalman–RTS smoother.
    This avoids cross-event bleed and reduces over-smoothing.
    Windows: [stim - pre_s, stim + post_s]. Outside those windows, the data is left unchanged.
    """
    T, n_trials = dff.shape
    X = dff.copy()
    for j in range(n_trials):
        y = dff[:, j]
        for st in stim_times:
            t0 = max(0, np.searchsorted(time, st - pre_s, side="left"))
            t1 = min(T, np.searchsorted(time, st + post_s, side="right"))
            if t1 - t0 < 2:
                continue
            # Estimate r from the pre-event portion when available
            b0 = max(0, np.searchsorted(time, st - pre_s, side="left"))
            b1 = max(b0, np.searchsorted(time, st, side="left"))
            base = y[b0:b1]
            r_hint = np.var(base[np.isfinite(base)]) if np.any(np.isfinite(base)) else None
            xs, _, _, _ = smooth_trial_auto(y[t0:t1], r_hint=r_hint, lam_grid=lam_grid)
            X[t0:t1, j] = xs
    return X

# -----------------------------
# Utility: PPR or train-mean normalization
# -----------------------------
def compute_ppr(amp: np.ndarray, mode: str = "P1", train_mean: bool = False):
    """
    Compute paired-pulse ratio (PPR) across pulses per trial.
    mode = 'P1'      -> PPR[i] = amp[i] / amp[0]
    mode = 'prev'    -> PPR[i] = amp[i] / amp[i-1]
    If train_mean is True, ignore ``mode`` and instead normalize each trial's
    amplitudes by its within-train mean, yielding ``amp / mean(amp)``.
    Returns array with same shape as ``amp``; baseline divisor positions are
    NaN for the classic PPR modes.
    """
    A = amp.astype(float).copy()
    if A.size == 0:
        return A
    if train_mean:
        with np.errstate(invalid='ignore', divide='ignore'):
            denom = np.nanmean(A, axis=1, keepdims=True)
            P = A / denom
        return P
    if mode.lower() == "p1":
        denom = A[:, [0]]
        with np.errstate(invalid='ignore', divide='ignore'):
            P = A / denom
        P[:, 0] = np.nan
    elif mode.lower() == "prev":
        P = np.full_like(A, np.nan)
        with np.errstate(invalid='ignore', divide='ignore'):
            P[:, 1:] = A[:, 1:] / A[:, :-1]
    else:
        raise ValueError("mode must be 'P1' or 'prev'")
    return P

# -----------------------------
# Optional: Gaussian smoothing per trial (SliceTCA-like)
# -----------------------------
def gaussian_smooth_trials(dff: np.ndarray, time: np.ndarray, sigma_s: float = 0.02):
    """
    Smooth each trial along time with a Gaussian filter of temporal sigma sigma_s.
    Falls back to a manual 1D Gaussian conv if scipy is unavailable.
    """
    if dff.size == 0:
        return dff
    dt = np.median(np.diff(time)) if len(time) > 1 else 0.01
    sigma_samples = max(0.0, float(sigma_s) / max(dt, 1e-6))
    if gaussian_filter1d is not None:
        return gaussian_filter1d(dff, sigma=sigma_samples, axis=0, mode="nearest")
    # Fallback: manual conv
    if sigma_samples == 0:
        return dff.copy()
    rad = int(max(3, np.ceil(3*sigma_samples)))
    x = np.arange(-rad, rad+1)
    k = np.exp(-0.5*(x/sigma_samples)**2)
    k /= k.sum()
    Y = np.empty_like(dff)
    for j in range(dff.shape[1]):
        y = dff[:, j]
        pad = np.r_[y[0]*np.ones(rad), y, y[-1]*np.ones(rad)]
        conv = np.convolve(pad, k, mode='valid')
        Y[:, j] = conv
    return Y

# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    # ---- File and stim parameters ----
    xlsx_path = r"C:\Users\Antoine.Valera\Desktop\Befafter\Before\241212_Fibre1_PortionB_bouton4.xlsx"
    train_start_s = 1.0
    isi_s = 0.05
    n_pulses = 10
    stim_times = train_start_s + isi_s * np.arange(n_pulses)
    normalize_to_train_mean = False  # Set True to normalize to train mean instead of pulse 1

    # ---- Load & ΔF/F0 ----
    time, Y_all = load_xlsx_trials(xlsx_path)
    dff = compute_dff(Y_all, time, train_start_s)
    baseline_mask = time < train_start_s

    # ---- Per-trial smoothing (no cross-trial mixing) ----
    # Option A: full-trace smoothing (current default)
    # dff_smooth, lam_used = smooth_all_trials(dff, baseline_mask)

    # Option B: event-by-event smoothing (less smoothing outside events)
    dff_smooth = smooth_events_kalman(dff, time, stim_times,
                                      pre_s=0.02, post_s=0.08,
                                      lam_grid=np.logspace(-3, 1, 9))
    lam_used = None

    # ---- Per-pulse amplitudes (within trials) ----
    amp_raw   = pulse_amplitudes(dff,        time, stim_times, window_s=0.08)
    amp_smooth= pulse_amplitudes(dff_smooth, time, stim_times, window_s=0.08)

    # -----------------------------
    # Plots
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    colors = plt.cm.tab10.colors
    patch_width = isi_s * 0.9

    # --- Top-left: raw ΔF/F0 traces ---
    for i in range(dff.shape[1]):
        axes[0,0].plot(time, dff[:, i], color="0.8", linewidth=0.8)
    mean_raw = _row_nanmean(dff)
    if mean_raw.size:
        axes[0,0].plot(time, mean_raw, color="k", linewidth=2)
    for i, st in enumerate(stim_times):
        c = colors[i % len(colors)]
        axes[0,0].axvline(st, linestyle=":", color=c, linewidth=1)
        axes[0,0].axvspan(st, st + patch_width, color=c, alpha=0.15)
    axes[0,0].set_title("ΔF/F₀ (raw): trials (gray) + mean (black)")
    axes[0,0].set_ylabel("ΔF/F₀")
    axes[0,0].set_xlim(time[0], time[-1])

    # --- Bottom-left: raw heatmap (trials × time) ---
    if dff.size:
        axes[1,0].imshow(dff.T, aspect="auto",
                         extent=[time[0], time[-1], 0, dff.shape[1]],
                         cmap="viridis", interpolation="nearest", origin="lower")
    for i, st in enumerate(stim_times):
        c = colors[i % len(colors)]
        axes[1,0].axvline(st, linestyle=":", color=c, linewidth=1)
        axes[1,0].axvspan(st, st + patch_width, color=c, alpha=0.15)
    axes[1,0].set_title("ΔF/F₀ (raw) heatmap")
    axes[1,0].set_xlabel("Time (s)")
    axes[1,0].set_ylabel("Trials")
    axes[1,0].set_xlim(time[0], time[-1])

    # --- Top-right: *smoothed* ΔF/F₀ traces ---
    for i in range(dff_smooth.shape[1]):
        axes[0,1].plot(time, dff_smooth[:, i], color="0.7", linewidth=0.9)
    mean_smooth = _row_nanmean(dff_smooth)
    if mean_smooth.size:
        axes[0,1].plot(time, mean_smooth, color="k", linewidth=2)
    for i, st in enumerate(stim_times):
        c = colors[i % len(colors)]
        axes[0,1].axvline(st, linestyle=":", color=c, linewidth=1)
        axes[0,1].axvspan(st, st + patch_width, color=c, alpha=0.15)
    axes[0,1].set_title("ΔF/F₀ (smoothed per trial): trials + mean")
    axes[0,1].set_xlim(time[0], time[-1])
    axes[0,1].set_yticks([])

    # --- Bottom-right: trials × pulses summary (lines + PPR/normalized mean) ---
    axes[1,1].cla()
    x_pulses = np.arange(1, n_pulses+1)
    if amp_smooth.size:
        # Plot per-trial amplitudes as lines with markers
        for tr in range(amp_smooth.shape[0]):
            axes[1,1].plot(x_pulses, amp_smooth[tr, :], '-o', color='0.65', alpha=0.7,
                           linewidth=1.0, markersize=3)
        axes[1,1].set_ylabel("ΔF/F₀ amplitude")
        # Compute and overlay PPR average (vs P1) or train-mean normalization on a twin Y axis
        ppr = compute_ppr(amp_smooth, mode='P1', train_mean=normalize_to_train_mean)
        if np.isfinite(ppr).any():
            ppr_mean = _col_nanmean(ppr)
            ax2 = axes[1,1].twinx()
            if normalize_to_train_mean:
                ax2.plot(x_pulses, ppr_mean, '-s', color='k', linewidth=2, markersize=5,
                         label='Mean-normalized amplitude')
                ax2.set_ylabel("Amplitude / train mean")
            else:
                ax2.plot(x_pulses, ppr_mean, '-s', color='k', linewidth=2, markersize=5,
                         label='PPR mean (vs P1)')
                ax2.set_ylabel("PPR (vs P1)")
            ax2.legend(loc='upper right', frameon=False)
        else:
            # No finite PPR available
            pass
    axes[1,1].set_xlabel("Pulse index")
    axes[1,1].set_title("Within-trial amplitudes + " + ("PPR mean" if not normalize_to_train_mean else "mean-normalized amplitude"))
    axes[1,1].set_xticks(x_pulses)
    axes[1,1].set_xticklabels([f"{i}" for i in x_pulses])

    plt.tight_layout()
    plt.show()
