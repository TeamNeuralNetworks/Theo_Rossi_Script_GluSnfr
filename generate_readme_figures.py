"""
Generate illustrative figures for README documentation.

This script processes example datasets and creates visual examples of:
1. Raw data and preprocessing
2. Template extraction and variants
3. NNLS fitting with variants
4. Residual correction for 50Hz data
5. Final amplitude extraction

Usage:
    python generate_readme_figures.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics

# Output directory for figures
OUTPUT_DIR = os.path.join(REPO_ROOT, "docs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Example datasets
EXAMPLE_20HZ = r"C:\Users\Antoine.Valera\Desktop\Testout\Stability_After\241212_Fibre2_PortionA_Bouton_6_bis.png"
EXAMPLE_50HZ = r"C:\Users\Antoine.Valera\Desktop\Testout\Theo_2_5_50Hz\20201022_linescan1_50Hz_10pulses_2.5mMCa_bouton4_traces_converted.png"


def find_xlsx_file(png_path):
    """Find the corresponding .xlsx file for a .png path."""
    # Replace .png with .xlsx
    xlsx_path = png_path.replace('.png', '.xlsx')
    if os.path.exists(xlsx_path):
        return xlsx_path

    # Try looking in parent directories
    base_dir = os.path.dirname(png_path)
    parent_dir = os.path.dirname(base_dir)

    # Check if there's a matching file in PPR_DATA_FINAL structure
    filename = os.path.basename(png_path).replace('.png', '.xlsx')

    # Search pattern: look for the file in nearby directories
    for root, dirs, files in os.walk(parent_dir):
        if filename in files:
            return os.path.join(root, filename)

    return None


def load_data(xlsx_path):
    """Load time and trials data from Excel file."""
    df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    _time = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
    _trials = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    valid = np.isfinite(_time)
    time = _time[valid]
    trials = _trials[valid, :]
    return time, trials


def figure1_raw_data_preprocessing(time, trials, train_start, isi, output_path):
    """Figure 1: Show raw data, bleach correction, and ΔF/F0 normalization."""
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Compute average
    y_avg = np.mean(trials, axis=1)

    # Plot 1: Raw individual trials
    ax1 = fig.add_subplot(gs[0, :])
    for i in range(min(10, trials.shape[1])):
        ax1.plot(time, trials[:, i], alpha=0.3, color='gray', lw=0.5)
    ax1.plot(time, y_avg, 'k-', lw=2, label='Average')
    for i in range(10):
        ax1.axvline(train_start + i * isi, color='r', ls=':', alpha=0.5, lw=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Raw Fluorescence')
    ax1.set_title('Step 1: Raw Data (individual trials + average)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Zoom on baseline
    ax2 = fig.add_subplot(gs[1, 0])
    baseline_mask = time < train_start
    ax2.plot(time[baseline_mask], y_avg[baseline_mask], 'k-', lw=1.5)
    ax2.axhline(np.mean(y_avg[baseline_mask]), color='b', ls='--', label='F0 (baseline mean)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Raw Fluorescence')
    ax2.set_title('Step 2: Baseline Period (F0 calculation)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: After bleach correction (simulated)
    ax3 = fig.add_subplot(gs[1, 1])
    # Simple exponential fit for illustration
    mask_pre = time < train_start
    if np.any(mask_pre):
        f0 = np.mean(y_avg[mask_pre])
        # Simulate bleach-corrected trace
        y_corrected = y_avg / f0
        ax3.plot(time, y_corrected, 'g-', lw=1.5, label='Bleach-corrected ΔF/F0')
        for i in range(10):
            ax3.axvline(train_start + i * isi, color='r', ls=':', alpha=0.5, lw=1)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('ΔF/F0')
        ax3.set_title('Step 3: After Bleach Correction & Normalization')
        ax3.legend()
        ax3.grid(alpha=0.3)

    # Plot 4: Smoothed trace (Savitzky-Golay)
    ax4 = fig.add_subplot(gs[2, :])
    from scipy.signal import savgol_filter
    y_smooth = savgol_filter(y_avg, window_length=9, polyorder=2)
    ax4.plot(time, y_avg, 'gray', alpha=0.5, lw=1, label='Raw average')
    ax4.plot(time, y_smooth, 'b-', lw=2, label='Savitzky-Golay smoothed')
    for i in range(10):
        ax4.axvline(train_start + i * isi, color='r', ls=':', alpha=0.5, lw=1)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Fluorescence')
    ax4.set_title('Step 4: Savitzky-Golay Smoothing (for weights & diagnostics)')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure2_template_extraction(output_path):
    """Figure 2: Illustrate template extraction and variant generation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Simulate iGluSnFR template
    t_ms = np.linspace(-5, 50, 500)
    t_s = t_ms / 1000.0

    # Parameters
    tau_rise = 0.0014  # 1.4 ms
    tau_fast = 0.008   # 8 ms
    tau_slow = 0.025   # 25 ms

    def iglusnfr_template(t, frac_fast):
        """iGluSnFR bi-exponential model."""
        t = np.maximum(t, 0)
        rise = 1 - np.exp(-t / tau_rise)
        decay_fast = np.exp(-t / tau_fast)
        decay_slow = np.exp(-t / tau_slow)
        decay = frac_fast * decay_fast + (1 - frac_fast) * decay_slow
        return rise * decay

    # Plot 1: Single template with components
    ax = axes[0, 0]
    template = iglusnfr_template(t_s, 0.5)
    ax.plot(t_ms, template, 'k-', lw=2, label='Full template (50/50)')
    ax.plot(t_ms, iglusnfr_template(t_s, 1.0), 'r--', lw=1.5, alpha=0.7, label='Fast component only')
    ax.plot(t_ms, iglusnfr_template(t_s, 0.0), 'b--', lw=1.5, alpha=0.7, label='Slow component only')
    ax.axvline(0, color='red', ls=':', alpha=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_title('iGluSnFR Template Components')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Template variants (different fast/slow ratios)
    ax = axes[0, 1]
    frac_fast_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(frac_fast_values)))
    for frac_fast, color in zip(frac_fast_values, colors):
        template = iglusnfr_template(t_s, frac_fast)
        ax.plot(t_ms, template, lw=2, color=color, label=f'{int(frac_fast*100)}% fast')
    ax.axvline(0, color='red', ls=':', alpha=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_title('Template Variants (varying fast/slow ratio)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Jitter variants
    ax = axes[1, 0]
    base_template = iglusnfr_template(t_s, 0.5)
    jitters = [-2, -1, 0, 1, 2]  # ms
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(jitters)))
    for jitter_ms, color in zip(jitters, colors):
        jitter_s = jitter_ms / 1000.0
        shifted_template = iglusnfr_template(t_s - jitter_s, 0.5)
        ax.plot(t_ms, shifted_template, lw=2, color=color, label=f'{jitter_ms:+.0f} ms')
    ax.axvline(0, color='red', ls=':', alpha=0.5, label='Stimulus')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_title('Jitter Variants (temporal shifts)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Grid search illustration
    ax = axes[1, 1]
    n_template = 10
    n_jitter = 9
    ratio_grid, jitter_grid = np.meshgrid(
        np.linspace(0, 1, n_template),
        np.linspace(-1, 1, n_jitter)
    )
    # Simulate "goodness of fit" landscape
    goodness = np.exp(-((ratio_grid - 0.7)**2 / 0.1) - ((jitter_grid - 0.5)**2 / 0.3))
    im = ax.contourf(ratio_grid, jitter_grid, goodness, levels=20, cmap='viridis')
    ax.plot(0.7, 0.5, 'r*', markersize=15, label='Best fit')
    ax.set_xlabel('Slow component fraction')
    ax.set_ylabel('Jitter (ms)')
    ax.set_title('2D Grid Search: Template × Jitter')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Fit quality')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure4_nnls_grid_search(output_path):
    """Figure 4: NNLS as grid search over template candidates."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

    # Simulate iGluSnFR template
    t_ms = np.linspace(-5, 100, 1000)
    t_s = t_ms / 1000.0

    # Parameters
    tau_rise = 0.0014  # 1.4 ms
    tau_fast = 0.008   # 8 ms
    tau_slow = 0.025   # 25 ms

    def iglusnfr_template(t, frac_fast):
        """iGluSnFR bi-exponential model."""
        t = np.maximum(t, 0)
        rise = 1 - np.exp(-t / tau_rise)
        decay_fast = np.exp(-t / tau_fast)
        decay_slow = np.exp(-t / tau_slow)
        decay = frac_fast * decay_fast + (1 - frac_fast) * decay_slow
        return rise * decay

    # Simulate 3-pulse data with overlapping responses
    stim_times = [0, 20, 40]  # ms
    true_amps = [1.0, 0.8, 0.6]
    true_ratio = 0.7

    # Generate synthetic data
    y_true = np.zeros_like(t_ms)
    for st, amp in zip(stim_times, true_amps):
        y_true += amp * iglusnfr_template((t_ms - st) / 1000.0, true_ratio)

    # Add noise
    np.random.seed(42)
    y_data = y_true + np.random.normal(0, 0.02, len(y_true))

    # Plot 1: Simulated overlapping data
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_ms, y_data, 'k-', lw=1.5, alpha=0.7, label='Measured data (with noise)')
    ax1.plot(t_ms, y_true, 'r--', lw=2, alpha=0.8, label='True signal')
    for st in stim_times:
        ax1.axvline(st, color='blue', ls=':', alpha=0.5, lw=1.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Fluorescence (a.u.)')
    ax1.set_title('Problem: Extract individual event amplitudes from overlapping responses')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(-5, 100)

    # Test different template ratios
    test_ratios = [0.3, 0.5, 0.7, 0.9]

    # Plot 2-5: Show fits with different template variants
    for idx, ratio in enumerate(test_ratios):
        ax = fig.add_subplot(gs[1, idx % 3]) if idx < 3 else fig.add_subplot(gs[2, 0])

        # Build kernel matrix for this ratio
        kernel = np.zeros((len(t_ms), len(stim_times)))
        for i, st in enumerate(stim_times):
            kernel[:, i] = iglusnfr_template((t_ms - st) / 1000.0, ratio)

        # NNLS fit
        from scipy.optimize import nnls
        amps_fitted, residual = nnls(kernel, y_data)
        y_fit = kernel @ amps_fitted

        # Plot
        ax.plot(t_ms, y_data, 'gray', alpha=0.4, lw=1, label='Data')
        ax.plot(t_ms, y_fit, 'b-', lw=2, label=f'NNLS fit (ratio={ratio:.1f})')
        for st in stim_times:
            ax.axvline(st, color='blue', ls=':', alpha=0.3, lw=1)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Fluorescence')
        ax.set_title(f'Template ratio={ratio:.1f}, Residual={residual:.4f}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(-5, 100)

        # Show extracted amplitudes
        amp_text = f"A=[{amps_fitted[0]:.2f}, {amps_fitted[1]:.2f}, {amps_fitted[2]:.2f}]"
        ax.text(0.02, 0.98, amp_text, transform=ax.transAxes,
                fontsize=7, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 6: Residual comparison across all ratios
    ax_residual = fig.add_subplot(gs[2, 1:])

    ratio_range = np.linspace(0, 1, 50)
    residuals = []

    for ratio in ratio_range:
        kernel = np.zeros((len(t_ms), len(stim_times)))
        for i, st in enumerate(stim_times):
            kernel[:, i] = iglusnfr_template((t_ms - st) / 1000.0, ratio)
        amps_fitted, residual = nnls(kernel, y_data)
        residuals.append(residual)

    ax_residual.plot(ratio_range, residuals, 'b-', lw=2)
    ax_residual.axvline(true_ratio, color='r', ls='--', lw=2, label=f'True ratio ({true_ratio})')
    best_idx = np.argmin(residuals)
    best_ratio = ratio_range[best_idx]
    ax_residual.plot(best_ratio, residuals[best_idx], 'g*', markersize=15,
                    label=f'Best fit ({best_ratio:.2f})')
    ax_residual.set_xlabel('Template ratio (fraction fast component)')
    ax_residual.set_ylabel('NNLS Residual')
    ax_residual.set_title('Grid Search: Find template variant with lowest residual')
    ax_residual.legend()
    ax_residual.grid(alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure3_parameter_bounds(output_path):
    """Figure 3: Illustrate parameter bounds system."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Example 1: Fixed value
    ax = axes[0, 0]
    fitted_values = [0.005, 0.007, 0.008, 0.009, 0.011]
    constrained_values = [0.008] * len(fitted_values)
    x = np.arange(len(fitted_values))
    ax.bar(x - 0.2, fitted_values, width=0.4, label='Raw fit', alpha=0.7, color='blue')
    ax.bar(x + 0.2, constrained_values, width=0.4, label='After bounds', alpha=0.7, color='green')
    ax.axhline(0.008, color='red', ls='--', lw=2, label='Bound: (0.008, 0.008)')
    ax.set_ylabel('tau_decay_fast (s)')
    ax.set_title('Fixed Value: (lower == upper)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Event {i+1}' for i in x])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Example 2: Range constraint
    ax = axes[0, 1]
    fitted_values = [0.015, 0.022, 0.028, 0.035, 0.045]
    constrained_values = [max(0.025, min(0.030, v)) for v in fitted_values]
    x = np.arange(len(fitted_values))
    ax.bar(x - 0.2, fitted_values, width=0.4, label='Raw fit', alpha=0.7, color='blue')
    ax.bar(x + 0.2, constrained_values, width=0.4, label='After bounds', alpha=0.7, color='green')
    ax.axhline(0.025, color='red', ls='--', lw=1.5, label='Lower: 0.025')
    ax.axhline(0.030, color='orange', ls='--', lw=1.5, label='Upper: 0.030')
    ax.set_ylabel('tau_decay_slow (s)')
    ax.set_title('Range Constraint: (0.025, 0.030)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Event {i+1}' for i in x])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Example 3: Unconstrained
    ax = axes[1, 0]
    fitted_values = [0.0008, 0.0012, 0.0015, 0.0018, 0.0022]
    x = np.arange(len(fitted_values))
    ax.bar(x, fitted_values, width=0.6, label='Fit values', alpha=0.7, color='blue')
    ax.set_ylabel('tau_rise (s)')
    ax.set_title('Unconstrained: (np.nan, np.nan)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Event {i+1}' for i in x])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Example 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [
        ['Parameter', 'Lower', 'Upper', 'Meaning'],
        ['tau_decay_fast', '0.008', '0.008', 'FIXED at 8ms'],
        ['tau_decay_slow', '0.025', '0.030', 'RANGE 25-30ms'],
        ['tau_rise', 'NaN', 'NaN', 'UNCONSTRAINED'],
        ['frac_fast', '0.0', '1.0', 'RANGE 0-100%'],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                     colWidths=[0.3, 0.15, 0.15, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    # Header row styling
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    ax.set_title('Parameter Bounds Configuration', fontsize=12, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure5_residual_correction(output_path):
    """Figure 5: Residual correction for 50Hz high-frequency data."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Simulate iGluSnFR template
    t_ms = np.linspace(-5, 120, 1000)
    t_s = t_ms / 1000.0

    # Parameters
    tau_rise = 0.0014
    tau_fast = 0.008
    tau_slow = 0.028

    def iglusnfr_template(t, frac_fast=0.5):
        """iGluSnFR bi-exponential model."""
        t = np.maximum(t, 0)
        rise = 1 - np.exp(-t / tau_rise)
        decay_fast = np.exp(-t / tau_fast)
        decay_slow = np.exp(-t / tau_slow)
        decay = frac_fast * decay_fast + (1 - frac_fast) * decay_slow
        return rise * decay

    # 50Hz = 20ms ISI
    stim_times = [0, 20, 40, 60, 80]  # ms
    true_amps = [1.0, 0.7, 0.5, 0.4, 0.35]

    # Generate synthetic data
    y_true = np.zeros_like(t_ms)
    for st, amp in zip(stim_times, true_amps):
        y_true += amp * iglusnfr_template((t_ms - st) / 1000.0, 0.6)

    # Add noise
    np.random.seed(42)
    y_data = y_true + np.random.normal(0, 0.015, len(y_true))

    # NNLS fit WITHOUT residual correction
    from scipy.optimize import nnls
    kernel = np.zeros((len(t_ms), len(stim_times)))
    for i, st in enumerate(stim_times):
        kernel[:, i] = iglusnfr_template((t_ms - st) / 1000.0, 0.6)

    amps_uncorr, _ = nnls(kernel, y_data)
    y_fit_uncorr = kernel @ amps_uncorr
    residual_uncorr = y_data - y_fit_uncorr

    # Plot 1: Before correction - show residuals contaminating next events
    ax = axes[0]
    ax.plot(t_ms, y_data, 'gray', alpha=0.5, lw=1, label='Data')
    ax.plot(t_ms, y_fit_uncorr, 'b-', lw=2, label='NNLS fit (no correction)')
    ax.plot(t_ms, residual_uncorr, 'r--', lw=1.5, alpha=0.7, label='Residuals')

    # Show individual event decay tails as dotted lines
    for i, (st, amp) in enumerate(zip(stim_times, amps_uncorr)):
        # Individual event contribution (decay tail)
        event_kernel = iglusnfr_template((t_ms - st) / 1000.0, 0.6)
        event_trace = amp * event_kernel
        # Only show decay tail (after peak)
        peak_idx = np.argmax(event_kernel)
        t_peak = st + t_ms[peak_idx]
        mask_decay = t_ms >= t_peak
        ax.plot(t_ms[mask_decay], event_trace[mask_decay], ':',
               color=f'C{i}', lw=1.5, alpha=0.6,
               label=f'Event {i+1} decay tail' if i < 2 else '')

    # Show residuals under next event
    for i in range(len(stim_times) - 1):
        st = stim_times[i]
        next_st = stim_times[i + 1]
        # Highlight residual region before next event
        mask = (t_ms >= st + 5) & (t_ms < next_st)
        if np.any(mask):
            ax.fill_between(t_ms[mask], 0, residual_uncorr[mask],
                          color='orange', alpha=0.3, label='Residual under next event' if i == 0 else '')

    for st in stim_times:
        ax.axvline(st, color='blue', ls=':', alpha=0.4, lw=1)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Fluorescence (a.u.)')
    ax.set_title('Before Correction: Residuals contaminate subsequent events (50Hz overlap)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-5, 120)

    # Annotate residual issue
    ax.annotate('Positive residual here...', xy=(15, residual_uncorr[np.argmin(np.abs(t_ms - 15))]),
               xytext=(15, 0.15), fontsize=9, ha='center',
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.annotate('...inflates amplitude here', xy=(20, y_data[np.argmin(np.abs(t_ms - 20))]),
               xytext=(30, 0.8), fontsize=9, ha='left',
               arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))

    # Simulate residual correction
    # Simplified: adjust each amplitude by subtracting residual contribution
    amps_corr = amps_uncorr.copy()
    for i in range(1, len(stim_times)):
        # Residual from previous events at this event's onset
        t_onset = stim_times[i] / 1000.0
        idx_onset = np.argmin(np.abs(t_s - t_onset))
        residual_at_onset = residual_uncorr[idx_onset]
        # Correct amplitude (simplified - actual algorithm is iterative)
        amps_corr[i] = max(0, amps_corr[i] - residual_at_onset * 0.7)

    y_fit_corr = kernel @ amps_corr
    residual_corr = y_data - y_fit_corr

    # Plot 2: After correction
    ax = axes[1]
    ax.plot(t_ms, y_data, 'gray', alpha=0.5, lw=1, label='Data')
    ax.plot(t_ms, y_fit_corr, 'g-', lw=2, label='NNLS fit (with correction)')
    ax.plot(t_ms, residual_corr, 'r--', lw=1.5, alpha=0.7, label='Residuals (reduced)')

    for st in stim_times:
        ax.axvline(st, color='blue', ls=':', alpha=0.4, lw=1)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Fluorescence (a.u.)')
    ax.set_title('After Correction: Residuals accounted for, cleaner amplitude estimates')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-5, 120)

    # Show amplitude comparison
    amp_text_uncorr = f"Uncorrected A: [{', '.join([f'{a:.2f}' for a in amps_uncorr])}]"
    amp_text_corr = f"Corrected A: [{', '.join([f'{a:.2f}' for a in amps_corr])}]"
    amp_text_true = f"True A: [{', '.join([f'{a:.2f}' for a in true_amps])}]"

    ax.text(0.02, 0.15, amp_text_uncorr + '\n' + amp_text_corr + '\n' + amp_text_true,
           transform=ax.transAxes, fontsize=8, va='bottom', ha='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure6_recut_oversampling(output_path):
    """Figure 6: Recut averaging and oversampling."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Simulate low sampling rate data
    np.random.seed(123)

    # True underlying smooth response (1000 Hz)
    t_fine = np.linspace(0, 50, 500)  # ms, high resolution
    tau_rise = 1.4
    tau_decay = 15
    true_response = (1 - np.exp(-t_fine / tau_rise)) * np.exp(-t_fine / tau_decay)

    # Low sampling rate acquisition (50 Hz = 20ms per sample)
    t_coarse = np.arange(0, 50, 2.0)  # Sample every 2ms (500 Hz - typical imaging)
    measured = np.interp(t_coarse, t_fine, true_response)
    measured += np.random.normal(0, 0.02, len(measured))

    # Plot 1: Low sampling rate problem
    ax = axes[0, 0]
    ax.plot(t_fine, true_response, 'r--', lw=2, alpha=0.7, label='True response')
    ax.plot(t_coarse, measured, 'ko-', lw=1, markersize=4, label='Measured (low sampling)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('ΔF/F0')
    ax.set_title('Problem: Low sampling rate misses peak')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axvline(t_fine[np.argmax(true_response)], color='r', ls=':', alpha=0.5, label='True peak')

    # Plot 2: Recut alignment from multiple trials
    ax = axes[0, 1]
    n_trials = 5
    colors = plt.cm.viridis(np.linspace(0, 1, n_trials))

    # Simulate trials with jitter
    recut_data = []
    for i in range(n_trials):
        jitter = np.random.uniform(-1, 1)  # ms
        trial = np.interp(t_coarse, t_fine - jitter, true_response)
        trial += np.random.normal(0, 0.02, len(trial))
        recut_data.append(trial)
        ax.plot(t_coarse, trial, 'o-', color=colors[i], alpha=0.5, lw=1, markersize=3, label=f'Trial {i+1}')

    # Average
    avg_recut = np.mean(recut_data, axis=0)
    ax.plot(t_coarse, avg_recut, 'k-', lw=2.5, marker='s', markersize=5, label='Average (recut)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('ΔF/F0')
    ax.set_title('Recut: Align and average multiple trials')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Plot 3: Oversampling via interpolation
    ax = axes[1, 0]

    # Original coarse samples
    ax.plot(t_coarse, avg_recut, 'ko-', lw=1, markersize=6, label='Original samples', zorder=3)

    # Oversampled (20x interpolation)
    from scipy.interpolate import interp1d
    interp_func = interp1d(t_coarse, avg_recut, kind='cubic')
    t_oversampled = np.linspace(t_coarse[0], t_coarse[-1], len(t_coarse) * 20)
    y_oversampled = interp_func(t_oversampled)

    ax.plot(t_oversampled, y_oversampled, 'b-', lw=2, alpha=0.7, label='Oversampled (20x)', zorder=2)
    ax.plot(t_fine, true_response, 'r--', lw=1.5, alpha=0.5, label='True response', zorder=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('ΔF/F0')
    ax.set_title('Oversampling: Interpolate to recover peak timing')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Why area matters (not just peak)
    ax = axes[1, 1]

    # Two responses with different sampling alignment
    shift1 = 0.0  # ms
    shift2 = 1.0  # ms (peak between samples)

    response1 = np.interp(t_coarse, t_fine - shift1, true_response)
    response2 = np.interp(t_coarse, t_fine - shift2, true_response)

    ax.plot(t_fine, true_response, 'r--', lw=2, alpha=0.7, label='True response')
    ax.plot(t_coarse, response1, 'bo-', lw=1.5, markersize=5, label='Sampling aligned with peak')
    ax.plot(t_coarse, response2, 'go-', lw=1.5, markersize=5, label='Sampling misses peak')

    # Show area under curve (trapz integration)
    area_true = np.trapz(true_response, t_fine)
    area1 = np.trapz(response1, t_coarse)
    area2 = np.trapz(response2, t_coarse)

    peak_true = np.max(true_response)
    peak1 = np.max(response1)
    peak2 = np.max(response2)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('ΔF/F0')
    ax.set_title('NNLS uses area (integral), not peak amplitude')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Add text box with area vs peak comparison
    text_str = (f"Peak amplitude:\n"
                f"  True: {peak_true:.3f}\n"
                f"  Aligned: {peak1:.3f} (error: {100*(peak1-peak_true)/peak_true:+.1f}%)\n"
                f"  Misaligned: {peak2:.3f} (error: {100*(peak2-peak_true)/peak_true:+.1f}%)\n\n"
                f"Area (integral):\n"
                f"  True: {area_true:.2f}\n"
                f"  Aligned: {area1:.2f} (error: {100*(area1-area_true)/area_true:+.1f}%)\n"
                f"  Misaligned: {area2:.2f} (error: {100*(area2-area_true)/area_true:+.1f}%)")

    ax.text(0.98, 0.97, text_str, transform=ax.transAxes,
           fontsize=7, va='top', ha='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure7_baseline_estimation(output_path):
    """Figure 7: Baseline estimation and F0 calculation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Simulate trace with photobleaching
    t = np.linspace(0, 10, 1000)  # seconds

    # True F0 + photobleaching + stimulation responses
    f0_base = 1000
    bleach = f0_base * np.exp(-t / 5.0)  # Exponential bleaching

    # Add responses (3 stimulations)
    stim_times = [2.0, 4.5, 7.0]
    responses = np.zeros_like(t)
    for st in stim_times:
        mask = t >= st
        responses[mask] += 200 * np.exp(-(t[mask] - st) / 0.3)

    # Total signal
    raw_signal = bleach + responses
    raw_signal += np.random.normal(0, 5, len(raw_signal))  # Add noise

    # Plot 1: Raw trace with photobleaching
    ax = axes[0, 0]
    ax.plot(t, raw_signal, 'k-', lw=1, alpha=0.7, label='Raw fluorescence')
    ax.plot(t, bleach, 'r--', lw=2, label='True baseline (bleaching)')
    for st in stim_times:
        ax.axvline(st, color='blue', ls=':', alpha=0.5, lw=1.5)

    # Show F0 window
    f0_window_end = stim_times[0] - 0.2
    f0_mask = t < f0_window_end
    ax.axvspan(0, f0_window_end, alpha=0.2, color='green', label='F0 window')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Fluorescence (a.u.)')
    ax.set_title('Raw trace: Photobleaching + responses')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: F0 estimation
    ax = axes[0, 1]
    f0_data = raw_signal[f0_mask]
    f0_time = t[f0_mask]

    ax.plot(f0_time, f0_data, 'k-', lw=1.5, alpha=0.7, label='Baseline window data')
    f0_mean = np.mean(f0_data)
    ax.axhline(f0_mean, color='g', ls='--', lw=2, label=f'F0 = {f0_mean:.1f} (mean)')
    ax.fill_between(f0_time, f0_mean - np.std(f0_data), f0_mean + np.std(f0_data),
                    alpha=0.3, color='green', label='±1 SD')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Fluorescence (a.u.)')
    ax.set_title('F0 estimation: Mean of pre-stimulus baseline')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Bleach correction
    ax = axes[1, 0]

    # Fit exponential to entire trace (robust to spikes)
    from scipy.optimize import curve_fit
    def exp_bleach(t, a, tau):
        return a * np.exp(-t / tau)

    # Use Huber weights to downweight spikes
    weights = np.ones_like(raw_signal)
    for st in stim_times:
        stim_mask = (t >= st) & (t <= st + 1.0)
        weights[stim_mask] = 0.1  # Downweight stimulus periods

    try:
        popt, _ = curve_fit(exp_bleach, t, raw_signal, p0=[f0_base, 5.0], sigma=1/weights)
        bleach_fit = exp_bleach(t, *popt)
    except:
        bleach_fit = bleach  # Use true if fit fails

    ax.plot(t, raw_signal, 'k-', lw=1, alpha=0.5, label='Raw')
    ax.plot(t, bleach, 'r--', lw=1.5, alpha=0.7, label='True bleaching')
    ax.plot(t, bleach_fit, 'orange', lw=2, label='Fitted bleaching')
    for st in stim_times:
        ax.axvline(st, color='blue', ls=':', alpha=0.5, lw=1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Fluorescence (a.u.)')
    ax.set_title('Bleach correction: Fit exponential baseline')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: ΔF/F0 normalized trace
    ax = axes[1, 1]

    # Correct for bleaching and normalize
    corrected = raw_signal - bleach_fit + f0_mean
    dff = (corrected - f0_mean) / f0_mean

    ax.plot(t, dff, 'b-', lw=1.5, label='ΔF/F0 (corrected)')
    ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
    for st in stim_times:
        ax.axvline(st, color='blue', ls=':', alpha=0.5, lw=1.5)

    # Show baseline is now flat
    baseline_region = (t < stim_times[0] - 0.2) | ((t > stim_times[0] + 1.5) & (t < stim_times[1] - 0.2))
    ax.plot(t[baseline_region], dff[baseline_region], 'go', markersize=2, alpha=0.5, label='Baseline ≈ 0')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ΔF/F0')
    ax.set_title('Final trace: Bleach-corrected and normalized')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure8_event_subtracted_residuals(output_path):
    """Figure 8: Event-subtracted residuals for goodness of fit assessment."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Simulate iGluSnFR data
    t_ms = np.linspace(-5, 100, 1000)
    t_s = t_ms / 1000.0

    tau_rise = 0.0014
    tau_fast = 0.008
    tau_slow = 0.028

    def iglusnfr_template(t, frac_fast=0.5):
        """iGluSnFR bi-exponential model."""
        t = np.maximum(t, 0)
        rise = 1 - np.exp(-t / tau_rise)
        decay_fast = np.exp(-t / tau_fast)
        decay_slow = np.exp(-t / tau_slow)
        decay = frac_fast * decay_fast + (1 - frac_fast) * decay_slow
        return rise * decay

    # 3 events
    stim_times = [0, 30, 60]  # ms
    true_amps = [1.0, 0.75, 0.6]

    # Generate synthetic data
    y_true = np.zeros_like(t_ms)
    for st, amp in zip(stim_times, true_amps):
        y_true += amp * iglusnfr_template((t_ms - st) / 1000.0, 0.6)

    np.random.seed(42)
    y_data = y_true + np.random.normal(0, 0.02, len(y_true))

    # NNLS fit
    from scipy.optimize import nnls
    kernel = np.zeros((len(t_ms), len(stim_times)))
    for i, st in enumerate(stim_times):
        kernel[:, i] = iglusnfr_template((t_ms - st) / 1000.0, 0.6)

    amps_fitted, _ = nnls(kernel, y_data)
    y_fit = kernel @ amps_fitted
    residual_global = y_data - y_fit

    # Plot 1: Full fit (spans first row)
    ax = plt.subplot2grid((2, 3), (0, 0), colspan=3, fig=fig)
    ax.plot(t_ms, y_data, 'k-', lw=1.5, alpha=0.7, label='Data')
    ax.plot(t_ms, y_fit, 'b-', lw=2, label='NNLS fit')
    ax.plot(t_ms, residual_global, 'r--', lw=1, alpha=0.7, label='Global residual')
    for st in stim_times:
        ax.axvline(st, color='blue', ls=':', alpha=0.4, lw=1)
    ax.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Fluorescence (a.u.)')
    ax.set_title('Global Fit: All events together')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 2-4: Event-subtracted residuals for each event
    for idx, (event_idx, st) in enumerate(zip([0, 1, 2], stim_times)):
        ax = axes[1, idx]

        # Subtract THIS event's contribution from the fit
        event_contribution = amps_fitted[event_idx] * kernel[:, event_idx]
        fit_without_event = y_fit - event_contribution

        # Event-subtracted residual: what remains when we remove this event
        event_subtracted_residual = y_data - fit_without_event

        # Zoom window around this event
        window_start = st - 5
        window_end = st + 40
        mask_window = (t_ms >= window_start) & (t_ms <= window_end)

        # Plot
        ax.plot(t_ms[mask_window], y_data[mask_window], 'k-', lw=1.5, alpha=0.7, label='Data')
        ax.plot(t_ms[mask_window], fit_without_event[mask_window], 'orange', lw=1.5, alpha=0.7,
               label='Fit (other events)')
        ax.plot(t_ms[mask_window], event_subtracted_residual[mask_window], 'g-', lw=2,
               label=f'Event {event_idx+1} isolated')
        ax.plot(t_ms[mask_window], event_contribution[mask_window], 'b:', lw=2,
               label=f'Event {event_idx+1} fitted')

        ax.axvline(st, color='blue', ls=':', alpha=0.5, lw=1.5)
        ax.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.5)

        # Calculate goodness of fit for this event
        # Residual after subtracting fitted event
        final_residual = event_subtracted_residual - event_contribution
        rms_residual = np.sqrt(np.mean(final_residual[mask_window]**2))

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Fluorescence (a.u.)')
        ax.set_title(f'Event {event_idx+1}: Subtracted residual (RMS={rms_residual:.4f})')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)

        # Add text explanation
        text_str = ("Event-subtracted residual:\n"
                   "1. Remove other events (orange)\n"
                   "2. Isolate THIS event (green)\n"
                   "3. Compare to template (blue)\n"
                   f"4. RMS = {rms_residual:.4f}")
        ax.text(0.02, 0.35, text_str, transform=ax.transAxes,
               fontsize=7, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure9_null_condition(output_path):
    """Figure 9: Null condition - moving window NNLS on noise + distribution."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Generate synthetic baseline noise
    np.random.seed(42)
    t_s = np.linspace(0, 1.0, 1000)  # 1 second of baseline @ 1000 Hz
    baseline_noise = np.random.normal(0, 0.02, len(t_s))  # Gaussian noise
    # Add some non-Gaussian structure (photobleaching step)
    baseline_noise[t_s > 0.5] += -0.01
    # Add small drift
    baseline_noise += 0.005 * np.sin(2 * np.pi * t_s * 3)

    # iGluSnFR template
    tau_rise = 0.0014
    tau_fast = 0.008
    tau_slow = 0.028

    def iglusnfr_template(t, frac_fast=0.5):
        """iGluSnFR bi-exponential model."""
        t = np.maximum(t, 0)
        rise = 1 - np.exp(-t / tau_rise)
        decay_fast = np.exp(-t / tau_fast)
        decay_slow = np.exp(-t / tau_slow)
        decay = frac_fast * decay_fast + (1 - frac_fast) * decay_slow
        return rise * decay

    # Moving window parameters
    isi = 0.05  # 20 Hz
    n_events = 5
    window_step = 0.01  # 10 ms steps

    # Compute null distribution
    from scipy.optimize import nnls
    null_amplitudes = []
    window_offsets = []

    for offset in np.arange(0, isi, window_step):
        # Generate fake stimulus times
        fake_stim_times = offset + np.arange(n_events) * isi
        # Only use if all times fit in baseline
        if fake_stim_times[-1] < t_s[-1] - 0.1:
            # Build kernel
            kernel = np.zeros((len(t_s), n_events))
            for i, st in enumerate(fake_stim_times):
                kernel[:, i] = iglusnfr_template(t_s - st, 0.6)

            # NNLS fit
            amps, _ = nnls(kernel, baseline_noise)
            null_amplitudes.extend(amps)
            window_offsets.append(offset)

    null_amplitudes = np.array(null_amplitudes)

    # Calculate threshold
    threshold_95 = np.percentile(null_amplitudes, 95)
    threshold_mean_2sd = np.mean(null_amplitudes) + 2 * np.std(null_amplitudes)

    # TOP PANEL: Pre-stimulus baseline with moving windows
    ax = axes[0]
    ax.plot(t_s, baseline_noise, 'k-', lw=1, alpha=0.7, label='Pre-stimulus baseline (noise)')

    # Show a few example fake stimulus positions
    colors_example = ['red', 'blue', 'green']
    for i, offset in enumerate(window_offsets[::2][:3]):
        fake_stim_times = offset + np.arange(n_events) * isi
        for st in fake_stim_times:
            ax.axvline(st, color=colors_example[i], ls=':', alpha=0.5, lw=1.5)
        # Label the first one
        if i == 0:
            ax.text(fake_stim_times[0], ax.get_ylim()[1] * 0.8, f'Window {i+1}\n(offset={offset:.3f}s)',
                   color=colors_example[i], fontsize=8, ha='left')

    ax.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ΔF/F0')
    ax.set_title('Moving Window Analysis: Fake stimulus times slide through pre-stimulus noise')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Add annotation
    text_str = ("Sliding window approach:\n"
                f"  • ISI = {isi*1000:.0f} ms (20 Hz)\n"
                f"  • {n_events} fake events per window\n"
                f"  • Step = {window_step*1000:.0f} ms\n"
                f"  • Total windows = {len(window_offsets)}")
    ax.text(0.98, 0.05, text_str, transform=ax.transAxes,
           fontsize=8, va='bottom', ha='right',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # MIDDLE PANEL: Example NNLS fits to noise
    ax = axes[1]

    # Show one example window
    offset_demo = window_offsets[0]
    fake_stim_times_demo = offset_demo + np.arange(n_events) * isi
    kernel_demo = np.zeros((len(t_s), n_events))
    for i, st in enumerate(fake_stim_times_demo):
        kernel_demo[:, i] = iglusnfr_template(t_s - st, 0.6)

    amps_demo, _ = nnls(kernel_demo, baseline_noise)
    fit_demo = kernel_demo @ amps_demo

    ax.plot(t_s, baseline_noise, 'k-', lw=1.5, alpha=0.7, label='Noise')
    ax.plot(t_s, fit_demo, 'r-', lw=2, alpha=0.8, label='NNLS fit to noise')
    for i, st in enumerate(fake_stim_times_demo):
        ax.axvline(st, color='blue', ls=':', alpha=0.5, lw=1)
        # Annotate amplitude
        ax.text(st, 0.03, f'A{i+1}={amps_demo[i]:.4f}',
               fontsize=8, rotation=90, va='bottom', ha='right', color='red')

    ax.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ΔF/F0')
    ax.set_title('NNLS Fit to Pure Noise (example window): Small but non-zero amplitudes')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(offset_demo - 0.05, fake_stim_times_demo[-1] + 0.15)

    # Add explanation
    text_str = ("NNLS fits noise fluctuations:\n"
                f"  • Mean amplitude: {np.mean(amps_demo):.4f}\n"
                f"  • Max amplitude: {np.max(amps_demo):.4f}\n"
                f"  • These are FALSE positives\n"
                f"  • Real events must exceed\n"
                f"    the null distribution")
    ax.text(0.98, 0.95, text_str, transform=ax.transAxes,
           fontsize=8, va='top', ha='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # BOTTOM PANEL: Null distribution histogram
    ax = axes[2]

    # Histogram
    ax.hist(null_amplitudes, bins=50, alpha=0.7, color='gray', edgecolor='black',
           label=f'Null distribution (n={len(null_amplitudes)} amplitudes)')

    # Threshold lines
    ax.axvline(threshold_95, color='red', ls='--', lw=2,
              label=f'95th percentile = {threshold_95:.4f}')
    ax.axvline(threshold_mean_2sd, color='orange', ls='--', lw=2,
              label=f'Mean + 2×SD = {threshold_mean_2sd:.4f}')

    # Add example real amplitudes
    real_amps = [0.15, 0.03, 0.02]  # Example from README
    colors_real = ['green', 'orange', 'red']
    labels_real = ['Significant (A1=0.15)', 'Marginal (A1=0.03)', 'Not significant (A1=0.02)']
    for amp, col, lab in zip(real_amps, colors_real, labels_real):
        ax.axvline(amp, color=col, ls='-', lw=2.5, alpha=0.8, label=lab)

    ax.set_xlabel('Fitted Amplitude (ΔF/F0)')
    ax.set_ylabel('Count')
    ax.set_title('Null Distribution: Amplitudes fitted to pure noise across all windows')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3, axis='y')

    # Add statistics
    text_str = (f"Null distribution statistics:\n"
                f"  • Mean: {np.mean(null_amplitudes):.4f}\n"
                f"  • SD: {np.std(null_amplitudes):.4f}\n"
                f"  • 95th percentile: {threshold_95:.4f}\n"
                f"  • Mean + 2×SD: {threshold_mean_2sd:.4f}\n"
                f"\n"
                f"Use threshold to test real events:\n"
                f"  • A > threshold → significant\n"
                f"  • A < threshold → likely noise")
    ax.text(0.98, 0.95, text_str, transform=ax.transAxes,
           fontsize=8, va='top', ha='right',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all README figures."""
    print("Generating README figures...")

    # Figure 2: Template extraction (no data needed)
    print("\n=== Figure 2: Template Variants ===")
    figure2_template_extraction(os.path.join(OUTPUT_DIR, "fig2_template_variants.png"))

    # Figure 3: Parameter bounds (no data needed)
    print("\n=== Figure 3: Parameter Bounds ===")
    figure3_parameter_bounds(os.path.join(OUTPUT_DIR, "fig3_parameter_bounds.png"))

    # Figure 4: NNLS grid search (no data needed)
    print("\n=== Figure 4: NNLS Grid Search ===")
    figure4_nnls_grid_search(os.path.join(OUTPUT_DIR, "fig4_nnls_grid_search.png"))

    # Figure 5: Residual correction
    print("\n=== Figure 5: Residual Correction ===")
    figure5_residual_correction(os.path.join(OUTPUT_DIR, "fig5_residual_correction.png"))

    # Figure 6: Recut and oversampling
    print("\n=== Figure 6: Recut and Oversampling ===")
    figure6_recut_oversampling(os.path.join(OUTPUT_DIR, "fig6_recut_oversampling.png"))

    # Figure 7: Baseline estimation
    print("\n=== Figure 7: Baseline Estimation ===")
    figure7_baseline_estimation(os.path.join(OUTPUT_DIR, "fig7_baseline_estimation.png"))

    # Figure 8: Event-subtracted residuals
    print("\n=== Figure 8: Event-Subtracted Residuals ===")
    figure8_event_subtracted_residuals(os.path.join(OUTPUT_DIR, "fig8_event_subtracted_residuals.png"))

    # Figure 9: Null condition
    print("\n=== Figure 9: Null Condition ===")
    figure9_null_condition(os.path.join(OUTPUT_DIR, "fig9_null_condition.png"))

    # Try to generate data-dependent figures if files exist
    print("\n=== Looking for data files ===")

    # 20Hz example
    xlsx_20hz = find_xlsx_file(EXAMPLE_20HZ)
    if xlsx_20hz and os.path.exists(xlsx_20hz):
        print(f"Found 20Hz data: {xlsx_20hz}")
        time, trials = load_data(xlsx_20hz)
        print("\n=== Figure 1: Preprocessing (20Hz) ===")
        figure1_raw_data_preprocessing(
            time, trials,
            train_start=0.998,
            isi=0.05,
            output_path=os.path.join(OUTPUT_DIR, "fig1_preprocessing_20hz.png")
        )
    else:
        print(f"Could not find 20Hz data file (looked for {EXAMPLE_20HZ})")

    print("\n[OK] Figure generation complete!")
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
