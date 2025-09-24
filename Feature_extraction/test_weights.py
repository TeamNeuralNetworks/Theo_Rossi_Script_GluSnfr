#!/usr/bin/env python3
"""
Test script for the new NNLS weight functionality.
Creates double-exponential synthetic data (8 ms & 50 ms decays) and tests different weight modes.
"""

import os, sys, numpy as np, matplotlib.pyplot as plt

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics

# Create synthetic data for testing
def create_synthetic_data():
    # Time parameters
    dt = 0.001  # 1ms resolution
    total_time = 2.0  # 2 seconds
    t = np.arange(0, total_time, dt)
    
    # Train parameters
    train_start = 0.5
    isi = 0.05  # 50ms intervals
    n_pulses = 5
    stim_times = train_start + isi * np.arange(n_pulses)
    
    # Create synthetic traces (multiple trials)
    n_trials = 3
    trials = np.zeros((len(t), n_trials))

    # Add synthetic events with double exponential decay (8 ms & 50 ms)
    tau_fast = 0.008   # 8 ms
    tau_slow = 0.050   # 50 ms
    fast_base = 1.0 * (0.8 ** np.arange(n_pulses))
    slow_base = 0.1 * (0.9 ** np.arange(n_pulses))

    for trial in range(n_trials):
        for i, st in enumerate(stim_times):
            # Decreasing amplitudes (PPR effect) for fast & slow components
            amp_fast = fast_base[i] + 0.05 * np.random.randn()
            amp_slow = slow_base[i] + 0.05 * np.random.randn()

            # Double exponential kernel
            mask = t >= st
            decay_time = t[mask] - st
            trials[mask, trial] += (
                amp_fast * np.exp(-decay_time / tau_fast)
                + amp_slow * np.exp(-decay_time / tau_slow)
            )

        # Add noise
        trials[:, trial] += 0.5 * np.random.randn(len(t))

    ground_truth = {
        'tau_fast': tau_fast,
        'tau_slow': tau_slow,
        'fast_amp0': fast_base[0],
        'slow_amp0': slow_base[0],
        'fast_series': fast_base,
        'slow_series': slow_base,
    }

    return t, trials, train_start, isi, n_pulses, ground_truth

# Test different weight modes
def test_weight_modes():
    print("Creating synthetic data...")
    (
        t,
        trials,
        train_start,
        isi,
        n_pulses,
        ground_truth,
    ) = create_synthetic_data()
    
    weight_modes = ['uniform', 'linear', 'exponential']
    results = {}

    print(
        "Ground truth double-exp:"
        f" fast tau={ground_truth['tau_fast']*1e3:.0f} ms (amp≈{ground_truth['fast_amp0']})"
        f", slow tau={ground_truth['tau_slow']*1e3:.0f} ms (amp≈{ground_truth['slow_amp0']})"
    )
    print(
        "  Fast component amplitudes:",
        [f"{a:.2f}" for a in ground_truth['fast_series']]
    )
    print(
        "  Slow component amplitudes:",
        [f"{a:.2f}" for a in ground_truth['slow_series']]
    )
    
    for mode in weight_modes:
        print(f"\nTesting weight mode: {mode}")
        
        res = extract_metrics(
            t, trials,
            train_start=train_start,
            isi=isi,
            n_pulses=n_pulses,
            options={
                'normalize_dff': False,  # Skip ΔF/F for synthetic data
                'bleach': False,         # Skip bleach correction
                'fit_source': 'global',
                'decay_progression_mode': 'linear',
                'event_model': 'double_exp',
                'nnls_weight_mode': mode,
                'nnls_weight_tau_s': 0.008 if mode == 'exponential' else None,  # 8ms
                'nnls_show_weights': True,
                'plot': {
                    'enabled': True,
                    'traces': ['nnls'],
                    'show_decay': True,
                    'trials': False,
                    'baseline': False,
                    'residuals': False,
                }
            }
        )
        
        results[mode] = res
        amps = res['average']['amp_nnls']
        print(f"Amplitudes ({mode}): {amps}")
        print(
            "  Fast/slow taus (ms):",
            f"{ground_truth['tau_fast']*1e3:.1f} / {ground_truth['tau_slow']*1e3:.1f}",
        )

    return results, ground_truth

if __name__ == "__main__":
    np.random.seed(42)  # For reproducible results
    results, ground_truth = test_weight_modes()
    
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    for mode in ['uniform', 'linear', 'exponential']:
        amps = results[mode]['average']['amp_nnls']
        print(
            f"{mode:12}: {[f'{a:.3f}' for a in amps]}"
            f" (fast/slow taus: {ground_truth['tau_fast']*1e3:.0f}/{ground_truth['tau_slow']*1e3:.0f} ms)"
        )
    
    print("\nWeight mode testing completed!")
    print("Check the plots to see the different weight patterns.")
    
    # Keep plots open
    try:
        plt.show()
    except Exception:
        pass

