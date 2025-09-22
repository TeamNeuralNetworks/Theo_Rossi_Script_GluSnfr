#!/usr/bin/env python3
"""
Test script for the new NNLS weight functionality.
Creates synthetic data and tests different weight modes.
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
    isi = 0.1  # 100ms intervals
    n_pulses = 5
    stim_times = train_start + isi * np.arange(n_pulses)
    
    # Create synthetic traces (multiple trials)
    n_trials = 3
    trials = np.zeros((len(t), n_trials))
    
    # Add synthetic events with exponential decay
    for trial in range(n_trials):
        for i, st in enumerate(stim_times):
            # Decreasing amplitude (PPR effect)
            amp = 1.0 * (0.8 ** i) + 0.1 * np.random.randn()
            
            # Exponential kernel
            mask = t >= st
            trials[mask, trial] += amp * np.exp(-(t[mask] - st) / 0.025)
        
        # Add noise
        trials[:, trial] += 0.05 * np.random.randn(len(t))
    
    return t, trials, train_start, isi, n_pulses

# Test different weight modes
def test_weight_modes():
    print("Creating synthetic data...")
    t, trials, train_start, isi, n_pulses = create_synthetic_data()
    
    weight_modes = ['uniform', 'linear', 'exponential']
    results = {}
    
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
                'nnls_weight_tau_s': 0.050 if mode == 'exponential' else None,  # 50ms
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
        print(f"Amplitudes ({mode}): {res['average']['amp_nnls']}")
    
    return results

if __name__ == "__main__":
    np.random.seed(42)  # For reproducible results
    results = test_weight_modes()
    
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    for mode in ['uniform', 'linear', 'exponential']:
        amps = results[mode]['average']['amp_nnls']
        print(f"{mode:12}: {[f'{a:.3f}' for a in amps]}")
    
    print("\nWeight mode testing completed!")
    print("Check the plots to see the different weight patterns.")
    
    # Keep plots open
    try:
        plt.show()
    except:
        pass