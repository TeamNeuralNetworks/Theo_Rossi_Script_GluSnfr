#!/usr/bin/env python3
"""
Simplified comparison of Double Exponential vs Cooperative Binding models
using the same data loading and preprocessing as the original notebook.

Focuses only on these two models of interest with essential statistics.
"""

import sys
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# DATA LOADING (from original notebook)
# =============================================================================

DEFAULT_DATA_DIR = r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Theo_1_5Ca\\"

def _scan_argv_for_dir(argv) -> str | None:
    for a in argv[1:]:
        if not a or a.startswith('-'):
            continue
        ap = os.path.abspath(a)
        if os.path.isdir(ap):
            return ap
    return None

def _resolve_input_dir(cli_arg: str | None) -> str:
    """Resolve input directory with precedence: positional existing dir > CLI arg (if valid) > GLUSNFR_IN_DIR env > default."""
    # If a specific CLI arg provided and valid
    if cli_arg and not cli_arg.startswith('-'):
        ap = os.path.abspath(cli_arg)
        if os.path.isdir(ap):
            return ap
    # Otherwise scan remaining argv for a directory (handles Jupyter injected flags)
    scan = _scan_argv_for_dir(sys.argv)
    if scan:
        return scan
    env_dir = os.environ.get("GLUSNFR_IN_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    return DEFAULT_DATA_DIR

def load_calcium_data(input_dir: str | None = None, *, verbose: bool = True):
    """Load data using the same approach as the original notebook.

    Parameters
    ----------
    input_dir : str | None
        Optional explicit directory of .xlsx files. If None, resolve using
        CLI/env/default precedence handled by caller (or resolved here if still None).
    verbose : bool
        If True, print diagnostic information about discovered files.
    """
    
    # Add likely source folders so the demo module is importable
    repo_root = Path(__file__).resolve().parents[1]
    cand_paths = [
        str(Path.cwd()),
        str(repo_root),
        str(repo_root / 'Model_Calibration'),
    ]
    for p in cand_paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    
    try:
        # Import the demo module (prefer package path)
        try:
            demo = importlib.import_module('Model_Calibration.demo_adjust_fit_events')
        except Exception:
            demo = importlib.import_module('demo_adjust_fit_events')
        
        # Determine which input directory to use
        if input_dir is None:
            # Try to reuse logic from demo module if available
            if hasattr(demo, "_resolve_input_dir"):
                try:
                    input_dir = demo._resolve_input_dir(None)  # type: ignore[attr-defined]
                except Exception:
                    input_dir = _resolve_input_dir(None)
            elif hasattr(demo, "DEFAULT_IN_DIR"):
                input_dir = getattr(demo, "DEFAULT_IN_DIR")
            else:
                input_dir = _resolve_input_dir(None)

        print(f"Using data directory: {input_dir}")

        # Basic diagnostics before processing
        import glob as _glob
        xlsx_pattern = os.path.join(input_dir, "*.xlsx")
        xlsx_files = _glob.glob(xlsx_pattern)
        if verbose:
            print(f"Scanning directory: {input_dir}")
            print(f"Found {len(xlsx_files)} .xlsx files (pattern: {xlsx_pattern})")
            if len(xlsx_files) == 0:
                print("No .xlsx files detected. Check that the path is correct or override with CLI arg or GLUSNFR_IN_DIR.")

        # Process the folder with configured settings
        RESULTS = demo.process_folder(
            input_dir,
            max_workers=getattr(demo, "MAX_WORKERS", 24)
        )
        
        if RESULTS:
            # Apply the same time shift as in original notebook
            RESULTS['time_grid'] = RESULTS['time_grid'] + 0.003  # +3 ms alignment
            
            print(f"Loaded data from {RESULTS['n_files']} files")
            print(f"Time range: {RESULTS['time_grid'][0]*1000:.1f} to {RESULTS['time_grid'][-1]*1000:.1f} ms")
            
            if demo.EVENT_INDEX is not None:
                print(f"Event {demo.EVENT_INDEX + 1} only")
            else:
                print("All events combined")
            
            return RESULTS
        else:
            raise ValueError(
                "No results returned from demo module. Possible causes: (1) directory has no valid Excel files, "
                "(2) files failed validation (corrupt or not OOXML .xlsx), (3) all files skipped during processing. "
                "Override the input directory via CLI or GLUSNFR_IN_DIR."
            )
            
    except ImportError as e:
        raise ImportError(f"Could not import demo_adjust_fit_events: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")


# =============================================================================
# DATA PREPROCESSING (from original notebook Cell 3A)
# =============================================================================

class SynapticCurrentAnalyzer:
    def clean_trace(self, trace):
        """Remove NaN values and interpolate if needed"""
        trace = np.array(trace, dtype=float)
        
        if np.all(np.isnan(trace)):
            return None
            
        valid_mask = ~np.isnan(trace)
        if np.sum(valid_mask) < 5:
            return None
            
        if not np.all(valid_mask):
            x = np.arange(len(trace))
            trace_clean = np.interp(x, x[valid_mask], trace[valid_mask])
        else:
            trace_clean = trace
            
        return trace_clean
    
    def baseline_correct(self, time_ms, trace):
        """Baseline correction using pre-stimulus data"""
        baseline_mask = time_ms < 0
        
        if not np.any(baseline_mask):
            n_baseline = max(1, len(time_ms) // 4)
            baseline_mask = np.zeros(len(time_ms), dtype=bool)
            baseline_mask[:n_baseline] = True
            
        baseline_data = trace[baseline_mask]
        baseline_data = baseline_data[np.isfinite(baseline_data)]
        
        if len(baseline_data) == 0:
            return trace, 0
            
        F0 = np.median(baseline_data)
        return trace - F0, F0
    
    def estimate_noise_level(self, time_ms, trace):
        """Estimate noise level from baseline period"""
        baseline_mask = time_ms < 0
        if not np.any(baseline_mask):
            baseline_mask = np.zeros(len(time_ms), dtype=bool)
            baseline_mask[:len(time_ms)//4] = True
            
        baseline_data = trace[baseline_mask]
        baseline_data = baseline_data[np.isfinite(baseline_data)]
        
        if len(baseline_data) < 3:
            return 0.001  # Default small value
            
        return np.std(baseline_data)


def preprocess_data(RESULTS):
    """Apply the same preprocessing as in the original notebook"""
    
    analyzer = SynapticCurrentAnalyzer()
    
    # Extract data from RESULTS
    time_grid = RESULTS['time_grid'] * 1000  # Convert to ms
    traces = RESULTS['traces']
    
    # Apply analysis window (-5ms to 50ms for better model fitting)
    analysis_mask = (time_grid >= -5) & (time_grid <= 50)
    time_analysis = time_grid[analysis_mask]
    
    # Process traces
    clean_traces = []
    noise_levels = []
    
    print("\n=== DATA PREPARATION ===")
    print(f"Analysis window: {time_analysis[0]:.1f} to {time_analysis[-1]:.1f} ms")
    
    for i, trace in enumerate(traces):
        trace_clean = analyzer.clean_trace(trace)
        if trace_clean is None:
            continue
            
        trace_cut = trace_clean[analysis_mask]
        trace_corrected, f0 = analyzer.baseline_correct(time_analysis, trace_cut)
        
        if np.all(np.isnan(trace_corrected)):
            continue
            
        # Estimate noise level for this trace
        noise_level = analyzer.estimate_noise_level(time_analysis, trace_corrected)
        
        clean_traces.append(trace_corrected)
        noise_levels.append(noise_level)
    
    print(f"Successfully processed: {len(clean_traces)}/{len(traces)} traces")
    
    if len(clean_traces) == 0:
        raise ValueError("No valid traces found!")
    
    # Calculate average trace and average noise level
    traces_array = np.array(clean_traces)
    y_avg = np.mean(traces_array, axis=0)
    avg_noise = np.mean(noise_levels)
    
    print(f"Average signal range: {y_avg.min():.4f} to {y_avg.max():.4f}")
    print(f"Average noise level: {avg_noise:.4f}")
    
    return time_analysis, traces_array, y_avg, avg_noise


# =============================================================================
# MODEL DEFINITIONS (only the two models of interest)
# =============================================================================

def model_double_exp_constrained(t, amp, tau_rise, tau_decay, t_peak):
    """Classic double exponential: (exp(-t/tau_decay) - exp(-t/tau_rise))"""
    t = np.asarray(t)
    result = np.zeros_like(t, dtype=float)
    
    mask = t >= t_peak
    if np.any(mask):
        t_shifted = (t[mask] - t_peak) / 1000
        
        tau_rise = max(tau_rise, 1e-6)
        tau_decay = max(tau_decay, 1e-6)
        
        if tau_decay > tau_rise:
            t_opt = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
            norm_factor = np.exp(-t_opt / tau_decay) - np.exp(-t_opt / tau_rise)
            
            rise_term = np.exp(-t_shifted / tau_rise)
            decay_term = np.exp(-t_shifted / tau_decay)
            
            if norm_factor > 1e-10:
                result[mask] = amp * (decay_term - rise_term) / norm_factor
            else:
                result[mask] = amp * (decay_term - rise_term)
    return result


def model_cooperative_binding(t, amp, tau_rise, tau_decay, n_coop, t_peak):
    """Cooperative binding: Hill-like rise + exponential decay"""
    t = np.asarray(t)
    result = np.zeros_like(t, dtype=float)
    
    mask = t >= t_peak
    if np.any(mask):
        t_shifted = (t[mask] - t_peak) / 1000
        
        # Cooperative rise phase
        tau_rise_safe = max(tau_rise, 1e-6)
        n_safe = max(n_coop, 0.5)
        
        # Hill-like binding kinetics
        normalized_t = t_shifted / tau_rise_safe
        rise_factor = (normalized_t ** n_safe) / (1 + normalized_t ** n_safe)
        
        # Exponential decay from bound state
        decay_factor = np.exp(-t_shifted / max(tau_decay, 1e-6))
        
        result[mask] = amp * rise_factor * decay_factor
    
    return result


# =============================================================================
# FITTING AND ANALYSIS
# =============================================================================

def fit_models_to_average(time_analysis, y_avg, avg_noise):
    """Fit both models to the average trace"""
    
    # Set up fitting region (0 to 30 ms)
    fit_mask = (time_analysis >= 0) & (time_analysis <= 30)
    t_fit = time_analysis[fit_mask]
    y_fit_data = y_avg[fit_mask]
    
    # Model specifications
    models_to_test = {
        'double_exp': {
            'func': model_double_exp_constrained,
            'params': ['amp', 'tau_rise', 'tau_decay', 't_peak'],
            'bounds': ([0, 0.0005, 0.001, 0], [np.inf, 0.010, 0.200, 10]),
            'p0_func': lambda y, t: [np.max(y), 0.002, 0.020, t[np.argmax(y)]]
        },
        'cooperative': {
            'func': model_cooperative_binding,
            'params': ['amp', 'tau_rise', 'tau_decay', 'n_coop', 't_peak'],
            'bounds': ([0, 0.001, 0.005, 0.5, 0], [np.inf, 0.020, 0.200, 5.0, 10]),
            'p0_func': lambda y, t: [np.max(y), 0.005, 0.030, 2.0, t[np.argmax(y)]]
        }
    }
    
    print("\n=== AVERAGE TRACE FITTING ===")
    print(f"Fitting on {len(t_fit)} points from {t_fit[0]:.1f} to {t_fit[-1]:.1f} ms")
    
    fit_results = {}
    
    for model_name, model_info in models_to_test.items():
        try:
            # Get initial parameters
            p0 = model_info['p0_func'](y_fit_data, t_fit)
            
            # Fit model
            popt, pcov = curve_fit(
                model_info['func'], 
                t_fit, y_fit_data,
                p0=p0,
                bounds=model_info['bounds'],
                maxfev=3000
            )
            
            # Generate predictions
            y_pred_full = model_info['func'](time_analysis, *popt)
            y_pred_fit = y_pred_full[fit_mask]
            
            # Calculate metrics
            residuals = y_fit_data - y_pred_fit
            r_squared = 1 - np.sum(residuals**2) / np.sum((y_fit_data - np.mean(y_fit_data))**2)
            rmse = np.sqrt(np.mean(residuals**2))
            
            fit_results[model_name] = {
                'params': popt,
                'param_names': model_info['params'],
                'y_pred_full': y_pred_full,
                'residuals': residuals,
                'r_squared': r_squared,
                'rmse': rmse,
                'success': True
            }
            
            # Print results
            print(f"\n{model_name.upper()}:")
            for i, (name, val) in enumerate(zip(model_info['params'], popt)):
                if 'tau' in name and 'peak' not in name:
                    print(f"  {name}: {val*1000:.2f} ms")
                elif 't_' in name:
                    print(f"  {name}: {val:.2f} ms")
                else:
                    print(f"  {name}: {val:.3f}")
            print(f"  R² = {r_squared:.3f}, RMSE = {rmse:.4f}")
            
        except Exception as e:
            fit_results[model_name] = {'success': False, 'error': str(e)}
            print(f"\n{model_name.upper()}: FAILED - {e}")
    
    return fit_results, models_to_test, fit_mask


def fit_individual_traces(time_analysis, traces_array, models_to_test, fit_mask):
    """Fit both models to individual traces"""
    
    print(f"\n=== INDIVIDUAL TRACE FITTING ===")
    print(f"Fitting {len(traces_array)} individual traces...")
    
    individual_results = {model_name: [] for model_name in models_to_test.keys()}
    
    t_fit = time_analysis[fit_mask]
    
    for trial_idx, trace in enumerate(traces_array):
        trace_fit = trace[fit_mask]
        
        # Skip traces with insufficient data
        if np.sum(np.isfinite(trace_fit)) < 5:
            for model_name in models_to_test.keys():
                individual_results[model_name].append({'success': False})
            continue
        
        for model_name, model_info in models_to_test.items():
            try:
                # Get initial parameters based on trace
                p0 = model_info['p0_func'](trace_fit, t_fit)
                
                # Fit model
                popt, pcov = curve_fit(
                    model_info['func'],
                    t_fit, trace_fit,
                    p0=p0,
                    bounds=model_info['bounds'],
                    maxfev=2000
                )
                
                # Generate predictions
                y_pred_full = model_info['func'](time_analysis, *popt)
                y_pred_fit = y_pred_full[fit_mask]
                
                # Calculate metrics
                residuals = trace_fit - y_pred_fit
                r_squared = 1 - np.sum(residuals**2) / np.sum((trace_fit - np.mean(trace_fit))**2)
                rmse = np.sqrt(np.mean(residuals**2))
                
                individual_results[model_name].append({
                    'success': True,
                    'params': popt,
                    'y_pred_full': y_pred_full,
                    'residuals': residuals,
                    'r_squared': r_squared,
                    'rmse': rmse
                })
                
            except Exception:
                individual_results[model_name].append({'success': False})
    
    # Print success rates
    for model_name in models_to_test.keys():
        n_success = sum(1 for r in individual_results[model_name] if r['success'])
        print(f"{model_name}: {n_success}/{len(traces_array)} successful fits ({100*n_success/len(traces_array):.1f}%)")
    
    return individual_results


def create_comparison_plots(time_analysis, y_avg, traces_array, fit_results, individual_results, models_to_test):
    """Create comprehensive comparison plots"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Panel 1: Average trace fits
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(time_analysis, y_avg, 'k-', linewidth=2, label='Data (average)')
    
    colors = ['red', 'blue']
    model_names = list(models_to_test.keys())
    
    for i, model_name in enumerate(model_names):
        if fit_results[model_name]['success']:
            ax1.plot(time_analysis, fit_results[model_name]['y_pred_full'], 
                    color=colors[i], linestyle='--', linewidth=2, 
                    label=model_name.replace('_', ' ').title())
    
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Signal')
    ax1.set_title('Average Trace Fits')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2-3: Individual fits for each model
    for i, model_name in enumerate(model_names):
        ax = plt.subplot(2, 4, 2 + i)
        
        # Show first 5 traces
        show_traces = min(5, len(traces_array))
        for j in range(show_traces):
            ax.plot(time_analysis, traces_array[j], color='gray', alpha=0.3, linewidth=1)
            result = individual_results[model_name][j]
            if result['success']:
                ax.plot(time_analysis, result['y_pred_full'], 
                       color=colors[i], alpha=0.7, linewidth=1.5)
        
        ax.axvline(0, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Signal')
        ax.set_title(f'{model_name.replace("_", " ").title()}\nIndividual Fits')
        ax.grid(True, alpha=0.3)
    
    # Panel 4: R² comparison
    ax4 = plt.subplot(2, 4, 4)
    r2_values = []
    for model_name in model_names:
        r2_list = [r['r_squared'] for r in individual_results[model_name] if r['success']]
        r2_values.append(r2_list)
    
    bp = ax4.boxplot(r2_values, labels=[name.replace('_', ' ').title() for name in model_names], 
                     patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i])
    
    ax4.set_ylabel('R²')
    ax4.set_title('R² Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Panel 5-6: Residual distributions
    all_residuals = {}
    for i, model_name in enumerate(model_names):
        ax = plt.subplot(2, 4, 5 + i)
        
        residuals_list = []
        for result in individual_results[model_name]:
            if result['success'] and 'residuals' in result:
                residuals_list.extend(result['residuals'])
        
        all_residuals[model_name] = residuals_list
        
        if residuals_list:
            ax.hist(residuals_list, bins=30, alpha=0.7, color=colors[i], 
                   edgecolor='black', density=True)
            
            # Overlay normal distribution
            x_norm = np.linspace(min(residuals_list), max(residuals_list), 100)
            normal_fit = stats.norm(np.mean(residuals_list), np.std(residuals_list))
            ax.plot(x_norm, normal_fit.pdf(x_norm), 'k-', linewidth=2)
            
            ax.set_xlabel('Residual Value')
            ax.set_ylabel('Density')
            ax.set_title(f'{model_name.replace("_", " ").title()}\nResidual Distribution')
            ax.grid(True, alpha=0.3)
    
    # Panel 7: Direct residual comparison
    ax7 = plt.subplot(2, 4, 7)
    for i, model_name in enumerate(model_names):
        if model_name in all_residuals and all_residuals[model_name]:
            ax7.hist(all_residuals[model_name], bins=25, alpha=0.6, color=colors[i], 
                    label=model_name.replace('_', ' ').title(), density=True)
    
    ax7.set_xlabel('Residual Value')
    ax7.set_ylabel('Density')
    ax7.set_title('Residual Comparison')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Panel 8: Summary statistics
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    summary_text = "MODEL COMPARISON\n\n"
    
    for model_name in model_names:
        successful = [r for r in individual_results[model_name] if r['success']]
        
        if successful:
            r2_vals = [r['r_squared'] for r in successful]
            rmse_vals = [r['rmse'] for r in successful]
            
            summary_text += f"{model_name.replace('_', ' ').title()}:\n"
            summary_text += f"  Success: {len(successful)}/{len(individual_results[model_name])}\n"
            summary_text += f"  Mean R²: {np.mean(r2_vals):.3f}\n"
            summary_text += f"  Mean RMSE: {np.mean(rmse_vals):.4f}\n"
            
            if model_name in all_residuals and all_residuals[model_name]:
                summary_text += f"  Residual std: {np.std(all_residuals[model_name]):.4f}\n"
            summary_text += "\n"
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return all_residuals


def statistical_comparison(all_residuals, model_names):
    """Perform statistical comparison of the two models"""
    
    print("\n=== STATISTICAL COMPARISON ===")
    
    for model_name in model_names:
        if model_name in all_residuals and all_residuals[model_name]:
            residuals = all_residuals[model_name]
            
            # Normality test (subsample if too many points)
            test_residuals = np.random.choice(residuals, min(5000, len(residuals)), replace=False)
            _, p_normal = stats.shapiro(test_residuals)
            
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Residual count: {len(residuals)}")
            print(f"  Mean: {np.mean(residuals):.4f}")
            print(f"  Std: {np.std(residuals):.4f}")
            print(f"  Shapiro p-value: {p_normal:.4f}")
            print(f"  Normality: {'✓ Normal' if p_normal > 0.05 else '✗ Non-normal'}")
    
    # Compare the two residual distributions
    if len(model_names) == 2:
        res1 = all_residuals[model_names[0]]
        res2 = all_residuals[model_names[1]]
        
        if res1 and res2:
            # Kolmogorov-Smirnov test
            _, p_ks = stats.ks_2samp(res1, res2)
            
            print(f"\nComparison between models:")
            print(f"  Kolmogorov-Smirnov p-value: {p_ks:.4f}")
            print(f"  Distributions: {'Significantly different' if p_ks < 0.05 else 'Not significantly different'}")
            
            # F-test for variance comparison
            f_stat = np.var(res1) / np.var(res2)
            print(f"  Variance ratio (model1/model2): {f_stat:.3f}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main analysis function"""
    
    print("=== CALCIUM RESPONSE MODEL COMPARISON ===")
    print("Double Exponential vs Cooperative Binding")
    
    try:
        # Resolve optional CLI directory argument
        cli_dir = sys.argv[1] if len(sys.argv) > 1 else None
        input_dir = _resolve_input_dir(cli_dir)

        # Load data
        RESULTS = load_calcium_data(input_dir)

        # Preprocess data
        time_analysis, traces_array, y_avg, avg_noise = preprocess_data(RESULTS)

        # Fit models to average trace
        fit_results, models_to_test, fit_mask = fit_models_to_average(time_analysis, y_avg, avg_noise)

        # Fit models to individual traces
        individual_results = fit_individual_traces(time_analysis, traces_array, models_to_test, fit_mask)

        # Create comparison plots
        all_residuals = create_comparison_plots(
            time_analysis, y_avg, traces_array, fit_results, individual_results, models_to_test
        )

        # Statistical comparison
        statistical_comparison(all_residuals, list(models_to_test.keys()))

        print("\n=== ANALYSIS COMPLETE ===")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()