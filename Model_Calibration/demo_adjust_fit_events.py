import os
import sys
import glob
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the repository root is on sys.path when running from this subfolder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics
from smoothing import build_median_recut_waveform

# Configuration parameters
TRAIN_START_S = 0.5
ISI_S = 0.05
N_PULSES = 10
PRE_MS = 5.0
POST_MS = 50.0
SAMPLE_HZ = 1000.0

# Processing settings
MAX_WORKERS = 24
EVENT_INDEX = None  # Set to 0 for event 1, 2 for event 3, etc. Use None for all events

# Paths - change these to your data locations  
IN_DIR = r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Theo_1_5Ca\\"


def _is_valid_xlsx(path: str) -> bool:
    """Check if file is a valid Excel file."""
    try:
        with zipfile.ZipFile(path) as z:
            return '[Content_Types].xml' in z.namelist()
    except Exception:
        return False


def _load_time_trials(xlsx_path: str):
    """Load time and trial data from Excel file."""
    df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    t_raw = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    ok = np.isfinite(t_raw)
    return t_raw[ok], X[ok, :]


def _clean_trials(X: np.ndarray) -> np.ndarray:
    """Remove trials that are entirely NaN."""
    if X is None or X.size == 0:
        return X
    col_ok = ~np.all(~np.isfinite(X), axis=0)
    return X[:, col_ok]


def _file_to_resampled_trace(xlsx_path: str, t_grid: np.ndarray):
    """Process a single Excel file and return resampled trace."""
    try:
        if not _is_valid_xlsx(xlsx_path):
            return None
            
        t, X = _load_time_trials(xlsx_path)
        X = _clean_trials(X)
        
        # Extract metrics from the data
        res = extract_metrics(
            t, X, train_start=TRAIN_START_S, isi=ISI_S, n_pulses=N_PULSES,
            options={'normalize_dff': False, 'bleach': False, 'plot': {'enabled': False}}
        )
        
        # Use processed trials from extract_metrics
        processed_trials = np.column_stack([trial_data['y_proc'] for trial_data in res['per_trial']])
        
        # Build median recut waveform
        stim = res['stim_times_s']
        if EVENT_INDEX is not None and len(stim) > EVENT_INDEX:
            stim_selected = [stim[EVENT_INDEX]]
            print(f"  Using event {EVENT_INDEX + 1} only for median calculation")
        else:
            stim_selected = stim
            print(f"  Using all {len(stim)} stimulus events (events 1-{len(stim)}) for median calculation")
        
        t_rel, med = build_median_recut_waveform(
            res['time_s'], processed_trials, np.asarray(stim_selected),
            pre_ms=PRE_MS, 
            post_ms=POST_MS, 
            align_by_peak=False
        )

        if t_rel is None or med is None:
            return None
            
        # Resample to grid
        r = np.full_like(t_grid, np.nan, dtype=float)
        m = (t_grid >= t_rel[0]) & (t_grid <= t_rel[-1])
        if np.any(m):
            r[m] = np.interp(t_grid[m], t_rel, med)
        return r
        
    except Exception as e:
        print(f"Error processing {xlsx_path}: {e}")
        return None


def process_folder(input_dir: str, max_workers: int = None):
    """Process all Excel files in a folder and create summary plot."""
    
    # Create time grid for resampling
    t_grid = np.arange(
        -PRE_MS/1000.0,
        POST_MS/1000.0 + 1e-12,
        1.0/SAMPLE_HZ
    )
    
    # Find all valid Excel files
    xlsx_files = [p for p in glob.glob(os.path.join(input_dir, "*.xlsx")) if _is_valid_xlsx(p)]
    print(f"Found {len(xlsx_files)} valid Excel files in {input_dir}")
    
    if not xlsx_files:
        print("No valid Excel files found!")
        return None
    
    # Process files in parallel using threads
    traces_resampled = []
    workers = max_workers or MAX_WORKERS
    
    print(f"Processing files with {workers} workers...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(_file_to_resampled_trace, file_path, t_grid): file_path 
            for file_path in xlsx_files
        }
        
        # Collect results
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            result = future.result()
            
            if result is not None and np.any(np.isfinite(result)):
                traces_resampled.append(result)
                print(f"✓ {os.path.basename(file_path)} -> Total: {len(traces_resampled)}")
            else:
                print(f"✗ Skipped {os.path.basename(file_path)}")
    
    if not traces_resampled:
        print("No valid traces found!")
        return None
    
    # Calculate average across all files
    S = np.vstack(traces_resampled)
    avg_all = np.nanmean(S, axis=0)
    
    print(f"\nSuccessfully processed {len(traces_resampled)} files")
    print(f"Time grid: {len(t_grid)} points from {t_grid[0]*1000:.1f} to {t_grid[-1]*1000:.1f} ms")
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual traces
    for i, trace in enumerate(traces_resampled):
        ax.plot(t_grid * 1000.0, trace, color='lightgray', alpha=0.5, linewidth=0.8)
    
    # Plot average
    ax.plot(t_grid * 1000.0, avg_all, color='red', linewidth=2.0, label=f'Average (N={len(traces_resampled)})')
    
    # Add stimulus time marker
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.0, label='Stimulus')
    
    # Formatting
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('ΔF (median)')
    title = f'Batch Processing Results: {len(traces_resampled)} files'
    if EVENT_INDEX is not None:
        title += f' - Event {EVENT_INDEX + 1} only'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'traces': traces_resampled,
        'time_grid': t_grid,
        'average': avg_all,
        'n_files': len(traces_resampled),
        'figure': fig,
        'axis': ax
    }


if __name__ == "__main__":
    # Process the folder
    results = process_folder(IN_DIR, max_workers=MAX_WORKERS)
    
    if results:
        print(f"\nProcessing complete! Processed {results['n_files']} files.")
        print("Results saved to 'results' variable.")
    else:
        print("Processing failed - no valid data found.")