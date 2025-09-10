"""
Adjust and visualize fit model by recutting, aligning, and averaging events.

Supports:
- Single file (.xlsx)
- Single folder (process all .xlsx)
- Multiple folders (batch)

For each dataset, the script:
- Loads time and trials (last column is time; others are trials)
- Runs the streamlined pipeline (`extract_metrics`) to get the average trace
  and its fitted model (`yhat_avg`)
- Re-cuts windows around all or a subset of events (with optional peak
  alignment)
- Averages the recut segments (median across selected events)
- Overlays the current fitted model (recut in the same way) on top of the
  averaged waveform

Usage (CLI):
  python adjust_fit_model.py INPUT [INPUT ...] \
      --train-start 0.5 --isi 0.05 --n-pulses 10 \
      --events 1-5,7 --align-by-peak \
      --pre-ms 2 --post-ms 200 \
      --out-dir C:\\out --save

Notes:
- INPUT may be a file (.xlsx) or a folder. When a folder is passed, all
  .xlsx files within are processed.
- The overlayed model is derived from the pipeline’s average-trace fit
  (`yhat_avg`) to reflect the current fitting model.
"""

from __future__ import annotations

import os
import sys
import glob
import zipfile
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the repository root is on sys.path when running from this subfolder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics
from Ultimate_iglusnfr_smoothing.smoothing import (
    build_median_recut_waveform,
    build_median_recut_figure,
    fit_template_decay,
)


# -------------------------
# Helpers
# -------------------------

def _is_valid_xlsx(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return '[Content_Types].xml' in z.namelist()
    except Exception:
        return False


def _load_time_trials_from_xlsx(xlsx_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load time and trials from an Excel file (sheet 0).

    Time is taken from the last column; trials are all preceding columns.
    Non-numeric entries are coerced to NaN and dropped row-wise via the time mask.
    """
    df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    if df.shape[1] < 2:
        raise ValueError(f"Expected ≥2 columns (data + time) in {xlsx_path}")
    t_raw = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    ok = np.isfinite(t_raw)
    time = t_raw[ok]
    trials = X[ok, :]
    return time, trials


def _parse_events_spec(spec: Optional[str], n_pulses: int) -> List[int]:
    """Parse an events spec like "1,3-5,8" into 0-based pulse indices.

    Returns all pulses if spec is None/empty.
    Clamps to [0, n_pulses-1] and de-duplicates in ascending order.
    """
    if spec is None or str(spec).strip() == "" or str(spec).strip().lower() == "all":
        return list(range(n_pulses))
    out = []
    for chunk in str(spec).split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            a, b = chunk.split('-', 1)
            try:
                i0 = int(a) - 1
                i1 = int(b) - 1
            except ValueError:
                continue
            if i0 > i1:
                i0, i1 = i1, i0
            out.extend(list(range(i0, i1 + 1)))
        else:
            try:
                out.append(int(chunk) - 1)
            except ValueError:
                continue
    # clamp and unique
    out = sorted({i for i in out if 0 <= i < int(n_pulses)})
    if not out:
        out = list(range(n_pulses))
    return out


def _recut_median_from_series(
    t: np.ndarray,
    y: np.ndarray,
    stim_times: np.ndarray,
    event_idx: List[int],
    *,
    pre_ms: float,
    post_ms: float,
    align_by_peak: bool,
    peak_win_ms: float = 25.0,
    peak_search_pre_ms: float = 2.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Recut windows around selected events on a single series and return median.

    Uses the same utility as the smoothing module; passing a single-column series
    simply produces one snippet per selected event, whose median is returned.
    """
    if t is None or y is None or np.size(t) == 0 or np.size(y) == 0:
        return None, None
    stim_sel = np.asarray(stim_times, float)[event_idx]
    t_rel, med = build_median_recut_waveform(
        t, np.asarray(y, float)[:, None], stim_sel,
        pre_ms=float(pre_ms), post_ms=float(post_ms),
        align_by_peak=bool(align_by_peak),
        peak_win_ms=float(peak_win_ms), peak_search_pre_ms=float(peak_search_pre_ms),
    )
    return t_rel, med


def _recut_median_from_trials(
    t: np.ndarray,
    Y_all: np.ndarray,
    stim_times: np.ndarray,
    event_idx: List[int],
    *,
    pre_ms: float,
    post_ms: float,
    align_by_peak: bool,
    peak_win_ms: float = 25.0,
    peak_search_pre_ms: float = 2.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Recut windows around selected events using ALL trials and return median.

    Produces one snippet per (event x trial) and returns the median across all
    snippets for a cleaner estimate of the event waveform.
    """
    if t is None or Y_all is None or np.size(t) == 0 or np.size(Y_all) == 0:
        return None, None
    stim_sel = np.asarray(stim_times, float)[event_idx]
    t_rel, med = build_median_recut_waveform(
        t, np.asarray(Y_all, float), stim_sel,
        pre_ms=float(pre_ms), post_ms=float(post_ms),
        align_by_peak=bool(align_by_peak),
        peak_win_ms=float(peak_win_ms), peak_search_pre_ms=float(peak_search_pre_ms),
    )
    return t_rel, med


def _figure_title(base: str, events_spec: str, align_by_peak: bool) -> str:
    ev = events_spec if events_spec else "all"
    align = "peak-aligned" if align_by_peak else "stim-aligned"
    return f"{base} | events: {ev} | {align}"


def _resample_to_grid(t_src: np.ndarray, y_src: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    if t_src is None or y_src is None or t_ref is None:
        return None
    t_src = np.asarray(t_src, float)
    y_src = np.asarray(y_src, float)
    t_ref = np.asarray(t_ref, float)
    y_ref = np.full_like(t_ref, np.nan, dtype=float)
    m = (t_ref >= t_src[0]) & (t_ref <= t_src[-1])
    if np.any(m):
        y_ref[m] = np.interp(t_ref[m], t_src, y_src)
    return y_ref


# -------------------------
# Core per-file processing
# -------------------------

def process_file(
    xlsx_path: str,
    *,
    train_start: float,
    isi: float,
    n_pulses: int,
    events_spec: Optional[str] = None,
    align_by_peak: bool = True,
    pre_ms: float = 2.0,
    post_ms: float = 200.0,
    normalize_dff: bool = True,
    bleach: bool = True,
    use_all_trials: bool = False,
    single_event_window: bool = False,
    guard_ms: float = 5.0,
    out_dir: Optional[str] = None,
    save: bool = False,
) -> Optional[str]:
    """Process a single Excel file and optionally save a figure.

    Returns path to saved figure if save=True and success, otherwise None.
    """
    if not _is_valid_xlsx(xlsx_path):
        print(f"[skip] Not a valid .xlsx package: {xlsx_path}")
        return None

    try:
        time, trials = _load_time_trials_from_xlsx(xlsx_path)
    except Exception as e:
        print(f"[skip] Failed to read Excel: {xlsx_path} -> {e}")
        return None

    # Run streamlined pipeline to get average trace and its model
    res = extract_metrics(
        time, trials,
        train_start=float(train_start), isi=float(isi), n_pulses=int(n_pulses),
        options={
            'normalize_dff': bool(normalize_dff),
            'bleach': bool(bleach),
            'plot': {'enabled': False}
        }
    )

    t = res['time_s']
    stim_times = res['stim_times_s']
    y_avg = res['average']['y_avg']
    yhat_avg = res['average']['yhat_avg']

    # Events selection
    idx = _parse_events_spec(events_spec, int(n_pulses))
    # Determine effective post window: optionally constrain to before next stimulus
    post_ms_eff = float(post_ms)
    if single_event_window and isi > 0:
        isi_ms = float(isi) * 1000.0
        post_ms_eff = max(0.0, min(float(post_ms), isi_ms - float(guard_ms)))

    # Recut source: either the average trace (default) or ALL trials for cleaner medians
    if use_all_trials:
        t_rel, med = _recut_median_from_trials(
            t, trials, stim_times, idx,
            pre_ms=pre_ms, post_ms=post_ms_eff, align_by_peak=align_by_peak,
        )
    else:
        t_rel, med = _recut_median_from_series(
            t, y_avg, stim_times, idx,
            pre_ms=pre_ms, post_ms=post_ms_eff, align_by_peak=align_by_peak,
        )
    _, med_model = _recut_median_from_series(
        t, yhat_avg, stim_times, idx,
        pre_ms=pre_ms, post_ms=post_ms_eff, align_by_peak=align_by_peak,
    )

    if t_rel is None or med is None:
        print(f"[warn] No recut median computed for {xlsx_path}")
        return None

    # Build figure with overlay
    base = os.path.splitext(os.path.basename(xlsx_path))[0]
    title = _figure_title(base, events_spec or "all", align_by_peak)
    fig = build_median_recut_figure(t_rel, med, fit_curve=med_model, title=title)

    if save and fig is not None:
        out_dir = out_dir or os.path.dirname(xlsx_path)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}_adjust_fit.png")
        try:
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"[ok] Saved: {out_path}")
            return out_path
        except Exception as e:
            print(f"[warn] Failed to save figure for {xlsx_path}: {e}")
    else:
        try:
            plt.show(block=False); plt.pause(0.05)
        except Exception:
            pass
    return None


# -------------------------
# Folder / multi-folder drivers
# -------------------------

def process_folder(
    in_dir: str,
    *,
    train_start: float,
    isi: float,
    n_pulses: int,
    events_spec: Optional[str] = None,
    align_by_peak: bool = True,
    pre_ms: float = 2.0,
    post_ms: float = 200.0,
    normalize_dff: bool = True,
    bleach: bool = True,
    use_all_trials: bool = False,
        single_event_window: bool = False,
        guard_ms: float = 5.0,
    out_dir: Optional[str] = None,
    aggregate: bool = False,
    aggregate_overlay: bool = False,
    sample_hz: float = 1000.0,
    save: bool = True,
) -> List[str]:
    """Process all .xlsx files in a folder; return list of saved figure paths."""
    in_dir = os.path.abspath(in_dir)
    out_dir = out_dir or in_dir
    paths = []
    agg_traces = []
    agg_trel = None
    for xlsx_path in glob.glob(os.path.join(in_dir, "*.xlsx")):
        if not aggregate:
            out_path = process_file(
                xlsx_path,
                train_start=train_start, isi=isi, n_pulses=n_pulses,
                events_spec=events_spec, align_by_peak=align_by_peak,
                pre_ms=pre_ms, post_ms=post_ms,
                normalize_dff=normalize_dff, bleach=bleach,
                use_all_trials=use_all_trials,
                single_event_window=single_event_window, guard_ms=guard_ms,
                out_dir=out_dir, save=save,
            )
            if out_path:
                paths.append(out_path)
        else:
            # Aggregate: compute median recut per file (without drawing), then combine
            try:
                time, trials = _load_time_trials_from_xlsx(xlsx_path)
            except Exception:
                continue
            # basic extract for stim times
            res = extract_metrics(
                time, trials,
                train_start=float(train_start), isi=float(isi), n_pulses=int(n_pulses),
                options={'normalize_dff': bool(normalize_dff), 'bleach': bool(bleach), 'plot': {'enabled': False}},
            )
            t = res['time_s']; stim_times = res['stim_times_s']; y_avg = res['average']['y_avg']
            if use_all_trials:
                t_rel, med = _recut_median_from_trials(
                    t, trials, stim_times, _parse_events_spec(events_spec, int(n_pulses)),
                    pre_ms=pre_ms, post_ms=post_ms if not single_event_window else max(0.0, min(post_ms, isi*1000.0 - guard_ms)),
                    align_by_peak=align_by_peak,
                )
            else:
                t_rel, med = _recut_median_from_series(
                    t, y_avg, stim_times, _parse_events_spec(events_spec, int(n_pulses)),
                    pre_ms=pre_ms, post_ms=post_ms if not single_event_window else max(0.0, min(post_ms, isi*1000.0 - guard_ms)),
                    align_by_peak=align_by_peak,
                )
            if t_rel is None or med is None:
                continue
            if agg_trel is None:
                agg_trel = t_rel
            med_resampled = _resample_to_grid(t_rel, med, agg_trel)
            agg_traces.append(med_resampled)
    if aggregate and agg_traces:
        # Build a common 1 kHz grid and resample all traces
        pre_s = float(pre_ms) / 1000.0
        post_eff_ms = post_ms if not single_event_window else max(0.0, min(post_ms, isi * 1000.0 - guard_ms))
        post_s = float(post_eff_ms) / 1000.0
        dt = 1.0 / float(sample_hz)
        t_grid = np.arange(-pre_s, post_s + 1e-12, dt)
        traces_resampled = []
        for tr, t_rel in zip(agg_traces, [agg_trel] * len(agg_traces)):
            traces_resampled.append(_resample_to_grid(t_rel, tr, t_grid))
        S = np.vstack(traces_resampled)
        avg_all = np.nanmean(S, axis=0)

        # Plot overlays + average on top (no fit here)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        # overlay all
        for r in traces_resampled:
            ax.plot(t_grid * 1000.0, r, color='0.5', alpha=0.35, linewidth=1.0)
        # average on top
        ax.plot(t_grid * 1000.0, avg_all, color='tab:blue', linewidth=2.0, label='Average')
        ax.axvline(0.0, color='k', linestyle=':', linewidth=1.0)
        ax.set_xlabel('Time (ms)'); ax.set_ylabel('ΔF (median)')
        ax.set_title(_figure_title(os.path.basename(in_dir), events_spec or 'all', align_by_peak))
        ax.legend(loc='best')

        # If overlay mode is requested, prefer showing without saving
        if aggregate_overlay or not save:
            try:
                plt.show(block=False); plt.pause(0.05)
            except Exception:
                pass
        else:
            try:
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{os.path.basename(in_dir)}_aggregate_overlay.png")
                fig.tight_layout(); fig.savefig(out_path, dpi=150)
                plt.close(fig)
                paths.append(out_path)
            except Exception:
                pass
    return paths


def process_inputs(
    inputs: List[str],
    *,
    train_start: float,
    isi: float,
    n_pulses: int,
    events_spec: Optional[str] = None,
    align_by_peak: bool = True,
    pre_ms: float = 2.0,
    post_ms: float = 200.0,
    normalize_dff: bool = True,
    bleach: bool = True,
    use_all_trials: bool = False,
        single_event_window: bool = False,
        guard_ms: float = 5.0,
    out_dir: Optional[str] = None,
    aggregate: bool = False,
    aggregate_overlay: bool = False,
    sample_hz: float = 1000.0,
    save: bool = True,
) -> List[str]:
    """Process a list of inputs (files or folders). Returns saved figure paths."""
    saved = []
    if aggregate:
    # Aggregate across all inputs (files and folders)
        agg_trel = None
        agg_traces = []
        label = "aggregate"
        def _collect_from_file(file_path: str):
            nonlocal agg_trel, agg_traces
            try:
                time, trials = _load_time_trials_from_xlsx(file_path)
            except Exception:
                return
            res = extract_metrics(
                time, trials,
                train_start=float(train_start), isi=float(isi), n_pulses=int(n_pulses),
                options={'normalize_dff': bool(normalize_dff), 'bleach': bool(bleach), 'plot': {'enabled': False}},
            )
            t = res['time_s']; stim_times = res['stim_times_s']; y_avg = res['average']['y_avg']
            post_eff = post_ms if not single_event_window else max(0.0, min(post_ms, isi*1000.0 - guard_ms))
            if use_all_trials:
                t_rel, med = _recut_median_from_trials(
                    t, trials, stim_times, _parse_events_spec(events_spec, int(n_pulses)),
                    pre_ms=pre_ms, post_ms=post_eff, align_by_peak=align_by_peak,
                )
            else:
                t_rel, med = _recut_median_from_series(
                    t, y_avg, stim_times, _parse_events_spec(events_spec, int(n_pulses)),
                    pre_ms=pre_ms, post_ms=post_eff, align_by_peak=align_by_peak,
                )
            if t_rel is None or med is None:
                return
            if agg_trel is None:
                agg_trel = t_rel
            agg_traces.append(_resample_to_grid(t_rel, med, agg_trel))

        for inp in inputs:
            if os.path.isdir(inp):
                label = os.path.basename(inp) if len(inputs) == 1 else "aggregate"
                for xlsx_path in glob.glob(os.path.join(inp, "*.xlsx")):
                    if xlsx_path.lower().endswith('.xlsx'):
                        _collect_from_file(xlsx_path)
            elif os.path.isfile(inp) and inp.lower().endswith('.xlsx'):
                _collect_from_file(inp)
        if agg_traces and agg_trel is not None:
            # Build uniform grid at sample_hz and overlay all + mean
            pre_s = float(pre_ms) / 1000.0
            post_eff_ms = post_ms if not single_event_window else max(0.0, min(post_ms, isi * 1000.0 - guard_ms))
            post_s = float(post_eff_ms) / 1000.0
            dt = 1.0 / float(sample_hz)
            t_grid = np.arange(-pre_s, post_s + 1e-12, dt)
            traces_resampled = [
                _resample_to_grid(agg_trel, tr, t_grid) for tr in agg_traces
            ]
            S = np.vstack(traces_resampled)
            avg_all = np.nanmean(S, axis=0)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            for r in traces_resampled:
                ax.plot(t_grid * 1000.0, r, color='0.5', alpha=0.35, linewidth=1.0)
            ax.plot(t_grid * 1000.0, avg_all, color='tab:blue', linewidth=2.0, label='Average')
            ax.axvline(0.0, color='k', linestyle=':', linewidth=1.0)
            ax.set_xlabel('Time (ms)'); ax.set_ylabel('ΔF (median)')
            ax.set_title(_figure_title(label, events_spec or 'all', align_by_peak))
            ax.legend(loc='best')

            if aggregate_overlay or not save:
                try:
                    plt.show(block=False); plt.pause(0.05)
                except Exception:
                    pass
            else:
                try:
                    out_dir_final = out_dir or (inputs[0] if os.path.isdir(inputs[0]) else os.path.dirname(inputs[0]))
                    os.makedirs(out_dir_final, exist_ok=True)
                    out_path = os.path.join(out_dir_final, f"{label}_aggregate_overlay.png")
                    fig.tight_layout(); fig.savefig(out_path, dpi=150)
                    plt.close(fig)
                    saved.append(out_path)
                except Exception:
                    pass
        return saved
    for inp in inputs:
        if os.path.isdir(inp):
            saved.extend(
                process_folder(
                    inp,
                    train_start=train_start, isi=isi, n_pulses=n_pulses,
                    events_spec=events_spec, align_by_peak=align_by_peak,
                    pre_ms=pre_ms, post_ms=post_ms,
                    normalize_dff=normalize_dff, bleach=bleach,
                    use_all_trials=use_all_trials,
                    single_event_window=single_event_window, guard_ms=guard_ms,
                    out_dir=out_dir, save=save,
                )
            )
        elif os.path.isfile(inp) and inp.lower().endswith('.xlsx'):
            out_path = process_file(
                inp,
                train_start=train_start, isi=isi, n_pulses=n_pulses,
                events_spec=events_spec, align_by_peak=align_by_peak,
                pre_ms=pre_ms, post_ms=post_ms,
                normalize_dff=normalize_dff, bleach=bleach,
                use_all_trials=use_all_trials,
                single_event_window=single_event_window, guard_ms=guard_ms,
                out_dir=out_dir, save=save,
            )
            if out_path:
                saved.append(out_path)
        else:
            print(f"[skip] Not a folder or .xlsx file: {inp}")
    return saved


# -------------------------
# CLI
# -------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Recut/align events, average them, and overlay the current fitted model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", help="File (.xlsx) and/or folder paths")
    p.add_argument("--train-start", type=float, default=0.5, help="First stimulus time (s)")
    p.add_argument("--isi", type=float, default=0.05, help="Inter-stimulus interval (s)")
    p.add_argument("--n-pulses", type=int, default=10, help="Number of pulses in the train")
    p.add_argument("--events", type=str, default="all", help="Events to include, e.g. '1-5,7'")
    p.add_argument("--align-by-peak", action="store_true", help="Align by peak within window instead of by stimulus time")
    p.add_argument("--pre-ms", type=float, default=2.0, help="Pre-stimulus window (ms)")
    p.add_argument("--post-ms", type=float, default=200.0, help="Post-stimulus window (ms)")
    p.add_argument("--no-dff", action="store_true", help="Disable ΔF/F0 normalization")
    p.add_argument("--no-bleach", action="store_true", help="Disable bleach correction")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory for figures (defaults to input folder)")
    p.add_argument("--save", action="store_true", help="Save figures instead of showing interactively")
    p.add_argument("--use-all-trials", action="store_true", help="Recut using all trials (event x trial snippets) for a cleaner median")
    p.add_argument("--single-event-window", action="store_true", help="Truncate post window to before next stimulus (use guard-ms margin)")
    p.add_argument("--guard-ms", type=float, default=5.0, help="Safety margin (ms) before the next stimulus when using single-event-window")
    p.add_argument("--aggregate", action="store_true", help="Aggregate across all files/folders into a single figure")
    p.add_argument("--aggregate-overlay", action="store_true", help="Overlay all selected traces and draw the average on top; do not save unless --save is explicitly set")
    p.add_argument("--sample-hz", type=float, default=1000.0, help="Resampling rate for aggregation (Hz)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    ap = _build_argparser()
    args = ap.parse_args(argv)

    normalize_dff = not args.no_dff
    bleach = not args.no_bleach

    saved = process_inputs(
        args.inputs,
        train_start=args.train_start,
        isi=args.isi,
        n_pulses=args.n_pulses,
        events_spec=args.events,
        align_by_peak=args.align_by_peak,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        normalize_dff=normalize_dff,
        bleach=bleach,
    use_all_trials=args.use_all_trials,
    single_event_window=args.single_event_window,
    guard_ms=args.guard_ms,
        out_dir=args.out_dir,
    aggregate=args.aggregate,
    aggregate_overlay=args.aggregate_overlay,
    sample_hz=args.sample_hz,
        save=args.save,
    )

    if args.save and saved:
        print(f"Saved {len(saved)} figure(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

