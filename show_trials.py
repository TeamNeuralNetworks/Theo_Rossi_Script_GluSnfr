"""
Overlay individual raw trial traces grouped by A1 class.

This script does not re-run extract_metrics. It uses summary_trials.xlsx to
select trial IDs and classes, then loads the matching trial columns directly
from the original recording .xlsx files. It may apply the same lightweight
preprocessing used upstream for trace export: NaN fill, bleach correction,
and dF/F0 normalization. It does not run NNLS or kinetic fitting.

Notebook one-liner example:

    from show_trials import show_trial_overlays
    show_trial_overlays()
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Feature_extraction.extract_metrics import apply_bleach_correction, fill_nans_timewise


DATA_ROOT = Path(r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL")
SUMMARY_TRIALS_PATH = DATA_ROOT / "summary_trials.xlsx"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "output"

DEFAULT_ISI = 0.05
DEFAULT_BASELINE = 0.998

WT_2_5_20HZ_CONDITIONS = [
    "WT_Theo",
    "WT_Theo_1scd",
    "WT_Anthime",
    "Stability_Before",
    "Stability_Before_05",
]
CA_1_5_CONDITIONS = ["Theo_1_5Ca", "Theo_1_5_50Hz"]
CA_2_5_CONDITIONS = WT_2_5_20HZ_CONDITIONS + ["Theo_2_5_50Hz"]
CA_4_CONDITIONS = ["Theo_4Ca", "Theo_4_50Hz"]
ALL_Q_CONDITIONS = CA_1_5_CONDITIONS + CA_2_5_CONDITIONS + CA_4_CONDITIONS
BASELINE_BY_CONDITION = {
    "Stability_Before_05": 0.498,
    "Stability_After_05": 0.498,
    "Theo_4Ca": 0.498,
    "Theo_1_5Ca": 0.498,
    "WT_Theo": 0.498,
    "Theo_1_5_50Hz": 0.498,
    "Theo_2_5_50Hz": 0.498,
    "Theo_4_50Hz": 0.498,
}
ISI_BY_CONDITION = {
    "Theo_1_5_50Hz": 0.02,
    "Theo_2_5_50Hz": 0.02,
    "Theo_4_50Hz": 0.02,
}


def load_trials_table(xlsx_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    time_raw = pd.to_numeric(df.iloc[:, -1], errors="coerce").to_numpy(float)
    all_before_time = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    has_avg_col = False
    if all_before_time.shape[1] >= 2:
        candidate_avg = all_before_time[:, -1]
        preceding = all_before_time[:, :-1]
        computed_avg = np.nanmean(preceding, axis=1)
        finite = np.isfinite(candidate_avg) & np.isfinite(computed_avg)
        if finite.sum() > 10:
            corr = np.corrcoef(candidate_avg[finite], computed_avg[finite])[0, 1]
            if corr > 0.99:
                has_avg_col = True

    trials = all_before_time[:, :-1] if has_avg_col else all_before_time
    valid = np.isfinite(time_raw)
    return time_raw[valid], trials[valid, :]


def resolve_raw_recording_path(data_root: Path, condition: str, file_stem: str) -> Path:
    folder = data_root / condition
    direct = folder / f"{file_stem}.xlsx"
    if direct.exists():
        return direct

    candidates = sorted(folder.glob(f"{file_stem}*.xlsx"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        exact_stem = [path for path in candidates if path.stem == file_stem]
        if len(exact_stem) == 1:
            return exact_stem[0]
    raise FileNotFoundError(f"Could not resolve raw recording for condition={condition!r}, file={file_stem!r}")


def compute_global_q(summary_trials: pd.DataFrame, q_mode: str = "median") -> float:
    q_df = summary_trials[summary_trials["condition"].isin(ALL_Q_CONDITIONS)].copy()
    q_df["status_norm"] = q_df["status"].astype(str).str.strip().str.lower()
    q_df = q_df[(q_df["condition"].isin(CA_1_5_CONDITIONS)) & (q_df["status_norm"] == "success")].copy()
    q_vals = pd.to_numeric(q_df["AMP1_UNCORR"], errors="coerce").to_numpy(float)
    q_vals = q_vals[np.isfinite(q_vals) & (q_vals > 0)]
    if q_vals.size == 0:
        raise RuntimeError("No valid 1.5 mM AMP1_UNCORR success values found to estimate Global_Q.")
    if str(q_mode).strip().lower() == "mean":
        return float(np.nanmean(q_vals))
    return float(np.nanmedian(q_vals))


def classify_a1_trials(
    summary_trials: pd.DataFrame,
    global_q: float,
    conditions: list[str],
    max_quanta: int | None = None,
) -> pd.DataFrame:
    df = summary_trials[summary_trials["condition"].isin(conditions)].copy()
    df["status_norm"] = df["status"].astype(str).str.strip().str.lower()
    df["AMP1_UNCORR_num"] = pd.to_numeric(df["AMP1_UNCORR"], errors="coerce")
    df["trial_input_col_1based"] = pd.to_numeric(df["trial_input_col_1based"], errors="coerce").astype("Int64")

    labels = []
    q_counts = []
    for _, row in df.iterrows():
        status_norm = row["status_norm"]
        amp1 = row["AMP1_UNCORR_num"]
        if status_norm == "failure":
            labels.append("failure")
            q_counts.append(0)
            continue
        if not np.isfinite(amp1):
            labels.append("unclassified")
            q_counts.append(np.nan)
            continue
        q_count = max(0, int(round(float(amp1) / float(global_q))))
        if max_quanta is not None:
            q_count = min(q_count, int(max_quanta))
        labels.append(f"{q_count}Q")
        q_counts.append(q_count)

    df["a1_q_count"] = q_counts
    df["a1_class"] = labels
    return df


def get_condition_baseline(condition: str) -> float:
    return float(BASELINE_BY_CONDITION.get(condition, DEFAULT_BASELINE))


def get_condition_isi(condition: str) -> float:
    return float(ISI_BY_CONDITION.get(condition, DEFAULT_ISI))


def preprocess_trials(
    time_s: np.ndarray,
    trials: np.ndarray,
    *,
    train_start: float,
    isi: float,
    bleach: bool,
    normalize_dff: bool,
) -> np.ndarray:
    baseline_mask = np.asarray(time_s, float) < float(train_start)
    if baseline_mask.sum() < 5:
        raise ValueError(
            f"Fewer than 5 baseline samples before train_start={float(train_start):.6f} s."
        )

    processed = np.zeros_like(trials, dtype=float)
    stim_times = float(train_start) + float(isi) * np.arange(10, dtype=float)
    peak_window_ms = max(5.0, float(isi) * 1000.0 - 2.0)
    for idx in range(trials.shape[1]):
        yj = fill_nans_timewise(np.asarray(trials[:, idx], float), np.asarray(time_s, float))
        if bleach:
            yj = apply_bleach_correction(
                np.asarray(time_s, float),
                yj,
                float(train_start),
                stim_times,
                post_zoom_s=0.30,
                peak_window_ms=peak_window_ms,
                pre_peak_ms=1.0,
                huber_delta=3.0,
                tau_range_factor=(0.25, 4.0),
                n_tau=25,
            )

        base = yj[baseline_mask]
        base = base[np.isfinite(base)]
        if not base.size:
            raise ValueError(f"Trial column {idx}: no finite values in baseline window.")
        f0 = float(np.nanmedian(base))
        if normalize_dff:
            if abs(f0) < 1e-9:
                raise ValueError(f"Trial column {idx}: F0 is zero or near-zero.")
            yj = (yj - f0) / f0
        else:
            yj = yj - f0
        processed[:, idx] = yj
    return processed


def load_raw_trials_for_file(
    data_root: Path,
    condition: str,
    file_stem: str,
    *,
    bleach: bool,
    normalize_dff: bool,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    xlsx_path = resolve_raw_recording_path(data_root, condition, file_stem)
    time_s, trials = load_trials_table(xlsx_path)
    train_start = get_condition_baseline(condition)
    isi = get_condition_isi(condition)
    processed_trials = preprocess_trials(
        np.asarray(time_s, float),
        np.asarray(trials, float),
        train_start=train_start,
        isi=isi,
        bleach=bleach,
        normalize_dff=normalize_dff,
    )
    aligned_time = np.asarray(time_s, float) - train_start + 1.0
    traces_by_trial = {
        int(idx + 1): np.asarray(processed_trials[:, idx], float)
        for idx in range(processed_trials.shape[1])
    }
    return aligned_time, traces_by_trial


def interpolate_trace(time_vals: np.ndarray, trace_vals: np.ndarray, common_time: np.ndarray) -> np.ndarray:
    keep = np.isfinite(time_vals) & np.isfinite(trace_vals)
    if keep.sum() < 2:
        return np.full(common_time.shape, np.nan, float)
    return np.interp(common_time, time_vals[keep], trace_vals[keep], left=np.nan, right=np.nan)


def gather_classified_traces(
    classified_trials: pd.DataFrame,
    data_root: Path,
    xlim: tuple[float, float],
    *,
    bleach: bool,
    normalize_dff: bool,
) -> dict[str, dict[str, object]]:
    per_class = defaultdict(list)
    cache: dict[tuple[str, str], tuple[np.ndarray, dict[int, np.ndarray]]] = {}

    for (condition, file_stem), group in classified_trials.groupby(["condition", "file"], sort=True):
        key = (str(condition), str(file_stem))
        if key not in cache:
            cache[key] = load_raw_trials_for_file(
                data_root,
                key[0],
                key[1],
                bleach=bleach,
                normalize_dff=normalize_dff,
            )
        aligned_time, traces_by_trial = cache[key]

        for _, row in group.iterrows():
            trial_id = row["trial_input_col_1based"]
            if pd.isna(trial_id):
                continue
            trial_id = int(trial_id)
            if trial_id not in traces_by_trial:
                continue
            trace_vals = np.asarray(traces_by_trial[trial_id], float)
            per_class[str(row["a1_class"])].append(
                {
                    "condition": key[0],
                    "file": key[1],
                    "trial_input_col_1based": trial_id,
                    "time": aligned_time,
                    "trace": trace_vals,
                }
            )

    out = {}
    for label, items in per_class.items():
        if not items:
            continue
        dt_candidates = []
        for item in items:
            t = np.asarray(item["time"], float)
            keep = np.isfinite(t)
            if keep.sum() >= 2:
                dt_candidates.append(np.nanmedian(np.diff(t[keep])))
        dt = float(np.nanmedian(dt_candidates)) if dt_candidates else 0.001
        common_time = np.arange(float(xlim[0]), float(xlim[1]) + dt * 0.5, dt, dtype=float)
        matrix = []
        for item in items:
            matrix.append(interpolate_trace(np.asarray(item["time"], float), np.asarray(item["trace"], float), common_time))
        matrix_arr = np.asarray(matrix, float)
        mean_trace = np.nanmean(matrix_arr, axis=0) if matrix_arr.size else np.full(common_time.shape, np.nan, float)
        out[label] = {
            "time": common_time,
            "matrix": matrix_arr,
            "mean": mean_trace,
            "items": items,
        }
    return out


def plot_class_overlay(
    label: str,
    class_payload: dict[str, object],
    *,
    xlim: tuple[float, float],
    alpha: float,
    line_width: float,
    mean_line_width: float,
    save_path: Path | None,
    show: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    time_vals = np.asarray(class_payload["time"], float)
    matrix = np.asarray(class_payload["matrix"], float)
    mean_trace = np.asarray(class_payload["mean"], float)

    for row_vals in matrix:
        keep = np.isfinite(time_vals) & np.isfinite(row_vals)
        if np.any(keep):
            ax.plot(time_vals[keep], row_vals[keep], color="0.65", alpha=alpha, lw=line_width)

    mean_keep = np.isfinite(time_vals) & np.isfinite(mean_trace)
    if np.any(mean_keep):
        ax.plot(time_vals[mean_keep], mean_trace[mean_keep], color="black", lw=mean_line_width, label="Mean")

    ax.axhline(0, color="0.75", lw=0.8, ls=":")
    ax.set_xlim(xlim)
    ax.set_xlabel("Aligned time (s)")
    ax.set_ylabel("dF/F")
    ax.set_title(f"A1 class: {label} (n={matrix.shape[0]} trials)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def show_trial_overlays(
    *,
    data_root: Path | str = DATA_ROOT,
    summary_trials_path: Path | str = SUMMARY_TRIALS_PATH,
    conditions: list[str] | None = None,
    classes: list[str] | None = None,
    xlim: tuple[float, float] = (0.9, 1.2),
    q_mode: str = "median",
    q_value: float | None = None,
    max_quanta: int | None = None,
    save_dir: Path | str | None = DEFAULT_OUTPUT_DIR,
    show: bool = True,
    alpha: float = 0.20,
    line_width: float = 0.8,
    mean_line_width: float = 1.7,
    bleach: bool = True,
    normalize_dff: bool = True,
) -> dict[str, dict[str, object]]:
    data_root = Path(data_root)
    summary_trials_path = Path(summary_trials_path)
    save_dir = Path(save_dir) if save_dir is not None else None
    if conditions is None:
        conditions = list(WT_2_5_20HZ_CONDITIONS)

    summary_trials = pd.read_excel(summary_trials_path)
    global_q = float(q_value) if q_value is not None else compute_global_q(summary_trials, q_mode=q_mode)
    classified_trials = classify_a1_trials(summary_trials, global_q=global_q, conditions=list(conditions), max_quanta=max_quanta)

    if classes is not None:
        classes_keep = {str(label) for label in classes}
        classified_trials = classified_trials[classified_trials["a1_class"].isin(classes_keep)].copy()

    payloads = gather_classified_traces(
        classified_trials,
        data_root=data_root,
        xlim=xlim,
        bleach=bleach,
        normalize_dff=normalize_dff,
    )
    ordered_labels = []
    if "failure" in payloads:
        ordered_labels.append("failure")
    ordered_labels.extend(sorted([label for label in payloads if label != "failure"], key=lambda s: (999 if not s.endswith("Q") else int(s[:-1]), s)))

    for label in ordered_labels:
        save_path = None
        if save_dir is not None:
            cond_tag = "_".join(list(conditions)) if len(conditions) <= 3 else "multi_condition"
            save_name = f"show_trials__{cond_tag}__{label}.png"
            save_path = save_dir / save_name
        plot_class_overlay(
            label,
            payloads[label],
            xlim=xlim,
            alpha=alpha,
            line_width=line_width,
            mean_line_width=mean_line_width,
            save_path=save_path,
            show=show,
        )

    print(f"Global_Q={global_q:.4f} ({'override' if q_value is not None else q_mode})")
    print(f"preprocessing: bleach={bool(bleach)} normalize_dff={bool(normalize_dff)}")
    for label in ordered_labels:
        print(f"{label}: {len(payloads[label]['items'])} trials")
    return payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay raw individual trial traces grouped by A1 class.")
    parser.add_argument("--summary-trials", default=str(SUMMARY_TRIALS_PATH))
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--conditions", nargs="*", default=list(WT_2_5_20HZ_CONDITIONS))
    parser.add_argument("--classes", nargs="*", default=None, help="Subset of labels to plot, e.g. failure 1Q 2Q")
    parser.add_argument("--xlim", nargs=2, type=float, default=(0.9, 1.2))
    parser.add_argument("--q-mode", choices=("median", "mean"), default="median")
    parser.add_argument("--q-value", type=float, default=None)
    parser.add_argument("--max-quanta", type=int, default=None)
    parser.add_argument("--save-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-bleach", action="store_true")
    parser.add_argument("--no-dff", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_trial_overlays(
        data_root=args.data_root,
        summary_trials_path=args.summary_trials,
        conditions=list(args.conditions),
        classes=args.classes,
        xlim=(float(args.xlim[0]), float(args.xlim[1])),
        q_mode=args.q_mode,
        q_value=args.q_value,
        max_quanta=args.max_quanta,
        save_dir=args.save_dir,
        show=not args.no_show,
        bleach=not args.no_bleach,
        normalize_dff=not args.no_dff,
    )


if __name__ == "__main__":
    main()
