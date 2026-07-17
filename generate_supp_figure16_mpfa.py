import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL")
OUTPUT_DIR = BASE_DIR / "output"
MIN_TRIALS = 5
N_BOOT = 1200
MLE_CAP = 20
QUANTAL_Q = 0.48
SEED = 16016
CALCIUM_COLORS = {"1.5 mM": "#3690d8", "2.5 mM": "#555555", "4 mM": "#d83a3a"}
CONDITIONS = {
    (20, "1.5 mM"): ["Theo_1_5Ca"],
    (20, "2.5 mM"): ["WT_Theo", "WT_Theo_1scd", "WT_Anthime", "Stability_Before", "Stability_Before_05"],
    (20, "4 mM"): ["Theo_4Ca"],
    (50, "1.5 mM"): ["Theo_1_5_50Hz"],
    (50, "2.5 mM"): ["Theo_2_5_50Hz"],
    (50, "4 mM"): ["Theo_4_50Hz"],
}


def base_id(value):
    value = str(value).strip()
    value = re.sub(r"\.xlsx?$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"_traces_converted$", "", value)
    value = re.sub(r"_\d+Hz_?|_\d+pulses_?|_\d+\.?\d*mMCa_?", "_", value, flags=re.IGNORECASE)
    value = re.sub(r"_set\d+", "", value, flags=re.IGNORECASE)
    return re.sub(r"_+", "_", value).strip("_ ")


def fiber_id(value):
    return re.sub(r"(?i)_bouton.*$", "", str(value))


def null_moments(value):
    try:
        values = np.asarray(json.loads(value), dtype=float)
        values = values[np.isfinite(values)]
    except (TypeError, ValueError, json.JSONDecodeError):
        values = np.array([], dtype=float)
    if values.size < 2:
        return pd.Series({"null_mean": np.nan, "null_var": np.nan})
    return pd.Series({"null_mean": float(values.mean()), "null_var": float(values.var(ddof=1))})


def load_trials():
    trials = pd.read_excel(BASE_DIR / "summary_trials.xlsx")
    null = pd.read_excel(BASE_DIR / "summary_trials_nnls_null.xlsx")
    keys = ["condition", "file", "trial", "trial_input_col_1based"]
    null[["null_mean", "null_var"]] = null["nnls_null_amps_json"].apply(null_moments)
    trials = trials.merge(null[keys + ["null_mean", "null_var"]], on=keys, how="left", validate="one_to_one")
    trials["BaseID"] = trials["file"].map(base_id)
    trials["FiberID"] = trials["BaseID"].map(fiber_id)
    for event in range(1, 11):
        unc = pd.to_numeric(trials[f"AMP{event}_UNCORR"], errors="coerce")
        if event == 1:
            amplitude = unc
        else:
            corr = pd.to_numeric(trials[f"AMP{event}_CORR"], errors="coerce")
            amplitude = pd.concat([corr, unc], axis=1).max(axis=1, skipna=True)
        trials[f"A{event}"] = amplitude - trials["null_mean"]
    return trials


def moments(group, event):
    values = pd.to_numeric(group[f"A{event}"], errors="coerce")
    valid = np.isfinite(values.to_numpy(dtype=float))
    values = values.loc[valid].to_numpy(dtype=float)
    if values.size < MIN_TRIALS:
        return None
    null_var = pd.to_numeric(group.loc[valid, "null_var"], errors="coerce").to_numpy(dtype=float)
    noise = float(np.nanmedian(null_var)) if np.isfinite(null_var).any() else 0.0
    mean = float(values.mean())
    observed_variance = float(values.var(ddof=1))
    variance = max(observed_variance - max(noise, 0.0), 0.0)
    if not np.isfinite(mean) or mean <= 0:
        return None
    return {
        "mean": mean,
        "variance": variance,
        "observed_variance": observed_variance,
        "noise_variance": noise,
        "trials": int(values.size),
    }


def bouton_table(trials, frequency, calcium):
    source = trials[trials["condition"].isin(CONDITIONS[(frequency, calcium)])]
    rows = []
    for bouton_id, bouton in source.groupby("BaseID", sort=False):
        row = {
            "Frequency_Hz": frequency,
            "Calcium": calcium,
            "BaseID": bouton_id,
            "FiberID": bouton["FiberID"].iloc[0],
        }
        complete = True
        for event in range(1, 11):
            result = moments(bouton, event)
            if result is None:
                complete = False
                break
            for key, value in result.items():
                row[f"{key}_A{event}"] = value
        if complete:
            rows.append(row)
    return pd.DataFrame(rows)


def normalized_trajectory(table):
    means = np.array([table[f"mean_A{event}"].mean() for event in range(1, 11)], dtype=float)
    variances = np.array([table[f"variance_A{event}"].mean() for event in range(1, 11)], dtype=float)
    fano = variances / means
    mean_percent = 100.0 * means / means[0]
    fano_percent = 100.0 * fano / fano[0]
    return means, variances, fano, mean_percent, fano_percent


def resample_by_fiber(table, rng):
    fibers = table["FiberID"].astype(str).to_numpy()
    unique = np.unique(fibers)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    indices = np.concatenate([np.flatnonzero(fibers == fiber) for fiber in sampled])
    return table.iloc[indices].reset_index(drop=True)


def bootstrap_trajectory(table, frequency, calcium_index):
    rng = np.random.default_rng(SEED + frequency + calcium_index)
    x_boot = np.full((N_BOOT, 10), np.nan)
    y_boot = np.full((N_BOOT, 10), np.nan)
    for sample in range(N_BOOT):
        sampled = resample_by_fiber(table, rng)
        _, _, _, x_boot[sample], y_boot[sample] = normalized_trajectory(sampled)
    return x_boot, y_boot


def estimate_q(trials):
    low_conditions = CONDITIONS[(20, "1.5 mM")] + CONDITIONS[(50, "1.5 mM")]
    low = trials[trials["condition"].isin(low_conditions)].copy()
    success = low["status"].astype(str).str.strip().str.lower().eq("success")
    estimates = {}
    for bouton_id, group in low.loc[success].groupby("BaseID"):
        values = pd.to_numeric(group["AMP1_UNCORR"], errors="coerce").dropna()
        if len(values) >= MIN_TRIALS:
            estimates[bouton_id] = float(values.median())
    return estimates, float(np.median(list(estimates.values())))


def trial_count(row, q_value, event=1):
    threshold = pd.to_numeric(pd.Series([row.get("thr_shared", np.nan)]), errors="coerce").iloc[0]
    unc = pd.to_numeric(pd.Series([row.get(f"AMP{event}_UNCORR", np.nan)]), errors="coerce").iloc[0]
    corr = unc if event == 1 else pd.to_numeric(
        pd.Series([row.get(f"AMP{event}_CORR", np.nan)]), errors="coerce"
    ).iloc[0]
    if not np.isfinite(threshold) or not np.isfinite(unc) or not np.isfinite(corr):
        return np.nan
    if min(unc, corr) < threshold:
        return 0.0
    amplitude = unc if event == 1 else max(unc, corr)
    return float(max(1, int(round(amplitude / q_value))))


def fit_binomial_mle(counts):
    counts = np.asarray(counts, dtype=int)
    if counts.size < 3 or counts.mean() <= 0:
        return np.nan, np.nan
    n_low = max(int(counts.max()), 1)
    best_n, best_p, best_ll = np.nan, np.nan, -np.inf
    for n_value in range(n_low, MLE_CAP + 1):
        p_value = float(counts.mean() / n_value)
        if not 0 < p_value < 1:
            continue
        log_likelihood = 0.0
        for count in counts:
            log_likelihood += (
                math.lgamma(n_value + 1)
                - math.lgamma(count + 1)
                - math.lgamma(n_value - count + 1)
                + count * math.log(p_value)
                + (n_value - count) * math.log1p(-p_value)
            )
        if log_likelihood > best_ll:
            best_n, best_p, best_ll = float(n_value), p_value, log_likelihood
    return best_n, best_p


def baseline_release_probability(trials, table, frequency, calcium, q_by_bouton, global_q):
    rows = []
    conditions = CONDITIONS[(frequency, calcium)]
    for _, bouton in table.iterrows():
        bouton_id = bouton["BaseID"]
        q_value = q_by_bouton.get(bouton_id, global_q)
        group = trials[(trials["BaseID"] == bouton_id) & (trials["condition"].isin(conditions))]
        counts = np.array([trial_count(row, q_value) for _, row in group.iterrows()], dtype=float)
        counts = counts[np.isfinite(counts)].astype(int)
        n_value, p_value = fit_binomial_mle(counts)
        rows.append({
            "Frequency_Hz": frequency,
            "Calcium": calcium,
            "BaseID": bouton_id,
            "FiberID": bouton["FiberID"],
            "Q": q_value,
            "MLE_N_A1": n_value,
            "MLE_P_A1": p_value,
            "At_cap_20": bool(n_value == MLE_CAP),
        })
    estimates = pd.DataFrame(rows)
    usable = estimates[np.isfinite(estimates["MLE_P_A1"]) & (~estimates["At_cap_20"])]
    p1 = float(usable["MLE_P_A1"].median()) if not usable.empty else np.nan
    return p1, estimates


def pca_reference_table():
    features = ["AMP1", "AMP2", "%Fail1", "%Fail2"] + [f"PPR{k}/1" for k in range(2, 11)]
    reference = pd.read_excel(OUTPUT_DIR / "PCA_Data_WT_Pooled_clustered_4.xlsx")
    values = reference[features].apply(pd.to_numeric, errors="coerce")
    valid = np.all(np.isfinite(values.to_numpy(dtype=float)), axis=1)
    reference = reference.loc[valid].reset_index(drop=True)
    scaled = StandardScaler().fit_transform(values.loc[valid])
    coordinates = PCA(n_components=2).fit_transform(scaled)
    reference["BaseID"] = reference["ID"].map(base_id)
    reference["PrC1"] = coordinates[:, 0]
    reference["PrC2"] = coordinates[:, 1]
    return reference[["ID", "BaseID", "FiberID", "PrC1", "PrC2"]]


def bootstrap_mpfa_maps(trials, q_value=QUANTAL_Q, n_boot=200):
    reference = pca_reference_table()
    source = trials[trials["condition"].isin(CONDITIONS[(20, "2.5 mM")])]
    rows = []
    for row_index, reference_row in reference.iterrows():
        bouton_id = reference_row["BaseID"]
        group = source[source["BaseID"] == bouton_id]
        output = reference_row.to_dict()
        output["Q"] = q_value
        output["Trials"] = len(group)
        rng = np.random.default_rng(SEED + int(row_index))
        for event in range(1, 11):
            amplitudes = pd.to_numeric(group[f"A{event}"], errors="coerce").to_numpy(dtype=float)
            null_variance = pd.to_numeric(group["null_var"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(amplitudes)
            amplitudes = amplitudes[valid]
            null_variance = null_variance[valid]
            noise = float(np.nanmedian(null_variance)) if np.isfinite(null_variance).any() else 0.0
            boot_n = []
            boot_p = []
            if len(amplitudes) >= MIN_TRIALS:
                point_mean = float(amplitudes.mean())
                point_observed_variance = float(amplitudes.var(ddof=1))
                point_corrected_variance = max(point_observed_variance - max(noise, 0.0), 0.0)
                corrected_denominator = q_value * point_mean - point_corrected_variance
                observed_denominator = q_value * point_mean - point_observed_variance
                point_n = point_mean * point_mean / corrected_denominator if corrected_denominator > 0 else np.nan
                point_p = point_mean / (q_value * point_n) if np.isfinite(point_n) else np.nan
                observed_n = point_mean * point_mean / observed_denominator if observed_denominator > 0 else np.nan
                observed_p = point_mean / (q_value * observed_n) if np.isfinite(observed_n) else np.nan
                for _ in range(n_boot):
                    sample = amplitudes[rng.integers(0, len(amplitudes), len(amplitudes))]
                    mean = float(sample.mean())
                    variance = max(float(sample.var(ddof=1)) - max(noise, 0.0), 0.0)
                    denominator = q_value * mean - variance
                    if mean > 0 and denominator > 0:
                        n_estimate = mean * mean / denominator
                        p_estimate = mean / (q_value * n_estimate)
                        if np.isfinite(n_estimate) and n_estimate > 0 and 0 <= p_estimate <= 1:
                            boot_n.append(n_estimate)
                            boot_p.append(p_estimate)
            else:
                point_mean = point_observed_variance = point_corrected_variance = np.nan
                point_n = point_p = observed_n = observed_p = np.nan
            output[f"Mean_A{event}"] = point_mean
            output[f"Observed_variance_A{event}"] = point_observed_variance
            output[f"Noise_corrected_variance_A{event}"] = point_corrected_variance
            output[f"N_A{event}"] = point_n
            output[f"P_A{event}"] = point_p
            output[f"N_observed_variance_A{event}"] = observed_n
            output[f"P_observed_variance_A{event}"] = observed_p
            output[f"Bootstrap_median_N_A{event}"] = float(np.median(boot_n)) if boot_n else np.nan
            output[f"Bootstrap_median_P_A{event}"] = float(np.median(boot_p)) if boot_p else np.nan
            output[f"Valid_bootstrap_fraction_A{event}"] = len(boot_n) / n_boot
        rows.append(output)
    return pd.DataFrame(rows)


def smooth_pca_field(coordinates, values, grid_x, grid_y):
    valid = np.isfinite(values)
    points = coordinates[valid]
    data = values[valid]
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    if len(points) < 3:
        return np.full(mesh_x.shape, np.nan)
    low, high = np.percentile(data, [5, 95])
    data = np.clip(data, low, high)
    pairwise = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(pairwise, np.inf)
    sigma = float(np.median(np.min(pairwise, axis=1)) * 2.0)
    grid_points = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    distances = np.sqrt(np.sum((grid_points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
    weight_sum = weights.sum(axis=1)
    supported = np.min(distances, axis=1) <= 3.0 * sigma
    result = np.full(len(grid_points), np.nan)
    use = supported & (weight_sum > 1e-12)
    result[use] = (weights[use] * data).sum(axis=1) / weight_sum[use]
    return result.reshape(mesh_x.shape)


def plot_mpfa_pca_maps(map_table):
    events = (1, 2, 5, 10)
    coordinates = map_table[["PrC1", "PrC2"]].to_numpy(dtype=float)
    pad = 0.5
    extent = [coordinates[:, 0].min() - pad, coordinates[:, 0].max() + pad,
              coordinates[:, 1].min() - pad, coordinates[:, 1].max() + pad]
    grid_x = np.linspace(extent[0], extent[1], 80)
    grid_y = np.linspace(extent[2], extent[3], 80)
    figure, axes = plt.subplots(2, 4, figsize=(14.2, 6.6), sharex=True, sharey=True, facecolor="white")
    figure.patch.set_facecolor("white")
    colorbar_axes = [
        figure.add_axes([0.925, 0.545, 0.012, 0.285]),
        figure.add_axes([0.925, 0.125, 0.012, 0.285]),
    ]
    specifications = [("N", "Spectral_r", 1.0, 8.0), ("P", "coolwarm", 0.0, 1.0)]
    for row, (parameter, cmap, vmin, vmax) in enumerate(specifications):
        for column, event in enumerate(events):
            ax = axes[row, column]
            values = map_table[f"{parameter}_A{event}"].to_numpy(dtype=float)
            valid = np.isfinite(values)
            field = smooth_pca_field(coordinates, values, grid_x, grid_y)
            ax.imshow(field, extent=extent, origin="lower", aspect="auto", cmap=cmap,
                      vmin=vmin, vmax=vmax, interpolation="bilinear", alpha=0.70)
            ax.scatter(coordinates[valid, 0], coordinates[valid, 1], c=values[valid], cmap=cmap,
                       vmin=vmin, vmax=vmax, s=17, edgecolors="none", zorder=3)
            ax.set_title(f"{parameter}: A{event} (n={valid.sum()})", fontsize=9)
            ax.set_aspect("equal", adjustable="box")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 1:
                ax.set_xlabel("PrC1")
            if column == 0:
                ax.set_ylabel("PrC2")
        scalar = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap)
        scalar.set_array([])
        figure.colorbar(scalar, cax=colorbar_axes[row], label=parameter)
    figure.suptitle(
        f"20 Hz, 2.5 mM Ca²⁺ — per-bouton MPFA moment maps in PCA space (q={QUANTAL_Q:.2f})",
        fontsize=12, fontweight="bold",
    )
    figure.subplots_adjust(left=0.06, right=0.90, bottom=0.10, top=0.88, wspace=0.16, hspace=0.24)
    pdf = OUTPUT_DIR / "FigS16b_2p5mM_20Hz_NP_MPFA_PCA_maps.pdf"
    png = OUTPUT_DIR / "FigS16b_2p5mM_20Hz_NP_MPFA_PCA_maps.png"
    figure.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white", transparent=False)
    plt.show()
    return figure, pdf, png


def plot_mpfa_sensitivity_maps(map_table):
    events = (1, 2, 5, 10)
    coordinates = map_table[["PrC1", "PrC2"]].to_numpy(dtype=float)
    pad = 0.5
    extent = [coordinates[:, 0].min() - pad, coordinates[:, 0].max() + pad,
              coordinates[:, 1].min() - pad, coordinates[:, 1].max() + pad]
    grid_x = np.linspace(extent[0], extent[1], 80)
    grid_y = np.linspace(extent[2], extent[3], 80)
    figure, axes = plt.subplots(2, 4, figsize=(14.2, 6.6), sharex=True, sharey=True, facecolor="white")
    figure.patch.set_facecolor("white")
    colorbar_axes = [
        figure.add_axes([0.925, 0.545, 0.012, 0.285]),
        figure.add_axes([0.925, 0.125, 0.012, 0.285]),
    ]
    specifications = [
        ("P_observed_variance", "P without noise subtraction", "coolwarm", 0.0, 1.0),
        ("Valid_bootstrap_fraction", "Valid bootstrap fraction", "viridis", 0.0, 1.0),
    ]
    for row, (prefix, label, cmap, vmin, vmax) in enumerate(specifications):
        for column, event in enumerate(events):
            ax = axes[row, column]
            values = map_table[f"{prefix}_A{event}"].to_numpy(dtype=float)
            valid = np.isfinite(values)
            field = smooth_pca_field(coordinates, values, grid_x, grid_y)
            ax.imshow(field, extent=extent, origin="lower", aspect="auto", cmap=cmap,
                      vmin=vmin, vmax=vmax, interpolation="bilinear", alpha=0.70)
            ax.scatter(coordinates[valid, 0], coordinates[valid, 1], c=values[valid], cmap=cmap,
                       vmin=vmin, vmax=vmax, s=17, edgecolors="none", zorder=3)
            ax.set_title(f"A{event} (n={valid.sum()})", fontsize=9)
            ax.set_aspect("equal", adjustable="box")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 1:
                ax.set_xlabel("PrC1")
            if column == 0:
                ax.set_ylabel(f"{label}\nPrC2")
        scalar = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap)
        scalar.set_array([])
        figure.colorbar(scalar, cax=colorbar_axes[row], label=label)
    figure.suptitle(
        f"20 Hz, 2.5 mM Ca²⁺ — MPFA variance-correction sensitivity (q={QUANTAL_Q:.2f})",
        fontsize=12, fontweight="bold",
    )
    figure.subplots_adjust(left=0.075, right=0.90, bottom=0.10, top=0.88, wspace=0.16, hspace=0.24)
    pdf = OUTPUT_DIR / "FigS16c_2p5mM_20Hz_MPFA_variance_sensitivity.pdf"
    png = OUTPUT_DIR / "FigS16c_2p5mM_20Hz_MPFA_variance_sensitivity.png"
    figure.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white", transparent=False)
    plt.show()
    return figure, pdf, png


def expected_lines(p1, x_max):
    x = np.linspace(0, x_max, 400)
    q_line = x
    n_line = np.full_like(x, 100.0)
    if np.isfinite(p1) and 0 < p1 < 1:
        p_line = 100.0 * (1.0 - p1 * x / 100.0) / (1.0 - p1)
        p_line[p_line < 0] = np.nan
    else:
        p_line = np.full_like(x, np.nan)
    return x, q_line, n_line, p_line


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return np.percentile(values, [2.5, 97.5]) if values.size else np.array([np.nan, np.nan])


def run():
    trials = load_trials()
    q_by_bouton, global_q = {}, QUANTAL_Q
    tables = {}
    bootstraps = {}
    mle_tables = []
    summary_rows = []
    calcium_order = ["1.5 mM", "2.5 mM", "4 mM"]

    for frequency in (20, 50):
        for calcium_index, calcium in enumerate(calcium_order):
            table = bouton_table(trials, frequency, calcium)
            tables[(frequency, calcium)] = table
            x_boot, y_boot = bootstrap_trajectory(table, frequency, calcium_index)
            bootstraps[(frequency, calcium)] = (x_boot, y_boot)
            p1, mle = baseline_release_probability(
                trials, table, frequency, calcium, q_by_bouton, global_q
            )
            mle_tables.append(mle)
            means, variances, fano, x, y = normalized_trajectory(table)
            for event in range(1, 11):
                x_ci = interval(x_boot[:, event - 1])
                y_ci = interval(y_boot[:, event - 1])
                x_sem = float(np.nanstd(x_boot[:, event - 1], ddof=1))
                y_sem = float(np.nanstd(y_boot[:, event - 1], ddof=1))
                summary_rows.append({
                    "Frequency_Hz": frequency,
                    "Calcium": calcium,
                    "Event": event,
                    "Boutons": len(table),
                    "Fibers": table["FiberID"].nunique(),
                    "Bouton_mean_average": means[event - 1],
                    "Bouton_variance_average": variances[event - 1],
                    "Fano_of_bouton_averages": fano[event - 1],
                    "Mean_percent_A1": x[event - 1],
                    "Mean_percent_CI_low": x_ci[0],
                    "Mean_percent_CI_high": x_ci[1],
                    "Mean_percent_SEM": x_sem,
                    "Fano_percent_A1": y[event - 1],
                    "Fano_percent_CI_low": y_ci[0],
                    "Fano_percent_CI_high": y_ci[1],
                    "Fano_percent_SEM": y_sem,
                    "Median_MLE_P_A1_excluding_cap20": p1,
                })

    summary = pd.DataFrame(summary_rows)
    mle_all = pd.concat(mle_tables, ignore_index=True)
    y_max = max(220.0, float(summary["Fano_percent_CI_high"].max()) * 1.08)

    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.4), sharey=True, facecolor="white")
    figure.patch.set_facecolor("white")
    event_colors = plt.cm.viridis(np.linspace(0.05, 0.92, 10))
    for row_index, frequency in enumerate((20, 50)):
        for column_index, calcium in enumerate(calcium_order):
            ax = axes[row_index, column_index]
            condition_rows = summary[
                (summary["Frequency_Hz"] == frequency) & (summary["Calcium"] == calcium)
            ].sort_values("Event")
            p1 = float(condition_rows["Median_MLE_P_A1_excluding_cap20"].iloc[0])
            panel_x_max = max(160.0, float(condition_rows["Mean_percent_CI_high"].max()) * 1.05)
            line_x, q_line, n_line, p_line = expected_lines(p1, panel_x_max)
            ax.plot(line_x, q_line, color="0.72", lw=1.35, ls="-", label="Q change")
            ax.plot(line_x, n_line, color="0.48", lw=1.35, ls="-", label="N change")
            ax.plot(line_x, p_line, color="0.15", lw=1.45, ls="-", label="P change")

            x = condition_rows["Mean_percent_A1"].to_numpy(dtype=float)
            y = condition_rows["Fano_percent_A1"].to_numpy(dtype=float)
            x_sem = condition_rows["Mean_percent_SEM"].to_numpy(dtype=float)
            y_sem = condition_rows["Fano_percent_SEM"].to_numpy(dtype=float)
            ax.plot(x, y, color=CALCIUM_COLORS[calcium], lw=1.25, alpha=0.75, zorder=3)
            for event in range(1, 11):
                index = event - 1
                marker = "D" if event == 1 else ("s" if event == 2 else "o")
                marker_size = 7.0 if event == 1 else (6.4 if event == 2 else 5.8)
                ax.errorbar(
                    x[index], y[index],
                    xerr=x_sem[index], yerr=y_sem[index],
                    fmt=marker, ms=marker_size, color=event_colors[index], mec="0.20", mew=0.55,
                    ecolor=CALCIUM_COLORS[calcium], elinewidth=0.85, capsize=2.2, alpha=0.95, zorder=4,
                )
                ax.annotate(f"A{event}", (x[index], y[index]), xytext=(3, 3),
                            textcoords="offset points", fontsize=6.2, color=event_colors[index])

            n_boutons = int(condition_rows["Boutons"].iloc[0])
            n_fibers = int(condition_rows["Fibers"].iloc[0])
            ax.set_title(f"{frequency} Hz — {calcium} Ca²⁺\n{n_boutons} boutons, {n_fibers} fibers; A1 P̂={p1:.2f}", fontsize=9.5)
            ax.set_xlim(0, panel_x_max)
            ax.set_ylim(0, y_max)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)
            if row_index == 1:
                ax.set_xlabel("Mean response (% of A1)")
            if column_index == 0:
                ax.set_ylabel("Variance / mean (% of A1)")
            if row_index == 0 and column_index == 0:
                ax.legend(frameon=False, fontsize=7, loc="upper right")

    figure.suptitle(
        "Fig. S16 — A1-normalized Fano–mean trajectories from A1 to A10",
        fontsize=12, fontweight="bold",
    )
    figure.text(0.5, 0.952, "Error bars show ±SEM from fiber-level bootstrap resampling",
                ha="center", va="top", fontsize=8, color="0.30")
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    pdf = OUTPUT_DIR / "FigS16_paired_calcium_MPFA.pdf"
    png = OUTPUT_DIR / "FigS16_paired_calcium_MPFA.png"
    figure.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white", transparent=False)
    plt.show()

    map_table = bootstrap_mpfa_maps(trials, q_value=QUANTAL_Q, n_boot=200)
    map_figure, map_pdf, map_png = plot_mpfa_pca_maps(map_table)
    sensitivity_figure, sensitivity_pdf, sensitivity_png = plot_mpfa_sensitivity_maps(map_table)

    workbook = OUTPUT_DIR / "FigS16_paired_calcium_MPFA_tables.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        summary.to_excel(writer, sheet_name="A1_normalized_trajectories", index=False)
        mle_all.to_excel(writer, sheet_name="A1_MLE_for_P_lines", index=False)
        map_table.to_excel(writer, sheet_name="MPFA_NP_PCA_2p5_20Hz", index=False)
        for frequency in (20, 50):
            for calcium in calcium_order:
                sheet = f"Boutons_{frequency}Hz_{calcium.split()[0].replace('.', 'p')}"
                tables[(frequency, calcium)].to_excel(writer, sheet_name=sheet, index=False)

    text_path = OUTPUT_DIR / "FigS16_paired_calcium_MPFA_summary.txt"
    lines = [
        "Fig. S16 — A1-normalized Fano–mean trajectories from A1 to A10", "",
        "For every bouton and event, trial mean and noise-corrected trial variance were estimated first.",
        "Bouton-level means and variances were then averaged; trials were never pooled across boutons.",
        "The plotted Fano estimate is mean(bouton variances) / mean(bouton means).",
        "Both axes are normalized to the corresponding A1 population estimate.",
        "Error bars are ±1 bootstrap SEM; fiber-level resampling accounts for boutons nested within fibers.", "",
        "Expected normalized trajectories:",
        "Q-only: y=x.",
        "N-only: y=100%.",
        "P-only: y=100*(1-P1*x/100)/(1-P1), using the median A1 MLE P after excluding N=20 cap fits.",
        f"Fixed quantal estimate used for the MLE P reference lines: q={QUANTAL_Q:.2f}.",
        "", "Additional 20 Hz, 2.5 mM MPFA N/P maps:",
        f"N=M²/(qM-V) and P=M/(qN), with q={QUANTAL_Q:.2f}.",
        "The primary maps show original per-bouton point estimates, not conditional bootstrap medians.",
        "Trial resampling is performed within boutons only to map the fraction of valid bootstrap solutions.",
        "The sensitivity map recalculates P from observed variance without subtracting null variance.",
        "The PCA coordinates and color ranges match the Fig. 3 WT reference maps.",
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Saved: {pdf}\nSaved: {png}\nSaved: {map_pdf}\nSaved: {map_png}\n"
        f"Saved: {sensitivity_pdf}\nSaved: {sensitivity_png}\n"
        f"Saved: {workbook}\nSaved: {text_path}"
    )
    print(summary.to_string(index=False))
    return figure, map_figure, sensitivity_figure, summary, map_table, tables, bootstraps, mle_all


if __name__ == "__main__":
    (S16_FIGURE, S16_NP_MAP_FIGURE, S16_SENSITIVITY_FIGURE, S16_SUMMARY, S16_NP_MAP_TABLE,
     S16_BOUTONS, S16_BOOTSTRAPS, S16_MLE) = run()
