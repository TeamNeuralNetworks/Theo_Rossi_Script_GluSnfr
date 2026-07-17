---
name: "Support Figure Notebook Refactor Context"
description: "Use when editing Theo_Rossi_Script_GluSnfr Support_figure.ipynb, figure helpers, plotting workflows, dataframe statistics, PCA panels, trace plots, profile plots, or notebook helper extraction."
---
# Support_figure.ipynb Context

The notebook already has a strong helper layer. Extend existing helpers rather than replacing them with parallel abstractions.

## Prefer Existing Helpers

- Profile helpers: `plot_profile_group_comparison`, `make_profile_group`, `plot_profile_analysis`, `summarize_profile_group_stats`.
- Boxplot helpers: `make_box_group`, `plot_boxplot_analysis`, `_boxplot_stat_summary`.
- PCA helpers: `plot_pca_background`, `plot_pca_overlay_points`, `plot_pca_analysis`, `style_pca_axes`.
- Trace helpers: `plot_traces`, `select_traces`, `compute_trace_stats`.
- Figure helpers: `make_figure_grid`, `_apply_clean_axes_style`, `finalize_figure`, `sanitize_figure_text`, `apply_external_legend`.
- Export helpers: `write_excel_sheets`, `_permission_retry_path`.

## Highest-Payoff Refactor Targets

- Extract repeated statistics wrappers: dataframe filter, numeric column extraction, missing-value removal, optional paired alignment, test execution, p-value/stat summary formatting.
- Keep statistics APIs explicit: do not hide whether a comparison is paired, unpaired, parametric, or nonparametric.
- Replace repeated globals guards with one helper accepting required global names and a context label.
- Route remaining direct `plt.subplots()` blocks through `make_figure_grid()` and `finalize_figure()` unless the layout is genuinely custom.
- For profile panels, prefer `make_profile_group` and `plot_profile_analysis` when they preserve clarity.

## Avoid

- Do not change scientific logic, figure design, saved output names, or already-good helper APIs just for style.
- Do not over-abstract calcium/frequency filtering; `filter_df_by_calcium()` and `get_calcium_conditions()` already cover that.
- Keep domain-specific columns such as `AMP1`, `PPR2/1`, and `FailRate1` explicit.
- Do not modify notebook outputs unless explicitly requested.
