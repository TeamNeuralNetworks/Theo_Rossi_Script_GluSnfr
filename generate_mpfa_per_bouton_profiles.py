"""Export MPFA train profiles after per-bouton estimation.

Run from ``Support_figure.ipynb`` after the cell that builds
``cond_results_E``. The script uses the notebook namespace so the MLE fits, Q
estimates, failure rule, and condition definitions stay aligned with figures.
"""

_required_mpfa_names = (
    'COND_DEFS_E', 'cond_results_E', 'trials_all', 'extract_base_name',
    'get_amp_row', 'N_STIM', 'OUTPUT_DIR', 'np', 'pd', 'Path',
)
_missing_mpfa_names = [name for name in _required_mpfa_names if name not in globals()]
if _missing_mpfa_names:
    raise RuntimeError(
        'Run the notebook through the multi-condition per-bouton MLE cell first; '
        f'missing: {", ".join(_missing_mpfa_names)}'
    )

MPFA_AGG_METHOD = 'mean'
MPFA_BOOTSTRAP_SAMPLES = 2000
MPFA_BOOTSTRAP_SEED = 20250316
MPFA_CONDITION_ORDER = (
    ('1.5 mM 20 Hz', '1.5mM_20Hz'),
    ('1.5 mM 50 Hz', '1.5mM_50Hz'),
    ('2.5 mM 20 Hz', '2.5mM_20Hz'),
    ('2.5 mM 50 Hz', '2.5mM_50Hz'),
    ('4.0 mM 20 Hz', '4mM_20Hz'),
    ('4.0 mM 50 Hz', '4mM_50Hz'),
)


def _mpfa_aggregate(values, axis=0):
    values = np.asarray(values, dtype=float)
    if MPFA_AGG_METHOD == 'median':
        return np.nanmedian(values, axis=axis)
    return np.nanmean(values, axis=axis)


def _mpfa_bootstrap_ci(values, rng):
    """Percentile CI from resampling the fixed bouton rows."""
    values = np.asarray(values, dtype=float)
    n_boutons = values.shape[0]
    boot = np.full((MPFA_BOOTSTRAP_SAMPLES, values.shape[1]), np.nan)
    for boot_idx in range(MPFA_BOOTSTRAP_SAMPLES):
        take = rng.integers(0, n_boutons, n_boutons)
        boot[boot_idx] = _mpfa_aggregate(values[take], axis=0)
    return np.nanpercentile(boot, [2.5, 97.5], axis=0)


def _mpfa_safe_corr(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    keep = np.isfinite(x_values) & np.isfinite(y_values)
    if keep.sum() < 3 or np.nanstd(x_values[keep]) == 0 or np.nanstd(y_values[keep]) == 0:
        return np.nan
    return float(np.corrcoef(x_values[keep], y_values[keep])[0, 1])


_mpfa_defs = {row[0]: row for row in COND_DEFS_E}
_mpfa_profiles = {'Stim': np.arange(1, int(N_STIM) + 1, dtype=int)}
_mpfa_metadata_rows = []
_mpfa_varmean_rows = []
_mpfa_condition_tables = {}
_mpfa_rng = np.random.default_rng(MPFA_BOOTSTRAP_SEED)

for _condition_label, _condition_key in MPFA_CONDITION_ORDER:
    if _condition_label not in cond_results_E or _condition_label not in _mpfa_defs:
        raise KeyError(f'Missing per-bouton MLE result for {_condition_label!r}.')

    _, _trial_conditions, _, _, _summary_name = _mpfa_defs[_condition_label]
    if _summary_name not in globals():
        raise KeyError(f'Missing summary dataframe {_summary_name!r}.')

    _summary = globals()[_summary_name].reset_index(drop=True)
    _result = cond_results_E[_condition_label]
    _N_all = np.asarray(_result['E_N'], dtype=float)
    _P_all = np.asarray(_result['E_P'], dtype=float)
    _n_rows = min(len(_summary), _N_all.shape[0], _P_all.shape[0])
    _summary = _summary.iloc[:_n_rows].copy()
    _N_all = _N_all[:_n_rows, :int(N_STIM)]
    _P_all = _P_all[:_n_rows, :int(N_STIM)]

    _trial_groups = {}
    _trial_subset = trials_all[trials_all['condition'].isin(_trial_conditions)]
    for _, _trial_row in _trial_subset.iterrows():
        _bouton_key = extract_base_name(str(_trial_row['file']).strip())
        _trial_groups.setdefault(_bouton_key, []).append(get_amp_row(_trial_row))

    _amp_all = np.full((_n_rows, int(N_STIM)), np.nan)
    _var_all = np.full_like(_amp_all, np.nan)
    _bouton_keys = []
    for _row_idx, _bouton_id in enumerate(_summary['ID'].astype(str)):
        _bouton_key = extract_base_name(_bouton_id.strip())
        _bouton_keys.append(_bouton_key)
        _trial_amps = np.asarray(_trial_groups.get(_bouton_key, []), dtype=float)
        if _trial_amps.ndim != 2 or _trial_amps.shape[1] < int(N_STIM):
            continue
        _complete_trials = np.all(np.isfinite(_trial_amps[:, :int(N_STIM)]), axis=1)
        _trial_amps = _trial_amps[_complete_trials, :int(N_STIM)]
        if _trial_amps.shape[0] < int(globals().get('MIN_TRIALS_D', 5)):
            continue
        _amp_all[_row_idx] = np.mean(_trial_amps, axis=0)
        if _trial_amps.shape[0] > 1:
            _var_all[_row_idx] = np.var(_trial_amps, axis=0, ddof=1)

    # One list for every exported series: non-boundary MLE and finite amplitude
    # estimates are required at every pulse.
    _valid = (
        np.all(np.isfinite(_N_all), axis=1)
        & np.all(np.isfinite(_P_all), axis=1)
        & np.all(np.isfinite(_amp_all), axis=1)
        & np.isfinite(_amp_all[:, 0])
        & (_amp_all[:, 0] > 0)
    )
    if not np.any(_valid):
        raise RuntimeError(f'No all-pulses-valid boutons remain for {_condition_label}.')

    _N = _N_all[_valid]
    _P = _P_all[_valid]
    _amp = _amp_all[_valid]
    _amp_var = _var_all[_valid]
    _ppr = _amp / _amp[:, [0]]
    _vmr = _amp_var / _amp
    _used_keys = np.asarray(_bouton_keys, dtype=object)[_valid]

    _N_agg = _mpfa_aggregate(_N)
    _P_agg = _mpfa_aggregate(_P)
    _ppr_agg = _mpfa_aggregate(_ppr)
    _vmr_agg = _mpfa_aggregate(_vmr)
    _N_ci = _mpfa_bootstrap_ci(_N, _mpfa_rng)
    _P_ci = _mpfa_bootstrap_ci(_P, _mpfa_rng)
    _ppr_ci = _mpfa_bootstrap_ci(_ppr, _mpfa_rng)

    _vmr_boot = np.full((MPFA_BOOTSTRAP_SAMPLES, int(N_STIM)), np.nan)
    _ppr_boot = np.full_like(_vmr_boot, np.nan)
    for _boot_idx in range(MPFA_BOOTSTRAP_SAMPLES):
        _take = _mpfa_rng.integers(0, _valid.sum(), _valid.sum())
        _vmr_boot_agg = _mpfa_aggregate(_vmr[_take], axis=0)
        _vmr_boot[_boot_idx] = _vmr_boot_agg / _vmr_boot_agg[0]
        _ppr_boot[_boot_idx] = _mpfa_aggregate(_ppr[_take], axis=0)
    _vmr_percent_ci = np.nanpercentile(_vmr_boot, [2.5, 97.5], axis=0) * 100.0
    _vmr_percent_sem = np.nanstd(_vmr_boot, axis=0, ddof=1) * 100.0
    _ppr_percent_sem = np.nanstd(_ppr_boot, axis=0, ddof=1) * 100.0

    _summary_used = _summary.loc[_valid]
    if 'FiberID' in _summary_used.columns:
        _n_fibers = int(_summary_used['FiberID'].astype(str).nunique())
    else:
        _n_fibers = int(len(_summary_used))
    _frequency = int(_condition_key.split('_')[1].replace('Hz', ''))
    _calcium = _condition_key.split('mM_')[0] + ' mM'

    _mpfa_profiles[f'Pr_{_condition_key}'] = _P_agg
    _mpfa_profiles[f'N_{_condition_key}'] = _N_agg
    _mpfa_profiles[f'PPR_{_condition_key}'] = _ppr_agg

    for _pulse_idx in range(int(N_STIM)):
        _ratios = _ppr[:, _pulse_idx]
        _mean_of_ratios = float(np.mean(_ratios))
        _ratio_of_means = float(np.mean(_amp[:, _pulse_idx]) / np.mean(_amp[:, 0]))
        _predicted_ppr = float(
            (_N_agg[_pulse_idx] / _N_agg[0])
            * (_P_agg[_pulse_idx] / _P_agg[0])
        )
        _mpfa_metadata_rows.append({
            'condition': _condition_key,
            'Stim': _pulse_idx + 1,
            'n_boutons_used': int(_valid.sum()),
            'agg_method': MPFA_AGG_METHOD,
            'Pr': _P_agg[_pulse_idx],
            'Pr_CI_low': _P_ci[0, _pulse_idx],
            'Pr_CI_high': _P_ci[1, _pulse_idx],
            'N': _N_agg[_pulse_idx],
            'N_CI_low': _N_ci[0, _pulse_idx],
            'N_CI_high': _N_ci[1, _pulse_idx],
            'PPR': _ppr_agg[_pulse_idx],
            'PPR_CI_low': _ppr_ci[0, _pulse_idx],
            'PPR_CI_high': _ppr_ci[1, _pulse_idx],
            'PPR_from_aggregate_N_Pr': _predicted_ppr,
            'PPR_prediction_in_CI': bool(
                _ppr_ci[0, _pulse_idx] <= _predicted_ppr <= _ppr_ci[1, _pulse_idx]
            ),
            'mean_of_ratios': _mean_of_ratios,
            'ratio_of_means': _ratio_of_means,
            'corr_A1_ratio': _mpfa_safe_corr(_amp[:, 0], _ratios),
            'VMR': _vmr_agg[_pulse_idx],
        })
        _mpfa_varmean_rows.append({
            'Frequency_Hz': _frequency,
            'Calcium': _calcium,
            'Event': _pulse_idx + 1,
            'Boutons': int(_valid.sum()),
            'Fibers': _n_fibers,
            'Bouton_mean_average': _mpfa_aggregate(_amp)[_pulse_idx],
            'Bouton_variance_average': _mpfa_aggregate(_amp_var)[_pulse_idx],
            'Fano_of_bouton_averages': _vmr_agg[_pulse_idx],
            'Mean_percent_A1': _ppr_agg[_pulse_idx] * 100.0,
            'Mean_percent_CI_low': _ppr_ci[0, _pulse_idx] * 100.0,
            'Mean_percent_CI_high': _ppr_ci[1, _pulse_idx] * 100.0,
            'Mean_percent_SEM': _ppr_percent_sem[_pulse_idx],
            'Fano_percent_A1': (_vmr_agg[_pulse_idx] / _vmr_agg[0]) * 100.0,
            'Fano_percent_CI_low': _vmr_percent_ci[0, _pulse_idx],
            'Fano_percent_CI_high': _vmr_percent_ci[1, _pulse_idx],
            'Fano_percent_SEM': _vmr_percent_sem[_pulse_idx],
            # Retain the legacy column name because the MATLAB demos consume
            # it, but use the same mean aggregation and fixed bouton list.
            'Median_MLE_P_A1_excluding_cap20': _P_agg[0],
            'Var_over_Imean': _vmr_agg[_pulse_idx],
            'Var_over_Imean_percent_A1': (_vmr_agg[_pulse_idx] / _vmr_agg[0]) * 100.0,
            'Var_over_Imean_percent_CI_low': _vmr_percent_ci[0, _pulse_idx],
            'Var_over_Imean_percent_CI_high': _vmr_percent_ci[1, _pulse_idx],
            'Var_over_Imean_percent_SEM': _vmr_percent_sem[_pulse_idx],
            'agg_method': MPFA_AGG_METHOD,
        })

    _mpfa_condition_tables[_condition_key] = pd.DataFrame({
        'bouton_id': _used_keys,
        **{f'Pr_{k + 1}': _P[:, k] for k in range(int(N_STIM))},
        **{f'N_{k + 1}': _N[:, k] for k in range(int(N_STIM))},
        **{f'PPR_{k + 1}': _ppr[:, k] for k in range(int(N_STIM))},
    })

_mpfa_ordered_columns = ['Stim']
for _profile_name in ('Pr', 'N', 'PPR'):
    _mpfa_ordered_columns.extend(
        f'{_profile_name}_{condition_key}'
        for _, condition_key in MPFA_CONDITION_ORDER
    )
MPFA_PROFILES = pd.DataFrame(_mpfa_profiles)[_mpfa_ordered_columns]
MPFA_DIAGNOSTICS = pd.DataFrame(_mpfa_metadata_rows)
MPFA_VAR_IMEAN = pd.DataFrame(_mpfa_varmean_rows).sort_values(
    ['Frequency_Hz', 'Calcium', 'Event']
).reset_index(drop=True)

_mpfa_repo_dir = Path('mpfa')
_mpfa_repo_dir.mkdir(parents=True, exist_ok=True)
_mpfa_output_dir = Path(OUTPUT_DIR)
_mpfa_output_dir.mkdir(parents=True, exist_ok=True)
_mpfa_profile_name = 'FigS16_mpfa_demo_profiles_matrix.csv'
_mpfa_diagnostic_name = 'FigS16_mpfa_per_bouton_diagnostics.csv'
_mpfa_varmean_name = 'FigS16_var_imean_profiles_6conditions.csv'

for _destination in (_mpfa_repo_dir, _mpfa_output_dir):
    MPFA_PROFILES.to_csv(_destination / _mpfa_profile_name, index=False)
    MPFA_DIAGNOSTICS.to_csv(_destination / _mpfa_diagnostic_name, index=False)
    MPFA_VAR_IMEAN.to_csv(_destination / _mpfa_varmean_name, index=False)

print('\n=== MPFA fixed-list aggregation diagnostics (pulse 2) ===')
print(MPFA_DIAGNOSTICS.loc[MPFA_DIAGNOSTICS['Stim'] == 2, [
    'condition', 'n_boutons_used', 'mean_of_ratios', 'ratio_of_means',
    'corr_A1_ratio', 'PPR_from_aggregate_N_Pr', 'PPR', 'PPR_prediction_in_CI',
]].to_string(index=False))
print(f'Saved: {_mpfa_repo_dir / _mpfa_profile_name}')
print(f'Saved: {_mpfa_repo_dir / _mpfa_diagnostic_name}')
print(f'Saved: {_mpfa_repo_dir / _mpfa_varmean_name}')
