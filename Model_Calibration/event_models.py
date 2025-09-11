"""
Reusable event models and helpers for calcium/iGluSnFR response fitting.

Provides model callables and their default parameter specs to avoid redundancy
across demos and notebooks.

Available models:
- 'double_exp': classic difference of exponentials with peak normalization
- 'cooperative': cooperative (Hill-like) rise with exponential decay
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple, List
import numpy as np


def model_double_exp_constrained(t, amp, tau_rise, tau_decay, t_peak):
    """Classic double exponential: (exp(-t/tau_decay) - exp(-t/tau_rise)).

    Parameters in seconds; t in milliseconds.
    Normalizes by value at theoretical peak to keep 'amp' interpretable.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_peak
    if np.any(m):
        ts = (t[m] - t_peak) / 1000.0
        tr = max(tau_rise, 1e-6)
        td = max(tau_decay, 1e-6)
        if td <= tr:
            # Fall back to raw difference if peak formula ill-defined
            y[m] = amp * (np.exp(-ts / td) - np.exp(-ts / tr))
        else:
            t_opt = tr * td / (td - tr) * np.log(td / tr)
            norm = np.exp(-t_opt / td) - np.exp(-t_opt / tr)
            rise = np.exp(-ts / tr)
            decay = np.exp(-ts / td)
            y[m] = amp * (decay - rise) / (norm if abs(norm) > 1e-10 else 1.0)
    return y


def model_cooperative_binding(t, amp, tau_rise, tau_decay, n_coop, t_peak):
    """Cooperative binding: Hill-like rise times exponential decay.

    Parameters in seconds (taus) and dimensionless n_coop; t in milliseconds.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_peak
    if np.any(m):
        ts = (t[m] - t_peak) / 1000.0
        tr = max(tau_rise, 1e-6)
        td = max(tau_decay, 1e-6)
        n = max(n_coop, 0.5)
        x = ts / tr
        rise = (x ** n) / (1.0 + x ** n)
        decay = np.exp(-ts / td)
        y[m] = amp * rise * decay
    return y


# -----------------------------
# Additional models (from 3B)
# -----------------------------

def model_single_exp_constrained(t, amp, tau_decay, t_peak):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_peak
    if np.any(m):
        ts = (t[m] - t_peak) / 1000.0
        y[m] = amp * np.exp(-ts / max(tau_decay, 1e-6))
    return y


def model_alpha_constrained(t, amp, tau, t_peak):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_peak
    if np.any(m):
        ts = (t[m] - t_peak) / 1000.0
        tau_s = max(tau, 1e-6)
        e_inv = 1.0 / np.e
        y[m] = amp * (ts / tau_s) * np.exp(-ts / tau_s) / e_inv
    return y


def model_gamma_constrained(t, amp, n, tau, t_peak):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_peak
    if np.any(m):
        ts = (t[m] - t_peak) / 1000.0
        tau_s = max(tau, 1e-6)
        n_safe = max(n, 0.1)
        x = ts / tau_s
        try:
            norm = (n_safe / np.e) ** n_safe
            y[m] = amp * (x ** n_safe) * np.exp(-x) / norm
        except Exception:
            y[m] = amp * (x ** n_safe) * np.exp(-x)
    return y


def model_bilinear_constrained(t, amp, t_rise, t_decay, t_peak):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    rise_mask = (t >= t_peak) & (t <= t_peak + t_rise)
    if np.any(rise_mask):
        t_rel = t[rise_mask] - t_peak
        y[rise_mask] = amp * (t_rel / max(t_rise, 0.1))
    decay_mask = t > (t_peak + t_rise)
    if np.any(decay_mask):
        ts = (t[decay_mask] - t_peak - t_rise) / 1000.0
        tau_s = max(t_decay / 1000.0, 1e-6)
        y[decay_mask] = amp * np.exp(-ts / tau_s)
    return y


def model_binding_kinetics(t, amp, kon, koff, tau_clear, t_peak):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_peak
    if np.any(m):
        ts = (t[m] - t_peak) / 1000.0
        kon_s = max(kon, 1.0)
        koff_s = max(koff, 1.0)
        binding = 1 - np.exp(-ts * kon_s)
        decay = np.exp(-ts * (koff_s + 1.0 / max(tau_clear, 1e-6)))
        y[m] = amp * binding * decay
    return y


def model_two_component_shared_rise(t, amp_fast, tau_rise, tau_fast, amp_slow, tau_slow, t_peak):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_peak
    if np.any(m):
        ts = (t[m] - t_peak) / 1000.0
        rise = 1 - np.exp(-ts / max(tau_rise, 1e-6))
        fast = amp_fast * np.exp(-ts / max(tau_fast, 1e-6))
        slow = amp_slow * np.exp(-ts / max(tau_slow, 1e-6))
        y[m] = rise * (fast + slow)
    return y


def model_desensitization(t, amp, tau_rise, tau_decay, tau_recovery, desens_factor, t_peak):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_peak
    if np.any(m):
        ts = (t[m] - t_peak) / 1000.0
        rise = 1 - np.exp(-ts / max(tau_rise, 1e-6))
        decay = np.exp(-ts / max(tau_decay, 1e-6))
        rec = 1 - desens_factor * (1 - np.exp(-ts / max(tau_recovery, 1e-6)))
        y[m] = amp * rise * decay * rec
    return y


def model_cooperative_plus_linear(t, amp_coop, tau_rise_coop, tau_decay_coop, n_coop, amp_linear, tau_decay_linear, t_peak):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_peak
    if np.any(m):
        ts = (t[m] - t_peak) / 1000.0
        tr = max(tau_rise_coop, 1e-6)
        n = max(n_coop, 0.5)
        norm_t = ts / tr
        rise_coop = (norm_t ** n) / (1 + norm_t ** n)
        decay_coop = np.exp(-ts / max(tau_decay_coop, 1e-6))
        coop = amp_coop * rise_coop * decay_coop
        rise_linear = 1 - np.exp(-ts / tr)
        decay_linear = np.exp(-ts / max(tau_decay_linear, 1e-6))
        linear = amp_linear * rise_linear * decay_linear
        y[m] = coop + linear
    return y


def model_diffusion_clearance(t, amp, tau_diff, tau_clear1, tau_clear2, frac_clear1, t_peak):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_peak
    if np.any(m):
        ts = (t[m] - t_peak) / 1000.0
        tau_d = max(tau_diff, 1e-6)
        rise = (ts / tau_d) * np.exp(-ts / tau_d)
        e_inv = 1.0 / np.e
        rise_norm = rise / e_inv
        clear1 = frac_clear1 * np.exp(-ts / max(tau_clear1, 1e-6))
        clear2 = (1 - frac_clear1) * np.exp(-ts / max(tau_clear2, 1e-6))
        y[m] = amp * rise_norm * (clear1 + clear2)
    return y


def model_double_cooperative(t, amp, tau_rise1, tau_decay1, n1, tau_rise2, tau_decay2, n2, t_peak):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_peak
    if np.any(m):
        ts = (t[m] - t_peak) / 1000.0
        tr1 = max(tau_rise1, 1e-6); n1s = max(n1, 0.5)
        tr2 = max(tau_rise2, 1e-6); n2s = max(n2, 0.5)
        x1 = ts / tr1; x2 = ts / tr2
        rise1 = (x1 ** n1s) / (1 + x1 ** n1s)
        rise2 = (x2 ** n2s) / (1 + x2 ** n2s)
        decay1 = np.exp(-ts / max(tau_decay1, 1e-6))
        decay2 = np.exp(-ts / max(tau_decay2, 1e-6))
        y[m] = amp * (0.5 * rise1 * decay1 + 0.5 * rise2 * decay2)
    return y


def model_heterogeneous_cooperative(t, amp, tau_rise1, tau_decay1, n1, frac1, tau_rise2, tau_decay2, n2, t_peak):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_peak
    if np.any(m):
        ts = (t[m] - t_peak) / 1000.0
        tr1 = max(tau_rise1, 1e-6); tr2 = max(tau_rise2, 1e-6)
        n1s = max(n1, 0.5); n2s = max(n2, 0.5)
        x1 = ts / tr1; x2 = ts / tr2
        rise1 = (x1 ** n1s) / (1 + x1 ** n1s)
        rise2 = (x2 ** n2s) / (1 + x2 ** n2s)
        decay1 = np.exp(-ts / max(tau_decay1, 1e-6))
        decay2 = np.exp(-ts / max(tau_decay2, 1e-6))
        comp1 = frac1 * rise1 * decay1
        comp2 = (1 - frac1) * rise2 * decay2
        y[m] = amp * (comp1 + comp2)
    return y


def model_two_component_cooperative(t, amp_fast, tau_rise_fast, tau_decay_fast, n_fast,
                                    amp_slow, tau_rise_slow, tau_decay_slow, n_slow, t_peak):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_peak
    if np.any(m):
        ts = (t[m] - t_peak) / 1000.0
        # fast component
        trf = max(tau_rise_fast, 1e-6); nf = max(n_fast, 0.5)
        xf = ts / trf
        rise_f = (xf ** nf) / (1 + xf ** nf)
        decay_f = np.exp(-ts / max(tau_decay_fast, 1e-6))
        comp_f = amp_fast * rise_f * decay_f
        # slow component
        trs = max(tau_rise_slow, 1e-6); ns = max(n_slow, 0.5)
        xs = ts / trs
        rise_s = (xs ** ns) / (1 + xs ** ns)
        decay_s = np.exp(-ts / max(tau_decay_slow, 1e-6))
        comp_s = amp_slow * rise_s * decay_s
        y[m] = comp_f + comp_s
    return y


def get_event_model(name: str) -> Dict:
    """Return a model spec dict for the given name.

    Dict keys:
    - 'func': callable(t_ms, *params)
    - 'params': list of parameter names
    - 'bounds': (lb, ub) passed to curve_fit
    - 'p0_func': callable(y_fit, t_fit) -> initial params
    - 'name': canonical name
    """
    nm = (name or '').strip().lower()
    if nm in ('double', 'double_exp', 'double-exponential', 'biexp'):
        return {
            'name': 'double_exp',
            'func': model_double_exp_constrained,
            'params': ['amp', 'tau_rise', 'tau_decay', 't_peak'],
            'bounds': ([0, 0.0005, 0.001, 0], [np.inf, 0.010, 0.200, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 0.002, 0.020, float(t[np.nanargmax(y)])],
            'complexity': 4,
        }
    if nm in ('coop', 'cooperative', 'cooperative_binding'):
        return {
            'name': 'cooperative',
            'func': model_cooperative_binding,
            'params': ['amp', 'tau_rise', 'tau_decay', 'n_coop', 't_peak'],
            'bounds': ([0, 0.001, 0.005, 0.5, 0], [np.inf, 0.020, 0.200, 5.0, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 0.005, 0.030, 2.0, float(t[np.nanargmax(y)])],
            'complexity': 5,
        }
    if nm in ('single', 'single_exp', 'single-exponential'):
        return {
            'name': 'single_exp',
            'func': model_single_exp_constrained,
            'params': ['amp', 'tau_decay', 't_peak'],
            'bounds': ([0, 0.001, 0], [np.inf, 0.200, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 0.020, float(t[np.nanargmax(y)])],
            'complexity': 3,
        }
    if nm in ('alpha',):
        return {
            'name': 'alpha',
            'func': model_alpha_constrained,
            'params': ['amp', 'tau', 't_peak'],
            'bounds': ([0, 0.001, 0], [np.inf, 0.100, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)) * np.e, 0.010, float(t[np.nanargmax(y)])],
            'complexity': 3,
        }
    if nm in ('gamma',):
        return {
            'name': 'gamma',
            'func': model_gamma_constrained,
            'params': ['amp', 'n', 'tau', 't_peak'],
            'bounds': ([0, 0.5, 0.001, 0], [np.inf, 8.0, 0.100, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)) * 3.0, 2.0, 0.010, float(t[np.nanargmax(y)])],
            'complexity': 4,
        }
    if nm in ('bilinear',):
        return {
            'name': 'bilinear',
            'func': model_bilinear_constrained,
            'params': ['amp', 't_rise', 't_decay', 't_peak'],
            'bounds': ([0, 0.1, 1, 0], [np.inf, 10, 100, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 2.0, 20.0, float(t[np.nanargmax(y)])],
            'complexity': 4,
        }
    if nm in ('binding_kinetics', 'binding'):
        return {
            'name': 'binding_kinetics',
            'func': model_binding_kinetics,
            'params': ['amp', 'kon', 'koff', 'tau_clear', 't_peak'],
            'bounds': ([0, 10, 1, 0.001, 0], [np.inf, 1000, 200, 0.200, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 200, 50, 0.030, float(t[np.nanargmax(y)])],
            'complexity': 5,
        }
    if nm in ('two_component', 'two-component', 'two_component_shared_rise'):
        return {
            'name': 'two_component',
            'func': model_two_component_shared_rise,
            'params': ['amp_fast', 'tau_rise', 'tau_fast', 'amp_slow', 'tau_slow', 't_peak'],
            'bounds': ([0, 0.0005, 0.001, 0, 0.010, 0], [np.inf, 0.010, 0.100, np.inf, 1.000, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y))*0.6, 0.002, 0.015, float(np.nanmax(y))*0.4, 0.080, float(t[np.nanargmax(y)])],
            'complexity': 6,
        }
    if nm in ('desens', 'desensitization'):
        return {
            'name': 'desensitization',
            'func': model_desensitization,
            'params': ['amp', 'tau_rise', 'tau_decay', 'tau_recovery', 'desens_factor', 't_peak'],
            'bounds': ([0, 0.001, 0.005, 0.020, 0, 0], [np.inf, 0.010, 0.100, 1.000, 0.8, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 0.003, 0.020, 0.100, 0.3, float(t[np.nanargmax(y)])],
            'complexity': 6,
        }
    if nm in ('coop_plus_linear', 'cooperative_plus_linear'):
        return {
            'name': 'coop_plus_linear',
            'func': model_cooperative_plus_linear,
            'params': ['amp_coop', 'tau_rise_coop', 'tau_decay_coop', 'n_coop', 'amp_linear', 'tau_decay_linear', 't_peak'],
            'bounds': ([0, 0.001, 0.005, 0.5, 0, 0.010, 0], [np.inf, 0.020, 0.200, 5.0, np.inf, 0.500, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y))*0.8, 0.005, 0.030, 2.0, float(np.nanmax(y))*0.2, 0.100, float(t[np.nanargmax(y)])],
            'complexity': 7,
        }
    if nm in ('diffusion_clearance', 'diffusion'):
        return {
            'name': 'diffusion_clearance',
            'func': model_diffusion_clearance,
            'params': ['amp', 'tau_diff', 'tau_clear1', 'tau_clear2', 'frac_clear1', 't_peak'],
            'bounds': ([0, 0.001, 0.005, 0.020, 0.1, 0], [np.inf, 0.020, 0.100, 0.500, 0.9, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y))*np.e, 0.003, 0.015, 0.080, 0.6, float(t[np.nanargmax(y)])],
            'complexity': 7,
        }
    if nm in ('double_cooperative', 'double_coop'):
        return {
            'name': 'double_cooperative',
            'func': model_double_cooperative,
            'params': ['amp', 'tau_rise1', 'tau_decay1', 'n1', 'tau_rise2', 'tau_decay2', 'n2', 't_peak'],
            'bounds': ([0, 0.001, 0.005, 0.5, 0.005, 0.020, 0.5, 0], [np.inf, 0.020, 0.100, 5.0, 0.100, 0.500, 5.0, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 0.003, 0.015, 2.0, 0.010, 0.080, 1.5, float(t[np.nanargmax(y)])],
            'complexity': 8,
        }
    if nm in ('hetero_coop', 'heterogeneous_cooperative'):
        return {
            'name': 'hetero_coop',
            'func': model_heterogeneous_cooperative,
            'params': ['amp', 'tau_rise1', 'tau_decay1', 'n1', 'frac1', 'tau_rise2', 'tau_decay2', 'n2', 't_peak'],
            'bounds': ([0, 0.001, 0.005, 0.5, 0.1, 0.005, 0.020, 0.5, 0], [np.inf, 0.020, 0.200, 5.0, 0.9, 0.100, 1.000, 5.0, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 0.003, 0.020, 2.0, 0.6, 0.010, 0.080, 1.5, float(t[np.nanargmax(y)])],
            'complexity': 9,
        }
    if nm in ('two_comp_coop', 'two_component_cooperative'):
        return {
            'name': 'two_comp_coop',
            'func': model_two_component_cooperative,
            'params': ['amp_fast', 'tau_rise_fast', 'tau_decay_fast', 'n_fast', 'amp_slow', 'tau_rise_slow', 'tau_decay_slow', 'n_slow', 't_peak'],
            'bounds': ([0, 0.001, 0.005, 0.5, 0, 0.005, 0.020, 0.5, 0], [np.inf, 0.020, 0.100, 5.0, np.inf, 0.100, 1.000, 5.0, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y))*0.6, 0.003, 0.015, 2.0, float(np.nanmax(y))*0.4, 0.010, 0.080, 1.5, float(t[np.nanargmax(y)])],
            'complexity': 9,
        }
    raise ValueError(f"Unknown model '{name}'")


def get_models(names: List[str]) -> Dict[str, Dict]:
    out = {}
    for n in names:
        spec = get_event_model(n)
        out[spec['name']] = spec
    return out
