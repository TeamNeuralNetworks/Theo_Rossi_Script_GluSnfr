# NNLS Weight Control Implementation

This document describes the new NNLS weight control functionality added to the `extract_metrics` function.

## Overview

The new implementation allows controlling weights during NNLS (Non-Negative Least Squares) fitting for event amplitude estimation. Three weight patterns are supported:

1. **Uniform** (default): All timepoints have equal weight
2. **Linear**: Weights decrease linearly from 1 to 0 between each stimulus and the next (sawtooth pattern)
3. **Exponential**: Exponential decay weights for each event

## New Options

### `nnls_weight_mode`
- **Type**: string
- **Default**: `'uniform'`
- **Options**: `'uniform'`, `'linear'`, `'exponential'`
- **Description**: Controls the weighting pattern for NNLS fitting

### `nnls_weight_tau_s`
- **Type**: float or None
- **Default**: `None` (auto)
- **Description**: Time constant for weight decay in seconds
  - For `'linear'`: Controls the linear decay slope (default: ISI)
  - For `'exponential'`: Controls the exponential decay time constant
    - When `fit_source='global'`: Uses estimated tau_d from kinetics fitting (default: 10ms if not available)
    - For other modes: Default 10ms
  - Linear decay is clipped at 0

### `nnls_show_weights`
- **Type**: boolean
- **Default**: `False`
- **Description**: When True, displays a plot showing the weight pattern for the train

## Weight Pattern Behavior

### Uniform
All timepoints receive weight = 1.0

### Linear
For each event, weights start at 1.0 at stimulus time and decrease linearly to 0 at the next stimulus time:
- Creates a sawtooth pattern across the stimulus train
- Slope controlled by `nnls_weight_tau_s` (default: ISI duration)
- Weights are clipped at 0 (non-negative)

### Exponential
For each event, weights start at 1.0 at stimulus time and decay exponentially:
- `weight = exp(-time_since_stimulus / tau)`
- Time constant controlled by `nnls_weight_tau_s`
- Default tau depends on `fit_source`:
  - `'global'`: Uses estimated tau_d from kinetics fitting
  - Other modes: 10ms

## Usage Examples

```python
# Basic exponential weighting with auto tau
options = {
    'nnls_weight_mode': 'exponential',
    'nnls_show_weights': True,
}

# Linear weighting with custom slope
options = {
    'nnls_weight_mode': 'linear',
    'nnls_weight_tau_s': 0.025,  # 25ms time constant
    'nnls_show_weights': True,
}

# Exponential weighting with explicit time constant
options = {
    'nnls_weight_mode': 'exponential',
    'nnls_weight_tau_s': 0.015,  # 15ms decay
    'nnls_show_weights': True,
}
```

## Implementation Details

### Key Functions Added

1. **`_calculate_nnls_weights()`**: Calculates weight patterns for different modes
2. **`_nnls_weighted()`**: Performs weighted NNLS solving
3. **Weight visualization**: Integrated into plotting system when `nnls_show_weights=True`

### Integration Points

- Modified `estimate_kinetics_from_average()` to accept and use weight parameters
- Updated all calls to this function to pass weight configuration
- Added weight parameter processing in main `extract_metrics()` function
- Integrated weight visualization into existing plotting system

### Automatic Parameter Selection

- **Linear mode**: Default tau = ISI (inter-stimulus interval)
- **Exponential mode with global fit**: Default tau = estimated tau_d from kinetics
- **Exponential mode with other fits**: Default tau = 10ms

## Testing

The implementation has been tested with:
1. Synthetic data with known ground truth
2. Weight calculation function verification
3. Weighted NNLS solver verification
4. Integration with existing `extract_metrics` workflow

## Files Modified

1. `Feature_extraction/extract_metrics.py`: Main implementation
2. `Feature_extraction/demo_single_file.py`: Updated with example usage
3. Created `Feature_extraction/test_weights.py`: Test script for verification

## Backward Compatibility

All changes are backward compatible. Existing code will continue to work with uniform weights (default behavior).