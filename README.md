# iGluSnFR Fluorescence Analysis: NNLS Template Fitting

A Python pipeline for extracting individual event amplitudes from overlapping fluorescence transients in iGluSnFR imaging data.

---

## Table of Contents

1. [The Problem: Overlapping Transients](#the-problem-overlapping-transients)
2. [The Solution: NNLS Grid Search](#the-solution-nnls-grid-search)
3. [Why NNLS Instead of Free Fitting?](#why-nnls-instead-of-free-fitting)
   - [What Does Non-Negative Mean?](#what-does-non-negative-mean)
4. [Step-by-Step Analysis Pipeline](#step-by-step-analysis-pipeline)
   - [Baseline Estimation and Preprocessing](#1-baseline-estimation-and-preprocessing)
   - [Reference Template Extraction (Recut)](#3-reference-template-extraction-recut)
   - [Why NNLS Uses Area Not Peak](#why-nnls-uses-area-integral-not-peak)
   - [Template Variants Generation](#4-template-variants-generation)
   - [Event-Subtracted Residuals](#7-event-subtracted-residuals-goodness-of-fit-assessment)
   - [Residual Correction (50Hz)](#8-residual-correction-50hz-only)
   - [Null Condition: Statistical Significance](#9-null-condition-statistical-significance-testing)
5. [Template Variants: Handling Biological Variability](#template-variants-handling-biological-variability)
6. [Jitter Variants: Temporal Alignment](#jitter-variants-temporal-alignment)
7. [Parameter Bounds: Constraining the Fit](#parameter-bounds-constraining-the-fit)
8. [Configuration Guide](#configuration-guide)
9. [Running the Analysis](#running-the-analysis)

---

## The Problem: Overlapping Transients

In high-frequency stimulation experiments (20-50 Hz), individual iGluSnFR responses overlap before returning to baseline. Traditional peak detection fails because:

- **Temporal overlap**: Fast stimulation (ISI < decay time constant) causes responses to sum
- **Variable kinetics**: The decay can have different fast/slow component ratios
- **Baseline contamination**: No clean baseline between events
- **Noise**: Trial-to-trial variability and photon noise

**Goal**: Extract accurate individual event amplitudes despite severe overlap.

---

## The Solution: NNLS Grid Search

### Core Concept

Instead of fitting parameters freely (which is ill-conditioned for overlapping events), we use **Non-Negative Least Squares (NNLS)** as a **grid search over a family of candidate templates**.

### How It Works

1. **Define template family**: Create multiple candidate templates by varying:
   - Fast/slow decay component ratios (e.g., 0%, 10%, 20%, ..., 100% fast)
   - Temporal jitter around stimulus times (e.g., -1ms, -0.75ms, ..., +1ms)

2. **Build kernel matrix**: For each event × template variant combination, create a basis function

3. **NNLS fitting**: Find non-negative amplitudes that best reconstruct the data as a linear combination of these basis functions

4. **Select best variant**: Choose the template variant (per event) that minimizes residuals

### Visual Example

![NNLS Grid Search](docs/figures/fig4_nnls_grid_search.png)

The top panel shows overlapping responses. Each middle panel shows NNLS fits using different template ratios. The bottom panel shows residuals across all ratios - the minimum identifies the best template variant.

---

## Why NNLS Instead of Free Fitting?

### Problem with Free Fitting

When transients overlap significantly:
- **Parameter coupling**: Amplitude, rise time, and decay are coupled - many parameter combinations produce similar fits
- **Local minima**: Non-convex optimization gets trapped
- **Negative amplitudes**: Unconstrained fitting may produce unphysical negative values
- **Overfitting**: Too many free parameters for limited data

### NNLS Advantages

| Feature | NNLS | Free Fitting |
|---------|------|--------------|
| **Convex optimization** | ✓ Guaranteed global minimum | ✗ Local minima |
| **Physical constraints** | ✓ Enforces non-negative amplitudes | ✗ May produce negative values |
| **Speed** | ✓ Fast linear solve | ✗ Iterative optimization |
| **Robustness** | ✓ Stable even with severe overlap | ✗ Unstable with overlap |
| **Degeneracy** | ✓ Grid search explores parameter space | ✗ Gets stuck in local optima |

### Grid Search Philosophy

By testing a **discrete set of plausible templates** (variants), we:
- Avoid the parameter coupling problem
- Efficiently explore the solution space
- Maintain interpretability (each variant has clear biological meaning)
- Get robust amplitude estimates even when exact kinetics are uncertain

Think of it as: **"Which template from my library best explains the data?"** rather than **"What exact parameters fit the data?"**

### What Does "Non-Negative" Mean?

The **Non-Negative** constraint in NNLS has crucial physical meaning:

**Amplitude ≥ 0**: Fluorescence can only increase (glutamate binding) or stay at baseline - it cannot go negative. This enforces:
- **Physical reality**: No "negative glutamate release"
- **Numerical stability**: Prevents oscillating solutions where positive and negative amplitudes cancel
- **Biological interpretation**: Each coefficient represents the strength of a real release event

**Why this matters**:
- Unconstrained fitting may find solutions like `A1=+2.0, A2=-0.5, A3=+1.5` that cancel perfectly to fit the data but are biologically meaningless
- NNLS rejects such solutions and finds `A1=1.5, A2=0.0, A3=1.0` - all non-negative
- This is especially important when events overlap severely (50Hz), where many parameter combinations can produce similar residuals

**Mathematical consequence**:
- NNLS formulation: `minimize ||K·a - y||²  subject to  a ≥ 0`
- The non-negativity constraint makes this a **convex quadratic program** with a unique global minimum
- Standard least squares without this constraint is **non-convex** when templates overlap, leading to degenerate solutions

---

## Step-by-Step Analysis Pipeline

### 1. Baseline Estimation and Preprocessing

#### F0 Calculation

The baseline fluorescence (F0) is estimated from the **pre-stimulus window** (typically 0.4-1.0 seconds before stimulation):

```python
F0 = mean(fluorescence[time < train_start - margin])
```

**Why this matters**:
- F0 represents the resting fluorescence level without glutamate
- All subsequent ΔF/F0 calculations are normalized to this value
- A noisy F0 estimate propagates error to all amplitude measurements

![Baseline Estimation](docs/figures/fig7_baseline_estimation.png)

#### Photobleaching Correction

Fluorescence imaging suffers from **photobleaching** - exponential decay of baseline fluorescence over time due to photodamage. This creates a downward-sloping baseline that contaminates amplitude measurements.

**Correction procedure**:
1. Fit exponential baseline: `F_bleach(t) = A × exp(-t/τ)`
2. Use **robust fitting** (Huber loss, IRLS) to downweight stimulus-evoked responses
3. Subtract fitted bleach curve and restore F0 level

```python
options = {
    'bleach': True,          # Enable bleach correction
    'bleach_huber_delta': 2.5,  # Robustness to spikes
}
```

Without bleach correction, late events appear smaller because the baseline has drifted downward.

#### ΔF/F0 Normalization

Convert absolute fluorescence to fractional change:

```
ΔF/F0(t) = (F(t) - F0) / F0
```

**Benefits**:
- Removes variation in sensor expression level
- Makes amplitudes comparable across boutons/recordings
- Units are intuitive (e.g., 50% increase = ΔF/F0 = 0.5)

```python
options = {
    'normalize_dff': True,   # Compute ΔF/F0
}
```

#### Savitzky-Golay Smoothing

Apply polynomial smoothing for denoising:

```python
options = {
    'sg_window': 9,   # Window size (samples)
    'sg_poly': 2,     # Polynomial order
}
```

**Uses**:
- Generate weights for NNLS (emphasize peaks)
- Peak detection
- Diagnostics and visualization

**Not used for**: NNLS fitting itself operates on the raw (non-smoothed) data to preserve amplitude information.

### 2. iGluSnFR Template Model

The iGluSnFR S72A response is modeled as a bi-exponential:

```
template(t) = (1 - exp(-t/τ_rise)) × [frac_fast × exp(-t/τ_fast) + (1-frac_fast) × exp(-t/τ_slow)]
```

**Parameters**:
- `τ_rise` ≈ 1.4 ms (rise time)
- `τ_fast` ≈ 8 ms (fast decay component)
- `τ_slow` ≈ 25-35 ms (slow decay component)
- `frac_fast` = 0.0 to 1.0 (fraction of fast component)

![Template Components](docs/figures/fig2_template_variants.png)

### 3. Reference Template Extraction (Recut)

Before fitting, we need to establish the template shape (τ_rise, τ_fast, τ_slow) for our event model. This is done by **"recutting"** individual events from the averaged trace.

#### The Recut Process

1. **Align events** across all trials to the stimulus time
2. **Average aligned snippets** to reduce noise
3. **Oversample** via interpolation (typically 20x)
4. **Fit kinetic parameters** to the averaged, oversampled template

![Recut and Oversampling](docs/figures/fig6_recut_oversampling.png)

#### Why Oversampling Works

Imaging data typically has **low sampling rates** (100-1000 Hz). The true peak of an event may fall between samples, causing:
- **Peak amplitude underestimation** (10-30% error)
- **Peak time jitter** (±1-2 samples)

**Solution**: After averaging multiple trials, **cubic interpolation** recovers the smooth underlying waveform:

```python
options = {
    'recut_oversample': 20,  # Interpolate 20x
    'recut_projection': 'mean',  # Average across trials
}
```

**Why averaging helps**: Trial-to-trial jitter in peak timing is **random**. When averaged:
- Noise cancels
- Peak timing becomes more consistent
- Interpolation reliably recovers the smooth peak

#### Why NNLS Uses Area (Integral), Not Peak

This is **critical** for low sampling rate data:

**Peak amplitude problem**:
- Peak position depends on sampling alignment
- If peak falls between samples: large error
- Different events may have different alignment → bias

**Area/Integral solution**:
- NNLS solves: `y(t) = Σ A_i × template(t - t_i)`
- Each amplitude `A_i` represents the **total contribution** (area under curve)
- Area is **robust to sampling** - trapezoidal integration averages over samples

**Demonstration** (from figure):
```
Peak amplitude error:  Aligned: -5%, Misaligned: -20%
Area (integral) error: Aligned: -2%, Misaligned: -3%
```

**Mathematical insight**: NNLS minimizes:
```
||y - K·a||² = Σ_t (y(t) - Σ_i a_i × k_i(t))²
```

This is equivalent to matching the **integrated signal** when time resolution is finite. Each sample point contributes a **trapezoidal area** to the sum, making amplitudes proportional to integral, not peak.

### 4. Template Variants Generation

Create a library of templates by varying `frac_fast`:

```python
options = {
    'use_template_variants': True,
    'template_variant_ratios': np.arange(0.0, 1.0, 0.1),  # Test 0%, 10%, 20%, ..., 100% fast
}
```

**Why?** Biological variability means the exact fast/slow ratio varies between synapses and conditions. Testing multiple ratios makes the fit robust to this variability.

**Critical feature: Independent ratio per event**
- Each event in the train can select its **own best-fit ratio**
- Ratios can **evolve throughout the train**
- Example: Event 1 uses 70% fast, Event 5 uses 40% fast (depression changes kinetics)
- Output: `dominant_ratio[10 events]` shows which ratio each event selected

This is fundamentally different from traditional fitting where all events must share the same kinetics.

### 5. Jitter Variants (Optional)

Account for temporal uncertainty in stimulus timing or response onset:

```python
options = {
    'jitter_variant_ms': np.arange(-1.0, 1.1, 0.25),  # Test -1ms to +1ms in 0.25ms steps
}
```

**Use cases**:
- Stimulus timing jitter
- Variable synaptic delay
- Action potential propagation time variability

When both template and jitter variants are enabled, the algorithm performs a **2D grid search** (template ratio × jitter).

### 5. NNLS Fitting with Weighted Residuals

Build kernel matrix and solve:

```python
# Kernel shape: [time_points, n_events × n_template_variants × n_jitter_variants]
K = build_kernel_matrix(time, stim_times, tau_rise, tau_decay_fast, tau_decay_slow, variants)

# Weighted NNLS (weights emphasize important regions like peaks)
amplitudes = weighted_nnls(K, data, weights)
```

**Weighting modes**:
- `'uniform'`: Equal weight to all time points
- `'savgol'`: Weight by smoothed signal (emphasizes peaks)
- `'linear'`: Linear decay from peak
- `'exponential'`: Exponential decay from peak

```python
options = {
    'nnls_weight_mode': 'savgol',  # Recommended for iGluSnFR
    'nnls_weight_tau_s': None,     # Auto-determined from data
}
```

### 6. Variant Selection

For each event, select the template variant (and jitter) that contributed most to the NNLS solution:

```
dominant_variant[i] = argmax(amplitudes[variants_for_event_i])
```

This gives you:
- **Amplitude** per event
- **Best-fit template ratio** per event (may vary across events!)
- **Best-fit jitter** per event (if jitter variants enabled)

### 7. Event-Subtracted Residuals: Goodness of Fit Assessment

#### Why Event-Subtracted Residuals?

Global residuals (data - full fit) don't tell you how well **individual events** are fitted. When events overlap, a globally good fit might still have poor fits for specific events. Event-subtracted residuals isolate each event to assess per-event fit quality.

#### How It Works

For each event in the train:

1. **Remove other events**: Subtract all other fitted events from the total fit
   ```
   other_events_fit = total_fit - this_event_fit
   isolated_data = raw_data - other_events_fit
   ```

2. **Isolate target event**: What remains is the data with other events removed
   ```
   isolated_event = isolated_data  # This should match this_event_fit if fit is good
   ```

3. **Compare to fitted template**: Calculate the residual
   ```
   event_residual = isolated_event - this_event_fit
   ```

4. **Calculate RMS**: Root mean square of the residual = goodness of fit
   ```
   RMS = sqrt(mean(event_residual²))
   ```

![Event-Subtracted Residuals](docs/figures/fig8_event_subtracted_residuals.png)

**Top panel**: Global fit showing all events together. Visually looks good.

**Bottom panels**: Per-event analysis reveals individual fit quality:
- **Orange line**: Data with other events removed (isolated)
- **Green line**: This event's isolated contribution
- **Blue dotted line**: Fitted template for this event
- **Gray shaded**: Event-subtracted residual (should be small)

#### Applications

**1. Template Variant Selection**
- When testing multiple template ratios, choose the one with lowest per-event RMS
- Each event can independently select its best template based on its own RMS

**2. Fitting Failure Detection**
- High RMS residual (e.g., > 0.05 for ΔF/F0) indicates poor template match
- May indicate:
  - Overlap contamination not handled by NNLS
  - Wrong model (e.g., non-iGluSnFR kinetics)
  - Artifacts or noise spikes during event

**3. Residual Correction Validation**
- Compare RMS before and after residual correction at 50Hz
- Confirms correction improved fit quality
- Identifies events that still have poor fits after correction

**4. Quality Control**
- Flag trials with consistently high event-subtracted residuals
- Exclude poorly fitted events from analysis
- Identify outliers that may need manual inspection

#### Metrics

```python
# Per-event goodness of fit
for i, event in enumerate(events):
    rms_residual[i] = sqrt(mean((isolated_data[i] - fit[i])**2))
    r_squared[i] = 1 - sum(residual²) / sum((data - mean(data))²)
```

**Typical thresholds** (for iGluSnFR ΔF/F0):
- **Excellent fit**: RMS < 0.02
- **Good fit**: RMS < 0.03
- **Acceptable fit**: RMS < 0.05
- **Poor fit**: RMS > 0.05 (consider excluding)

### 8. Residual Correction (50Hz Only)

For very high-frequency stimulation (ISI < 25 ms), even NNLS fitting can be contaminated by overlap. An optional **residual-based correction** adjusts amplitudes when the template shape doesn't perfectly match the data.

#### The Problem

At 50Hz (20ms ISI), the slow decay component (τ_slow ≈ 25-30ms) hasn't fully decayed before the next stimulus arrives. Even with the best template variant:
- **Positive residuals** (data > fit) accumulate between events
- These residuals **artificially inflate** the next event's amplitude
- The NNLS fit attributes residual fluorescence to the next event

![Residual Correction](docs/figures/fig5_residual_correction.png)

**Visual demonstration** (top panel):
- Red dashed line shows residuals (data - fit)
- Orange shaded regions highlight positive residuals **under the next event**
- Arrows show how residual at t=15ms inflates the amplitude fitted at t=20ms

#### The Solution

Iterative residual-based correction:

1. **Initial NNLS fit** with template variants
2. **Measure residuals** at each event onset (dotted line value at next stimulus time)
3. **Adjust amplitudes** by subtracting residual contribution
4. **Re-fit** with adjusted initial guess
5. **Iterate** until residuals minimize

```python
options = {
    'residual_correction_mode': 'residuals',  # Use residual signatures
    'correct_residuals_dff_50hz': True,       # Enable for 50Hz data
}
```

**Result** (bottom panel):
- Corrected amplitudes closer to ground truth
- Reduced systematic bias in later events
- Cleaner residuals (smaller, more centered at zero)

**When to use**:
- 50Hz stimulation (ISI = 20ms)
- Any condition where ISI < 1.5 × τ_slow
- When residual plots show systematic positive residuals before next stimulus

**When NOT to use**:
- 20Hz or slower (ISI ≥ 50ms) - no correction needed
- When template variants already achieve small residuals
- Single events (no overlap)

### 9. Null Condition: Statistical Significance Testing

#### Why Test for Significance?

Not all fitted amplitudes represent real synaptic events. At the noise floor:
- NNLS may fit noise fluctuations as small "events"
- First event (A1) near noise level creates artificially large PPR values
- Need statistical threshold to distinguish signal from noise

#### How It Works: Moving Window NNLS on Pre-Stimulus Noise

Instead of assuming Gaussian noise, we **empirically measure** the distribution of amplitudes that NNLS fits to pure noise:

1. **Extract pre-stimulus baseline** (time < train_start)
   ```python
   noise_window = data[time < train_start - margin]
   ```

2. **Generate synthetic "stimulus times"** in the noise window
   ```python
   # Example: 10 fake events spaced by ISI in the baseline
   fake_stim_times = np.arange(0, baseline_duration, ISI)
   ```

3. **Run NNLS on noise** with same template variants/jitter as real data
   ```python
   noise_amps = nnls_fit(noise_window, fake_stim_times, templates)
   ```

4. **Repeat with sliding window** to build distribution
   ```python
   # Shift fake stimuli by small steps (e.g., 10ms)
   for offset in np.arange(0, ISI, step=0.01):
       fake_stim_times_shifted = fake_stim_times + offset
       noise_amps_i = nnls_fit(noise_window, fake_stim_times_shifted, templates)
       null_distribution.append(noise_amps_i)
   ```

5. **Calculate threshold** from the null distribution
   ```python
   # Example: 95th percentile or mean + 2*SD
   threshold = np.percentile(null_distribution, 95)
   # Or: threshold = mean(null_distribution) + 2 * std(null_distribution)
   ```

6. **Test real amplitudes** against threshold
   ```python
   is_significant = (real_amplitude > threshold)
   p_value = sum(null_distribution >= real_amplitude) / len(null_distribution)
   ```

![Null Condition Distribution](docs/figures/fig9_null_condition.png)

**Top panel**: Pre-stimulus baseline with moving window analysis. Fake stimulus times slide through noise at different offsets.

**Middle panel**: NNLS fits noise as if it were real events. Amplitudes are small but non-zero due to noise fluctuations.

**Bottom panel**: Null distribution (histogram) from all sliding windows. The 95th percentile defines the detection threshold. Real event amplitudes are tested against this threshold.

#### Applications

**1. A1 Significance Testing**
- Most critical: First event (A1) determines PPR = A2/A1
- If A1 is below threshold → PPR is unreliable
- Flag trials where `A1 < threshold` as "not detected"

**2. Amplitude Flooring**
- Instead of rejecting trials, floor A1 to threshold:
  ```python
  A1_floored = max(A1, threshold)
  PPR = A2 / A1_floored  # Prevents giant PPR from tiny A1
  ```

**3. Per-Event Quality Control**
- Any event amplitude below threshold is likely noise
- Flag or exclude these events from paired-pulse analysis
- Example: Event 7 amplitude = 0.01, threshold = 0.03 → exclude

**4. Trial Exclusion**
- Reject trials where multiple events are below threshold
- Indicates poor recording quality or failed stimulation

#### Configuration

```python
options = {
    # Enable null condition testing
    'compute_null_distribution': True,

    # Sliding window parameters
    'null_n_fake_events': 10,      # Number of fake events per window
    'null_window_step_s': 0.010,   # Step size for sliding window (10ms)

    # Threshold calculation
    'null_percentile': 95,          # Use 95th percentile as threshold
    # Alternative: 'null_n_std': 2.0  # Or use mean + 2*SD

    # Application
    'amplitude_floor_to_noise': True,  # Floor A1 to threshold
    'reject_below_threshold': False,   # Or reject trials entirely
}
```

#### Why Moving Windows?

**Problem**: A single NNLS fit to noise at fixed positions might be unrepresentative.

**Solution**: Slide the fake stimulus times through the baseline at small steps (e.g., 5-10ms). This samples all possible alignments between noise fluctuations and template positions.

**Result**: The null distribution captures the **worst-case** scenario where noise fluctuations happen to align with template peaks, giving the most conservative threshold.

#### Why This Is Better Than Gaussian Assumptions

Traditional approach:
```python
# Assume Gaussian noise
noise_std = std(baseline)
threshold = 2 * noise_std  # Assumes normal distribution
```

**Problems**:
- iGluSnFR data has **non-Gaussian noise** (photobleaching steps, autofluorescence changes)
- NNLS fitting is **nonlinear** - noise doesn't simply scale
- Overlapping templates create **structure in noise** that Gaussian model misses

**NNLS-based null distribution**:
- Accounts for NNLS fitting behavior on noise
- Captures template overlap effects on noise amplitudes
- Uses same variants/jitter as real fits → matched test
- Empirical distribution automatically handles non-Gaussian noise

#### Typical Results

For iGluSnFR at 20-50Hz:
```
Noise baseline SD: 0.02 (ΔF/F0)
Null distribution mean: 0.005
Null distribution 95th percentile: 0.025

Real event amplitudes:
  A1 = 0.15 → significant (p < 0.001)
  A1 = 0.03 → marginal (p ≈ 0.05)
  A1 = 0.02 → not significant (p > 0.05)
```

---

## Template Variants: Handling Biological Variability

### Why Multiple Templates?

The fast/slow decay ratio varies due to:
- **Synaptic variability**: Different boutons have different kinetics
- **Calcium dependence**: Buffering affects apparent kinetics
- **Temperature**: Affects unbinding rates
- **Expression level**: Sensor concentration affects saturation

### How It Works

Instead of forcing all events to use the same template, we:
1. Generate 10 templates (e.g., 0%, 10%, 20%, ..., 100% fast component)
2. NNLS tests all 10 templates **for each event independently**
3. Each event selects its own best-fit template (ratio can differ per event)
4. Result: `amplitudes[10 events]` and `dominant_ratio[10 events]`

**Key advantage: Ratios evolve throughout the train**
- Event 1 might use 70% fast (strong, fast release)
- Event 5 might use 30% fast (depressed, slower kinetics)
- This captures **physiological changes** in synaptic kinetics due to:
  - Short-term depression (slower rebinding kinetics)
  - Facilitation (altered calcium buffering)
  - Depletion (reduced vesicle pool affects sensor kinetics)

Traditional single-template fitting forces all events to share the same kinetics, which is biologically unrealistic during plasticity.

### Configuration

```python
# Example: Test 5 different ratios
options = {
    'use_template_variants': True,
    'template_variant_ratios': [0.0, 0.25, 0.5, 0.75, 1.0],
}
```

**When to use**:
- Always recommended for iGluSnFR
- Essential when kinetics vary within a train (depression/facilitation affects kinetics)
- Not needed if you have independent measurement of exact kinetics

![Template Variants Grid](docs/figures/fig2_template_variants.png)

---

## Jitter Variants: Temporal Alignment

### Why Temporal Jitter?

Stimulus times may not perfectly align with fluorescence response due to:
- **Stimulator jitter**: Electrical stimulation timing variability
- **Synaptic delay**: Variable neurotransmitter release timing
- **Measurement error**: Manual stimulus time annotation

### How It Works

For each event:
1. Test multiple temporal offsets (e.g., -1ms, -0.5ms, 0ms, +0.5ms, +1ms)
2. NNLS finds best amplitude for each offset
3. Select offset with lowest residual

### Configuration

```python
# Example: Search ±1ms in 0.25ms steps (9 jitter values)
options = {
    'jitter_variant_ms': np.arange(-1.0, 1.1, 0.25),
}
```

### Combined Grid Search

When using both template and jitter variants:
- **Grid size**: `n_events × n_templates × n_jitters`
- **Example**: 10 events, 10 templates, 9 jitters = 900 basis functions
- **Output**: Best (template, jitter) pair per event

```python
# Full 2D grid search
options = {
    'use_template_variants': True,
    'template_variant_ratios': np.arange(0.0, 1.0, 0.1),  # 10 templates
    'jitter_variant_ms': np.arange(-1.0, 1.1, 0.25),      # 9 jitters
}
# Total: 10 events × 10 templates × 9 jitters = 900 variants tested simultaneously
```

**When to use jitter variants**:
- When stimulus timing is manually annotated (±1-2ms uncertainty)
- When using electrical stimulation with jitter
- When synaptic delay is variable
- NOT needed for precise optogenetic stimulation with known timing

---

## Parameter Bounds: Constraining the Fit

### Classical Bounds System

Instead of ad-hoc constraints, use explicit `(lower, upper)` bounds for each parameter:

```python
options = {
    'parameter_bounds': {
        'tau_decay_fast': (0.008, 0.008),    # FIXED at 8ms
        'tau_decay_slow': (0.025, 0.030),    # RANGE: 25-30ms
        'tau_rise': (np.nan, np.nan),        # UNCONSTRAINED
        'frac_fast': (0.0, 1.0),             # RANGE: 0-100%
    }
}
```

### Bound Types

| Syntax | Meaning | Use Case |
|--------|---------|----------|
| `(value, value)` | **Fixed** | Force parameter to exact value |
| `(lower, upper)` | **Range** | Constrain within plausible range |
| `(np.nan, np.nan)` | **Unconstrained** | Let optimizer decide |

### Example Use Cases

**1. Fixed kinetics (50Hz data)**

For high-frequency stimulation, kinetics estimation from recut is contaminated. Fix to literature values:

```python
'parameter_bounds': {
    'tau_decay_fast': (0.008, 0.008),  # Literature: 8ms ± 1ms → fix to avoid nonsense
    'tau_decay_slow': (0.025, 0.030),  # Constrain slow component to reasonable range
}
```

**2. Unconstrained (20Hz data)**

For lower frequency where events are better separated:

```python
'parameter_bounds': {
    'tau_rise': (np.nan, np.nan),       # Let it fit freely
    'tau_decay_fast': (0.006, 0.012),   # Constrain to plausible range
    'tau_decay_slow': (0.020, 0.040),   # Wider range allowed
}
```

**3. Prevent unphysical values**

```python
'parameter_bounds': {
    'tau_rise': (0.0005, 0.003),   # Must be between 0.5-3ms (prevents negative or huge values)
    'frac_fast': (0.0, 1.0),       # Fraction must be between 0-100%
}
```

![Parameter Bounds](docs/figures/fig3_parameter_bounds.png)

---

## Configuration Guide

### Quick Start Presets

#### 20Hz Stimulation (Standard)

```python
options = {
    # Preprocessing
    'normalize_dff': True,
    'bleach': True,
    'sg_window': 9,
    'sg_poly': 2,

    # Model
    'event_model': 'iglusnfr',
    'use_template_variants': True,
    'template_variant_ratios': np.arange(0.0, 1.0, 0.1),

    # NNLS
    'nnls_weight_mode': 'savgol',
    'huber_delta': 5.5,
    'irls_iters': 20,

    # Parameter constraints
    'parameter_bounds': {
        'tau_decay_fast': (0.006, 0.012),
        'tau_decay_slow': (0.020, 0.040),
        'tau_rise': (np.nan, np.nan),
    },

    # Plotting
    'plot': {
        'enabled': True,
        'traces': ['raw', 'nnls'],
        'residuals': True,
    }
}
```

#### 50Hz Stimulation (High Frequency)

```python
options = {
    # Preprocessing
    'normalize_dff': True,
    'bleach': True,
    'sg_window': 9,
    'sg_poly': 2,

    # Model
    'event_model': 'iglusnfr',
    'use_template_variants': True,
    'template_variant_ratios': np.arange(0.0, 1.0, 0.1),

    # Grid search with jitter
    'jitter_variant_ms': np.arange(-1.0, 1.1, 0.25),

    # NNLS
    'nnls_weight_mode': 'savgol',
    'huber_delta': 2.5,  # Lower for high-frequency (less margin for outliers)
    'irls_iters': 20,

    # FIX kinetics (can't estimate from overlap)
    'parameter_bounds': {
        'tau_decay_fast': (0.008, 0.008),  # FIXED
        'tau_decay_slow': (0.025, 0.030),  # Narrow range
        'tau_rise': (np.nan, np.nan),
    },

    # Residual correction for severe overlap
    'correct_residuals_dff_50hz': True,
    'residual_correction_mode': 'residuals',

    # Amplitude flooring
    'amplitude_floor_to_noise': True,  # Prevent giant PPR from tiny A1

    # Onset detection
    'use_onset_detection': True,  # Exclude contaminated pre-onset baseline

    # Plotting
    'plot': {
        'enabled': True,
        'traces': ['raw', 'nnls'],
        'residuals': True,
    }
}
```

### Key Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `event_model` | `'double_exp'` | Use `'iglusnfr'` for iGluSnFR-specific model |
| `use_template_variants` | `False` | Enable template variant grid search |
| `template_variant_ratios` | `None` | Array of fast/slow ratios to test |
| `jitter_variant_ms` | `None` | Temporal jitter range in milliseconds |
| `nnls_weight_mode` | `'uniform'` | Weighting: `'uniform'`, `'savgol'`, `'linear'`, `'exponential'` |
| `huber_delta` | `5.5` | Robust fitting threshold (lower = stricter outlier rejection) |
| `irls_iters` | `6` | Iteratively Reweighted Least Squares iterations |
| `amplitude_floor_to_noise` | `False` | Floor amplitudes to noise level before PPR calculation |
| `correct_residuals_dff_50hz` | `False` | Enable residual correction for 50Hz |
| `use_onset_detection` | `False` | Detect response onset to exclude contaminated baseline |
| `parameter_bounds` | `{}` | Dict of `{param: (lower, upper)}` constraints |

---

## Running the Analysis

### Single File Analysis

```python
from Feature_extraction.extract_metrics import extract_metrics
import pandas as pd
import numpy as np

# Load data
xlsx_path = "path/to/your/data.xlsx"
df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
time = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
trials = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)

# Configure analysis
options = {
    'normalize_dff': True,
    'bleach': True,
    'event_model': 'iglusnfr',
    'use_template_variants': True,
    'template_variant_ratios': np.arange(0.0, 1.0, 0.1),
    'jitter_variant_ms': np.arange(-1.0, 1.1, 0.25),
    'parameter_bounds': {
        'tau_decay_fast': (0.008, 0.008),
        'tau_decay_slow': (0.025, 0.030),
    },
    'nnls_weight_mode': 'savgol',
    'plot': {'enabled': True, 'traces': ['raw', 'nnls'], 'residuals': True}
}

# Run analysis
results = extract_metrics(
    time, trials,
    train_start=0.5,      # seconds
    isi=0.02,             # 50Hz = 20ms ISI
    n_pulses=10,
    options=options
)

# Extract results
amplitudes = results['average']['amp_nnls']
pprs = results['average']['ppr_nnls']
dominant_ratios = results.get('nnls_dominant_template_ratio', None)
dominant_jitters = results.get('nnls_dominant_jitter_s', None)

print(f"Amplitudes: {amplitudes}")
print(f"PPRs: {pprs}")
print(f"Best template ratios per event: {dominant_ratios}")
print(f"Best jitters per event (s): {dominant_jitters}")
```

### Batch Processing

See [demo_batch_process.py](Feature_extraction/demo_batch_process.py) for complete batch processing examples.

---

## Output Structure

```python
results = {
    'average': {
        'amp_nnls': np.ndarray,          # Amplitudes (NNLS fit)
        'ppr_nnls': np.ndarray,          # Paired-pulse ratios
        'amp_nnls_corr': np.ndarray,     # Corrected amplitudes (if residual correction)
        'ppr_nnls_corr': np.ndarray,     # Corrected PPRs
        'tau_decay': np.ndarray,         # Decay time constants per event
    },
    'per_trial': {
        'amp_nnls': np.ndarray,          # Shape: [n_trials, n_events]
        'ppr_nnls': np.ndarray,          # Shape: [n_trials, n_events]
    },
    'nnls_dominant_template_ratio': np.ndarray,  # Best ratio per event (if variants)
    'nnls_dominant_jitter_s': np.ndarray,        # Best jitter per event (if jitter variants)
    'threshold_amp1': np.ndarray,                # A1 detection thresholds per trial
    'pval_amp1': np.ndarray,                     # A1 significance p-values per trial
    'figure': matplotlib.figure.Figure,          # Main analysis figure
    'trial_figs': List[matplotlib.figure.Figure], # Per-trial figures (if requested)
}
```

---

## Troubleshooting

### Problem: Negative or zero amplitudes

**Solution**: Check that `use_template_variants=True` and test a wide range of ratios. NNLS enforces non-negative, but wrong template shapes can drive amplitudes to zero.

### Problem: Unrealistic PPR values (e.g., PPR > 5)

**Cause**: A1 is near noise level, making PPR = A2/A1 huge.

**Solution**: Enable amplitude flooring:
```python
options = {'amplitude_floor_to_noise': True}
```

### Problem: Fitted kinetics don't match literature

**Cause**: For high-frequency stimulation, overlap prevents accurate kinetics estimation.

**Solution**: Fix kinetics using parameter bounds:
```python
options = {
    'parameter_bounds': {
        'tau_decay_fast': (0.008, 0.008),
        'tau_decay_slow': (0.028, 0.028),
    }
}
```

### Problem: Large residuals after fitting

**Cause 1**: Template shape mismatch - try more template variants.
**Cause 2**: For 50Hz, severe overlap requires residual correction.

**Solution**:
```python
options = {
    'template_variant_ratios': np.arange(0.0, 1.0, 0.05),  # Finer grid
    'correct_residuals_dff_50hz': True,  # Enable correction
}
```

### Problem: Slow performance

**Cause**: Large grid search (many variants × many jitters).

**Solution**:
- Reduce grid resolution: `np.arange(0.0, 1.0, 0.2)` instead of `0.1`
- Use jitter variants only when necessary
- For initial testing, disable variants: `use_template_variants=False`

---

## Key References

1. **iGluSnFR kinetics**: Marvin et al., Nature Methods (2013) - S72A variant
2. **NNLS deconvolution**: Lawson & Hanson (1995) - Solving Least Squares Problems
3. **Template matching for calcium imaging**: Vogelstein et al., J Neurosci Methods (2010)
4. **Robust fitting (Huber loss)**: Huber (1981) - Robust Statistics

---

## License

MIT License - see LICENSE file for details

---

## Contact

For questions or issues, please open an issue on the GitHub repository.
