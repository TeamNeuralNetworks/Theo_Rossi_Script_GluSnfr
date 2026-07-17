# Train Model Notes

## Purpose

This note records the modeling guidelines discussed for the train-mode extension built on top of the MPFA and relative-change dashboard.

The key point is that the train plots should distinguish clearly between:

- latent biological variables that evolve across the train;
- apparent event-wise estimates recovered from many-trial fits performed independently at each stimulus number.

## Current default scenario

Unless changed explicitly, the train module should use these starting conditions:

- starting $P_1 = 0.25$;
- total $N = 8$ release sites;
- $Q = 0.5\ \Delta F/F_0$ per released quantum (72A-fast iGluSnFR);
- no imposed increase in total $N$;
- refill = $1$ vesicle per stimulus;
- stimulation frequency = $20$ Hz.

At $20$ Hz, a refill of $1$ vesicle per stimulus corresponds to an effective refill flux of $20$ vesicles/s.

## What the dynamic plots should mean

All recorded signals in this project are direct **72A-fast iGluSnFR
fluorescence** ($\Delta F/F_0$), not postsynaptic currents. $Q$ is the
per-quantum fluorescence scale. The fluorescence readout is treated as
release-linear ($F = L$): the reported amplitude is proportional to released
glutamate, with no history-dependent response-loss mechanism.

The dynamic $N$ or $P_r$ curves shown in train mode should **not** be interpreted immediately as the true latent total $N$ or true latent $P_r$.

What those curves report, when estimated from data, is better described as:

- apparent $N_k$;
- apparent $P_k$;

where each value is obtained by evaluating stimulus $k$ independently across many trials.

In practice, that means:

1. collect many repeated trains;
2. isolate all responses at stimulus number $k$;
3. fit that stimulus alone with the same event-wise MPFA or MLE statistics;
4. report the fitted quantity as an **apparent** parameter for stimulus $k$.

Therefore, the dynamic plots are event-wise statistical summaries, not direct readouts of the latent presynaptic control variables.

## Why apparent and latent quantities differ

Even if the underlying biological control variable follows a simple monotonic rule, the observed response at each pulse is shaped by empty sites and refill.

For example:

- a latent increase in $P_r$ may be partially masked by depletion of available vesicles;
- a latent increase in total competent sites may not be visible immediately if many sites are empty;
- a flat apparent $P_k$ curve does not prove that the latent $P_r$ is flat.

This is why the train model should separate:

- latent total capacity or release drive;
- instantaneous occupancy of ready sites;
- apparent per-stimulus fit outputs.

## Recommended state decomposition

The train model should be written with at least three levels.

### 1. Latent drive variables

These describe what the synapse would like to do if empty-site limitations were absent.

- latent release probability drive: $P_{\mathrm{latent},k}$;
- latent total competent sites: $N_{\mathrm{latent},k}$.

### 2. Occupancy state

This tracks the number of vesicles or sites actually ready at each pulse.

- ready pool before pulse $k$: $R_k$;
- refill between pulses;
- depletion caused by release at pulse $k$.

### 3. Apparent event-wise outputs

These are the quantities recovered from many-trial fits stimulus by stimulus.

- apparent $P_{\mathrm{app},k}$;
- apparent $N_{\mathrm{app},k}$;
- apparent mean, variance, cumulative release.

## Recommended latent evolution law

For the latent variables, a saturating exponential is a good default choice.

Instead of assuming a linear change over pulse number, use an inverse exponential approaching an asymptote:

$$
x_k = x_{\infty} + (x_1 - x_{\infty}) e^{-(k-1)/\tau_x}
$$

where $x_k$ can represent either latent release probability or latent total competent sites.

### Latent release probability

$$
P_{\mathrm{latent},k}
=
P_{\max}
-
(P_{\max}-P_1)e^{-(k-1)/\tau_P}
$$

for facilitation toward a maximum probability.

If depression is intended instead, the same form can be written with a lower asymptote:

$$
P_{\mathrm{latent},k}
=
P_{\infty}
+
(P_1-P_{\infty})e^{-(k-1)/\tau_P}.
$$

### Latent total competent sites

$$
N_{\mathrm{latent},k}
=
N_{\max}
-
(N_{\max}-N_1)e^{-(k-1)/\tau_N}
$$

if recruitment occurs progressively toward a maximum competent-site count.

For the present default scenario, the latent total $N$ is fixed:

$$
N_{\mathrm{latent},k} = 8.
$$

## Occupancy and refill dynamics

The latent drive should then be filtered through occupancy dynamics.

A minimal mean-field form is:

$$
R_{k+1} = \min\left(N_{\mathrm{latent},k+1},\; R_k - \text{release}_k + \text{refill}_k\right).
$$

In the current implementation, replenishment is expressed in vesicles per
stimulus and evolves toward a steady state:

$$
RR_k = RR_\infty - (RR_\infty-RR_1)e^{-(k-1)/\tau_{RR}}.
$$

Thus the safe dynamic controls are latent $N_{\max}$, $\tau_N$, and the
replenishment trajectory $(RR_1,RR_\infty,\tau_{RR})$. The competent pool
$R_k$—not latent $N_{\max}$—is the apparent $N$ used in the event-wise
binomial moments. Depleted sites that have not refilled do not contribute to
$R_k$ (equivalently, they are sites with release probability zero).

Across related conditions, these controls should not be estimated as six
unrelated parameter sets. The demo uses the following hierarchy:

- both train frequencies share the pre-train $N_1/P_1$ state at each calcium;
- absolute latent $N_{\max}$ and $P_{\max}$ are non-decreasing with calcium
  and with stimulation frequency;
- constant replenishment is the default nested model, and evolving RR is kept
  only when its AIC improvement supports the two additional parameters;
- RR magnitude is bounded by the observed competent-pool scale, because refill
  far above the latent ceiling is observationally degenerate.

If the stimulus interval is $\Delta t$, then a more biophysical version could use:

$$
R_{k+1} = N_{\mathrm{latent},k+1} - \left(N_{\mathrm{latent},k+1} - R_k^{\mathrm{post}}\right)e^{-\Delta t/\tau_{\mathrm{refill}}}.
$$

That version is preferable once the frequency dependence becomes important.

## What the MLE-style event-wise statistics should target

If stimulus $k$ is analyzed independently across many trials, the fitted statistic should be labeled explicitly as an apparent quantity:

- apparent $P_{\mathrm{app},k}$;
- apparent $N_{\mathrm{app},k}$;
- apparent $Q_{\mathrm{app},k}$ if needed.

These are the values visible in the per-event MLE statistics, not the latent control trajectory itself.

This distinction is essential because the same latent trajectory can generate different apparent trajectories depending on depletion and refill.

## Consequence for the train-mode UI

The train-mode window should eventually expose two conceptual layers.

### Layer A: latent control parameters

- $P_1$, $P_{\max}$, $\tau_P$;
- $N_1$, $N_{\max}$, $\tau_N$;
- refill rate or refill time constant;
- stimulus frequency or ISI.

### Layer B: apparent outputs

- apparent $P_{\mathrm{app},k}$ curve;
- apparent $N_{\mathrm{app},k}$ curve;
- pulse-by-pulse mean response;
- pulse-by-pulse variance;
- cumulative and normalized cumulative release.

The UI should make clear that the displayed apparent curves are downstream consequences of the latent controller plus occupancy dynamics.

## Recommended modeling workflow

1. Fix the A1 baseline from the main dashboard.
2. Define latent $P_{\mathrm{latent},k}$ and, if needed, latent $N_{\mathrm{latent},k}$.
3. Simulate depletion and refill to obtain the ready pool $R_k$.
4. Generate many trial-level trains.
5. Re-fit each stimulus independently across trials.
6. Report those recovered values as apparent per-stimulus quantities.
7. Compare latent-model predictions to the apparent train statistics.

## Practical interpretation rule

If a train plot says that $N$ or $P_r$ changes over pulse number, the default interpretation should be:

> this is the apparent event-wise estimate that would be obtained by fitting that stimulus independently across many trials.

It should **not** be described automatically as the true latent $N$ or true latent $P_r$ unless the occupancy model and inverse problem have already been solved.

## Short takeaway

The train extension should ultimately become a two-stage model:

- a latent saturating control law, likely inverse exponential toward $N_{\max}$ or $P_{\max}$;
- a depletion/refill observation layer that produces the apparent event-wise $N$ and $P_r$ curves.

That distinction should be preserved in both the code and the displayed documentation.
