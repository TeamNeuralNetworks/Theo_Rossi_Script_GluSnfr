# TODO — estimate per bouton, aggregate last

**Where this runs:** the machine with `PPR_DATA_FINAL` + `Support_figure.ipynb`
(the NNLS/MLE pipeline). Not runnable on the MPFA/MATLAB machine — that side only
consumes the exported CSVs.

**Who needs it:** the MATLAB side (`mpfa_fit_demo.m`, `mpfa_make_figures.m`,
`mpfa_demo_invert_varmean.m`) currently mixes estimands and cannot be fixed
downstream. This has to be fixed at the source.

---

## 1. The problem in one line

The three PPR estimates disagree because each one **collapses the bouton
population at a different point in the calculation**, and those operations do not
commute. It is not a physical effect.

Observed at 2.5 mM / 20 Hz, pulse 2 (all from the same amplitudes):

| quantity | value | how the population is collapsed |
|---|---|---|
| measured PPR (`PPR2/1` column) | **1.380** | per bouton, then unweighted mean of ratios |
| variance-to-mean inversion | **2.365** | ensemble first (ratio of means) |
| MLE-based train model | **2.668** | ensemble N and P first, then multiplied |

The var/mean and MLE routes agree with each other (R² = 0.91–0.97 at 50 Hz) and
both sit 1.3–2.6× above the measured PPR. **Two chains agreeing does not validate
them** — it only means they share the same aggregation order.

### Why they differ (the algebra)

With per-bouton ratio $r_b = A_{k,b}/A_{1,b}$:

$$\underbrace{\frac{\sum_b A_{k,b}}{\sum_b A_{1,b}}}_{\text{ensemble / inversion}}
= \underbrace{\bar r}_{\text{measured PPR}}\left[1 + \frac{\operatorname{cov}(A_1, r)}{\bar A_1 \, \bar r}\right]$$

i.e. the ensemble ratio is an **$A_1$-weighted** mean of exactly the same
per-bouton ratios. The gap is a covariance term, nothing else.

The model drops two further covariances:
$\overline{NP} = \bar N \bar P + \operatorname{cov}(N,P)$, and since `Q_bouton` is
per bouton, $\overline{NPQ} \neq \bar N\,\bar P\,\bar Q$.

These are the same heterogeneity terms the framework already formalises
($Q_{app}=Q(1+CV_Q^2)$, $N_{app}=N/(1+CV_P^2)$, $P_{app}=P(1+CV_P^2)/(1+CV_Q^2)$).
The ensemble N/P are **apparent** parameters of a heterogeneous population; the
per-bouton PPR is a different estimand. No correction factor downstream can
reconcile them — which is precisely why a free `K_sat` parameter appeared to
"work": it was absorbing a covariance term and calling it sensor saturation.

---

## 2. Run these two diagnostics FIRST (cheap, and one of them can falsify the story)

Do not restructure anything until these are done. There is a competing
explanation — **different bouton subsets** — that produces the same sign and size
of gap with no covariance story at all. The var/mean CSV reports 192 boutons for
2.5 mM / 20 Hz; it is unknown whether the profiles matrix used the same 192.

**Diagnostic A — same-list check.**
On one identical bouton list, compute both:
- `mean_of_ratios = mean_b(A_k,b / A_1,b)`
- `ratio_of_means = sum_b(A_k,b) / sum_b(A_1,b)`

*If they still differ ~1.7× → covariance/aggregation is the cause.*
*If they now agree → the original gap was subset selection, and the covariance
story below is wrong.*

**Diagnostic B — sign check (falsification).**
Compute `corr(A_1,b , r_b)` across boutons, per condition.
- The aggregation explanation **requires this to be positive** (big-A1 boutons
  facilitate more).
- Classic synaptic physiology predicts the **opposite** (low-P / small-A1 boutons
  facilitate more).
- **If this comes out negative, the explanation above is wrong** and the gap must
  be selection or something else. Stop and re-diagnose.

Report both numbers per condition before proceeding.

---

## 3. Objective

**Estimate everything per bouton, then aggregate once, at the very end, over one
fixed bouton list.**

For each condition (Ca × frequency) and each bouton *b*:

1. Per-pulse amplitude `A_b,k` (existing NNLS `AMP1_UNCORR` + `AMP{k}_CORR`).
2. Per-bouton quantal size `Q_b` (existing `Q_bouton`).
3. Per-bouton MLE per pulse → `N_b,k`, `P_b,k` (existing `fit_binom_mle`,
   `get_count_row_with_failures`).
4. Per-bouton derived series, formed **before** any averaging:
   - `PPR_b,k = A_b,k / A_b,1`
   - `VMR_b,k = var_t(A_b,k) / mean_t(A_b,k)` (across trials, within bouton)
5. **Only then** aggregate across boutons (report both mean and median, plus a
   bootstrap CI over boutons).

The invariant to preserve: **Pr, N and PPR must be aggregated the same way, over
the same bouton list.** That single rule is what makes the held-out PPR test
meaningful.

---

## 4. Fixed bouton list (fair comparison)

Use **only boutons that survive the MLE** for every series — i.e. exclude the
ones already dropped by the MLE criteria (notably the `cap20` exclusion implied
by `Median_MLE_P_A1_excluding_cap20`, plus any failure-threshold/min-trials
rejections).

Rationale: today the MLE-derived columns are computed on a filtered set while the
amplitude-derived columns may use everything, so part of the 1.3–2.6× may be pure
population difference. Restricting all series to the MLE-valid set removes that
confound.

Export the bouton count actually used per condition so the MATLAB side can report
it (`Boutons` in the current var/mean CSV is 21/192/21/15/44/18 — verify these
after filtering).

---

## 5. What to export

One CSV, replacing `FigS16_mpfa_demo_profiles_matrix.csv`, with all series
aggregated identically over the same MLE-valid bouton list:

```
Stim, Pr_<cond>..., N_<cond>..., PPR_<cond>...
```
for the six conditions in this order:
`1.5mM_20Hz, 1.5mM_50Hz, 2.5mM_20Hz, 2.5mM_50Hz, 4mM_20Hz, 4mM_50Hz`
(the existing column order — `mpfa_fit_demo.m` reads 1+c / 7+c / 13+c).

Please also add, per condition (new file or extra columns):
- `n_boutons_used`
- `agg_method` (`mean` or `median`) — whichever you choose, apply it to **all**
  series
- per-pulse CI (bootstrap over boutons) for Pr, N and PPR
- `corr_A1_ratio` from Diagnostic B

Keep `FigS16_var_imean_profiles_6conditions.csv` as-is for now, but ideally
regenerate `Bouton_mean_average` / `Bouton_variance_average` over the **same**
MLE-valid list so the var/mean route and the MLE route are finally comparable.

---

## 6. Acceptance criteria

1. Diagnostics A and B reported per condition.
2. All series aggregated identically over one MLE-valid bouton list.
3. On the new export, `(N_k/N_1)·(P_k/P_1)` should reproduce the exported `PPR_k`
   to within the bootstrap CI — **without any free parameter**. That is the whole
   point: if a `K_sat`-like fudge factor is needed again, the aggregation is
   still inconsistent somewhere.
4. `mpfa_fit_demo.m` should then be reverted to read its PPR target from the
   profiles matrix (it currently reads `Mean_percent_A1` from the var/mean CSV as
   a workaround for exactly this bug — see the header note in that file).

---

## 7. Notes / provenance gap

- The generator for **both** `FigS16_*` CSVs could not be found in
  `Support_figure.ipynb` or the NNLS repo. Those exact column names
  (`Mean_percent_A1`, `Bouton_mean_average`, `Fano_of_bouton_averages`,
  `Median_MLE_P_A1_excluding_cap20`) appear nowhere in the tracked code. Whatever
  produced them should be committed alongside this work, otherwise this is not
  reproducible.
- Verified by arithmetic on the file itself:
  `Mean_percent_A1 = 100 × Bouton_mean_average_k / Bouton_mean_average_1` (exact),
  `Fano_of_bouton_averages = Bouton_variance_average / Bouton_mean_average` (exact),
  and `Var_over_Imean` is a byte-for-byte duplicate of the Fano column.
- Relevant existing code: `ppr_profile_stats` (does
  `means = [1.0] + ppr_data[valid_cols].mean()` → the unweighted mean-of-ratios),
  `ppr_profiles_matrix`, `amplitude_profiles_matrix`, `fit_binom_mle`,
  `get_count_row_with_failures`, `Q_bouton`.
