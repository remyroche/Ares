# R5 posterior-contract tests and ablations — 2026-08-12

## Decision status

No arm was promoted. This is a matched challenger audit only. Promotion still
requires explicit user approval.

## Population and frozen contracts

- Long side only.
- Ten strict held months: October 2025 through July 2026.
- 1,172,161 identical scored candidate IDs in every arm.
- 1,165,677 valid frozen SimplePolicyOptimiser outcomes.
- Nine-month rolling trust-model training window, 60,000 equal-month sampled rows.
- Canonical Cell-day trim-15% EV input with a 28-calendar-day causal window.
- Policy outcome is already materialized: next-bar entry, frozen optimized trailing
  policy, H12 timeout, and 100-bps cost exactly once.
- Geometry/K9 identity is frozen; raw K9 posterior coordinates are excluded.
- Portfolio replay is the identical global auction: eight concurrent positions,
  two new entries per 15-minute bar, one per asset, 80% margin cap, 7x leverage.

## Questions tested

1. Does the top-30%-training/all-candidate-inference mismatch create bad extrapolation?
2. Does separating a neutral mean target from tail-risk weighting help?
3. Does independent-experience support improve shrinkage authority?
4. Does local aleatoric plus epistemic uncertainty calibrate better than one global noise term?
5. Does prior-OOS calibration of the clipped posterior into raw policy net improve admission?
6. Does a conservative `mean >= 50 bps AND P(EV > 0) >= 0.60` rule help?
7. Should positive and negative corrections receive asymmetric authority?

## Arms

| Arm | Contract |
|---|---|
| A0 current | Current R5: top-30%-only training, outcome-weighted mean, row-count support, all-candidate inference |
| A1 domain gated | A0, but authority/admission is limited to timestamp-local top 30% |
| A2 mixed weighted | 75% timestamp-top-30 + 25% lower-score reference rows; original outcome-weighted mean |
| A3 mixed neutral | A2 with uniform mean weights; aggressive false-positive/loss weights remain risk-only |
| A4 independent local | A3 plus Kish/diversity effective support and local-leaf aleatoric + tree-disagreement epistemic variance |
| A5 calibrated | A4 posterior mapped to raw policy net by train-only Huber calibration; month `t` uses earlier resolved OOS months only |
| A6 calibrated P60 | A5, admitting only `calibrated mean >= 50` and `calibrated P(EV>0) >= 0.60` |
| A7 demotion only | A5 negative corrections apply fully; positive corrections are suppressed |
| A8 capped promotion | A5 negative corrections apply fully; positive corrections are capped at +50 bps |
| A9 strict promotion | A8 promotion additionally requires `P(EV>0) >= 0.70` and effective support >= 300 |

## Pooled score/admission economics

All values are frozen-policy net bps/trade. Tail percentages are taken globally
from each arm's admitted population, never per timestamp.

| Arm | Admitted | All admitted | Top 1% | Top 2% | Top 5% | Positive rate |
|---|---:|---:|---:|---:|---:|---:|
| A0 current | 22,918 | +127.47 | +493.91 | +421.45 | +354.43 | 56.88% |
| A1 domain gated | 21,536 | +137.18 | +489.07 | +428.69 | +360.50 | 58.53% |
| A2 mixed weighted | 19,721 | +127.19 | +392.06 | +389.08 | +336.46 | 57.84% |
| A3 mixed neutral | 40,199 | +106.01 | +446.75 | +384.22 | +320.78 | 55.65% |
| A4 independent local | 41,177 | +102.49 | +454.92 | +389.54 | +319.85 | 55.31% |
| A5 prequential calibrated | 25,178 | +140.63 | +293.32 | +305.61 | +274.22 | 59.15% |
| A6 calibrated P60 | 14,533 | +186.65 | +296.31 | +305.30 | +296.75 | 62.66% |
| A7 demotion only | 21,899 | +118.80 | +94.23 | +85.31 | +131.27 | 58.27% |
| A8 capped promotion | 24,891 | +138.90 | +174.69 | +264.04 | +272.76 | 59.10% |
| A9 strict promotion | 21,899 | +118.80 | +220.71 | +167.35 | +144.33 | 58.27% |

## Month-level all-admitted net EV

| Month | A0 | A1 domain gate | A5 calibrated | A6 P60 | A8 capped promotion |
|---|---:|---:|---:|---:|---:|
| 2025-10 | +201.74 | +203.62 | +168.51 | +200.64 | +168.51 |
| 2025-11 | +128.50 | +128.78 | +109.19 | +128.47 | +97.50 |
| 2025-12 | +94.39 | +94.63 | +76.78 | +111.66 | +75.52 |
| 2026-01 | +193.40 | +193.89 | +207.66 | +304.49 | +207.48 |
| 2026-02 | +202.46 | +202.46 | +248.70 | +320.62 | +248.46 |
| 2026-03 | +128.11 | +128.11 | +256.51 | +288.86 | +256.42 |
| 2026-04 | +126.09 | +152.10 | +165.04 | +279.47 | +161.99 |
| 2026-05 | +60.57 | +83.01 | +85.76 | +134.37 | +85.71 |
| 2026-06 | +140.19 | +140.19 | +143.42 | +112.12 | +143.42 |
| 2026-07 | +198.32 | +198.32 | +397.23 | +821.83 | +397.23 |

A6 has only 2 admitted candidates in June and 14 in July. Its very large July
mean is therefore not broad evidence.

## Portfolio-constrained replay

| Arm | Trades | Trades/day | Net bps/trade | Positive | Sortino | Max DD | Worst week |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 current | 3,972 | 13.07 | +161.22 | 62.19% | 0.490 | -43.81% | -31.51% |
| A1 domain gated | 3,951 | 13.00 | +159.29 | 62.47% | 0.482 | -43.81% | -31.51% |
| A2 mixed weighted | 3,681 | 12.11 | +158.09 | 61.67% | 0.451 | -59.90% | -12.17% |
| A3 mixed neutral | 5,183 | 17.05 | +145.44 | 62.30% | 0.518 | -64.53% | -19.62% |
| A4 independent local | 5,168 | 17.00 | +143.87 | 62.33% | 0.507 | -66.67% | -13.80% |
| A5 prequential calibrated | 4,142 | 13.63 | +168.02 | 63.79% | 0.482 | -49.07% | -33.87% |
| A6 calibrated P60 | 3,217 | 10.58 | +194.80 | 66.09% | 0.608 | -41.74% | +5.63% |
| A7 demotion only | 3,782 | 12.44 | +158.92 | 62.67% | 0.451 | -51.12% | -31.94% |
| A8 capped promotion | 4,095 | 13.47 | +164.46 | 63.15% | 0.465 | -49.07% | -33.87% |
| A9 strict promotion | 3,791 | 12.47 | +159.51 | 62.75% | 0.454 | -51.12% | -31.94% |

Portfolio caveat: A6 accepts only one portfolio trade in June (−100 bps) and
seven in July. Its pooled risk metrics are strong, but recent deployment support
is too thin for automatic promotion.

## Diagnostics

### Domain mismatch

A0 admitted 1,382 rows outside the timestamp-local top-30% domain. Those rows
averaged −23.75 bps, versus +137.18 bps inside the trained domain. Hard gating
therefore improves score-level admission economics, worst month (+60.57 to
+83.01 bps), and positive rate. It does not improve the stateful portfolio:
the auction already avoided most bad rows and the altered opportunity sequence
slightly reduces portfolio net EV by 1.93 bps/trade.

### Mean/risk separation

Uniform mean weighting does not help. A3 admits many more candidates but loses
21.46 bps/trade versus A0 and weakens the extreme tail. The aggressive weights
were not the sole reason the current mean worked.

### Effective support

The independent-experience formula is mechanically more defensible than row
count, but support remains economically inverted among A4 admissions:

| Support quintile median | Net bps/trade |
|---:|---:|
| 278.7 | +256.74 |
| 304.3 | +171.45 |
| 351.4 | +55.91 |
| 507.0 | +31.16 |
| 862.2 | −3.09 |

High support currently identifies common market cells, not necessarily reliable
positive-edge cells. It must not be treated as monotone trust authority without
conditioning on cell economics/regime.

### Uncertainty

| Arm | Nominal 50% | Nominal 80% | Nominal 90% | Median SD |
|---|---:|---:|---:|---:|
| A0 global residual noise | 72.06% | 92.04% | 95.93% | 361.1 bps |
| A4 local aleatoric + epistemic | 59.85% | 83.46% | 90.48% | 259.4 bps |

Local uncertainty is materially better calibrated. This part of A4 works even
though A4's posterior mean/ranking does not.

### Prequential calibration

The Huber slope declines causally from 0.917 in November 2025 to 0.622 in July
2026, while the intercept moves from +9.54 to −35.04 bps. This confirms that the
clipped residual posterior overstates raw policy-net scale increasingly over
time. Calibration improves all-admitted and constrained-portfolio EV, but its
month-specific affine maps reduce pooled extreme-tail comparability.

### Asymmetric promotion versus demotion

- Demotion-only is too destructive: it removes useful positive corrections and
  loses both score-level and portfolio EV.
- Capping promotions at +50 bps restores most of symmetric calibration's value,
  but remains below A5 on all-admitted and portfolio EV.
- Requiring both 70% positive probability and support >= 300 effectively turns
  the arm back into demotion-only; the present support measure is unsuitable as
  a promotion authority.
- The useful asymmetry in this audit is at admission, not at correction sign:
  A6 retains symmetric calibrated means but demands stronger evidence before
  admitting a trade.

## Correctness tests

`18 passed`:

- trust-model feature/geometry semantics;
- independent-experience support bounds;
- local uncertainty output validity;
- mixed-domain reference contract;
- future/held outcome mutation cannot alter prequential calibration;
- calibration starts cold and then consumes prior resolved OOS rows only;
- inference-bundle compatibility.

## Conclusion

No unambiguous promotion candidate exists yet:

- A1 is the cleanest structural repair and improves unconstrainted admission,
  but it is neutral/slightly negative after portfolio constraints.
- A4 validates the uncertainty repair but rejects the independent-support mean.
- A5 improves portfolio net EV by +6.80 bps/trade, but worsens drawdown and
  extreme-tail ranking.
- A6 improves portfolio net EV by +33.58 bps/trade, Sortino, drawdown, and
  positive rate, but its June/July support collapses to 1/7 executed trades.

The next safe step is an untouched forward shadow comparison of A0, A1, A5,
and A6, with an explicit minimum-activity gate. No canonical configuration or
production artifact was changed by this workstream.

## Follow-up: domain depth, A5 combination, and activity-preserving A6

The A1 timestamp-local domain was swept at 30/25/20/15/10%. Each threshold is
implemented as `position < ceil(candidates_at_timestamp * fraction)`, using
only the decision-time upstream score.

| A1 domain | Raw admitted | Raw net bps/trade | Raw worst month | Portfolio trades | Portfolio net bps/trade | Sortino |
|---|---:|---:|---:|---:|---:|---:|
| Top 30% | 21,536 | +137.18 | +83.01 | 3,951 | +159.29 | 0.482 |
| Top 25% | 21,116 | +139.34 | +86.34 | 3,942 | +160.18 | 0.486 |
| Top 20% | 20,626 | +141.71 | +89.44 | 3,932 | +160.99 | 0.488 |
| Top 15% | 19,850 | +145.40 | +93.71 | 3,920 | +162.54 | **0.492** |
| Top 10% | 18,250 | **+151.86** | **+100.44** | 3,863 | **+162.55** | 0.465 |

Top 15% is retained as the research winner. Top 10% has the highest broad raw
EV, but its portfolio advantage is only 0.006 bps/trade; top 15% has higher
Sortino, lower drawdown, 57 more portfolio trades, and better raw top-1% and
top-5% EV. Top 10% remains the high-selectivity sensitivity control.

Combining the selected top-10% domain with A5 causal calibration gives:

| Arm | Raw admitted | Raw net | Raw worst month | Portfolio trades | Trades/day | Portfolio net | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| A5 | 25,178 | +140.63 | +76.78 | 4,142 | 13.63 | +168.02 | -49.07% |
| A5 + top-10 domain | 20,654 | +152.60 | +82.80 | 4,023 | 13.23 | +171.64 | -57.97% |
| A5 + **top-15 domain** | 22,680 | **+147.24** | **+80.02** | **4,076** | **13.41** | **+172.37** | **-49.07%** |

The selected top-15 combination improves raw EV by +6.61 bps/trade and
portfolio EV by +4.35 bps/trade versus A5, preserves A5's drawdown, and executes
66 fewer trades. It is retained as a challenger, not promoted.

The A6 probability threshold was then refined:

| P(EV>0) threshold | Raw admitted | Raw net | Portfolio trades | Trades/day | Portfolio net | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| 0.550 | 25,178 | +140.63 | 4,142 | 13.63 | +168.02 | -49.07% |
| 0.570 | 23,645 | +147.71 | 4,071 | 13.39 | +169.52 | -49.07% |
| **0.5725** | 22,903 | **+151.77** | **4,034** | **13.27** | **+171.08** | **-45.98%** |
| 0.575 | 22,083 | +155.17 | 3,958 | 13.02 | +171.27 | -47.03% |
| 0.600 | 14,533 | +186.65 | 3,217 | 10.58 | +194.80 | -41.74% |

Relative to A5, every binding probability threshold necessarily removes trades;
0.550 is non-binding and therefore exactly reproduces A5. Relative to the A0
incumbent's 3,972 trades, 0.5725 preserves activity at 4,034 trades while adding
+9.87 bps/trade and improving drawdown versus A5. It is the activity-preserving
A6 research threshold. The difference from 0.575 is small, but 0.575 falls below
the incumbent trade count.

Recent support remains thin because the underlying A5 calibration admits few
June/July candidates: A6-0.5725 executes five June and 22 July trades. The lower
threshold fixes aggregate trade count, not the recent admission drought.

### Top-15 drought gate

The selected A5-plus-top-15 challenger does **not** pass the required activity
gate:

- October 2025-May 2026: 321-635 constrained trades/month;
- June 2026: 17 raw admissions and 5 constrained trades;
- July 2026: 34 raw admissions and 22 constrained trades;
- June 8-14, June 15-21, June 22-28, and the June 29-July 5 bridge week:
  zero constrained trades;
- July weekly executions are only 6, 8, 3, and 5 after the dry bridge week.

This is a genuine upstream/calibration admission drought: the portfolio cannot
create trades from only 17/34 admitted candidates. The long-period extension
and promotion assessment are therefore deferred. Extending backward could make
aggregate statistics look better while leaving the current failure unresolved.

### Bounded A5 repair

A5 was repaired by removing its authority over admission. The fixed floor is
A0 timestamp-top-15 admission; A5 may only modify ordering inside that pool.
The principal score is:

```text
admit = A0 admitted AND timestamp-local rank <= top 15%
score = A0 expected bps + 0.20 * (A5 calibrated bps - A0 expected bps)
```

Thus A5 cannot remove an A0 opportunity and cannot cause an additional drought.

| A5 use | Portfolio trades | Net bps/trade | Sortino | Max DD | Worst week |
|---|---:|---:|---:|---:|---:|
| A0 top-15 control | 3,920 | +162.54 | 0.492 | -43.81% | -31.51% |
| 10% A5 blend | 3,924 | +162.94 | 0.488 | -43.81% | -31.51% |
| 15% A5 blend | 3,924 | +163.52 | 0.492 | -43.81% | -31.51% |
| **20% A5 blend** | **3,924** | **+164.92** | **0.500** | **-43.81%** | **-31.51%** |
| 25% A5 blend | 3,921 | +164.86 | 0.499 | -44.55% | -31.51% |

The 20-25% neighborhood forms a plateau; 20% is the research winner because it
has marginally higher EV and avoids the 25% arm's drawdown deterioration.
Relative to A0 top-15, 20% improves constrained EV by +2.38 bps/trade and
improves eight of ten monthly means; April is -5.37 bps worse and July is
unchanged. It retains exactly the same two dry weeks—June 8-14 and June 15-21—
that already exist in A0 top-15. It does not create the additional June 22-July
5 drought caused by A5-owned admission.

Full A5 reranking and A5 promotion unions do not advance. Full reranking damages
raw tails; union admission dilutes raw EV and worsens portfolio drawdown. The
bounded 20% reranker remains a challenger requiring longer-period and untouched
forward validation before any promotion.

## Artifacts

- Aggregate score/admission audit:
  `data_perp/artifacts/strict_r3_r5_posterior_contract_ablation_long_2025oct_2026jul_20260812_v1`
- Portfolio arms:
  `data_perp/artifacts/strict_r3_r5_posterior_contract_portfolio_<arm>_long_2025oct_2026jul_20260812_v1`
- Monthly fold families:
  `data_perp/artifacts/strict_r3_r5posterior_audit_mixed_weighted_long_<month>_20260812_v1`
  `data_perp/artifacts/strict_r3_r5posterior_audit_mixed_neutral_long_<month>_20260812_v1`
  `data_perp/artifacts/strict_r3_r5posterior_audit_mixed_neutral_independent_local_long_<month>_20260812_v2`
- Domain/probability selection ledger:
  `data_perp/artifacts/strict_r3_r5_domain_probability_sweep_long_2025oct_2026jul_20260812_v8`
- Raw and monthly follow-up metrics:
  `data_perp/artifacts/strict_r3_r5_domain_probability_summary_long_2025oct_2026jul_20260812_v2`
- Follow-up portfolio arms:
  `data_perp/artifacts/strict_r3_r5_domain_probability_portfolio_<arm>_long_2025oct_2026jul_20260812_v1`
- Bounded A5 selection/summary:
  `data_perp/artifacts/strict_r3_a5_bounded_integration_long_2025oct_2026jul_20260812_v2`
  `data_perp/artifacts/strict_r3_a5_bounded_integration_summary_long_2025oct_2026jul_20260812_v2`
- Bounded A5 portfolio arms:
  `data_perp/artifacts/strict_r3_a5_bounded_portfolio_<arm>_long_2025oct_2026jul_20260812_v1`
