# Frozen specialist → residual funnel (2026-08-10)

This is a sequential funnel, not a full factorial search.  All specialist views
come from the frozen seven-view, side-specific contract; no view was rediscovered
inside a transport fold.

The previously frozen incumbent remains the CMI-selected six-head + AE/GMM arm
under the binary H12-net>+50 baseline. Its matched replay was top-1 +31.57,
top-5 −50.92, and top-10 −92.89 bps/trade. The ATR2/all-seven results below are
a target/query challenger and should not be presented as the same arm.

## Contract and evaluation

- OOS transport folds: July–August, September–October, and November 2024.
- Specialist views: seven per side, 68 fields per view, frozen from the audited
  cross-fold contract.
- Specialist queries: the selected construction is 4-hour bucket × side.
- Residual target: ordinal per-row net residual in bps around the prequential base
  expected-net map, fit with native LightGBM LambdaRank.
- Final ranking: pooled global score, then top 1%, 5%, and 10% tails.  It is not
  a per-timestamp top-k evaluation.
- Costs: 100 bps is applied once in the outcome labels/exit replay.

## 1. Specialist target/query/HPO funnel

The five predeclared specialist target contracts were all evaluated with the same
4-hour×side query and the same eight-trial broad HPO space.  Values below are on
the development validation (May–June 2024), before transport replay.

| Target | HPO winner | Top-1 net | Top-5 net | Top-10 net | Month mean | Month std | Worst month |
|---|---:|---:|---:|---:|---:|---:|---:|
| Binary H12 net > +50 bps | trial 2 | −33.74 | −59.54 | −77.57 | −61.67 | 4.68 | −66.35 |
| ATR spacing 1.5 | trial 6 | −30.26 | −52.99 | −60.68 | −53.21 | 15.06 | −68.27 |
| **ATR spacing 2.0** | **trial 2** | **−34.17** | **−51.21** | **−62.62** | **−50.10** | **16.34** | **−66.43** |
| Absolute spacing 2.0% | trial 4 | −89.04 | −63.25 | −77.36 | −65.27 | 3.14 | −68.41 |
| Triple barrier SL3/TP5 | trial 4 | −33.44 | −54.62 | −66.61 | −54.26 | 16.44 | −70.70 |

The additional truncation replay held the ATR2 winner parameters fixed and
compared LambdaRank truncation levels 5, 10, and 20.  Level 20 was best in that
bounded replay (top-5 net −53.77 bps versus −54.51 at level 10); this is recorded
as a specialist-stage challenger and was not silently substituted into the
downstream stack without a matched residual replay.

ATR2 wins the declared primary development criterion (global top-5 net EV); the
specialist stage itself is not profitable and is only a representation-selection
stage for the residual learner.

### Specialist HPO winner parameters (ATR2)

`max_depth=4`, `num_leaves=16`, `min_child_samples=776`,
`min_sum_hessian_in_leaf=28.08`, `min_gain_to_split=0.00333`,
`feature_fraction=0.840`, `bagging_fraction=0.730`, `lambda_l1=0.000123`,
`lambda_l2=1.746`, `max_bin=127`, economic-step label gains.

## 2. Residual query construction and HPO

The ATR2 specialist winner was carried unchanged into the residual stage.  The
residual query grid was exact timestamp×side, 1-hour×side, and 4-hour×side;
four broad HPO trials were run per query on the first transport fold as a bounded
development proxy.

| Residual query | Best proxy top-5 net | Best proxy top-1 net |
|---|---:|---:|
| Exact timestamp×side | −153.82 | −74.55 |
| 1-hour×side | −159.75 | −74.22 |
| **4-hour×side** | **−126.22** | **−63.85** |

The selected residual HPO parameters are:
`max_depth=5`, `num_leaves=52`, `min_child_samples=893`,
`min_sum_hessian_in_leaf=1.13`, `min_gain_to_split=0.00893`,
`feature_fraction=0.788`, `bagging_fraction=0.867`, `lambda_l1=0.0309`,
`lambda_l2=0.170`, `max_bin=63`, moderate-tail label gains.

### Specialist query construction replay

With ATR2 held fixed, the specialist-only May–June replay favored exact
timestamp×side (top-5 net −47.64 versus −53.08 bps for 4-hour×side). The
downstream residual comparison is the decisive test:

| Specialist query fed to residual | Top-1 net | Top-5 net | Top-10 net | Worst month |
|---|---:|---:|---:|---:|
| Exact timestamp×side | −6.56 | +7.18 | −42.53 | −184.69 |
| **4-hour×side** | **−7.30** | **+8.89** | **−37.63** | **−171.07** |

The 4-hour query is retained because the production decision is the downstream
residual stack: it wins global top-5 net and has the less-bad worst month.
This distinguishes specialist standalone ranking from stack-level selection.

### Residual relevance-grade ablation

| Grade definition | Top-1 net | Top-5 net | Top-10 net | Worst month |
|---|---:|---:|---:|---:|
| Default (−150/−50/+50/+150 bps) | −7.30 | **+8.89** | −37.63 | −171.07 |
| Tight (−100/−50/+50/+100 bps) | −3.49 | +8.62 | −38.05 | −177.17 |
| Wide (−200/−75/+75/+200 bps) | −3.24 | +6.90 | −39.09 | −180.65 |
| Symmetric-50 (−50/−25/+25/+50 bps) | **+0.24** | +8.06 | **−34.91** | **−163.79** |

The default ordinal target remains the strict global top-5 winner. Symmetric-50
is a stability/ top-1 challenger, but does not replace the default under the
primary exact top-5 rule.

### Transport replay of the selected residual stack

| Tail | Gross bps/trade | Net bps/trade |
|---|---:|---:|
| Top 1% | 92.70 | −7.30 |
| **Top 5%** | **108.90** | **+8.89** |
| Top 10% | 62.37 | −37.63 |

Monthly top-5 net EV: July −51.55, August −171.07, September −58.25,
October −81.07, November +11.00 bps/trade.  Thus the pooled top-5 pass is
not yet execution-ready: the worst month is materially negative.

### Side decomposition of the selected control

| Side | Top-1 net | Top-5 net | Top-10 net | Worst monthly top-5 |
|---|---:|---:|---:|---:|
| Long | −24.06 | **+17.31** | **+11.49** | −144.78 |
| Short | −191.70 | −195.74 | −171.67 | −237.83 |

The pooled improvement is therefore long-driven. The short side is a hard
execution-readiness failure and must not be hidden by global aggregation.

The matched residual-only control was −59.63 / −93.11 / −98.77 bps at top
1%/5%/10%; the specialist residual stack therefore improves the top-5 tail by
102.0 bps/trade versus that control, while remaining unstable by month.

## 3. Base-tail and top-40% specialist gates

The base-tail head uses the canonical R3 classes (clear = robust-clear b25,
adverse = lower-touch first, weak = remainder), the frozen 32-feature per-side
base contract, and the incumbent base learner parameters.  The specialist gate
uses the broad base score’s top 40% within timestamp×side for specialist training;
test candidates remain the full population.

| Arm | Top-1 net | Top-5 net | Top-10 net | Month mean | Month std | Worst month |
|---|---:|---:|---:|---:|---:|---:|
| **Control (selected stack)** | **−7.30** | **+8.89** | **−37.63** | −70.19 | 58.95 | −171.07 |
| Base-tail head | −3.13 | +5.16 | −38.93 | −72.84 | 62.89 | −181.87 |
| Specialists trained on top 40% | −3.87 | +7.16 | −37.86 | −71.61 | 65.01 | −185.18 |
| Base-tail + top-40% specialists | −1.31 | +6.44 | −38.79 | −72.14 | 66.40 | −189.50 |

Under the declared global top-5-first selection rule, none of the gates advances.
The base-tail head slightly improves top-1, but sacrifices top-5 and worsens
worst-month behavior; it is not a safe promotion.

Top-5 side check (net bps/trade): control long +17.31 / short −195.74;
base-tail +17.51 / −199.10; top-40% specialists +15.93 / −197.77;
combined +18.32 / −199.47.  None repairs the short-side failure.

## 4. Coarse 15-minute exit grid

The final selected control score was replayed on the historical 15-minute source
(`15m_ohlcv_perp`) over the 12-hour horizon.  This is a coarse execution proxy,
not minute-level execution.  Grid: stop `{1, 1.5, 2, 2.5, 3}` ATR, activation
`{0.5, 1, 1.5, 2, 3}` ATR, giveback `{0.25, 0.5, 0.75, 1}` ATR.

The best grid point by top-5 net, then top-1 net, is:

`SL = 3.0 ATR; trailing activation = 0.5 ATR; giveback = 0.25 ATR`

| Tail | Gross bps/trade | Net bps/trade |
|---|---:|---:|
| Top 1% | 231.99 | +131.99 |
| **Top 5%** | **131.99** | **+31.99** |
| Top 10% | 121.25 | +21.25 |

These exit results should not be conflated with the fixed H12 label results: the
exit grid changes the path-to-PnL mapping, is evaluated on the same global score
ordering, and uses the coarser 15-minute path convention.

## Decision

Keep the control stack: frozen ATR2 specialists → 4-hour×side residual LambdaRank.
Do not promote the base-tail or top-40% specialist gates.  The next validation
requirement is a genuinely untouched later period and a causal admission rule;
the current July–November transport still has a negative worst month despite a
small pooled top-5 improvement.

Artifacts:

- Specialist HPO: `data_perp/artifacts/frozen_specialist_query_hpo_20260810_v1/`
- Specialist query replay: `data_perp/artifacts/frozen_specialist_query_construction_20260810_v1/`
- Specialist truncation: `data_perp/artifacts/frozen_specialist_truncation_ablation_20260810_v1/`
- Specialist-query residual impact: `data_perp/artifacts/frozen_specialist_query_residual_impact_20260810_v1/`
- Residual HPO/replay: `data_perp/artifacts/frozen_residual_query_hpo_20260810_v1/`
- Residual grade ablation: `data_perp/artifacts/frozen_residual_grade_ablation_20260810_v1/`
- Base-tail/gates: `data_perp/artifacts/frozen_base_tail_top40_gate_20260810_v1/`
- Exit grid: `data_perp/artifacts/frozen_selected_stack_exit_grid_20260810_v1/`

Additional requested meta/specialist ablations are consolidated in
`docs/ADDITIONAL_META_SPECIALIST_ABLATIONS_20260810.md`.
