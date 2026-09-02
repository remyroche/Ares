# Combined regime + CMI + specialist-error residual diagnosis

## Combined arm

The combined arm used:

- five regime-grouped specialist heads: volatility, trend, transition, entropy,
  and composite;
- the frozen 160-field specialist contract;
- every raw/derived field used to form those regime groups in the residual meta
  input;
- five greedy binned-CMI additions restricted to configured meta-key families;
- 7-day prior-only specialist IC, hit-rate, and hit-rate-surprise features;
- q4h×side residual queries, ordinal per-row net residual target, and max depth 4.

| Tail | Net bps/trade | Gross bps/trade |
|---|---:|---:|
| Top 1% | −9.37 | 90.63 |
| Top 5% | **+8.67** | 108.67 |
| Top 10% | −44.63 | 55.37 |

Monthly global top-5 net was:

| July | August | September | October | November |
|---:|---:|---:|---:|---:|
| −57.90 | −223.12 | −64.25 | −86.63 | **+13.70** |

Side top-5 net:

- Long: **+22.44 bps/trade**;
- Short: **−199.05 bps/trade**.

The combined arm therefore does not beat the 7-day-only error-history arm
(+9.71 top-5, worst month −172.39), and it materially worsens worst-month risk.

Artifacts:

- `data_perp/artifacts/combined_regime_cmi_error_meta_20260810_v1/`
- `data_perp/artifacts/combined_regime_cmi_error_meta_20260810_v1/feature_contract.json`
- `data_perp/artifacts/combined_regime_cmi_error_meta_20260810_v1/side_month_metrics.parquet`

## What causes the bad months?

### 1. The pooled positive result is not a positive typical month

The global top-5 contains approximately:

- 71.8% November observations, versus 19.0% of the total population;
- 14.0% October;
- only 3.4% August.

The global top-5 net by month is −59.68, −86.14, −42.57, −76.61, and +38.52
for July through November. Thus pooled top-k EV is positive because the global
ranking concentrates on the one favorable regime/month, not because the ranking
is stable month by month.

### 2. Cross-side score scale is unstable

The global ranking is almost entirely long-driven overall, but August is the
exception: approximately 74.8% of its global top-5 is short. That month’s short
top-5 is −249.46 bps/trade. November instead has essentially no short trades in
the global top-5, while its long top-5 is only −19.67 on the side-local tail and
the pooled global tail is positive because it selects the extreme November long
scores.

This is a score comparability/admission failure, not simply a lack of raw signal.
The same numerical score has different side/month meaning:

- August long score 95th percentile: approximately −94.1;
- August short score 95th percentile: approximately −91.5;
- November long score 95th percentile: approximately −78.9;
- November short score 95th percentile: approximately −97.8.

The August short scores look competitive globally even though their realized tail
is deeply negative.

### 3. The score-to-net relationship reverses in bad months

Spearman correlation of combined score with realized net in the top-level
transport months:

| Month | Long | Short |
|---|---:|---:|
| July | −0.039 | −0.161 |
| August | −0.196 | −0.292 |
| September | +0.052 | −0.063 |
| October | +0.060 | −0.093 |
| November | +0.060 | −0.001 |

August is a genuine ranking reversal for both sides, especially short. November
is not a uniform positive state; it is mainly a favorable long conversion regime.

### 4. The residual layer cannot correct a base economic mismatch

For the selected top-5:

| Month / side | Realized net | Base expected net | Conversion error |
|---|---:|---:|---:|
| July long | −55.92 | −87.95 | +32.03 |
| July short | −172.95 | −97.31 | −75.64 |
| August long | −162.82 | −89.18 | −73.64 |
| August short | −249.46 | −90.43 | −159.03 |
| September long | −50.37 | −85.87 | +35.50 |
| October long | −74.34 | −84.45 | +10.11 |
| November long | −19.67 | −76.29 | +56.61 |
| November short | −238.62 | −97.15 | −141.46 |

The residual learner is most wrong exactly where the month fails: August short
and November short are severe under-conversion cases. Adding context features
does not help if the side-local expected-net map and residual semantics do not
transport.

### 5. Regime states have shifted economic meaning by side and era

The causal regime distributions do drift, but the larger issue is payoff
semantic drift within the same state. Examples from the train/calibration/test
audit:

- November long volatility state 1 moved from calibration −128.1 to test
  +30.7 bps;
- November short volatility state 1 moved from calibration −65.1 to test
  −220.6 bps;
- November short trend state 3 moved from calibration −75.2 to test −206.4;
- November short transition state 3 remained prevalent but moved from
  calibration −82.9 to test −197.9.

The state labels are therefore useful descriptors, but they are not stable
payoff classes. A single residual model is being asked to map one state to
different economics across eras and sides.

## Diagnosis

The bad months are caused by a combination of:

1. global top-k concentration in one favorable month;
2. side-score scale shifts that admit shorts in the wrong months;
3. negative score-to-net rank correlation during regime transitions;
4. base expected-net calibration failure, especially on shorts;
5. regime semantic/payoff drift that the current residual features identify but
   do not reliably transport.

This is primarily a conversion/admission and cross-side calibration problem,
not evidence that the specialist representation has no information.

## Recommended next experiments

1. Use a prior-only side × regime expected-net map with hierarchical shrinkage,
   then rank after conversion to common bps; never compare raw side scores.
2. Add a strict admission gate requiring the side-local 21-day map to estimate
   at least +50 bps net before global pooling.
3. Train the residual on conversion error after this side/regime map, with a
   conservative shrinkage-to-base rule when regime support is weak.
4. Add a side-specific regime support/OOD feature: distance from the training
   regime distribution and minimum state support.
5. Select by equal-month or worst-month-gated objectives during development;
   pooled global top-k alone is explicitly rewarding November concentration.
6. Treat the short side as a separate research problem until it clears a
   positive causal admission test.

Detailed machine-readable outputs are in `bad_month_side_summary.parquet`,
`regime_drift.parquet`, `bad_month_diagnostic.parquet`, and
`bad_month_cause_summary.json` under the combined artifact directory.
