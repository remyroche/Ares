# Strict-R3 T6/T9 meta-head selection — current canonical decision

**Status:** `RETAIN_FROZEN_T6_T9_NO_LIVE_PROMOTION`  
**Scope:** Long-only, offline research. This is the canonical handover for the
2026-08-26 new-base T6/T9 selection cycle. It does not change inference,
MC1, admission, portfolio, execution, or any exchange process.

## Decision

The current frozen correction heads remain the research-canonical contract:

```text
S11 = 0.75 × enhanced-base rank + 0.20 × frozen T6 rank + 0.05 × frozen T9 rank
```

The F72-style T6 candidate is rejected. The F72-style T9 candidate is retained
only as a research challenger: its two-month development advantage is too
small to establish temporal portability. No head is promoted to live use.

Machine-readable decision:
[strict_r3_meta_t6t9_f72_selection_decision_20260826_v1.json](../config/strict_r3_meta_t6t9_f72_selection_decision_20260826_v1.json).

## Evaluation contract

- **Base stream:** target-free B/E/T source
  `strict_r3_frozen_threeway_targetfree_20260826_v2`.
- **Route:** retain the deterministic timestamp-local top 30% of the enhanced
  base score before either correction head is fitted or scored.
- **Folds:** six complete months of resolved training labels followed by a
  28-day reserve; held months June and July 2026.
- **Labels:** semantic head targets are available strictly before each fitting
  cutoff. Canonical rich-policy outcomes are joined only after target-free held
  scores have been persisted, solely to calculate metrics.
- **Query:** 4-hour UTC cycle × side. Long-only rows are used in this decision.
- **Caveat:** the new B/E/T source first supports a full six-month fit in June
  2026. Two held months are adequate for a controlled candidate audit, but not
  for a cross-era portability or live-promotion claim.

## Base-geometry contract available to T6 and T9

Every held row has all 22 fixed geometry fields, with 100% finite coverage and
non-zero variation in both held months. The heads receive more than the final
base rank:

1. raw B, efficiency, timing, and enhanced-base values;
2. timestamp-local B/E/T ranks and enhanced-base rank;
3. `E−T`, `E−B`, `T−B`, component standard deviation, min/max/median/range;
4. routed-query count, score dispersion/range, and top-versus-next/top-2 gaps.

This preserves candidate-specific agreement/disagreement geometry as well as
the timestamp’s opportunity surface. These fields are target-free and are
constructed identically before training and held scoring.

## Retained baseline metrics

The baseline receipt is
`strict_r3_meta_newbase_t6t9_baseline_metrics_20260826_v3`. Metrics are
realised canonical rich-policy net bps per trade after score construction.

| S11 metric | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|
| Fixed *k* per timestamp | +191.75 | +140.20 | +102.72 | +68.95 | +29.15 |
| Global percentile tail | +408.70 | +349.63 | — | +232.61 | +141.28 |

The global tails are diagnostic only; timestamp-local selection is the primary
training/HPO metric. Its Top-2 stability is:

| Period resolution | Q1 | Q5 | Q25 | Median |
|---|---:|---:|---:|---:|
| Week (9 periods) | +49.43 | +55.61 | +92.32 | +102.01 |
| Month (2 periods) | +89.29 | +93.51 | +114.63 | +141.04 |

The frozen T6/T9 ranks are conditional-correction coordinates, not standalone
alpha. Their value is evaluated only within the combined score and later
requires a separate MC1/admission/portfolio replay before any replacement.

## Feature-selection process used for the challengers

The process mirrors the successful B0 F72 sequence, adapted for distinct
supportive targets and Top-3-focused downstream use:

```text
1,407 causal numeric fields
→ cross-month hygiene (>=95% aggregate and >=90% every-month coverage)
→ 1,023 eligible fields
→ full-model gain + tail SHAP + univariate rescue + randomized stability
→ Screen120
→ strict-OOF economic and Top-3-boundary MDA
→ semantic-family MDA
→ 120 / 90 / 70 / 50 / 35 / 25 subset ladder
→ one family add-back pass (up to four fields/family)
→ 24-trial HPO
```

Conditional mutual importance is deliberately not a selection term: T6 and
T9 use materially different targets. A feature has to improve the target’s
own strict-OOF *combined* score, rather than merely a standalone head score.

The HPO objective is:

```text
0.80 × (0.70 × timestamp Top-3 + 0.20 × Top-2 + 0.10 × Top-1)
+ 0.10 × weekly Q25 Top-2
+ 0.10 × monthly Q25 Top-2
```

It uses query-safe subsampling, 30-round early stopping and
`MedianPruner(startup=4, warmup-fold=1)`. All candidates use LightGBM L2
regression on their ordinal target values, then convert raw outputs to
timestamp-local ranks.

## T6 result — reject

**Target:** five bins of rank-error relative to the base score.  
**Selected causal fields:** 35; **fixed geometry fields:** 22.

| Combined-score result | Frozen T6 | F72 T6 candidate | Delta |
|---|---:|---:|---:|
| Mean Top-1 | +188.48 | +173.48 | −14.99 |
| Mean Top-2 | +141.89 | +134.26 | −7.63 |
| Mean Top-3 | +106.26 | +104.91 | −1.35 |
| Weekly Q25 Top-2 | +129.67 | +107.39 | −22.28 |
| Monthly Q25 Top-2 | +141.89 | +134.26 | −7.63 |
| Selection objective | +124.44 | +118.27 | −6.17 |

The candidate’s final parameters were depth 5, 22 leaves, learning rate
0.043443, minimum child 812, feature fraction 0.787158, subsample 0.838231,
L1 0.082971, L2 0.333865, minimum split gain 0.000398, and an 1,800-tree
ceiling with early stopping. Its selected raw causal fields are:

```text
vov_mad_60, trend_pct_mkt_resid, cs_rank_ret_24h, prog_eff_24,
dir_path_long_2h, asset_atr_level, choppiness_index_20,
oi_value_log_7d_robust_z, roc_div,
distance_to_support_daily_donchian_atr, log_bars_since_above_1atr,
bars_in_high_vol_state_log_norm, adx_di_minus_10,
distance_to_resistance_daily_donchian_atr, log_realized_vol, adx_7,
distance_to_support_weekly_donchian_atr, trend_retest_success_rate,
vol_high, oi_trend_10d_robust_z, asset_minus_mkt_oi_1d_cp_z_8_32_96,
path_efficiency_12, corr_eth_24h, rv_48h, vol_concentration_12, rv_120h,
price_x_oi_7d, adx_di_minus_7, atr_percentile,
price_minus_oi_recovery_72h, choppiness_cp_absratio_8_32,
asset_minus_universe_median_ret_24h, upside_semivariance_24,
atr_expansion_ts_resid, z_compression_expansion
```

Its selection artifacts are in
`strict_r3_meta_t6t9_f72_selection_20260826_v5_t6`. The candidate is not
used downstream.

## T9 result — research challenger only

**Target:** ordinalised five-state exit quality.  
**Selected causal fields:** 25; **fixed geometry fields:** 22.

| Combined-score result | Frozen T9 | F72 T9 candidate | Delta |
|---|---:|---:|---:|
| Mean Top-1 | +191.51 | +190.08 | −1.43 |
| Mean Top-2 | +140.45 | +141.83 | +1.39 |
| Mean Top-3 | +101.46 | +104.19 | +2.74 |
| Weekly Q25 Top-2 | +125.33 | +125.84 | +0.51 |
| Monthly Q25 Top-2 | +140.45 | +141.83 | +1.39 |
| Selection objective | +121.18 | +123.01 | +1.83 |

The candidate’s final parameters were depth 4, 35 leaves, learning rate
0.078777, minimum child 523, feature fraction 0.772471, subsample 0.729401,
L1 0.004790, L2 8.813355, minimum split gain 0.000718, and an 1,800-tree
ceiling with early stopping. Its selected raw causal fields are:

```text
dist_prior_day_low, pct_breakout_t, shannon_entropy_ret_8,
spread_proxy_wick_to_range_robust_z, vov_mad_60, shannon_entropy_ret_16,
spike_score, leverage_build, downside_semivariance_24,
liquidity_ratio_peer_resid, asset_vol_level, upside_semivariance_8,
range_per_volume, ffd_rv_2h_04, adx_di_minus_7, avg_pair_corr_24h,
breadth_dispersion, btc_ret_24h_pct, btc_ret_48h_pct, btc_ret_4h_pct,
btc_rv_ratio_1h24h_pct, btc_rv_ratio_4h24h_pct,
corr_concentration_24h, cross_asset_downside_corr_4h,
cs_dispersion_ret_24h
```

Its 2026-06 result is higher, but its combined objective is lower in July.
With only two held months, that is not portable replacement evidence. The
candidate remains an immutable research artifact at
`strict_r3_meta_t6t9_f72_selection_20260826_v1_t9`.

## Current frozen head contract

| Head | Physical slot | Target | Query | Features | Weight in S11 |
|---|---|---|---|---:|---:|
| T6 | `cap80_ordinary` | rank-error ordinal | 4-hour UTC × side | 102 | 20% |
| T9 | `cap120_equal_month` | five-state exit quality | 4-hour UTC × side | 73 | 5% |

The current physical-slot contract is
`strict_r3_o3v2_t6t9_consensus_contract_20260825_v1/selected_physical_slots.json`.
It remains the only permissible T6/T9 source for the current canonical
research stack.

## Reproducibility

| Purpose | Artifact or runner |
|---|---|
| New B/E/T target-free base source | `strict_r3_frozen_threeway_targetfree_20260826_v2` |
| Frozen new-base T6/T9 baseline | `strict_r3_meta_newbase_t6t9_baseline_20260826_v1` |
| Baseline metrics | `strict_r3_meta_newbase_t6t9_baseline_metrics_20260826_v3` |
| T6 selection and HPO | `strict_r3_meta_t6t9_f72_selection_20260826_v5_t6` |
| T9 selection and HPO | `strict_r3_meta_t6t9_f72_selection_20260826_v1_t9` |
| Runner | `scripts/run_strict_r3_meta_t6t9_f72_selection_v1.py` |
| Metric reporter | `scripts/report_strict_r3_meta_newbase_metrics_v1.py` |
| Rich-policy outcome ledger | `strict_r3_enhanced_base_rich_policy_labels_reconciled_20260823_v1/canonical_reconciled_policy_labels.parquet` |

Future work requires a compatible routed B/E/T history that creates more held
eras before either candidate can be retested for MC1/admission/portfolio
replacement. The completed two-month candidates must not be tuned further on
this period.
