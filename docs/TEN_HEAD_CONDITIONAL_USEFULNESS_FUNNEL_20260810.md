# Ten-head conditional-usefulness residual funnel

## Decision status

This is a long-only research result. The final August–October period was not used for target, query, feature, or HPO selection. Promotion requires a strict conditional improvement in global Top‑1%, Top‑2%, Top‑5%, and worst-month Top‑5% net EV; no result is made canonical merely because it wins a pooled proxy.

## Fixed contract

- Source: `/Users/remyroche/Documents/Ares/data_perp/artifacts/strict_r3_schema_v2_source_panel_long_2022_2026_20260809_v1/canonical_source_panel.parquet`; prequential base ledger: `/Users/remyroche/Documents/Ares/data_perp/artifacts/strict_r3_full_inference_2025_2026_v2/predictions.parquet`.
- Development selection months: 2025-05, 2025-06, 2025-07; untouched final months: 2025-08, 2025-09, 2025-10.
- Feature contract: 120 frozen causal fields; source side: long only.
- Base inputs: independently prequential `base_rank` and `base_anchor_bps`. Residual target is `policy_net_bps − base_anchor_bps`.
- Outcome policy: next-hour entry; 12-hour 15-minute path; SL 3 ATR; trailing activation 0.5 ATR; giveback 0.25 ATR; 100 bps cost applied exactly once.
- Consensus: median of five feature caps (40/60/80/100/120) × ordinary/equal-month LambdaRank heads, then `0.75 × base_rank + 0.25 × consensus_rank`, globally ranked across the fixed candidate population.
- Frozen ranker defaults retained by every final head: 120 trees, learning rate 0.035, depth 5, 31 leaves, 300 minimum-child samples, 0.82 feature/bagging fractions, L1 0.02, L2 2.0, max-bin 127, gains `[0, .25, 1, 3, 7]`, truncation 10.
- Each head-score CDF is fitted on its mature training scores only; train labels must satisfy `policy_label_available_ts < held-month start`.

## Data and source-contract audit

- Requested source rows: 937,652; joined to prequential base ledger: 937,652; policy-valid residual rows: 474,401.
- The earliest authorized training-window field audit found 111/120 varying fields. Fields that were temporarily constant remain in the frozen 120-field contract and can only be removed by the conditional MDA stage.

## Search funnel

- Query pre-screen shortlist: q1_cycle_4h_side, q0_exact_timestamp_side, q1_cycle_12h_side.
- Full-stack conditional target/query candidates per head: 6. These included both target and query changes.
- Conditional MDA screen cap: 90,000 candidate rows; selected field subsets were then re-fitted and gated on the complete development population.
- Per-head HPO: 6 Optuna trials/head, TPE + aggressive MedianPruner; 2,000-tree ceiling with 30-round chronological early stopping for search, then full-mature-window refit at median selected tree count for promotion recheck.

### Target semantics screen (development only)

| Target | Grade/net ρ | Grade 0 net | Grade 4 net | Spread | Entropy |
|---:|---:|---:|---:|---:|---:|
| resid_wide_200_75 | +0.61 | -344.75 | +189.01 | +533.75 | +1.54 |
| resid_default_150_50 | +0.60 | -301.37 | +119.93 | +421.29 | +1.59 |
| resid_tight_100_50 | +0.56 | -257.94 | +62.73 | +320.66 | +1.51 |
| resid_symmetric_50_25 | +0.51 | -219.22 | +33.08 | +252.30 | +1.28 |

## Global downstream net EV (bps/trade)

| Arm | Top‑1% | Top‑2% | Top‑5% | Worst month @5% | Top‑5 rows |
|---:|---:|---:|---:|---:|---:|
| Development control | +64.17 | +26.49 | -16.61 | -54.57 | 5545 |
| Development frozen winner | +103.54 | +64.29 | +5.16 | -36.51 | 5545 |
| Final untouched control | +0.99 | -5.11 | -39.95 | -76.38 | 3627 |
| Final untouched frozen winner | +16.14 | +4.44 | -35.29 | -59.99 | 3627 |

### Frozen-winner change versus exact matched final control

| Δ Top‑1% | Δ Top‑2% | Δ Top‑5% | Δ worst month @5% | Δ conditional utility |
|---:|---:|---:|---:|---:|
| +15.15 | +9.55 | +4.66 | +16.39 | +14.03 |

## Per-head conditional selection

| Head | Frozen target | Frozen query | Fields | Best T/Q Δutility | Best HPO Δutility | HPO passed | MDA fields | Promoted stage(s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cap40_ordinary | resid_default_150_50 | q0_exact_timestamp_side | 40 | +7.69 | -4.38 | no | 40 | target_query |
| cap40_equal_month | resid_default_150_50 | q0_exact_timestamp_side | 40 | +2.71 | +0.61 | no | 40 | target_query |
| cap60_ordinary | resid_default_150_50 | q1_cycle_4h_side | 15 | +5.78 | -0.86 | no | 15 | conditional_mda |
| cap60_equal_month | resid_default_150_50 | q0_exact_timestamp_side | 30 | +6.45 | -0.61 | no | 30 | conditional_mda |
| cap80_ordinary | resid_default_150_50 | q0_exact_timestamp_side | 80 | +3.91 | +0.41 | no | 80 | target_query |
| cap80_equal_month | resid_default_150_50 | q1_cycle_4h_side | 80 | +1.67 | -1.66 | no | 80 | none |
| cap100_ordinary | resid_default_150_50 | q0_exact_timestamp_side | 100 | +5.97 | -0.99 | no | 100 | target_query |
| cap100_equal_month | resid_default_150_50 | q1_cycle_4h_side | 100 | +0.67 | -1.61 | no | 100 | none |
| cap120_ordinary | resid_default_150_50 | q1_cycle_4h_side | 120 | +5.79 | +0.70 | no | 120 | none |
| cap120_equal_month | resid_default_150_50 | q0_exact_timestamp_side | 51 | +6.94 | -3.22 | no | 51 | conditional_mda |

### What advanced

- Target semantics: no alternative residual target passed conditional downstream promotion; every frozen head retains `resid_default_150_50`.
- Query construction: four heads advanced from the 4-hour × side query to exact timestamp × side; all other heads retained the 4-hour × side query.
- Conditional MDA: only `cap60_ordinary` (60→15), `cap60_equal_month` (60→30), and `cap120_equal_month` (120→51) passed their full-development subset recheck.
- Ranker HPO: 39 completed and 21 pruned trials; no HPO challenger passed the strict conditional promotion recheck, so every final head keeps the frozen defaults above.

### Conditional feature findings

The following are the strongest individual downstream conditional-MDA signals per head. They are diagnostics; only a full-subset refit that passed the strict full-development gate was retained.

| Head | Top conditional-MDA fields (bps utility loss when permuted) |
|---:|---:|
| cap100_equal_month | mkt_oi_dispersion_24h (+1.18), distance_to_resistance_atr (+0.92), mark_perp_dislocation (+0.59) |
| cap100_ordinary | mark_perp_dislocation (+4.36), distance_to_resistance_atr (+2.23), ob_trade_size_to_l1_depth_z_24h (+1.74) |
| cap120_equal_month | mark_perp_dislocation (+1.71), prior_volatility (+1.48), distance_to_resistance_atr (+1.41) |
| cap120_ordinary | prior_volatility (+1.24), mkt_pct_oi_drawdown_24h_lt_minus5pct (+1.18), volume_percentile (+0.61) |
| cap40_equal_month | mark_perp_dislocation (+2.61), mkt_pct_price_up_oi_up_1h (+0.75), mkt_ret_15m (+0.73) |
| cap40_ordinary | mark_perp_dislocation (+3.86), xasset_mkt_depth_to_qv_z (+0.96), q_lower_tail__oi_3d_x_funding (+0.61) |
| cap60_equal_month | mark_perp_dislocation (+2.03), xs_dispersion__oi_value_1d_chg_z_90d (+0.52), breadth_chg_15m (+0.31) |
| cap60_ordinary | q_lower_tail__xasset_ob_liquidity_ts_resid (+0.29), log_bars_since_below_3atr (+0.29), q_tail_asym__vol_z_4h (+0.18) |
| cap80_equal_month | mark_perp_dislocation (+1.66), memory_asymmetry_1ATR (+0.85), median_alt_minus_btc (+0.83) |
| cap80_ordinary | mark_perp_dislocation (+2.99), grind_score_surprise (+0.99), median_alt_minus_btc (+0.90) |

### Head necessity before selection

| Head | Necessity | Utility change if removed |
|---:|---:|---:|
| cap80_equal_month | +7.11 | -7.11 |
| cap100_equal_month | +6.42 | -6.42 |
| cap40_equal_month | +4.06 | -4.06 |
| cap120_equal_month | +3.32 | -3.32 |
| cap60_equal_month | +3.08 | -3.08 |
| cap120_ordinary | +2.11 | -2.11 |
| cap40_ordinary | +0.48 | -0.48 |
| cap60_ordinary | -1.03 | +1.03 |
| cap100_ordinary | -1.40 | +1.40 |
| cap80_ordinary | -4.64 | +4.64 |

### Frozen leave-one-head-out attribution

Positive necessity means the complete frozen stack becomes worse when that head is removed. The deltas are the economics of removal, so negative values are evidence of a helpful head.

#### Development frozen winner

| Head | Necessity | Δ Top‑1 if removed | Δ Top‑2 if removed | Δ Top‑5 if removed | Δ worst month @5% if removed |
|---:|---:|---:|---:|---:|---:|
| cap120_equal_month | +3.14 | -2.77 | -1.03 | -2.99 | +0.54 |
| cap100_ordinary | +3.06 | -3.19 | -3.93 | -1.08 | +1.30 |
| cap60_equal_month | +1.90 | -1.86 | -2.74 | -1.39 | +1.50 |
| cap40_ordinary | +1.76 | +0.08 | -3.24 | -2.08 | +2.06 |
| cap100_equal_month | +1.62 | +0.16 | -3.06 | +0.63 | -2.35 |
| cap40_equal_month | +1.14 | -0.99 | -1.94 | -1.78 | +1.72 |
| cap80_ordinary | -0.14 | +2.02 | -0.72 | -0.03 | -3.54 |
| cap80_equal_month | -1.29 | +2.98 | -0.64 | +1.02 | -0.22 |
| cap60_ordinary | -1.46 | +1.56 | +1.21 | +2.24 | -0.79 |
| cap120_ordinary | -2.86 | +4.12 | +0.94 | +3.13 | -0.49 |

#### Final untouched frozen winner

| Head | Necessity | Δ Top‑1 if removed | Δ Top‑2 if removed | Δ Top‑5 if removed | Δ worst month @5% if removed |
|---:|---:|---:|---:|---:|---:|
| cap40_equal_month | +3.65 | -1.30 | -3.97 | -0.96 | -2.98 |
| cap80_ordinary | +3.64 | -2.94 | -4.13 | -1.70 | -1.52 |
| cap40_ordinary | +2.60 | +0.06 | -2.13 | -0.47 | -2.72 |
| cap80_equal_month | +2.01 | -4.23 | -2.07 | +1.76 | +0.78 |
| cap120_equal_month | +0.88 | +0.55 | -0.35 | -0.11 | -0.12 |
| cap100_ordinary | +0.87 | -1.61 | +1.57 | -0.44 | +0.96 |
| cap60_equal_month | +0.67 | +0.77 | -1.37 | +0.24 | -3.24 |
| cap100_equal_month | +0.05 | -1.35 | +0.77 | +0.14 | -0.08 |
| cap60_ordinary | -2.56 | +1.68 | -0.29 | +3.15 | +4.34 |
| cap120_ordinary | -2.63 | +1.65 | +1.42 | +2.20 | +1.98 |

## Time robustness

| Arm | Month | Top‑1% | Top‑2% | Top‑5% | Top‑5 rows |
|---:|---:|---:|---:|---:|---:|
| final_control | 2025-08 | -26.01 | -40.97 | -76.38 | 1235 |
| final_control | 2025-09 | +8.72 | -1.78 | -37.63 | 1192 |
| final_control | 2025-10 | -16.90 | +10.77 | -9.29 | 1202 |
| final_frozen_winner | 2025-08 | -2.34 | -32.12 | -59.99 | 1235 |
| final_frozen_winner | 2025-09 | +31.24 | +12.50 | -36.42 | 1192 |
| final_frozen_winner | 2025-10 | +1.49 | +10.93 | -10.45 | 1202 |

## Interpretation and next decisions

1. The frozen winner passes the *relative* final comparison: it improves Top‑1, Top‑2, Top‑5, the Top‑5 worst month, and conditional utility versus the exact control.
2. It is not execution-ready at a broad Top‑5% admission rate: final Top‑5 remains negative. Treat it as the research winner and preserve the canonical control for any rule requiring absolute positive Top‑5 net EV.
3. The durable evidence is query/feature-contract refinement, not a richer residual target or larger ranker. The next target research should address broad-tail economic separation rather than add HPO capacity.
4. This audit remains long-only. Repeat the same frozen methodology side-locally once an equivalent short prequential base-score ledger is available.

## Artifacts

- Artifact directory: `data_perp/artifacts/ten_head_conditional_usefulness_20260810_v1`.
- `target_query_conditional_trials.parquet`: every per-head target/query full-stack replacement.
- `conditional_feature_mda.parquet`: per-field full-stack conditional usefulness screen.
- `per_head_conditional_hpo_trials.parquet`: all completed and pruned HPO trials.
- `downstream_metrics.parquet` and `final_conditional_comparison.parquet`: global/montly matched economics.
- `development_frozen_winner_head_necessity.parquet` and `final_frozen_winner_head_necessity.parquet`: per-head leave-one-out attribution.
- `frozen_head_configs.json`: exact winning ten-head contract.
