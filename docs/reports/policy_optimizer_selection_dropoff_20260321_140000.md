# Policy Optimizer Selection Drop-Off Report

Run ID: `20260321_140000`

Generated: 2026-04-29

## Summary

The `policy_params/strategy_for_inference.json` artifact selecting only `2/8`
simple-position-sizer strategies is explainable from the persisted artifacts,
but the artifacts make the drop-off harder to audit than necessary.

The meta model and simple position sizer can both look good while policy
selection remains small because they evaluate different questions:

| Stage | What it measures | Why it can diverge from policy selection |
|---|---|---|
| Meta model | OOF event-level classification/ranking quality | Good top-slice hit-rate does not guarantee path-level profitability after exits, costs, and concurrency constraints. |
| Simple position sizer | OOF score concentration and pre-policy PnL | It is upstream of policy replay and does not prove TP/SL/trailing policy economics. |
| Policy optimizer | Trade-path economics after matched meta OOF, policy replay, and deployment gates | This can reject otherwise good rankers when realized path PnL deteriorates. |

The selected `2/8` is also not two independent signal families: both selected
rows are the base and `_tbm` variants of the same `dist_ema_fast...rsi_slope...`
family with identical policy metrics.

## Reconciliation

| Step | Count | Artifact / evidence |
|---|---:|---|
| Simple sizer OOF strategies | 8 | `data/artifacts/20260321_140000/oof/simple_sizer_oof_all.parquet` |
| Simple sizer downstream gate | 7 pass / 1 blocked | `ridge_sizer/strategy_params.json` |
| Policy candidate/meta OOF match | 6 matched / 1 skipped | one passing sizer candidate has no current `meta_oof` bucket |
| Policy optimized candidates | 6 | `policy_params/best_policy_params.json` |
| Policy deployment selected | 2 selected / 4 rejected | `policy_params/strategy_for_inference.json` |
| Holdout deployment selected | 2 selected | root `strategy_for_inference.json` |

## Simple Sizer Gate

Seven simple-sizer candidates passed the downstream economics gate. One was
blocked because its average PnL/trade was below the configured `0.2%` threshold.

| Strategy | Sizer net PnL | Wallet PnL | Avg PnL/trade pct | Gate |
|---|---:|---:|---:|---|
| `bars_in_high_vol_state_log_norm_-0_610...ema50_slope...` | 53.2944 | 5.9214 | 0.7086 | pass |
| `dist_ema_fast...rsi_slope...` | 27.7500 | 3.1444 | 0.7126 | pass |
| `dist_ema_fast...rsi_slope..._tbm` | 27.7500 | 3.1444 | 0.7126 | pass |
| `bars_in_high_vol_state_log_norm_0_453...volume_trend...` | 16.7919 | 1.9186 | 0.4305 | pass |
| `bars_in_high_vol_state_log_norm_0_453...volume_trend..._tbm` | 16.7919 | 1.9186 | 0.4305 | pass |
| `dist_prior_day_high...vol_z...` | 12.6336 | 1.5954 | 0.3297 | pass |
| `dist_prior_day_high...vol_z..._tbm` | 12.6336 | 1.5954 | 0.3297 | pass |
| `dist_prior_day_low...loc_prev_week_range...` | 6.8140 | 0.8819 | 0.1774 | blocked |

## The Missing High-PnL Sizer Candidate

The strongest simple-sizer candidate was:

`bars_in_high_vol_state_log_norm_-0_61002535_dist_weekly_vwap_-0_38453856_loc_ema_stack_pos_48_0_66408622_loc_prev_day_range_pos_24_0_41419065_atr_compression_ratio_0_97724169_ema50_slope_0_00070510805`

It passed the sizer gate with `net_pnl=53.2944`, `wallet_pnl=5.9214`, and
`avg_pnl_per_trade_pct=0.7086`, but policy optimization skipped it because there
is no matching current meta OOF bucket. The current meta OOF set contains the
`bars_in_high_vol_state_log_norm_0_453...volume_trend..._tbm` bucket, not the
`-0_610...dist_weekly_vwap...ema50_slope...` bucket.

This is the main reason the apparent `7` sizer-pass candidates became `6`
policy-optimized candidates. It also indicates a contract mismatch: a sizer
strategy was persisted even though the current base/meta OOF contract cannot
support policy optimization for it.

## Policy Results

Six candidates were policy-optimized. Two were selected; four failed deployment
economics after policy replay.

| Strategy | Policy side | Full policy PF | Full avg net PnL/trade | Deployment result |
|---|---|---:|---:|---|
| `dist_ema_fast...rsi_slope...` | long | 1.4158 | 0.00297 | selected |
| `dist_ema_fast...rsi_slope..._tbm` | long | 1.4158 | 0.00297 | selected |
| `bars_in_high_vol_state_log_norm_0_453...volume_trend...` | unknown | 0.4388 | -0.00244 | rejected |
| `bars_in_high_vol_state_log_norm_0_453...volume_trend..._tbm` | unknown | 0.4388 | -0.00244 | rejected |
| `dist_prior_day_high...vol_z...` | unknown | 0.0474 | -0.00265 | rejected |
| `dist_prior_day_high...vol_z..._tbm` | unknown | 0.0474 | -0.00265 | rejected |

The four rejected rows had positive simple-sizer metrics, but their final policy
average net PnL/trade was negative. They therefore failed the deployment
economics gate regardless of their earlier meta/sizer quality.

## Issues Found

1. **Sizer/meta contract mismatch.** The highest-PnL simple-sizer candidate has
   no matching current meta OOF bucket, so it cannot be policy-optimized.

2. **Silent policy skip.** The policy optimizer skips unmatched candidates before
   writing `strategy_for_inference.json`, so the final artifact shows
   `2 selected / 4 rejected` rather than reconciling all `7` sizer-pass
   candidates.

3. **Side is not persisted for some rejected candidates.** The rejected candidates
   have `side=unknown`, even though their matched meta OOF filenames encode
   `long` or `short`. This did not change selection because those rows already
   failed PnL gates, but it weakens auditability.

4. **Selected rows are duplicate variants.** The two selected rows are base and
   `_tbm` variants of the same strategy family with identical metrics. The
   current top-2-per-side cap counts them separately.

## Recommended Fixes

1. Have `simple_position_sizer.py` persist only strategies that can be matched to
   current base/meta OOF artifacts, or add an explicit `meta_oof_available` field.

2. Have `policy_optimiser.py` include skipped candidates in
   `rejected_strategies` with reasons such as `no_matching_meta_oof`,
   `no_finite_sizer_oof`, or `empty_trade_outcomes`.

3. Infer side from matched meta OOF keys when sizer rows leave `side` blank.

4. Deduplicate base and `_tbm` variants by core strategy id before applying the
   deployment top-2-per-side cap, unless the variants have materially different
   predictions or policy metrics.

5. Add a reconciliation table to the policy artifact:
   `sizer_pass -> meta_oof_matched -> policy_optimized -> deployment_selected`.

