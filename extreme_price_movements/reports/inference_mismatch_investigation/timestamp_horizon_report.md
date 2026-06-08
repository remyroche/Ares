# Timestamp, Label Horizon, and Tradability Audit

## Scope

Run: `20260525_010004_nopenalty`

Market: Kraken perps

This report reconciles the timing contract between model/OOS policy optimisation and live inference. It focuses on what is known from code inspection and replay evidence so far; it does not yet claim full execution realism.

## Current Timing Contract

### Prediction Timestamp

- Historical policy rank references are keyed by `timestamp` and `symbol` in `simple_policy_optimiser/rank_reference/*.parquet`.
- Live inference scores candidates on the latest closed hourly bar and writes `signal_bar_ts`, `decision_ts`, `feature_source_max_ts`, and `feature_available_ts` to `prediction_ledger.parquet`.
- Historical replay through `scripts/historical_inference_parity.py --feature-load-path inference_candidate` uses the same `_select_candidates_and_load_features(...)` candidate path as live inference.

### Feature Availability Cutoff

- Strict replay loads panels ending at the policy reference timestamp and computes features from data up to that timestamp.
- The replay path now sets `effective_lookback_hours` before panel loading, so long-window transformed features and benchmark residuals have their intended warmup.
- Live/replay feature generation now separates mask features and model features:
  - `mask` cache namespace for LGBM candidate masks.
  - `model` cache namespace for model scoring features.
- The mask-only strict replay no longer computes model features when prediction parity is explicitly skipped.

### Entry Timestamp

- `simple_policy_optimiser.py` applies `_apply_delayed_entry_execution_model(...)` after persisting rank references and before deployment threshold discovery.
- Default delayed entry is `EPM_SIMPLE_POLICY_DELAYED_ENTRY_MINUTES=10`.
- Delayed entry uses the 1m candle at `timestamp + 10 minutes`, floored to minute.
- Eligible rows default to `rank_pct >= 0.50`.
- The entry fill proxy uses `EPM_SIMPLE_POLICY_DELAYED_ENTRY_REF` default `open` and intraminute alpha default `0.5`:
  - Long fill: delayed ref plus `0.5 * max(high - ref, 0)`.
  - Short fill: delayed ref minus `0.5 * max(ref - low, 0)`.
- The optimiser records `delayed_entry_ts`, `entry_delay_minutes`, `entry_gap_bps`, `entry_slippage_proxy_bps`, delay-window OHLC, range, max adverse/favorable move, and close gap.

### Live Entry Timestamp

- Live/shadow execution records `decision_to_entry_seconds` and `signal_to_entry_seconds`.
- Market entries use the order fill when available; shadow entries use the caller reference price.
- Both live and shadow paths compute:
  - `theoretical_entry_price`
  - `policy_entry_price`
  - `ohlcv_entry_price`
  - `entry_delay_adverse_bps`
  - `entry_delay_effect_bps`
  - `entry_delay_abs_bps`
- `prediction_ledger.py` and `trade_logger.py` both include spread, slippage, adverse signal gap, entry delay, and fee diagnostic columns.

## Evidence

- `simple_policy_optimiser.py` lines around `_apply_delayed_entry_execution_model(...)` show the delayed-entry model is now `t+10m`, not `t+7m`.
- `simple_policy_optimiser.py` applies delayed-entry execution before `discover_deployment_rank_threshold_simple_grid(...)`, so deployment thresholds are based on delayed-entry proxy paths.
- The current saved candidate artifacts have been regenerated under the t+10 contract:
  - `data_perp/artifacts/20260525_010004_nopenalty/simple_policy_optimiser/simple_policy_candidates.parquet`
  - `extreme_price_movements/reports/inference_mismatch_investigation/simple_policy_optimiser_trained_universe_full_deployable/simple_policy_optimiser/simple_policy_candidates.parquet`
  - `extreme_price_movements/reports/inference_mismatch_investigation/simple_policy_optimiser_trained_universe_verify/simple_policy_optimiser/simple_policy_candidates.parquet`
- Direct parquet inspection on 2026-06-04 shows 27,485 candidate rows in each of those artifacts and `delayed_entry_ts - timestamp = 10 minutes` for every row.
- The regenerated artifacts use `entry_execution_source=delayed_1m_intraminute_proxy` for 25,564 rows and `theoretical_15m_open` fallback for 1,921 rows where the t+10 1m proxy was not available.
- `trade_executor.py` records realized entry price, expected entry price, order fee conversion, entry delay fields, and signal-to-entry time for market/stop execution modes.
- `prediction_ledger.py` includes feature, artifact, decision, friction, spread/slippage, entry delay, and drift diagnostic fields needed for later live/OOS reconciliation.

## Confirmed Fixes Related To Timing

- Historical replay now uses the actual inference candidate path.
- Historical replay no longer computes model features during mask-only checks.
- The live LGBM mask fast path now resolves spot-style market-basket config symbols to perp symbols by base asset before computing market-basket features.

## Open Checks

1. Verify that deployment thresholds were discovered from the regenerated delayed-entry-adjusted paths for every active deployed head.
2. Compare live `signal_to_entry_seconds` against the optimiser's intended 10-minute delay.
3. Quantify OOS performance at zero delay, 10-minute proxy delay, and observed live delays.
4. Verify exit timestamp and stop-trigger/fill handling against the path arrays used by `simple_policy_optimiser`.
5. Confirm that adverse hourly-close gap is included in live EV/friction gates for the currently running code path, not only on disk.
6. Explain the 1,921 `theoretical_15m_open` fallback rows and decide whether they should be excluded from threshold discovery or marked separately in deployment metrics.

## Current Conclusion

No label-horizon mismatch is proven yet. The earlier timestamp/execution artifact mismatch has been repaired for the active candidate artifacts: the current code default is t+10 and the regenerated OOS candidate artifacts are also t+10. Execution realism remains incomplete until the regenerated t+10 candidates are re-scored against deployment thresholds, observed live delays, spread, slippage, fees, liquidity filters, and rejected market/stop orders.
