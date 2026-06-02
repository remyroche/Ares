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
- The current saved candidate artifact has not been regenerated under this default: `simple_policy_optimiser/simple_policy_candidates.parquet` has `delayed_entry_ts - timestamp = 7 minutes` for all 20,026 rows.
- The local execution-1m cache also contains one saved minute per symbol-hour at minute `:07` for the audited May window, which is consistent with the older t+7 artifact.
- `trade_executor.py` records realized entry price, expected entry price, order fee conversion, entry delay fields, and signal-to-entry time for market/stop execution modes.
- `prediction_ledger.py` includes feature, artifact, decision, friction, spread/slippage, entry delay, and drift diagnostic fields needed for later live/OOS reconciliation.

## Confirmed Fixes Related To Timing

- Historical replay now uses the actual inference candidate path.
- Historical replay no longer computes model features during mask-only checks.
- The live LGBM mask fast path now resolves spot-style market-basket config symbols to perp symbols by base asset before computing market-basket features.

## Open Checks

1. Regenerate simple-policy candidates and deployment metrics under the current t+10 delayed-entry setting.
2. Verify that every regenerated policy candidate file contains delayed-entry fields and that deployment thresholds are discovered on delayed-entry-adjusted paths for all four heads.
3. Compare live `signal_to_entry_seconds` against the optimiser's intended 10-minute delay.
4. Quantify OOS performance at zero delay, 10-minute proxy delay, and observed live delays.
5. Verify exit timestamp and stop-trigger/fill handling against the path arrays used by `simple_policy_optimiser`.
6. Confirm that adverse hourly-close gap is included in live EV/friction gates for the currently running code path, not only on disk.

## Current Conclusion

No label-horizon mismatch is proven yet. A timestamp/execution artifact mismatch is proven: the current code default is t+10, but the saved OOS candidate artifact is still t+7. Execution realism remains incomplete until OOS candidates are regenerated under t+10 and then re-scored under observed live delays, spread, slippage, fees, and rejected market/stop orders.
