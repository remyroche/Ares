# Exit Policy + Fees Review (Meta Model + Optimiser)

## End-to-end flow
- Meta models are trained on `y_ret` generated from triple-barrier outcomes (`__y_ret__`), and ranking/IC diagnostics are computed on those raw returns.
- The TP/SL optimiser runs in four steps (TP/SL calibration, loss limiter, profit exit, sizing), then evaluates holdout net returns.
- Live/backtest execution uses `simulate_trade_hourly` with a staged exit state machine and applies round-trip fees as `2 * fee_bps` in the backtest path.

## Fee handling (current)
- Training diagnostics include a fixed top-decile cost adjustment (`cost=0.005`) in `_detailed_oof_metrics`.
- Optimiser fee is now threaded from a single `fee_pct` in `run_optimise_step` (default 0.005) into step 30 and step 40, and holdout net returns.
- Net return in tpsl metrics and sizing uses fee proportional to position size: `(raw_ret * pos_size) - (fee_pct * pos_size)`.
- This patch extends fee threading to step 10 and step 20 test-metric reporting so all optimiser stages use the same configured fee.

## Important caveat still present
- Step 40 sizing still evaluates on the input `exit_price` path from `bucket_df`, while step 20/30 changes are not fully state-threaded into sizing inputs. This can blur the measured incremental edge of the full stacked exit policy.

## Practical interpretation
- If performance remains weak after fee-consistent evaluation, likely edge is weak.
- If metrics improve materially, prior under/overcharging of fees and mixed fee assumptions likely masked edge.
- For definitive attribution, next step is to make steps 10→20→30→40 fully stateful on a single evolving trade path.
