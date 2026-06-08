# Execution And Exit Parity Todo

Goal: prove that live execution and exits are faithful to the policy optimiser/backtest contract before treating poor live performance as model decay.

## Todo

1. Same-signal-bar would-open parity
   - Re-run the deployed decision policy for each live signal bar.
   - Use the same deployed models, feature path, rank normalization, masks, thresholds, tradable universe, and live-only gates.
   - Prove every live-opened trade would open in replay.
   - Prove every replay-opened trade either opened live or has a logged rejection reason.
   - Status 2026-06-06: in progress, not yet proven.
   - Evidence: `execution_exit_parity_20260606/live_decision_replay_reconciliation.csv`.
   - Current result: 255 latest decision rows, 12 live traded, 5 replay accepted, 17 full replay/live mismatches.
   - Direct rank-gate check is closer: 25 would-open rows, 13 mismatches; mismatches are explainable by stale/live symbol/stop-distance gates or portfolio replay state gaps.
   - Current result after replayability guard: 0/447 legacy ledger rows are exact portfolio-state replayable because they do not contain pre-decision open-position/cooldown/wallet snapshots.
   - Patch: future ledger rows now persist compact `portfolio_replay_state_v1` snapshots, state hashes, open-position counts, cooldowns, wallet, open notional, and portfolio priority when a portfolio manager is available.
   - Remaining gap: rerun live/shadow inference and replay new rows; only rows with `exact_portfolio_state_replayable=true` can prove stateful would-open parity.

2. Entry price parity
   - For each accepted trade, compare policy/theoretical entry, t+5 expected entry, live decision price, order timestamp, fill timestamp, and realized fill price.
   - Attribute gaps into hourly close to decision price, decision price to fill price, spread proxy, slippage proxy, fees, and stale-signal delay.
   - Fail loudly when `signal_to_entry_seconds > 600` or when required entry fields are missing.
   - Status 2026-06-06: partially covered, future logging patched.
   - Evidence: `execution_exit_parity_20260606/ledger_replay_field_coverage.csv` and `spread_slippage_reconciliation.csv`.
   - Current result: 12 traded rows; 5 have complete signal-close and decision-to-fill attribution; older rows miss those fields.
   - Current result: live spread/slippage fields reconcile with policy fields for rows where both exist, but all 12 historical traded rows miss fee bps and position id.
   - Patch: future prediction-ledger rows now fall back to snapshot fee bps and derive `position_id` from order id when needed.
   - Patch: future prediction-ledger rows include portfolio state fields needed to separate live portfolio/cooldown rejection from rank/threshold rejection.

3. Closed-trade exit replay parity
   - Replay each closed live trade with the same stop/trailing policy and the cached minute or 5m bars.
   - Compare live exit reason, exit timestamp, and exit price against replay.
   - Classify mismatches as missing candle resolution, market-stop gap, stale state, wrong anchor, or implementation bug.
   - Status 2026-06-06: partially covered, not fully faithful on fill price.
   - Evidence: `execution_exit_parity_20260606/closed_trade_exit_replay/live_closed_trade_exit_replay.csv`.
   - Current result: 13 unique closed trades; policy-entry replay reproduces 10 stop crosses and misses 2 live fills from cached bars; one row has invalid time window.
   - Current result: replayed exit prices differ materially from live fills, consistent with market-stop gap/slippage not captured by the current bar-cross approximation.
   - Patch: simple-policy optimiser and closed-trade exit replay now share `execution_fill_model.stop_exit_fill_price`, so stop-cross fill assumptions are no longer duplicated across code paths.

4. Spread/slippage expectation reconciliation
   - Compare live observed spread/slippage/delay haircut against simple-policy assumptions.
   - Use p66 spread baseline and only excess friction for live EV haircut.
   - Write per-trade and aggregate reports.
   - Status 2026-06-06: covered for entry rows where fields exist.
   - Evidence: `execution_exit_parity_20260606/spread_slippage_reconciliation.csv`.
   - Current result: policy-vs-live friction delta is 0 bps where both policy and live fields exist, because live now logs the same expected friction path. Actual fill vs theoretical entry remains large on historical stale rows.

5. Runtime and ledger guards
   - Require selected feature values, raw predictions, rank-normalized scores, thresholds, gate outcomes, policy entry, t+5 expected entry, decision/fill timing, spread/slippage proxies, fee bps, stop level, trailing activation level, trailing stop level, exit reason, and rejection reason.
   - Mark rows unavailable or fail loudly when the data required for 1:1 replay is missing.
   - Status 2026-06-06: field coverage guard added for reconciliation; live ledger writer patched for fee, position id, and portfolio-state replay fields.
   - Evidence: `ledger_replay_field_coverage.csv` plus tests `test_ledger_replay_field_coverage_flags_missing_traded_entry_fields`, `test_ledger_replay_field_coverage_accepts_portfolio_state_snapshot`, `test_stop_exit_fill_model_scalar_and_array_match_long_short`, and `test_prediction_ledger_row_falls_back_to_snapshot_fee_and_order_id_for_replay`.
   - Remaining gap: runtime hard failure should be added once a clean cycle produces fully populated rows, to avoid breaking on legacy ledger rows.

6. Evidence report
   - Summarize commands, artifacts, pass/fail counts, mismatches, and residual execution risks in the mismatch investigation report.
   - Status 2026-06-06: this file is the current execution/exit parity status note.
