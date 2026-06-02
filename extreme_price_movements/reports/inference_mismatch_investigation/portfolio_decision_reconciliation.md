# Portfolio Decision Reconciliation

Status: updated, 2026-06-02.

## Scope

Run: `20260525_010004_nopenalty`

Source artifact:

`data_perp/artifacts/20260525_010004_nopenalty/portfolio_policy_replay/per_candidate_replay_decisions.parquet`

This audit checks whether the portfolio replay artifact is balanced across strategies and whether final portfolio gates explain accepted versus rejected candidates after per-strategy rank thresholds have already been applied.

## Replay Coverage

- Rows: 20,026
- Time range: `2026-01-22 10:00:00 UTC` through `2026-05-21 23:00:00 UTC`
- Accepted rows: 3,028
- Rejected rows: 16,998
- Accepted replay days: 96
- Accepted trades per accepted day:
  - mean: 31.54
  - median: 33
  - min: 2
  - max: 47

## Accepted Trades by Strategy

| Side | Strategy family | Candidate rows | Accepted | Accept rate |
|---|---|---:|---:|---:|
| short | short-loc | 6,108 | 1,009 | 16.52% |
| long | long-dist | 7,429 | 868 | 11.68% |
| short | short-dist | 3,626 | 678 | 18.70% |
| long | long-loc | 2,863 | 473 | 16.52% |

Accepted replay trades are not concentrated in one strategy in the current artifact. The prior symptom where accepted portfolio trades were almost entirely the fourth long-dist strategy is not present in this replay.

## Accepted Trades by Month and Side

| Month | Long | Short |
|---|---:|---:|
| 2026-01 | 8 | 47 |
| 2026-02 | 186 | 305 |
| 2026-03 | 359 | 604 |
| 2026-04 | 463 | 496 |
| 2026-05 | 325 | 235 |

The replay shifts from short-heavy early in the OOS period toward more balanced and then long-heavier May acceptance. This should be compared against market regime and live symbol availability before treating it as model decay.

## Rejection Reasons

| Reason | Long | Short |
|---|---:|---:|
| accepted | 1,341 | 1,687 |
| below_dynamic_threshold | 2,464 | 2,869 |
| symbol_already_open | 2,117 | 1,765 |
| symbol_in_cooldown | 1,741 | 564 |
| max_new_entries_per_bar_reached | 974 | 1,124 |
| max_concurrent_per_strategy_reached | 927 | 1,081 |
| max_concurrent_positions_reached | 728 | 644 |

## Live Ledger Caveat

The local live ledger at `data_perp/exchanges/krakenfutures/live_state/prediction_ledger.parquet` is not a clean current-run replay comparison table:

- It has 149 rows from `2026-05-17 00:04:04 UTC` through `2026-05-28 12:26:31 UTC`.
- It mixes three artifact generations: `112` rows for `20260321_140000`, `36` rows for `20260523_015947`, and `1` row for `20260525_010004_nopenalty`.
- The only row referencing `20260525_010004_nopenalty` is the AR short row at signal bar `2026-05-28 11:00:00 UTC`, with strategy id `loc_ema_stack_pos_24_...`, which is not part of the current six-head replay set and is outside the frozen holdout replay window.

Therefore the portfolio replay can be reconciled internally, but current live-vs-replay final decision parity needs a fresh ledger run with the current artifact and the newer execution diagnostics enabled.

## Live vs Replay Reconciliation Artifacts

Added `scripts/reconcile_live_replay_decisions.py` to generate reproducible live-ledger versus replay-decision checks.

Frozen post-policy holdout comparison:

- Report: `extreme_price_movements/reports/inference_mismatch_investigation/live_vs_frozen_holdout_decision_reconciliation.md`
- Replay decisions: `data_perp/artifacts/20260525_010004_nopenalty/policy_holdout_frozen_replay_1m_fallback/portfolio_policy_replay/per_candidate_replay_decisions.parquet`
- Exact matches on `signal_bar_ts/timestamp + symbol + side + strategy_id`: `0/149`.
- Loose matches on `signal_bar_ts/timestamp + symbol + side`: `0/149`.
- Current-artifact live rows: `1`, exact matches: `0`.

Main policy replay comparison:

- Report: `extreme_price_movements/reports/inference_mismatch_investigation/main_replay_live_reconciliation/live_vs_main_replay_decision_reconciliation.md`
- Replay decisions: `data_perp/artifacts/20260525_010004_nopenalty/portfolio_policy_replay/per_candidate_replay_decisions.parquet`
- Exact matches: `0/149`.
- Loose matches: `1/149`, but it is not a current-artifact exact strategy match.
- Current-artifact live rows: `1`, exact matches: `0`.

This proves the current ledger cannot be used to explain live degradation against the current six-head replay. It can still summarize live gate types: `93` portfolio rejections, `22` rank rejections, `19` liquidity rejections, `4` price-gap rejections, and `11` traded rows, but those rows are not a clean current-package sample.

## Current Conclusion

Per-strategy rank parity is proven for sampled OOS rows, and portfolio replay acceptance is distributed across all heads. The current live ledger is not comparable enough to answer final live-vs-replay decision parity. The next unresolved decision-path question is not rank generation; it is whether a fresh current-package live-test ledger sees the same source eligibility, sparse-feature coverage, dynamic thresholds, cooldown/concurrency state, adverse price gap, spread/stale ticker gates, and portfolio capacity state as the replay.

## Run-Scoped Shadow Live Check

Status: blocked before scoring, 2026-06-02.

A one-shot shadow run with `EPM_RUN_SCOPED_PREDICTION_LEDGER=1` confirmed that the new run-scoped ledger path is used:

`data_perp/exchanges/krakenfutures/live_state/prediction_ledgers/20260525_010004_nopenalty/prediction_ledger.parquet`

However, no current-package ledger rows were produced because live inference failed closed before feature scoring. The optimized portfolio contract loaded six selected strategies, but the selected deployment rows all have empty `lgbm_regime_mask` contracts. The fallback perps mask CSV contains four old strategy rows and filters to `0` rows for the selected six. Runtime correctly refused to fall back to stale masks:

`LGBM strategy regime masks missing for accepted strategies`

This means the next live-vs-replay decision reconciliation requires a valid six-strategy mask contract first. Until that is repaired, "no new positions" from this package is expected: the current deployment is monitor-only/fail-closed by design.

## Mask Contract Repair

Status: repaired and locally verified, 2026-06-02.

The missing-contract failure was traced to the policy-source handoff, not to the live mask evaluator:

- `policy_oos_retrain_strategy_source_perps.csv` had lossy safe ids in `canonical_key` and no parseable rule column.
- The original parseable rules for the same strategy ids were present in `strategy_selection/candidate_strategies.json`.
- `params_store.load_inference_candidate_mask_params_per_bucket(...)` now supports this split contract: explicit `strategy_id` is kept as the deployment/model id, while `base_event_trigger` is used as the parseable LGBM regime mask.
- The current source CSV was backfilled with `strategy_id`, `base_event_trigger`, `trade_side`, `feature_keys_json`, and `market_mode=perps`.
- The current deployment JSON copies under `data_perp/artifacts/20260525_010004_nopenalty` were backfilled with embedded `lgbm_regime_mask` contracts.

Validation through the inference loader:

- selected strategy cores: `6`
- embedded mask aliases loaded: `12`
- coverage guard: `ok`
- parseable mask rules: `6/6`

The next live-vs-replay reconciliation should now get past the pre-base LGBM regime-mask coverage gate.

## Fresh Shadow Cycle After Mask Repair

Status: passed mask-contract loading, blocked by market kill switch before scoring, 2026-06-02 10:31 UTC.

Command shape:

`python3 -u -m extreme_price_movements.inference.run_inference --shadow --perps --data-root data_perp --run-id 20260525_010004_nopenalty --run-scoped-prediction-ledger --run-once --challenger-interval 0`

Observed:

- Kraken Futures market loading succeeded after network approval.
- Optimized portfolio policy loaded with six strategy-contract entries.
- Deployment strategy filter resolved six selected aliases.
- `strategy_for_inference.json` loaded six policy-param rows and 12 strategy aliases.
- Training-live parity contract loaded for six strategies.
- Embedded LGBM strategy masks loaded from `strategy_for_inference.json`: `loaded_rows=12`, `status=loaded`.
- Run-scoped ledger path resolved to `data_perp/exchanges/krakenfutures/live_state/prediction_ledgers/20260525_010004_nopenalty/prediction_ledger.parquet`.
- Hourly fetch used the trained-symbol model context: `fetch=149`, `workers=32`, `microdata_workers=24`, elapsed `24.3s`.
- Panel load took `19.3s`.

The cycle then stopped before feature scoring because the market kill switch was active:

- reason: `MARKET_AVG_1H_MOVE_GT_5PCT`
- BTC 1h move: `6.21%`
- ETH 1h move: `2.55%`
- average absolute 1h move: `5.02%`
- halt until: `2026-06-02T22:15:00Z`

This is a valid protective block and should not be bypassed for live-path parity. No current run-scoped ledger rows were produced because scoring did not run.

## Follow-Up Checks

1. Rerun a fresh live-test or shadow cycle after the market kill switch self-recovers, keeping the ledger on the run-scoped path.
2. Include only the six active strategy heads and persist the current run id on every ledger row.
3. Join live ledger rows to replay-style decision rows by signal-bar timestamp, symbol, side, and strategy.
4. For every live rejection, persist the first failed gate and the relevant threshold/input value.
5. Compare live counts by gate against replay counts by gate over a matched timestamp window and identical initial portfolio state.
6. Verify that dynamic threshold and cooldown/concurrency state are initialized equivalently in replay and live-test modes.
