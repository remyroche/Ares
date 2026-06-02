# Ranking, Threshold, and Selection Reconciliation

Status: updated, 2026-06-02.

## Scope

This audit reconciles the OOS/simple-policy ranking handoff against the live inference gate. The system does not use a plain raw-probability top-k selector at deployment. It scores each strategy head, maps the score through saved rank-normalization references, then applies per-strategy deployment thresholds and the global portfolio auction constraints.

## Rank Sources

- Per-strategy rank source: `simple_policy_optimiser/rank_reference/<strategy>.parquet`
  - score column: `calibrated_score`
  - rank column: `rank_pct`
  - schema: `policy_rank_reference_v1`
- Cross-strategy auction rank source: `simple_policy_optimiser/rank_reference/cross_strategy_auction.parquet`
  - score column: `calibrated_score`
  - rank column: `normalized_rank_score`
- Runtime rank-normalization contract:
  - `policy_rank_source = policy_rank_reference_percentile`
  - `cross_strategy_rank_source = cross_strategy_auction_reference`
  - `cross_strategy_reference_required = true`

## Deployment Thresholds

Current deployed six-head package, excluding `long_dist_ema20_leverage`:

| Strategy | Side | Deployment threshold | Policy-OOS rows | Exported candidates | Candidate net hit | Portfolio accepted | Accepted net hit | Accepted mean net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `long_asset_vol_level_pct_...` | long | 0.68 | 30,880 | 7,787 | 63.38% | 583 | 75.47% | 3.98% |
| `long_bars_in_high_vol_state_log_norm_...pullback...` | long | 0.65 | 48,779 | 7,088 | 60.00% | 470 | 71.28% | 4.14% |
| `long_bars_in_high_vol_state_log_norm_...loc_range...` | long | 0.63 | 32,456 | 12,282 | 60.28% | 1,307 | 71.00% | 3.73% |
| `long_loc_bb_channel_pos_48_...` | long | 0.64 | 40,330 | 8,327 | 56.84% | 558 | 62.19% | 3.31% |
| `short_asset_minus_mkt_oi_1d_peer_resid_...` | short | 0.60 | 26,522 | 4,391 | 67.43% | 509 | 76.23% | 4.22% |
| `short_bollinger_band_width_...` | short | 0.68 | 24,519 | 823 | 64.88% | 120 | 77.50% | 2.35% |

The local-candidate net-hit guard is enabled at `min_net_hit_rate = 0.50`, `min_rows = 50`, rank column `auction_rank_score`. In the current package the thresholds are no longer the earlier excessive long thresholds (`0.94`/`0.97`); they sit between `0.60` and `0.68`.

## Threshold-Band Economics

The optimiser now writes `simple_policy_optimiser/rank_threshold_band_report.csv` and `.json`. These reports separate marginal local bands from cumulative-at-threshold rows so higher-ranked rows cannot hide a bad threshold band.

Global cross-strategy auction local bands from the current six-head candidate table:

| Auction rank band | Rows | Net hit | Mean net | Median net | Mean gross |
|---|---:|---:|---:|---:|---:|
| 0.80-0.85 | 10,175 | 57.41% | 2.82% | 1.21% | 3.02% |
| 0.85-0.90 | 10,174 | 60.06% | 3.19% | 1.54% | 3.39% |
| 0.90-0.95 | 10,174 | 62.82% | 3.52% | 1.93% | 3.72% |
| 0.95-1.00 | 10,175 | 63.67% | 3.95% | 2.17% | 4.15% |

So the exported threshold band is not negative-then-compensated by higher ranks. The lower 0.80-0.85 auction band is already positive on mean and median net return, but its hit rate is below the desired 60% for some long heads.

## Frozen Post-Policy Holdout

`scripts/evaluate_frozen_policy_holdout.py` now evaluates the deployed six-head package on `2026-05-22T00:00:00Z` through `2026-05-27T17:00:00Z` without re-optimising thresholds or portfolio parameters. It uses train-meta-frozen model state, saved per-strategy rank references, the saved cross-strategy auction rank reference, saved deployment thresholds, the saved portfolio policy config, and the t+10 delayed-entry execution proxy where Kraken 1m candles are available. The delayed-entry model now tries exact t+10 first, then +1/+2/+3 minute fallbacks before using the theoretical 15m open.

Frozen holdout result:

| Metric | Value |
|---|---:|
| Local candidates before auction floor | 1,335 |
| Candidates after auction floor | 729 |
| Accepted portfolio trades | 73 |
| Trades/day | 11.70 |
| Mean accepted net return | 2.19% |
| Mean accepted gross return | 2.39% |
| Final wallet | 11583.00 |
| Max drawdown | -0.13% |

Per-strategy local holdout rows still show uneven hit-rate quality: long asset-vol/compression `52.33%`, long high-vol pullback/funding `51.13%`, long high-vol location/range `36.63%`, long local BB/channel `46.43%`, short asset-OI `64.56%`, and short Bollinger/price-RV `41.49%`. The portfolio remains positive on this short untouched holdout because the auction and portfolio constraints select a smaller subset, but this is not enough evidence to claim every local threshold is above the desired 60% net-hit target in all regimes.

Execution coverage: the first run fell back to `theoretical_15m_open` because the Kraken 1m loader dropped explicit zero-volume carry candles. After preserving those candles for `1m` and adding t+10/+1/+2/+3 fallback, the frozen holdout candidate metadata reports `703/729` rows using `delayed_1m_intraminute_proxy`, with complete 11-candle t+10 windows. All delayed rows used exact t+10 (`entry_delay_fallback_minutes=0`); the remaining `26` rows still use the theoretical fallback because no usable t+10 through t+13 candle was found.

Accepted-trade coverage: joining `per_candidate_replay_decisions.parquet` back to the candidate table shows `71/73` accepted trades used the delayed 1m proxy and `2/73` accepted trades used `theoretical_15m_open`. The delayed-only accepted subset has `63.38%` net hit, `2.19%` mean net return, and `2.40%` mean gross return, so the frozen holdout portfolio result is not being driven by the two theoretical-open accepted rows.

## Strict Replay Evidence

After the cache split, all four active heads were replayed through the actual inference candidate path with predictions enabled. The replay uses:

- `--feature-load-path inference_candidate`
- exchange-scoped artifact root `data_perp/exchanges/krakenfutures`
- policy rank-reference samples
- full trained perp universe
- source feature override `20260523_015947`

Results:

| Strategy | Candidate side/count after source/model checks | Score max abs diff vs policy reference | Rank max abs diff vs policy reference |
|---|---:|---:|---:|
| long-dist | long 12 | `2.62e-08` | `3.94e-05` |
| long-loc | long 15 | `1.26e-08` | `4.86e-05` |
| short-dist | short 9 | `1.31e-08` | `3.94e-05` |
| short-loc | short 84 | `1.53e-08` | `3.47e-05` |

This proves that, for sampled OOS rows, the actual inference candidate path can reproduce the saved policy score/rank references to float precision once it uses the same selected-feature handoff.

## Remaining Ranking Risks

- The deployment path still has several gates after per-strategy rank:
  - source-panel freshness and tradability filter.
  - sparse model-feature rejection.
  - per-symbol concurrency.
  - per-strategy concurrency.
  - global auction and portfolio capacity constraints.
  - stale/adverse price gap checks.
- Therefore score/rank parity alone is necessary but not sufficient for live trade parity. The next audit step is to reconcile the final portfolio decision table against `portfolio_policy_replay/per_candidate_replay_decisions.parquet` and live `prediction_ledger.parquet` rejection reasons.
