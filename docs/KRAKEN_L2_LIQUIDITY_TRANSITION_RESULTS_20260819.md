# Kraken L2 liquidity-transition results

## Scope and retention

This is an offline-only causal research result. It does not change the live
Strict-R3 score, admission, policy, or execution code.

The retained substrate is `data/execution/tardis_orderbook_recap_v1`:

- 215 exact Kraken Spot symbol/day identities across 2025-01-01, 2025-04-01,
  2025-07-01, 2025-10-01, 2026-01-01, 2026-04-01, and 2026-07-01;
- 309,274 compact causal per-minute order-book rows;
- the superseding compact v2 panel has 553 columns after causal
  OI/volume/volatility context joins, depth-normalized L2 event rates, and a
  completed-candle BTC benchmark join;
- raw L2/snapshot/trade files permanently removed only after all 215 recaps
  passed exact identity, positive-row, required-L2-field, and no-trade-field
  checks. The deletion receipt is
  `data/execution/tardis_orderbook_recap_v1/reports/raw_prune_receipt_20260819.json`.

Final retention decision (2026-08-20): retain **order-book recaps only**. The
raw archive is empty, no compact trade-print archive is retained, and the
temporary trade-aggregate research branch was removed. The retained
order-book-recap root is 2.1 GiB, contains no `.csv.gz`, `.jsonl`, or `.zip`
raw files, and is covered by
`reports/liquidity_transition_v2_orderbook_btc_audit`.

The retained L2 inputs include spread, microprice, depth, imbalance,
size-specific executable sell cost, L2 cancel/replenish, depth-normalized
cancel/replenishment/failure rates, book-event pressure, causal transitions,
actual-position cost interpolation, local returns/drawdown/volatility,
cross-sectional market state, liquidity rank, and backward/as-of OI/volume/
volatility context. BTC 1/5/15-minute returns come from a 520-KB, completed
one-minute OHLCV recap; the downloaded daily archives were temporary and were
not retained. Context is available on 96.3% of rows overall; the initial hour
of 2025-01-01 is deliberately null because the source carries a conservative
one-hour availability lag.

The raw-trade retention change means `trade_intensity`, executed
`sell_order_flow_imbalance`, minute trade-volume ratios, and
position-to-trade-volume ratios are intentionally absent from this compact
version. They must remain unavailable rather than be imputed. The compact
`book_flow_imbalance_50bps` is explicitly a quote-event proxy, not a trade-flow
claim. The external BTC return recap is backward-joined only after each
completed candle close; no BTC value is fabricated. Market return/breadth are
calculated from the complete observed L2 universe at each minute before
filtering.

## Validation protocol

Each target uses three expanding chronological UTC-date folds, with a
deterministic 12,000-row cap per date. There is no random row split.

Models:

- training-derived Bayesian shrinkage;
- robust-scaled Ridge;
- shallow LGBM regression (depth 3, 400 trees, learning rate 0.035, 500 minimum
  child rows, 0.8 row/feature sampling, L2=15).

MDA is a full semantic-feature-family cyclic permutation inside each
symbol/date trajectory, reported separately by held date fold. It is not a
row-random permutation and is never used as a live feature.

The executable-cost target is the future change in sell cost for a 500-quote
unit order. Spread targets are top-of-book widening in bps. Both use strict
contiguous future-minute label validity.

## Direct-target results

Mean held-fold MAE in bps. LGBM is the winner in every direct target/horizon.

| Horizon | Cost: Bayesian | Cost: Ridge | Cost: LGBM | Spread: Bayesian | Spread: Ridge | Spread: LGBM |
|---:|---:|---:|---:|---:|---:|---:|
| 1m | 0.823 | 4.642 | **0.712** | 1.179 | 4.306 | **0.992** |
| 5m | 0.939 | 3.313 | **0.783** | 1.310 | 5.525 | **1.077** |
| 10m | 0.969 | 8.016 | **0.803** | 1.363 | 5.347 | **1.113** |
| 15m | 0.976 | 5.549 | **0.811** | 1.392 | 5.295 | **1.130** |
| 30m | 1.001 | 5.145 | **0.837** | 1.420 | 6.935 | **1.163** |

LGBM's worst individual held-fold MAE remains bounded: 0.785–0.959 bps for
cost and 1.080–1.248 bps for spread across the five horizons. Ridge is not a
robust fallback: its heavy-tail RMSE ranges from 19.8 to 172.1 bps on direct
targets, despite ordinary mean MAE sometimes appearing moderate.

## Date-fold grouped MDA: LGBM

The expected liquidity-transition result appears consistently: derivatives of
the book have far more predictive value than simply feeding the instantaneous
book snapshot.

| Target | Horizon | Most important family | Mean loss increase | Second family | Mean loss increase |
|---|---:|---|---:|---|---:|
| Cost deterioration | 1m | book transition | 0.212 | external causal context | 0.019 |
| Cost deterioration | 5m | book transition | 0.250 | external causal context | 0.032 |
| Cost deterioration | 10m | book transition | 0.267 | external causal context | 0.037 |
| Cost deterioration | 15m | book transition | 0.240 | external causal context | 0.052 |
| Cost deterioration | 30m | book transition | 0.242 | external causal context | 0.053 |
| Spread widening | 1m | book transition | 0.389 | current book | 0.004 |
| Spread widening | 5m | book transition | 0.479 | current book | 0.008 |
| Spread widening | 10m | book transition | 0.483 | current book | 0.010 |
| Spread widening | 15m | book transition | 0.483 | current book | 0.014 |
| Spread widening | 30m | book transition | 0.475 | current book | 0.015 |

The book-transition result is stable across all three held-date folds. For
cost it ranges 0.199–0.319 loss increase; for spread it ranges 0.364–0.514.
This supports the intended interpretation: spread/depth/cost changes and
cancel/replenishment behavior are materially more useful than snapshot level
alone. OI/volume/volatility context is incremental for cost deterioration,
especially from 15–30 minutes, but is much less incremental for immediate
spread widening.

The new, separately permuted `book_flow_rates` family is **not incremental**:
its mean held-fold MAE increase is only -0.000009 to +0.000010 bps across the
ten direct tasks, with non-positive folds in most cost horizons. Keep the raw
L2 transition family; do not promote the new normalized rate features merely
because they are intuitively plausible.

The newly complete `btc_benchmark` family is also **not incremental** in this
seven-date sample: its grouped held-fold loss increase is 0 to 0.000006 bps
across all ten direct tasks, and adding it changes LGBM mean MAE by at most
0.0014 bps. Keep the compact BTC recap for a complete causal contract, but do
not promote it as a liquidity-transition driver without broader evidence.

## Requirement audit

`reports/liquidity_transition_v2_orderbook_btc_audit/feature_contract_audit.json`
is the authoritative compact-contract receipt. It confirms all retained L2,
derivative, local-state, market-state, and existing OI/volume/volatility
families plus all direct 1/5/10/15/30-minute labels. Under its declared
`orderbook_only` retention profile, `retained_contract_complete` is true. It
also documents the intentional absence of individual-trade fields following
raw-data pruning.

The final order-book-only audit deliberately reports the trade-recap family as
absent. This is a retention choice, rather than missing-data imputation: no
trade-intensity, executed-flow, or trade-volume feature may be inferred from
quotes. The compact BTC benchmark is instead a separate, completed-candle
source and passes its coverage gate.

## Compact aggregate-activity completion cohort

The original contract also calls for executed-flow and trade-capacity fields.
Those cannot be truthfully reconstructed from book changes. A separate dense
cohort therefore uses a **6.3-MiB per-minute aggregate recap**, with no
individual trade prints: buy/sell quote volume, total quote volume, trade
count, signed sell order-flow imbalance, and their 1/3/5-minute changes.

It covers 12 symbols across the same seven dates (120,852 rows).  The source
aggregate is accepted only when its `activity_available_ts` is no later than
the next-minute `decision_ts`; minute gaps remain null.  The 1-minute
position/volume measure records an explicit zero-volume flag and applies a
causal, capped activity floor only to make its otherwise-infinite
participation-pressure value model-safe.  Its full-recap audit reports:

```text
all_inference_families_complete = true
retained_contract_complete      = true
all_supervision_labels_present  = true
```

This is a different, denser 12-symbol population, so its raw MAE is not a
like-for-like improvement claim against the 99-symbol book-only panel.  Within
that identical cohort, however, the matched ablation is conclusive: removing
only the 14 `trade_flow` fields changes LGBM mean held-fold MAE by at most
0.0011 bps in every one of the ten direct tasks.  Grouped date-fold MDA agrees:
the trade-flow family ranges from -0.000020 to +0.000037 bps loss increase per
fold.  Actual trade aggregates therefore complete the requested feature
contract but do not currently advance as an incremental predictor.

## Artifacts

- compact source receipt:
  `data/execution/tardis_orderbook_recap_v1/reports/raw_prune_receipt_20260819.json`
- causal feature panels:
  `data/execution/tardis_orderbook_recap_v1/features/liquidity_transition_panel_v2`
- compact causal BTC recap and checksum manifest:
  `data/execution/tardis_orderbook_recap_v1/context/binance_btcusdt_1m_recap_v1.parquet`
- compact aggregate-activity recap and checksum manifest:
  `data/execution/tardis_orderbook_recap_v1/context/kraken_per_minute_activity_recap_v1.parquet`
- direct fold metrics/MDA with BTC:
  `data/execution/tardis_orderbook_recap_v1/models/liquidity_transition_v2_orderbook_btc/*`
- consolidated metrics/MDA:
  `data/execution/tardis_orderbook_recap_v1/models/liquidity_transition_v2_orderbook_btc/consolidated_metrics.parquet`
  and `consolidated_grouped_mda.parquet`
- isolated book-flow-rate MDA:
  `data/execution/tardis_orderbook_recap_v1/models/liquidity_transition_v2_grouped_mda/*`
- requirement and coverage receipt:
  `data/execution/tardis_orderbook_recap_v1/reports/liquidity_transition_v2_orderbook_btc_audit/*`
- full compact-recap activity-cohort audit and matched ablation:
  `data/execution/tardis_orderbook_recap_v1/reports/liquidity_transition_activity_recap_v1_contract_audit/*`,
  `liquidity_transition_activity_trade_flow_matched_ablation.parquet`, and
  `liquidity_transition_activity_trade_flow_grouped_mda.parquet`

## Decision

The direct 1/5/10/15/30-minute cost and spread models advance as an offline
liquidity-transition research component. The leading initial candidate is the
shallow LGBM cost-deterioration model, with book-transition derivatives and
causal OI/volume/volatility context retained. The depth-normalized L2 event
rates, BTC benchmark, and compact aggregate trade-flow family do not advance.
It is not approved for live use: the next step is a
separately predeclared exact order-size/latency/fill replay and an untouched
period before any execution-policy integration discussion. A compact causal
BTC OHLCV source and compact activity recap now complete the requested
feature-contract portion of this research workstream.
