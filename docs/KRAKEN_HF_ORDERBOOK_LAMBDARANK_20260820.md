# Kraken HF order-book recap / spread LambdaRank research

## Scope and data contract

This is an execution-friction research panel only.  It does not alter the
canonical alpha, admission, portfolio, or live-execution stack.

`Abraxasccs/kraken-market-data` contains Kraken **spot** order-book snapshots;
it has no futures order-book tree.  Every retained row is therefore explicitly
labelled `spot_fallback_no_futures_in_abraxasccs_dataset`.

The producer streams source Parquet bytes, reduces each snapshot to a compact
L2 state, then discards the raw payload in process memory.  It never requests
the dataset's trade files.  The completed retention audit reports:

- 54 valid UTC-day partitions; 67,302 compact rows; 14 symbols;
- 57,157 valid-book rows (84.93%);
- 5,109 raw source files / 490,507,848 bytes streamed then discarded;
- no persisted `bids_json`, `asks_json`, or trade columns; no raw-like files;
- 15-minute and 30-minute label coverage of 82.54% and 80.98%, respectively.

The panel is only 86 MB on disk.  Incomplete future horizon labels are null and
are excluded from fitting, never encoded as benign spread outcomes.

## Point-in-time construction

The source collection minute can drift.  For each fixed 15-minute decision
boundary, the producer selects the latest source snapshot available before the
boundary.  Features and labels require contiguous retained source intervals;
no missing snapshot is bridged or fabricated.

The model sees 102 causal book-state and transition fields.  To make these
portable across tight/deep and wide/thin symbols, the contract includes seven
strictly-prior, per-asset normalization fields:

- `spread_bps_to_asset_prior_median`;
- bid/ask displayed depth at 10, 50, and 100 bps divided by its strictly-prior
  asset-local median.

There is no trade-volume input by design: the retained source is order-book
only.  Displayed L2 depth is the available point-in-time liquidity proxy.

## LambdaRank arms

All runs use four expanding chronological UTC-date folds, depth 3, 400 trees,
learning rate 0.035, 15 leaves maximum, conservative child support, and
timestamp-local queries.  A query requires at least five valid symbols.
Labels are five ordinal within-query grades with gains `[0, 1, 3, 7, 15]`.

| Arm | Native rank target | Cross-asset treatment |
|---|---|---|
| Absolute future spread | Literal future full spread in bps | Strictly-prior spread/depth-relative **features**; target remains absolute |
| Spread delta | Future spread − current spread | Divide target by strictly-prior per-asset robust MAD scale (floor 1 bp) |
| Asset-relative deviation | Future spread − strictly-prior median spread | Divide target by the same strictly-prior robust scale |

## Strict chronological OOS results

`top-k raw` is the realised future raw target for the ranker's top `k` symbols
per timestamp.  Uplift is relative to the full same-timestamp candidate pool.

| Horizon | Target | NDCG@1 | Raw Spearman | Pool bps | Top-1 bps / uplift | Top-3 bps / uplift | Top-5 bps / uplift |
|---|---|---:|---:|---:|---:|---:|---:|
| 15m | Absolute future spread | 0.986 | 0.871 | 7.84 | 43.50 / +35.66 | 23.01 / +15.17 | 16.13 / +8.30 |
| 15m | Spread delta | 0.862 | 0.475 | -0.00 | 3.13 / +3.13 | 2.12 / +2.12 | 1.46 / +1.46 |
| 15m | Asset-relative deviation | 0.753 | 0.233 | 0.14 | 3.04 / +2.90 | 1.66 / +1.52 | 1.05 / +0.91 |
| 30m | Absolute future spread | 0.985 | 0.869 | 7.89 | 42.84 / +34.95 | 23.07 / +15.18 | 16.12 / +8.23 |
| 30m | Spread delta | **0.867** | **0.484** | -0.01 | **3.59 / +3.60** | **2.39 / +2.40** | **1.58 / +1.59** |
| 30m | Asset-relative deviation | 0.744 | 0.196 | 0.14 | 2.39 / +2.26 | 1.38 / +1.24 | 0.92 / +0.78 |

The literal absolute-spread arm works largely because it recognizes structural
wide books.  It is useful as a static friction/risk diagnostic, but should not
be presented as a predictor of *incremental* spread deterioration.  The
30-minute normalized spread-delta arm is the research champion for that use.

It is portable across the descriptive liquidity groups built from each asset's
initial strictly-prior spread baseline: its within-query raw-delta Spearman is
0.442 (tight/deep), 0.457 (middle), and 0.429 (wide/thin); corresponding top-1
uplifts are +0.22, +0.88, and +3.84 bps.

## Interpretation and next use

The ranker can be used as a causal **friction-warning input**: a high predicted
delta can demote a candidate's execution-adjusted EV or increase a conservative
slippage reserve.  It must not be treated as evidence for a direct trade signal
or live deployment yet: there are no futures snapshots, no individual trade
flow/actual volume, and this study has not linked the warning score to realised
Kraken fill slippage under the live policy.

Relevant artifacts:

- `scripts/execution/materialize_hf_kraken_orderbook_recap.py`
- `scripts/execution/audit_hf_kraken_orderbook_recap.py`
- `scripts/execution/evaluate_liquidity_transition.py`
- `data/execution/hf_kraken_orderbook_recap_v1/contract_audit.json`
- `data/execution/hf_kraken_orderbook_recap_v1/models/lambdarank_h30_spread_delta_v3/`

