# Kraken L2 execution-risk research

## Purpose

This is a self-contained, offline research pipeline for predicting imminent
Kraken order-book spread and executable-slippage deterioration. It can support
a separately authorised *earlier-exit* overlay, but does **not** modify the
canonical Strict-R3 close-price exit policy, its parameters, or its
policy-optimisation objective.

The parent policy remains the source of an exit timestamp `t*`. The research
oracle may consider only `t*−5m`, `t*−3m`, `t*−2m`, `t*−1m`, and `t*`. A future
overlay may only pull an already-closed exit forward by zero to three minutes;
it may not change stops, take-profit, trailing activation, sizing, or entry
logic.

## Contracts

The implementation follows the repository's [dataset contract](../agents/dataset_contract.md),
[leakage prevention rules](../agents/leakage_prevention.md), and
[backtest protocol](../agents/backtest_protocol.md).

- All timestamps are UTC.
- Feature state uses only messages locally observed at or before its timestamp.
- Future L2 deterioration is a label, never an inference feature.
- A missing/crossed/pre-snapshot book is unavailable; it is never forward-filled.
- Spot base-quantity semantics are isolated in `src/execution/book_walk.py`.
  Kraken Futures must use a separately validated quantity adapter.
- Canonical close-price PnL and execution cost are represented separately, so a
  parent policy cost is applied exactly once.

## Components

| Component | Role | Output |
|---|---|---|
| `scripts/execution/build_tardis_free_manifest.py` | Queries official Tardis exchange metadata and combines it with a user-supplied exact internal-to-dataset symbol mapping. It emits only first UTC days of months. | Immutable download manifest + receipt |
| `scripts/execution/download_tardis_kraken_free.py` | Resumable, atomic raw downloader. It cannot overwrite a verified raw file. | Raw `.csv.gz` + download-status manifest |
| `src/execution/tardis_book.py` | Atomic incremental L2 reconstruction. Snapshot resets, zero amount deletes, positive amount replaces the price level. | Complete state iterator |
| `scripts/execution/build_kraken_execution_states.py` | Streams temporary raw L2 into compact per-minute order-book recaps and optional message-level states. | `kraken_execution_surface` + quality audit |
| `scripts/execution/validate_tardis_snapshot5.py` | Checks either a normalized pilot or raw `book_snapshot_5` records against the last complete reconstructed state at or before each local timestamp. | Reconstruction audit |
| `scripts/execution/build_execution_oracle.py` | Joins a fixed canonical parent-policy exit to executable L2 cost at predeclared earlier candidates. | Oracle + robust summary |
| `scripts/execution/train_execution_risk.py` | Trains only chronological controls: empirical shrinkage, OHLCV-only shallow LGBM, and L2-aware shallow LGBM. | Held-month predictions, model artifact, manifest |
| `scripts/execution/report_execution_risk.py` | Renders evidence from supplied receipts. | Research report |

## Data acquisition

The mapping is deliberately explicit. Create a CSV with:

```csv
internal_symbol,dataset_symbol,valid_from,valid_to
BTC/USD,BTC/USD,2024-01-01T00:00:00Z,
```

Use metadata, not fuzzy matching:

```bash
python scripts/execution/build_tardis_free_manifest.py \
  --mapping config/execution/kraken_spot_tardis_mapping.csv \
  --start 2024-01-01T00:00:00Z --end 2026-01-01T00:00:00Z \
  --data-types incremental_book_L2 \
  --out data/execution/tardis/manifests/kraken_free.parquet

python scripts/execution/download_tardis_kraken_free.py \
  --manifest data/execution/tardis/manifests/kraken_free.parquet
```

`book_snapshot_5` is a small reconstruction pilot. Its raw schema is retained as
downloaded. The validator supports a bounded raw-stream mode which walks every
L2 message but persists only a deterministic snapshot sample, avoiding a
multi-million-row message-state dump:

```bash
python scripts/execution/validate_tardis_snapshot5.py \
  --raw-incremental data/execution/tardis/raw/kraken/incremental_book_L2/ETH__USD/2025-01-01.csv.gz \
  --raw-snapshot5 data/execution/tardis/raw/kraken/book_snapshot_5/ETH__USD/2025-01-01.csv.gz \
  --sample-stride 100 \
  --out data/execution/tardis/reports/kraken_snapshot5_validation_ETH_20250101.parquet
```

Every retained recap is compared only after all complete L2 source messages
whose local timestamp is no later than that snapshot have been applied. A
missing, empty, or crossed reconstructed book is recorded as unmatched, rather
than silently backfilled.

### Compact retention contract

Raw Tardis data is an ephemeral materialisation input, not a retained research
asset. The compact contract retains only per-minute, causal **order-book**
recaps: top of book, spread, microprice, depth, imbalance, executable cost for
the declared notional grid, and L2 cancel/replenish summaries. Individual trade
prints are deliberately excluded. When the full feature contract requires
executed-flow fields, a separate compact per-minute **aggregate** recap may
retain only quote-volume totals, trade counts, and signed flow imbalance; it
must never retain individual trade timestamps, prices, sizes, or IDs. Future
1/2/3/5/10/15/30 minute outcomes are offline labels only.

Before raw deletion, every source symbol/day must have exactly one complete
compact recap with the required L2 field contract and no `trade_*` or
`sell_order_flow_imbalance` field. The raw L2, snapshot, and trade directories
may then be permanently removed; the manifest, checksum/coverage receipt, and
compact recap remain as the reproducible substrate.

The deployed research retention state follows this contract: raw archives have
been pruned after validation. It retains compact order-book recaps plus their
receipts and a 6.3-MiB aggregate-activity recap where needed for the full
feature contract. Individual trade prints and any message-level trade archive
are not retained.

## Completed bounded raw pilot (2025-01-01)

The exact Kraken Futures frozen-universe mapping and raw acquisition receipt is
`agents/receipts/20260819_kraken_l2_exact_mapping_raw_pilot.json`.

- 170 source perpetual symbols; 136 exact `BASE/USD:USD → BASE/USD` Tardis
  Kraken-Spot mappings; 34 explicitly unmapped (no aliases or fuzzy matching).
- Immutable raw pilot: ETH/USD, SOL/USD, XRP/USD; each has
  `incremental_book_L2`, `book_snapshot_5`, and `trades` data. All nine gzip
  files passed integrity validation.
- Compact state surfaces retain 1,440 source minutes per symbol. ETH/USD and
  SOL/USD have 1,440 valid minutes; XRP/USD has 1,432 valid minutes and keeps
  eight crossed/empty minutes unavailable.
- Raw `book_snapshot_5` validation at a 0.5-bps top-of-book tolerance sampled
  24,347 ETH, 13,583 SOL, and 41,321 XRP snapshots while walking every raw L2
  message. The respective matched-within-tolerance rates were 99.988%,
  99.845%, and 99.917%. XRP’s 302 unmatched samples correspond to explicitly
  invalid reconstructed book states, not forward-filled levels.

This evidence validates the source reconstruction only. It is not economic
evidence for an exit overlay, and it does not alter the canonical policy,
admission, live inference, or live execution stack.

## State surface

The compact order-book surface includes: bid/ask/mid/spread; microprice;
notional depth within 10/25/50/100/200 bps; imbalance; sell/buy executable VWAP
and cost for a historical quote-notional grid; L2 cancel/replenish summaries;
and 1/3/5-minute causal spread/depth/cost transitions. It can write
message-level states so an oracle chooses the first valid state at 0, 250 ms,
1 s, or 5 s after a candidate exit time.

```bash
python scripts/execution/build_kraken_execution_states.py \
  --manifest data/execution/tardis/manifests/kraken_free_download_status.parquet \
  --notional-source data_perp/artifacts/<historical_portfolio>.parquet \
  --write-message-states
```

Future-only labels include 1/2/3/5/10/15/30-minute executable book-cost deterioration,
spread widening, 50/100/150/250-bps cost-tail flags, 10/25/50/100-bps
spread-tail flags, and maximum cost/spread deterioration over the next
3/5/10/15/30 minutes. A path with a missing state is invalid for a
maximum-path label.

The primary short-horizon prediction targets are deliberately separated:

```text
book_cost_delta       = future side/size-specific executable VWAP cost − current cost
max_book_cost_delta   = maximum executable cost over the complete next 3/5m path − current cost
spread_delta          = terminal quoted-spread widening
max_spread_delta      = maximum quoted-spread widening over the complete next 3/5m path
```

`book_cost_*` is the tradeable slippage proxy because it is evaluated at a
declared quote notional and liquidation side. `spread_*` is a complementary
market-liquidity warning. Neither is an actual fill prediction until joined to
an explicit order-size and latency contract.

## Fixed-policy oracle

Materialize exits with the existing canonical policy first, then run:

```bash
python scripts/execution/build_execution_oracle.py \
  --exits data_perp/artifacts/<canonical_policy_exits>.parquet \
  --prices data_perp/artifacts/<canonical_minute_closes>.parquet \
  --states-root data/execution/tardis/processed/kraken_execution_message_states \
  --notionals 100 250 500 1000 2500 \
  --out data/execution/tardis/reports/execution_oracle.parquet
```

For each earlier exit candidate the oracle reports:

```text
preemption_gain_bps
  = (candidate close PnL − canonical close PnL)
  + (canonical executable cost − candidate executable cost)
```

It also records missing/late book state and insufficient-depth flags. The
canonical close remains authoritative; the L2 mid is audit-only.

## Bounded model comparison

Run a separate model arm per fixed configuration, always with the latest
whole-month chronological holdout:

```bash
python scripts/execution/train_execution_risk.py \
  --surface-root data/execution/tardis/features/kraken_execution_surface \
  --ohlcv data_perp/artifacts/<causal_ohlcv>.parquet \
  --arm l2_aware --task quantile --horizon-minutes 3 --max-depth 3 \
  --validation-months 3 --out-dir data/execution/tardis/models/l2_q75_h3
```

The empirical baseline uses training-derived bins that are persisted with the
model. LGBM is intentionally shallow (depth 2–4); there is no broad HPO and
no random row-level split. Compare:

1. empirical shrinkage;
2. OHLCV-only;
3. L2-aware;
4. optionally, a small quantile/classifier sensitivity.

The L2-aware model must add stable held-month value over both controls before
any separately authorised integration discussion.

## Required tests and reporting

Run:

```bash
python -m pytest -q tests/execution
python -m py_compile src/execution/*.py scripts/execution/*.py
```

The report must cover data and reconstruction quality, cost distributions,
fixed-policy oracle results, held-month predictive evidence, L2 incremental
value, economic tail slices, and a recommendation. Generate it with:

```bash
python scripts/execution/report_execution_risk.py \
  --state-receipt data/execution/tardis/reports/kraken_execution_state_build_audit.json \
  --oracle data/execution/tardis/reports/execution_oracle.parquet \
  --training-manifest data/execution/tardis/models/l2_q75_h3/run_manifest.json \
  --out docs/KRAKEN_L2_EXECUTION_RISK_REPORT.md
```

No results exist until a manifest, raw download, reconstruction, canonical
exits, and chronological held-out training have actually been materialized.
