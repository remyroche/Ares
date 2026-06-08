# Data Source and Microstructure Reconciliation

Status: refreshed 2026-06-04.

## Scope

Run: `20260525_010004_nopenalty`

Market: Kraken perps

This audit compares persisted live-fetched hourly OHLCV against cached historical hourly references and compares hourly OHLCV against locally cached execution 1m rows aggregated back to hourly. It also checks whether the simple-policy t+10 execution path can reuse cached 1m rows instead of redownloading complete windows.

## Command

```bash
python3 scripts/verify_live_ohlcv_parity.py \
  --data-root data_perp \
  --run-id 20260525_010004_nopenalty \
  --symbols auto \
  --max-symbols 0 \
  --start 2026-05-01T00:00:00Z \
  --end 2026-05-31T00:00:00Z \
  --max-mismatch-examples 300
```

Outputs:

`data_perp/artifacts/20260525_010004_nopenalty/live_ohlcv_parity/`

## Store Paths Compared

- Live hourly: `data_perp/exchanges/krakenfutures/ohlcv/1h`
- Cached historical hourly: `data_perp/exchanges/krakenfutures/exchanges/krakenfutures/ohlcv/1h`
- Execution 1m: `data_perp/exchanges/krakenfutures/execution_1m/ohlcv/1m`
- Nested execution 1m: `data_perp/exchanges/krakenfutures/exchanges/krakenfutures/execution_1m/ohlcv/1m`

The parity script reads raw partition files directly and bypasses production sidecar overlays, so this comparison is about saved candle values.

## Summary

| Pair | Symbols | Overlap rows | Symbols with mismatch | Mismatch rows | Mismatch rate |
|---|---:|---:|---:|---:|---:|
| live hourly vs historical hourly | 306 | 306 | 3 | 3 | 0.98% |
| live hourly vs execution 1m aggregate | 306 | 22,340 | 227 | 5,705 | 25.54% |
| live hourly vs nested execution 1m aggregate | 306 | 0 | 0 | 0 | n/a |

## Hourly Live vs Historical Hourly

The broad symbol sample remains mostly clean: 303 of 306 overlapping symbol rows match exactly within tolerance. The historical-hourly cache still has only one overlapping timestamp per symbol in this comparison, so this is a broad-symbol but shallow-time test.

The three mismatching symbols all mismatch on `2026-05-22T05:00:00Z`:

| Symbol | Largest price diff | Volume diff |
|---|---:|---:|
| `ETH/USD:USD` | open diff 6.899902, close diff 5.700195 | 372.700012 |
| `GOOGLX/USD:USD` | open diff 0.709991, close diff 0.610016 | 1593.220000 |
| `LTC/USD:USD` | open diff 0.160000, close diff 0.040001 | 123.639999 |

Interpretation:

- This still does not look like broad live-hourly data corruption.
- The mismatches are localized to one cached historical timestamp, so the next investigation target is cache provenance for those rows, not the live fetcher generally.
- Because the historical overlap is shallow, this audit is not enough to prove all historical/live data paths are interchangeable over long spans.

## Hourly Live vs Execution 1m Aggregate

Execution 1m coverage is now more informative than the earlier sparse-only audit:

- Rows checked: 22,340 symbol-hours.
- Complete 60-minute hours: 2,832.
- Incomplete hours: 19,508.
- Unique 1m rows per symbol-hour: min 1, median 14, max 60.
- Rows by unique 1m minutes: 7,759 with 1 minute, 11,749 with 14 minutes, 2,832 with 60 minutes.

The full aggregate mismatch rate is 25.54%, but this is dominated by incomplete 1m hours. Among complete 60-minute hours, only 2 of 2,832 rows mismatch:

| Symbol | Timestamp | Difference |
|---|---|---|
| `MNT/USD:USD` | `2026-05-17T15:00:00Z` | low and close differ by 0.00070 |
| `USUAL/USD:USD` | `2026-05-21T08:00:00Z` | open/low differ by 0.00055 and volume differs by 32,300 |

Interpretation:

- Complete cached 1m hours mostly aggregate back to hourly OHLCV correctly.
- Incomplete 1m hours must not be used to infer hourly parity or full intrahour path statistics.
- The current policy candidates do not need full-hour reconstruction: they need the t+10 delayed-entry window and candidate metadata confirms those windows are complete for rows using the 1m proxy.

## Candidate t+10 Execution Cache

`data_perp/artifacts/20260525_010004_nopenalty/simple_policy_optimiser/simple_policy_candidates_metadata.json` now shows:

- configured delayed entry: 10 minutes.
- candidate rows: 27,485.
- all candidate rows have `delayed_entry_ts - timestamp = 10 minutes`.
- `delayed_1m_intraminute_proxy`: 25,564 rows.
- `theoretical_15m_open`: 1,921 fallback rows.
- delayed 1m coverage: 93.01%.
- delayed 1m rows have 11 candles in the delay window.

Code inspection of `_load_policy_1m_klines_cached(...)` in `simple_policy_optimiser.py` confirms the intended reuse behavior:

- It first checks the in-process `_POLICY_1M_KLINES_CACHE`.
- It then loads persisted rows from `PartitionedOHLCVStore`.
- It computes missing minute ranges against the exact required timestamps.
- It downloads only missing ranges when `EPM_SIMPLE_POLICY_1M_DOWNLOAD` is enabled.
- Fresh rows are saved with `save_partitioned(..., defer_compact=True)` and touched years are compacted afterward.

## Live Observation Coverage

`prediction_ledger.parquet` contains the live observation fields needed to tune replay assumptions, including spread, expected fill slippage, orderbook slippage proxy, expected total entry friction, entry gap, realized entry price, theoretical/policy/ohlcv entry price fields, decision-to-entry time, signal-to-entry time, and portfolio rejection reason.

Current coverage is sparse because only one row in the latest ledger was traded:

- prediction rows: 87.
- traded rows: 1.
- friction/delay/spread fields non-null: 1.
- `signal_to_entry_seconds`: 5102.37 seconds on that traded row.
- `entry_delay_adverse_bps`: 30.50 bps.
- `expected_fill_slippage_bps`: 7.52 bps.
- `spread_bps`: 1.00 bps.
- `realized_entry_price`: present for the traded row.
- `inference_trades.sqlite`: currently empty, so ledger is the usable live-observation store for this run.

## Current Conclusion

Live hourly candles mostly match cached historical hourly candles where overlap exists. The evidence does not support a broad live-hourly data corruption root cause.

Execution 1m data is no longer just a single-minute sparse cache: it includes a mixture of exact delayed-entry windows, 14-minute windows, and complete 60-minute blocks. Complete 60-minute aggregates almost always match hourly candles, but incomplete windows still dominate mismatch counts and must not be treated as full intrahour history.

The active simple-policy candidate artifacts are now t+10 and use cached 1m rows for 93.01% of candidates. The remaining 6.99% fallback rows should stay visible in metrics and should not silently drive threshold conclusions as if they had the same execution evidence quality as 1m-backed rows.

## Follow-Up Checks

1. Trace the three live-vs-historical hourly mismatches at `2026-05-22T05:00:00Z` to exact partition files and mtimes.
2. Expand historical-hourly overlap beyond one row per symbol when a deeper cached historical-hourly slice is available.
3. Keep a completeness gate for delayed-entry replay: required minutes, actual minutes, duplicate minutes, and fallback reason.
4. Report policy metrics separately for `delayed_1m_intraminute_proxy` and `theoretical_15m_open` rows.
5. Continue persisting live spread/slippage/delay/friction fields in the prediction ledger; the current sample size is too small to tune replay assumptions robustly.
