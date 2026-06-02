# Data Source and Microstructure Reconciliation

Status: in progress, 2026-06-01.

## Scope

Run: `20260525_010004_nopenalty`

Market: Kraken perps

This audit compares locally persisted live-fetched OHLCV against cached historical hourly references and against execution 1m bars aggregated back to hourly. It is intended to answer whether live candles are numerically equivalent to historical candles already on disk, and whether the 1m execution data path is coherent enough to tune delayed-entry, spread, and slippage assumptions.

## Commands

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

Outputs were written under:

`data_perp/artifacts/20260525_010004_nopenalty/live_ohlcv_parity/`

## Store Paths Compared

- Live hourly: `data_perp/exchanges/krakenfutures/ohlcv/1h`
- Cached historical hourly: `data_perp/exchanges/krakenfutures/exchanges/krakenfutures/ohlcv/1h`
- Execution 1m: `data_perp/exchanges/krakenfutures/execution_1m/ohlcv/1m`
- Nested execution 1m: `data_perp/exchanges/krakenfutures/exchanges/krakenfutures/execution_1m/ohlcv/1m`

The parity script reads raw partition files directly and intentionally bypasses production sidecar overlays so the comparison is about saved candle values.

## Summary

| Pair | Symbols | Overlap rows | Symbols with mismatch | Mismatch rows | Mismatch rate |
|---|---:|---:|---:|---:|---:|
| live hourly vs historical hourly | 306 | 306 | 4 | 4 | 1.31% |
| live hourly vs execution 1m aggregate | 306 | 8,986 | 220 | 3,833 | 42.66% |
| live hourly vs nested execution 1m aggregate | 306 | 0 | 0 | 0 | n/a |

## Hourly Live vs Historical Hourly

The broad sample is mostly clean: 302 of 306 symbol-overlap rows match exactly within the configured tolerances.

The four mismatching symbols all mismatch on the same cached historical reference timestamp, `2026-05-22 05:00:00 UTC`:

| Symbol | Max open diff | Max high diff | Max low diff | Max close diff | Max volume diff |
|---|---:|---:|---:|---:|---:|
| `ETH/USD:USD` | 6.899902 | 5.599854 | 4.100098 | 5.700195 | 372.700012 |
| `GOOGLX/USD:USD` | 0.709991 | 0.000000 | 0.099976 | 0.610016 | 1593.220000 |
| `LTC/USD:USD` | 0.160000 | 0.020000 | 0.009998 | 0.040001 | 123.639999 |
| `ROSE/USD:USD` | 0.000000 | 0.000000 | 0.000060 | 0.000060 | 1.000000 |

Interpretation:

- This does not look like a general live-hourly fetch issue.
- The historical-hourly cache currently has only one overlapping row per symbol in this comparison, so the test is a broad symbol test but still a shallow time test.
- The ETH mismatch remains material for that one row: about 19-32 bps on OHLC and a large volume difference.
- The next check should inspect whether those four historical rows came from a different cache path, exchange alias, or post-hoc data correction.

## Hourly Live vs Execution 1m Aggregate

This path is not clean. Of 8,986 overlapping symbol-hour rows, 3,833 differ beyond tolerance. Most overlapping symbols have at least one mismatch.

The largest recurring issues include:

- Price differences in many small or thin contracts, often tens to hundreds of bps in the mismatch examples.
- Volume differences are frequent and sometimes extreme, including cases where the 1m aggregate has zero volume while the hourly candle has nonzero volume.
- Only 226 of 306 symbols have overlapping execution-1m rows in the compared window.

The follow-up completeness probe explains why this aggregate cannot be interpreted as full hourly parity:

- Rows checked: 8,986 symbol-hours.
- Complete 60-minute hours: 0.
- Incomplete hours: 8,986.
- Unique saved 1m rows per symbol-hour: min 1, median 1, max 1.
- Every checked execution-1m row is a single sampled minute inside the hour, not a full 60-minute candle set.
- The inspected symbols have rows at minute `:07`, consistent with an older t+7 delayed-entry proxy artifact rather than the current code default of t+10.

Interpretation:

- This is not evidence that complete 1m candles disagree with hourly candles. The store does not contain complete 1m candles for these hours.
- It is evidence that the local execution-1m store is a sparse delayed-entry sample cache, not a reusable full 1m history store.
- Because the simple policy optimiser depends on 1m candles for delayed-entry and adverse-entry-gap modelling, the sparse store is enough for a single delayed-entry proxy but not for reconstructing the intra-delay path. Spread/slippage fitting must rely on the live observations we persist around signal, order, and fill time, because quote/orderbook snapshots are not available.

## Candidate Artifact Delay

`simple_policy_optimiser/simple_policy_candidates.parquet` currently contains `delayed_entry_ts = timestamp + 7 minutes` for all 20,026 rows. The current code default is `EPM_SIMPLE_POLICY_DELAYED_ENTRY_MINUTES=10`.

This means the current candidate artifact and derived deployment metrics are still based on t+7 delayed-entry candles, not the requested t+10 policy. The code has been changed, but this artifact must be regenerated before the report can claim t+10 OOS execution realism for `20260525_010004_nopenalty`.

## Current Conclusion

Live hourly candles mostly match cached historical hourly candles where overlap exists. The evidence does not support a broad hourly-live data corruption root cause.

The execution 1m path is not full 1m history; it is currently sparse delayed-entry sample data. This is now a confirmed data-collection limitation and artifact-staleness issue for execution realism, because the saved policy candidates are still t+7 while the current code default is t+10.

## Follow-Up Checks

1. Expand historical-hourly overlap beyond one row per symbol by fetching or locating a deeper cached historical-hourly slice, then rerun the same parity script.
2. For the four hourly mismatches, trace file paths and partition mtimes for the live and historical rows to identify whether they came from different endpoints or stale caches.
3. For execution 1m, collect at least the exact delayed-entry proxy minute needed by the policy replay. Full intra-delay path modelling is out of scope for now because it would require a separate model and denser data.
4. Add a completeness gate before delayed-entry replay: expected minutes, actual minutes, zero-volume minutes, duplicate minutes, and timestamp offsets.
5. Confirm volume units for hourly and 1m paths before using 1m aggregate volumes in replay fitting.
6. Regenerate simple-policy candidates and deployment metrics with the current t+10 delayed-entry setting.
7. Persist live observed entry gap, fee, realized spread/slippage proxy, rejection reason, and stop trigger/fill gap fields for every actual or rejected market/stop decision. Quote/orderbook snapshot fitting is not available, so policy tuning should use these live observations plus the delayed-entry candle proxy.
