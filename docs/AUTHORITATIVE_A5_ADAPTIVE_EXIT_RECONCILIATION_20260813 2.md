# Authoritative A5 adaptive-exit reconciliation — 2026-08-13

## Decision

The fair comparison is complete.  The adaptive controller is **not promoted**
automatically.  It is positive against the authoritative canonical baseline
both with the original 8,453 trades held fixed and after rerunning the
capacity-aware auction.

## What was repaired

The earlier complete-15-minute replay was not a valid canonical baseline:

1. it imported `simple_policy_optimiser` before setting the canonical
   zero-gap/zero-spread environment, creating an approximately 19-bps penalty
   on fine-path outcomes;
2. it replaced the canonical hourly-proxy source with newly downloaded
   15-minute paths.

The corrected runner sets all three policy environment controls before the
simulator import and hard-gates fine-path parity at 0.002 bps.  Actual maximum
absolute error is 0.001192 bps, caused by float32 path serialization.  The
baseline auction then reproduces the canonical replay exactly:

- 53,282 admitted candidates;
- identical candidate indices and accept/reject decisions;
- identical rejection reasons and exit timestamps;
- zero net/gross-return delta;
- 8,453 trades at +163.0863 net bps/trade.

## Source-aligned contract

The canonical admitted ledger contains two distinct outcome substrates:

- `existing_15m_or_exact`: 34,469 candidates overall and 5,500 canonical
  accepted trades;
- `hourly_ohlc_proxy`: 18,813 candidates overall and 2,953 canonical accepted
  trades.

Each source is controlled on its own authoritative resolution:

- fine-path rows use the F4-disagreement-gated continuous controller;
- hourly-proxy rows use the already completed source-matched F1-path28 OOF
  forecast, converted to continuous activation with 0.75 shrinkage and
  0.5x--1.25x bounds.

The adaptive OOF schedule begins in April 2025.  Consequently, 6,800 of the
8,453 fixed canonical trades have a source-matched OOF controller: 4,374 fine
and 2,426 hourly.  January--March 2025 remain unchanged.

The execution contract remains:

- entry at the first 15-minute open one hour after signal close;
- one exit decision per completed hour, effective on the next 15-minute bar;
- stop fixed at 4.152000643 ATR;
- baseline activation 2.326224920 ATR;
- giveback fixed at 0.102371990 ATR;
- H12 timeout;
- 100-bps cost exactly once.

## Layer 1 — identical canonical trades

This comparison holds the original 8,453 accepted candidate IDs fixed.
Admission, ordering and capacity cannot change.

| Period | Trades | Baseline net | Adaptive net | Uplift |
|---|---:|---:|---:|---:|
| 2025--2026 | 8,453 | +163.09 | **+183.63** | **+20.55** |
| 2025 | 6,031 | +155.58 | **+177.02** | **+21.44** |
| 2026 | 2,422 | +181.77 | **+200.09** | **+18.33** |

All values are net bps/trade.  Using a 0.01-bps numerical-flat tolerance,
across 81 calendar weeks uplift is positive in 59, negative in 4 and flat in
18; the flat weeks include the January--March 2025 fail-closed warm-up. Median
weekly uplift is +15.26 bps and worst weekly uplift is -44.76 bps.

### Fixed-trade monthly results

| Month | Trades | Baseline | Adaptive | Uplift |
|---|---:|---:|---:|---:|
| 2025-01 | 576 | +162.00 | +162.00 | 0.00 |
| 2025-02 | 581 | +160.00 | +160.00 | 0.00 |
| 2025-03 | 491 | +171.14 | +171.14 | 0.00 |
| 2025-04 | 513 | +227.05 | +255.37 | +28.31 |
| 2025-05 | 578 | +127.66 | +181.95 | +54.29 |
| 2025-06 | 274 | +43.98 | +77.79 | +33.80 |
| 2025-07 | 522 | +151.51 | +182.99 | +31.48 |
| 2025-08 | 531 | +134.52 | +163.01 | +28.48 |
| 2025-09 | 560 | +111.67 | +128.66 | +16.99 |
| 2025-10 | 593 | +191.82 | +213.30 | +21.48 |
| 2025-11 | 315 | +206.92 | +265.30 | +58.38 |
| 2025-12 | 497 | +148.33 | +152.20 | +3.87 |
| 2026-01 | 436 | +206.96 | +226.44 | +19.47 |
| 2026-02 | 264 | +202.91 | +252.62 | +49.70 |
| 2026-03 | 487 | +175.34 | +193.41 | +18.07 |
| 2026-04 | 509 | +243.43 | +275.60 | +32.17 |
| 2026-05 | 576 | +130.01 | +126.29 | **-3.72** |
| 2026-06 | 51 | +4.43 | +6.97 | +2.54 |
| 2026-07 | 99 | +121.53 | +117.55 | **-3.98** |

## Layer 2 — capacity-aware auction

The auction is rerun from the same 53,282 admissions.  Only exit timestamps may
change.  Earlier exits release symbol/concurrency/capital capacity and can
therefore change which later already-admitted candidates enter.

| Metric | Canonical frozen | Adaptive hybrid | Change |
|---|---:|---:|---:|
| Accepted trades | 8,453 | **8,663** | +210 |
| Trades/calendar day | 14.72 | **15.09** | +0.37 |
| Net bps/trade | +163.09 | **+181.61** | +18.52 |
| Positive-trade rate | 63.88% | **68.22%** | +4.34 pp |
| Sortino | 0.465 | **0.554** | +0.089 |
| Worst wallet week | -35.09% | **-11.67%** | +23.42 pp |
| Max drawdown | -76.53% | -76.53% | unchanged |
| Mean open positions | 6.56 | 6.50 | -0.06 |
| Full-stop rate | 5.30% | **2.08%** | -3.22 pp |
| Timeout rate | 25.85% | **20.94%** | -4.91 pp |

Dynamic weekly net-EV change is positive in 57 weeks, negative in 9 and zero in
15.  Median weekly change is +8.04 bps.  The worst change is -20.95 bps in the
week of 2026-05-11; July 20--26 is also weaker by -12.05 bps.

The dynamic auction is weaker than canonical in May 2026 (+122.36 versus
+130.01 bps/trade) and July 2026 (+117.55 versus +121.53).  Those recent-period
failures are the main reason not to promote from this development result alone.

## Model interpretation

The fine-path arm remains `F4_disagreement_abstain_p80`, chosen on 2025 only.
On all supported fine-path OOF candidates it adds +37.17 bps/trade in 2025 and
+16.13 in untouched 2026.  The hourly source uses the prior F1-path28 OOF
forecast but replaces its discrete action mapping with the same continuous
activation authority. Stop and giveback are never modified.

## Reproduction

```bash
python3 scripts/run_canonical_a5_15m_adaptive_exit_funnel.py \
  --out-dir data_perp/artifacts/canonical_a5_source_aligned_hybrid_adaptive_exit_funnel_20260813_v4 \
  --max-train-states 40000 \
  --adaptive-source existing_15m_or_exact \
  --parity-tolerance-bps 0.002

python3 scripts/reconcile_authoritative_a5_adaptive_exit.py \
  --out-dir data_perp/artifacts/authoritative_a5_adaptive_exit_reconciliation_20260813_v4
```

Relevant outputs:

- corrected OOF/adaptive run:
  `data_perp/artifacts/canonical_a5_source_aligned_hybrid_adaptive_exit_funnel_20260813_v4`;
- fixed canonical-trade comparison:
  `data_perp/artifacts/authoritative_a5_adaptive_exit_reconciliation_20260813_v4/fixed_canonical_trade_comparison.parquet`;
- monthly/weekly fixed-trade metrics:
  `data_perp/artifacts/authoritative_a5_adaptive_exit_reconciliation_20260813_v4/fixed_trade_metrics.parquet`;
- dynamic-capacity decisions and risk metrics:
  `data_perp/artifacts/authoritative_a5_adaptive_exit_reconciliation_20260813_v4/portfolio_SOURCE_MATCHED_FINE_F4_HOURLY_F1`;
- parity and lineage receipt:
  `data_perp/artifacts/authoritative_a5_adaptive_exit_reconciliation_20260813_v4/run_manifest.json`.
