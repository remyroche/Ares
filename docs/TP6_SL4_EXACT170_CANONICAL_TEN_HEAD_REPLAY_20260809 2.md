# Exact-170 TP6/SL4/H12 canonical ten-head consensus replay

Date: 2026-08-09

## Executive result

The full ten-head architecture was run on the exact 170-symbol TP6/SL4/H12
population with monthly prequential refits from February through July 2026.
The pooled global ranking is economically negative after the declared 100-bps
cost floor:

| Arm | Top 1% gross / net (bps) | Top 5% gross / net (bps) | Top 10% gross / net (bps) | Rank IC |
|---|---:|---:|---:|---:|
| Base only | 58.16 / **-41.84** | 42.43 / **-57.57** | 18.57 / **-81.43** | 0.0457 |
| Ten-head consensus only | 11.23 / **-88.77** | 6.94 / **-93.06** | 8.50 / **-91.50** | 0.0270 |
| Canonical 75/25 blend | 29.69 / **-70.31** | 17.14 / **-82.86** | 17.78 / **-82.22** | 0.0523 |

The blend improves rank IC over the base, but it reduces gross tail separation
relative to the base and does not clear costs.  This is an exact-universe
stress result, not evidence that the canonical 2025 long-only result transfers
to this 2026 population.

## Contract used

- Population: exact 170-symbol candidate IDs from the materialized TP6/SL4/H12 label ledger.
- Entry: signal close, then one completed hour, exact next-minute open.
- Geometry: +6 ATR take-profit / -4 ATR stop-loss, 12-hour horizon, adverse tie precedence.
- Cost: 100 bps applied exactly once.
- Base target: strict R3, class 0 adverse-first, class 1 weak/timeout/marginal upper, class 2 robust clear (pre-adverse MFE clears cost +25 bps; B25/T50).
- Base: 220 trees, learning rate 0.035, depth 5, 24 leaves, 2,400 minimum child samples, feature fraction 0.85, L2 20; latest matured 240,000-row fit cap.
- Residual target: realised TP6/SL4 net bps minus the train-only isotonic base-score net map, ordinal grades `[-150, -50, +50, +150]`.
- Consensus: ten native LambdaRank heads, caps 40/60/80/100/120 crossed with ordinary/equal-month weighting (median of the ten held-month percentile ranks).
- Ranker: 120 trees, learning rate 0.035, depth 5, 31 leaves, 300 minimum child samples, feature/bagging fractions 0.82, L1 0.02, L2 2, max bin 127, truncation 10, gains `[0, 0.25, 1, 3, 7]`.
- Queries: 4-hour UTC bucket × side. Selection: monthly side percentile ranks, then `0.75 * base + 0.25 * consensus`, then one pooled global ranking; no per-timestamp top-k selection.
- Fold schedule: February–July 2026. January is a warm-up month because no earlier exact-contract labels are available for a causal monthly fit.

## Monthly canonical-blend metrics

| Held month | Rows in top 5% | Gross bps/trade | Net bps/trade | Rank IC |
|---|---:|---:|---:|---:|
| 2026-02 | 1,976 | -73.05 | -173.05 | -0.0650 |
| 2026-03 | 2,283 | 9.03 | -90.97 | 0.0178 |
| 2026-04 | 3,234 | 65.51 | -34.49 | 0.1352 |
| 2026-05 | 8,806 | -6.49 | -106.49 | 0.0661 |
| 2026-06 | 7,811 | 46.06 | -53.94 | 0.0573 |
| 2026-07 | 2,563 | 25.97 | -74.03 | 0.0178 |

## Side-local canonical-blend metrics

| Side | Top 1% gross / net | Top 5% gross / net | Top 10% gross / net | Rank IC |
|---|---:|---:|---:|---:|
| Long | 5.91 / **-94.09** | 3.65 / **-96.35** | 6.75 / **-93.25** | 0.0437 |
| Short | 50.25 / **-49.75** | 27.04 / **-72.96** | 24.86 / **-75.14** | 0.0579 |

The short side carries nearly all of the remaining gross tail separation.  The
long side is economically unusable under this exact population and cost
contract.

## Coverage gate and interpretation

The feature materializer retained the exact 120-field side-local contract, but
the available 2026 source panel does not supply all fields.  Across the union
of the long/short contracts, 187 requested fields were retained; 11 long
fields and 20 short fields are entirely unavailable from the supplied causal
source panel.  Many additional OI/funding/order-book fields are sparse.  The
raw panel’s maximum finite coverage is about 89.4% of candidate rows, and the
side-specific 90%-of-120 row gate is not met (long about 9.9% of rows; short
0% under the strict gate). Missing values were imputed from training medians
for model execution, never filled with future outcomes.

Consequently, the run is a valid diagnostic of the ten-head logic on the exact
population, but it is **not a strict feature-parity apples-to-apples result**
against the 2025 canonical handover. The next blocking task is to recover or
materialize the full 120-field causal source panel for this exact universe;
repeating the model fit without that source repair would only measure the
sparse-contract fallback.

## Artifacts

- Feature panel: `data_perp/artifacts/tp6_sl4_exact170_canonical120_features_20260809_v4/canonical120_features.parquet`
- Feature coverage: `data_perp/artifacts/tp6_sl4_exact170_canonical120_features_20260809_v4/feature_coverage.parquet`
- Predictions: `data_perp/artifacts/tp6_sl4_exact170_canonical_consensus_20260809_v5/predictions.parquet`
- Pooled/monthly/side metrics: `data_perp/artifacts/tp6_sl4_exact170_canonical_consensus_20260809_v5/metrics.parquet`
- Fold lineage: `data_perp/artifacts/tp6_sl4_exact170_canonical_consensus_20260809_v5/fold_metrics.parquet`
- Run manifest: `data_perp/artifacts/tp6_sl4_exact170_canonical_consensus_20260809_v5/run_manifest.json`
- Reusable runner: `scripts/run_tp6_sl4_exact170_canonical_consensus.py`

## Decision

`CANONICAL_TEN_HEAD_LOGIC_RAN_EXACT170`, but
`STRICT_FEATURE_PARITY_GATE_FAILED`.  The negative net result is therefore a
credible population/cost stress signal and a strong warning about the long
side, but it should not be used to reject the canonical architecture until the
120-field causal input lineage is complete.
