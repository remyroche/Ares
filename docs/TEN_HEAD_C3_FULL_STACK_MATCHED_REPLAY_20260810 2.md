# Frozen Ten-Head → C3 Full-Stack Matched Replay

This is a causal composition replay, not a new HPO. It freezes the conditional ten-head winner and compares it with the pre-existing consensus on the same full candidate population, fixed SL3 / activation 0.5 / giveback 0.25 / H12 / 100-bps-once policy labels.

## Scope and causal boundary

- Evaluation months: 2025-08, 2025-09, 2025-10.
- The frozen heads were re-scored on every candidate before outcome coverage is inspected; their label-valid August–October scores reproduce the original frozen artifact.
- Each monthly C3 downstream fit uses a preceding 3-month raw market-geometry burn-in and a nominal six-month resolved-score window (the compatible score ledger begins in March, so the August fit has five populated months), one matching geometry bundle for train/reference/held rows, exact-H12 Severe-200, +100-bps policy-residual correctness, and a same-model prior-42-day CDF.
- Causal 21-day expected-net admission and the 8-concurrent / 2-new-per-15m / 1-per-asset auction are evaluated afterward.
- This retains the fixed-policy label contract of the ten-head experiment. It is therefore not a reproduction of Part A's separately selected SL4.152 / activation2.326 / giveback0.102 policy.

## Frozen-head reproduction

- Matched label-valid rows: 72,539
- Maximum absolute consensus-rank difference: 2.98e-08
- Maximum absolute upstream-score difference: 7.45e-09

## Final C3 score: pooled-global net bps/trade

| Tail | Control net (valid / coverage) | Frozen ten-head net (valid / coverage) | Delta |
|---:|---:|---:|---:|
| Top 0.5% | -47.87 (202 / 12.2%) | +156.24 (78 / 4.7%) | +204.10 |
| Top 1% | -32.09 (412 / 12.4%) | +127.91 (146 / 4.4%) | +160.00 |
| Top 2% | -24.36 (851 / 12.8%) | +70.48 (365 / 5.5%) | +94.84 |
| Top 5% | -31.04 (2,342 / 14.1%) | +29.99 (1,284 / 7.7%) | +61.03 |
| Top 10% | -55.69 (5,608 / 16.9%) | -11.90 (3,725 / 11.2%) | +43.79 |

## Causal admission and portfolio

| Arm | Accepted trades | Trades/day | Net bps/trade | Gross bps/trade |
|---|---:|---:|---:|---:|
| control | 370 | 4.02 | +3.94 | +15.84 |
| frozen_ten_head | 1,010 | 10.98 | +8.85 | +16.97 |

## Coverage and model-fit checks

- Final-window source outcome coverage: 72,539 valid fixed-policy paths out of 331,986 scored candidates (21.9%).
- This inherited low outcome coverage is not an admission feature or score filter; it makes the global-tail diagnostics sparse and prevents promotion from this replay alone.
- Candidate scoring rows across the March–October history: 847,821.
- C3 downstream fits completed: 12 (two arms × six May–October monthly fits; May–July provide causal admission history).
- Every C3 audit records train/reference/held separation, geometry identity, resolved-only supervised support, and no held outcome consumption during scoring.

Detailed numbers are in the associated parquet artifacts; this document intentionally does not promote either arm, because the Aug–October period was already opened by the upstream ten-head final comparison.
