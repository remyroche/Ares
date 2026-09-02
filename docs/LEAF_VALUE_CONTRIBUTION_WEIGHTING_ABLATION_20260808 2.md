# Leaf value/contribution weighting ablation

## Question

Does the residual correctness layer improve if structural leaves are weighted by
their frozen value and/or their emitted base-score contribution, instead of
treating every mapped path contribution equally?

## Frozen contract

- Long side only; 64-family semantic contract: `long_structural_family_semantic_020_top64_strict_20260808_v1`.
- All 64 families were available to the residual learner (`authority_k=0`); this isolates leaf weighting from the separate authority-size ablation.
- Entry is the 15-minute open one hour after the feature timestamp; 48-bar horizon; SL=3 ATR; trailing activation=0.5 ATR; giveback=0.25 ATR; one 100-bps cost.
- Evaluation is pooled global top-k, with monthly diagnostics; it is not a per-timestamp quota.
- The residual target remains the per-row policy-net residual around the Cap-120/equal-month base score, ordinalized at -50/+50 bps for the MLP correctness head.

## Weight definitions

`raw` is the existing control. For every path in the frozen base model, the other
arms multiply the signed emitted contribution by a bounded strength factor
computed from the fold's training leaf catalog only:

- `value`: `clip(abs(tree_leaf_value) / median_train_abs_leaf_value, 0.25, 4)`.
- `contribution`: `clip(abs(emitted_contribution) / median_train_abs_emitted_contribution, 0.25, 4)`.
- `value_x_contribution`: geometric combination `sqrt(value_factor * contribution_factor)`.

The sign is retained. No realised policy outcome, calibration label, or outer
test row is used to fit a weighting scale. Weighted signed contributions,
weighted absolute shares, per-family weight factors, trust/history fields, and
the residual features are persisted.

## Pooled OOS results (net bps/trade)

| weighting | residual arm | top 0.5% | top 1% | top 5% | top 10% |
|---|---|---:|---:|---:|---:|
| raw | H support/OOD abstain | -22.24 | -35.58 | **-55.59** | -65.71 |
| value | H support/OOD abstain | -53.46 | -47.47 | -56.39 | -67.35 |
| contribution | H support/OOD abstain | -38.06 | -43.51 | -58.24 | -68.51 |
| value × contribution | H support/OOD abstain | -48.73 | -50.46 | -61.22 | -67.71 |

For the dynamic MLP arm J, the same top-5 results were -60.31 (raw), -59.84
(value), -61.81 (contribution), and -63.76 (combined). Thus value weighting
produced a small J top-5 movement but did not beat the raw H arm or the raw
control at any decision tail.

## Monthly H top-5 net bps/trade

| month | raw | value | contribution | value × contribution |
|---|---:|---:|---:|---:|
| 2024-05 | -69.51 | -70.56 | -71.77 | -71.75 |
| 2024-06 | -101.00 | -100.04 | -101.23 | -102.49 |
| 2024-07 | -64.43 | -68.31 | -75.07 | -80.46 |
| 2024-08 | -59.32 | -35.57 | -39.05 | -45.83 |
| 2024-09 | -69.81 | -66.66 | -68.07 | -72.64 |
| 2024-10 | -95.04 | -98.49 | -101.68 | -99.51 |
| 2024-11 | **+16.70** | -8.42 | -3.82 | -5.32 |

The raw arm has mean monthly H top-5 net -63.20 bps, median -69.51, worst
-101.00, and one positive month. Value weighting has mean -64.01 and no
positive month; contribution weighting -65.81 and no positive month; the
combined weighting -68.29 and no positive month.

## Coverage and factor diagnostics

| weighting | weighted mass mean | assignment-quality mean | leaf-factor median / mean / p95 |
|---|---:|---:|---:|
| raw | 88.40% | 33.87% | 1.00 / 1.00 / 1.00 |
| value | 89.77% | 35.26% | 1.00 / 1.10 / 2.07 |
| contribution | 90.23% | 35.67% | 1.00 / 1.11 / 2.14 |
| value × contribution | 89.97% | 35.43% | 1.00 / 1.10 / 2.10 |

The apparent mass increase is a consequence of changing the mass definition;
it is not evidence of better economic attribution. The value and contribution
channels are highly related in this base model, so their combination mostly
increases concentration rather than adding an independent signal.

## Decision

Do not promote leaf-value or leaf-contribution weighting. Keep the raw signed
contribution contract and use leaf strength as an audit/trust feature only. The
weighted variants are reproducible, causal, and useful for diagnostics, but
they worsen pooled top-5 economics and generally reduce month portability.

Artifacts:

- [value-weighted replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/long_family_leaf_value_weighted_semantic020_top64_20260808_v2)
- [contribution-weighted replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/long_family_leaf_contribution_weighted_semantic020_top64_20260808_v1)
- [combined replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/long_family_leaf_value_x_contribution_semantic020_top64_20260808_v1)
- [patched replay script](/Users/remyroche/Documents/Ares/scripts/run_long_family_conditional_correctness.py)
