# Causal S/R outputs as MC1 residual-map inputs — challenger result

## Decision

Adding causal, strictly OOF support/resistance outputs to the paired MC1 residual mapper is a material **offline challenger** improvement over its matched non-S/R control.  It is not promoted to the canonical or live stack: it has not yet been composed with the canonical E2 entry head and H4 continuation/rich-exit stack on the same source-valid candidate population.

## Exact comparison

Both arms preserve the same target-free candidate identities, source-aligned rich-policy labels, monthly OOS schedule, dual BCF/current MC1 maps, 30-bps dual admission, top-two-per-timestamp routing and global portfolio auction.  The only change is an appended causal S/R feature block in the already-existing paired residual mapper.

The mapper target is the existing residual target:

`policy_net_bps - mean(bcf_mc1_expected_bps, current_mc1_expected_bps)`.

Each monthly fit uses only labels resolved before the held month.  The selected authority is full residual authority (`w100`); `w050` was retained as a stress control.

The S/R additions are the twelve output fields below, joined strictly by candidate ID and decision timestamp.  A missing S/R snapshot is represented as model-native missing plus `sr_snapshot_available`; it is never an eligibility gate.

- accepted-break probabilities, conditional/prior strengths, reaction magnitudes and zone distances for support and resistance;
- long support-hold, resistance-break, downside-break and resistance-rejection outputs;
- long structure balance, support/resistance distances; and
- `sr_snapshot_available`.

The source is `causal_sr_heads_oof_20260830_v3_entrypivotfix`; its held-month heads use only interactions resolved before the decision month.  Oracle/non-causal sources are explicitly rejected.

## Portfolio-constrained residual-mapper results (not canonical E2 + H4)

| Scope | Matched control w100 | + causal S/R w100 | Delta (S/R - control) |
|---|---:|---:|---:|
| Jun–Jul selection: accepted trades | 1,073 | 1,002 | -71 |
| Jun–Jul selection: net EV/trade | +17.36 bps | +36.21 bps | +18.85 bps |
| Jun–Jul selection: total net bps | +18,628.91 | +36,287.27 | +17,658.35 |
| Jun–Jul selection: max drawdown | -75.84% | -42.72% | +33.12 pp |
| Jun–Jul selection: Sortino | 0.0532 | 0.1441 | +0.0909 |
| August residual-mapper holdout: accepted trades | 588 | 575 | -13 |
| August residual-mapper holdout: net EV/trade | -5.50 bps | +43.42 bps | +48.92 bps |
| August residual-mapper holdout: total net bps | -3,231.35 | +24,968.98 | +28,200.33 |
| August residual-mapper holdout: max drawdown | -60.04% | -32.93% | +27.10 pp |
| August residual-mapper holdout: worst week | -42.36 bps | -6.98 bps | +35.38 bps |
| August residual-mapper holdout: Sortino | -0.0304 | 0.1647 | +0.1952 |
| Jun–Aug all OOS: accepted trades | 1,661 | 1,577 | -84 |
| Jun–Aug all OOS: net EV/trade | +9.27 bps | +38.84 bps | +29.57 bps |
| Jun–Aug all OOS: total net bps | +15,397.57 | +61,256.25 | +45,858.68 |
| Jun–Aug all OOS: max drawdown | -89.79% | -45.03% | +44.76 pp |
| Jun–Aug all OOS: Sortino | 0.0229 | 0.1518 | +0.1289 |

Monthly `w100` net EV/trade:

| Month | Control | + causal S/R |
|---|---:|---:|
| June 2026 | +66.76 bps | +80.03 bps |
| July 2026 | -11.42 bps | +10.57 bps |
| August 2026 (untouched) | -5.50 bps | +43.42 bps |

The canonical composed `E2_q50_agreement + H4 + Giveback-20` stack is a
different entry/exit composition.  Its matched August result is positive:
539 accepted trades, +54.36 bps/trade and +29,300.6 total net bps.  It is not
the control in this S/R residual-mapper experiment.

## Causality and data limitations

- The join preserves all target-free candidate identities and contains no outcome fields.
- Post-run audit passed: no selected candidate duplicates, no more than two selected candidates per timestamp, no labels in selection files, and no oracle source.
- Eight unreadable immutable rich-policy label parts were excluded at whole-symbol granularity from **every** compared arm.  They were not imputed or replaced: `APE`, `GAS`, `MTL`, `ORDI`, `SHIB`, `SPYX`, `TURBO`, and `W`.

## Artifacts and reproduction

- Runner: `scripts/run_causal_sr_mc1_residual_ablation.py`
- Contract tests: `tests/test_causal_sr_mc1_input_contract.py` (3 passed)
- Results: `data_perp/artifacts/causal_sr_mc1_residual_input_ablation_20260830_v1`
- Core result files: `run_manifest.json`, `portfolio_summary.parquet`, `fold_trace.parquet`, `sr_merge_coverage.parquet`, and per-arm decisions/accepted/equity files.

## Required next gate before promotion

Run one predeclared composed challenger with the frozen E2 entry model and H4 continuation/rich exit on this identical source-valid population.  Compare it against the canonical E2+H4 control under the same portfolio state, costs and policy labels.  Do not change the live bundle until that composition passes its held-out OOS test.
