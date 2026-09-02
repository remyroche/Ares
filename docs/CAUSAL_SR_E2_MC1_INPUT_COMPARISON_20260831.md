# MC1 input comparison: causal S/R, all-candidate 15m E2, and both

## Decision

Causal S/R outputs remain the only material input addition to the matched
paired-MC1 residual mapper. The newly retrained all-candidate 15-minute E2
input is slightly better than the narrow MC1 control on aggregate, but it is
negative in the August holdout and dilutes the S/R arm when both are supplied.
Do not add the 15m E2 input to the S/R-MC1 challenger, canonical stack, or live
bundle.

This is not a replacement for the canonical pairwise `E2_q50_agreement + H4 +
Giveback-20` entry/exit stack. The E2 input tested here is an independent
candidate-level 15-minute prediction with no direct entry authority.

## Contract

All arms preserve target-free candidate IDs, source-aligned rich-policy labels,
four preceding complete calendar months, the paired residual target, residual
L1 model geometry, clipping, dual BCF/current MC1 >=30-bps admission,
BCF-priority top-two routing, costs, and global portfolio auction.

The tested all-candidate E2 input uses all 70 fixed 15-minute features and a
LightGBM L1 model: depth 4, 15 leaves, 350 trees, learning rate .03, lambda 4,
seed 1729. Its target is rich-policy net bps (with the 100-bps policy cost
embedded once). Each nonmissing training-row E2 value is a raw prediction from
an inner model fitted strictly before that row's month; earlier training rows
are missing with an availability flag. Held scores use the outer prior-resolved
fit. No same-row label calibration is an MC1 feature.

## Portfolio-constrained walk-forward

June--July is the declared comparison period; August is the holdout.

| Scope / arm | Trades | Net EV/trade | Total net EV | Max DD | Worst week | Sortino |
|---|---:|---:|---:|---:|---:|---:|
| Jun--Jul: control | 1,073 | +17.36 bps | +18,628.9 bps | -75.84% | -39.42 bps | 0.0532 |
| Jun--Jul: + S/R | 1,002 | +36.21 bps | +36,287.3 bps | -42.72% | -30.02 bps | 0.1441 |
| Jun--Jul: + E2-15m | 1,070 | +18.19 bps | +19,458.6 bps | -72.67% | -36.02 bps | 0.0572 |
| Jun--Jul: + S/R + E2-15m | 1,007 | +34.79 bps | +35,031.5 bps | -41.34% | -31.05 bps | 0.1363 |
| August: control | 588 | -5.50 bps | -3,231.3 bps | -60.04% | -42.36 bps | -0.0304 |
| August: + S/R | 575 | +43.42 bps | +24,969.0 bps | -32.93% | -6.98 bps | 0.1647 |
| August: + E2-15m | 586 | -4.36 bps | -2,553.8 bps | -60.86% | -40.94 bps | -0.0248 |
| August: + S/R + E2-15m | 579 | +38.36 bps | +22,210.1 bps | -35.99% | -11.95 bps | 0.1495 |
| Jun--Aug: control | 1,661 | +9.27 bps | +15,397.6 bps | -89.79% | -42.36 bps | 0.0229 |
| Jun--Aug: + S/R | 1,577 | +38.84 bps | +61,256.2 bps | -45.03% | -30.02 bps | 0.1518 |
| Jun--Aug: + E2-15m | 1,656 | +10.21 bps | +16,904.8 bps | -88.68% | -40.94 bps | 0.0273 |
| Jun--Aug: + S/R + E2-15m | 1,586 | +36.09 bps | +57,241.6 bps | -41.48% | -31.05 bps | 0.1411 |

## Interpretation

- S/R is the clear winner: +29.57 bps/trade and +45,858.7 total bps versus
  control over Jun--Aug; it is positive in every held month.
- E2-15m alone adds only +0.94 bps/trade and +1,507.2 total bps to control,
  while remaining negative in August. This is not sufficient portability.
- There is no useful S/R/E2 synergy. Adding E2-15m to S/R loses 2.75
  bps/trade and 4,014.7 total bps over Jun--Aug. It loses 5.06 bps/trade and
  2,758.9 total bps in August versus S/R alone.

## Causality and audit

- The S/R source is `causal_sr_heads_oof_20260830_v3_entrypivotfix`; oracle or
  noncausal paths are rejected.
- E2 scores are finite for every MC1-assessed held candidate: 678 in June,
  1,280 in July, and 1,726 in August. The latest prequential training score
  precedes each held month: May 31, June 30, and July 31 respectively.
- All selections are target-free, have no duplicate IDs, and retain the
  two-per-timestamp cap.
- Eight unreadable immutable label parts are excluded equally from all arms:
  APE, GAS, MTL, ORDI, SHIB, SPYX, TURBO, and W. They are not imputed.
- A repeated identical run reproduced the portfolio summary exactly (0.0
  maximum numeric delta).

## Artifacts

- Runner: `scripts/run_causal_sr_e2_mc1_input_ablation.py`
- Tests: `tests/test_causal_sr_e2_mc1_input_contract.py` and
  `tests/test_causal_sr_mc1_input_contract.py` (6 passed)
- Result: `data_perp/artifacts/causal_sr_e2_mc1_input_ablation_20260831_v2`

The surviving `M + S/R` arm remains an offline challenger. It needs a
predeclared composition against the canonical E2 + H4 stack on one identical
source-valid population, then later untouched evidence, before promotion.
