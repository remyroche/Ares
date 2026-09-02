# Short P0 exact-path target funnel — decision

## Decision

No full-population replacement target advances beyond the frozen short P0/M4
control.  The new exact-path labels resolve the diagnosis, but the T1–T4
targets do not produce a portable causal admission rule under the unchanged
41-field M4 feature contract and the exact M4 train-OOF-p80 admission
convention.

`T5_conditional_low_regret` is the sole positive finding.  It is **not** a
deployable target: both its training and its reported evaluation condition on
the realised fact that `MFE_H12 > 200 bps`.  It instead establishes that,
conditional on a meaningful opportunity existing, the existing causal state
contains useful information about whether the policy will harvest it.

The next justified architecture is therefore a strict-OOF two-stage research
experiment:

```text
causal opportunity estimate
       ×
causal conditional-conversion / low-regret estimate
       ->
train-only common-bps policy-net mapping
```

It must be evaluated as one full target-free candidate pipeline.  It may not
use realised MFE as a live route or admission condition.  Short live trading
does not advance from this result.

## Final artifact

The authoritative receipt is:

`data_perp/artifacts/strict_r3_short_p0_path_target_funnel_2024may_2026jul_20260821_v4`

It copies the v3 fitted scores byte-for-byte and fixes only two metric-scope
issues:

1. the new arms now use exactly the historical M4 operation—take the p80 of
   raw chronological-OOF scores, then map that cut through isotonic calibration;
2. T5 evaluation now uses its declared `MFE_H12 > 200 bps` condition.

The v4 manifest hashes the v3 source prediction and fold-audit files.  It
does not refit, tune, rescore, modify feature values, change the policy, or
change candidate identities.

Earlier artifacts are research-only and superseded for metric interpretation:

- v1 never sealed a manifest after an early interrupted invocation;
- v2 used `p80(calibrated score)` rather than the M4-compatible
  `calibrate(p80(raw score))` operation;
- v3 contains the correct fitted scores but pre-repair metric aggregation.

## Scope and invariants

- Population: frozen short P0 rank-1 target-free hourly candidates.
- Entry: signal close + one hour, exact frozen decision-minute open.
- Outcome: complete post-decision H12 exact one-minute path.
- Policy: short SL `3 ATR`, trailing activation `0.5 ATR`, giveback `0.25
  ATR`, H12 timeout, 100-bps cost exactly once.
- Feature contract: the frozen 41 M4 base fields only.
- Rich path quantities are supervised labels only and are forbidden from
  inference fields.
- A fit row requires `label_available_at < held_month_start`; label availability
  is exactly decision time + 12 hours.
- Invalid/incomplete paths are scored for coverage only.  They never fit a
  target, never become a zero label, and do not contribute to outcome metrics.
- Score-to-policy-net calibration uses expanding chronological OOF predictions
  from the current training fold only.

The artifact contains 19,696 P0 candidates over May 2024–July 2026, of which
14,092 have valid exact-path labels.  T1–T4 have 27 complete monthly folds;
T5 starts in June 2024 because its conditional training support is initially
below the 500-row minimum.

## Causal train-p80 results

All figures below use the corrected candidate-level p80 threshold.  2025 H1
is shown because it is the time range shared with the stored M4 control;
T1–T4 also have later 2025 support.

| Arm | 2024 May–Dec EV/trade | 2025 H1 EV/trade | 2026 Jan–Jul EV/trade | Result |
| --- | ---: | ---: | ---: | --- |
| T0 frozen M4 | -15.04 bps | +92.48 bps | +92.51 bps | Control only; cross-era failure remains. |
| T1 cost-clear MFE magnitude | -46.25 | +54.09 | -53.25 | Reject. |
| T2 fast cost-clear | -14.70 | +42.04 | -13.99 | Reject; slightly less negative in 2024 but unstable. |
| T3 MFE3h − 0.25×MAE | -22.72 | +41.02 | -37.05 | Reject. |
| T3 MFE3h − 0.50×MAE | -45.26 | +41.02 | -37.05 | Reject. |
| T3 MFE3h − 1.00×MAE | -40.31 | n/a | -37.05 | Reject; sparse/flat mapped support. |
| T4 min(opportunity, convertibility) | -24.63 | +18.66 | -26.58 | Reject. |

None of T1–T4 improves both the 2024 portability failure and the positive
2025–26 M4 behaviour.  Stronger target-label AUC in parts of 2026 does not
override this economics result: score-to-policy-net mapping remains unstable.

## Conditional T5 diagnostic

T5 is evaluated only among paths with realised `MFE_H12 > 200 bps`.

| Period | Trades | Policy net EV/trade | Positive months | Worst month | Mean score→policy-net Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024 May–Dec | 925 | +137.24 bps | 7 / 7 | +48.52 bps | +0.131 |
| 2025 H1 | 772 | +384.55 | 6 / 6 | +306.16 | +0.366 |
| 2025 H2 | 409 | +420.77 | 6 / 6 | +220.74 | +0.457 |
| 2026 Jan–Jul | 351 | +364.27 | 7 / 7 | +214.53 | +0.356 |

This is useful evidence of conversion learnability, but not evidence of a
live policy.  A live system cannot know at decision time whether the path will
later cross the 200-bps MFE condition.

## Why this matters

The preceding exact-path cross-era audit showed broadly comparable eventual
short MFE but much weaker 2024 capture and larger early adverse movement.
The target funnel adds an important refinement:

- direct full-population opportunity and combined targets are not portable
  with the frozen 41 inputs;
- conditional conversion/reliability is recoverable once an opportunity is
  present.

This supports factorising the problem rather than replacing M4 with another
single collapsed target.  A richer causal feature search is justified only
for that predeclared two-stage architecture, with target-specific MDA and a
later untouched OOS block.

## Relevant implementation

- `scripts/materialize_strict_r3_short_p0_rich_path_labels.py`
- `scripts/run_strict_r3_short_p0_opportunity_conversion_diagnostic.py`
- `scripts/run_strict_r3_short_p0_path_target_funnel.py`
- `extreme_price_movements/tests/test_short_p0_rich_path_labels.py`
- `extreme_price_movements/tests/test_short_p0_opportunity_conversion_diagnostic.py`
- `extreme_price_movements/tests/test_short_p0_path_target_funnel.py`

