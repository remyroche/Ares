# Long-only cost-aware R3 target hurdle ablation (2026-08-07)

## Contract

This is a strict chronological OOF screen on the current mixed exact-minute /
15-minute-proxy training surface, restricted to `side_name=long`.

- 1,338,630 valid long rows across 31 months.
- Entry, H12 path, cost, and label maturity remain unchanged.
- The base contract is the frozen 32-field long F0 list.
- Score is `P(clear) - 0.5 P(adverse)`.
- Twelve chronological folds; training labels satisfy
  `label_available_ts < held_out_fold_start`.
- All target definitions use pre-adverse MFE and ATR only as training labels;
  no outcome/path field is an inference feature.

## Base-target screen

| Target | Rank IC | Top-5 net uplift | Top-30 net uplift | Top-40 net uplift | Top-40 recall | Positive top-5 months | Monthly decile violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| B25 | 0.4574 | **+10.49** | +5.14 | +4.09 | 48.25% | 15/31 | 33 |
| B50 | 0.4566 | +10.24 | +5.12 | +3.79 | 49.30% | 15/31 | 32 |
| B75 | 0.4552 | +9.23 | +4.87 | +3.76 | 50.24% | 17/31 | 36 |
| B100 | 0.4560 | +8.99 | +4.69 | +3.70 | 51.20% | 15/31 | 30 |
| B125 | 0.4551 | +9.56 | +4.66 | +3.84 | 52.25% | 17/31 | 41 |
| B150 | 0.4567 | +8.47 | +4.58 | +3.63 | **53.30%** | 15/31 | 32 |
| `max(150 bps, 1 ATR)` | 0.4417 | **+11.15** | +4.88 | +3.75 | 48.19% | **18/31** | 44 |
| `150 bps + 0.5 ATR` | 0.4367 | +9.98 | **+5.35** | **+4.12** | 48.78% | 17/31 | 41 |

All arms have positive rank IC in all 31 months. The fixed-bps ladder shows a
clear trade-off: larger hurdles increase top-40 recall but reduce top-5 net
uplift. The `max(150 bps, 1 ATR)` result has the highest pooled top-5 uplift,
but its lower IC and 44 monthly decile violations fail the current monotone
ranking-quality preference. The additive ATR arm has the best top-30/40 net
uplift but the weakest IC and also violates monotonicity frequently.

**Decision:** retain B25 as the canonical long base target. Keep
`max(150 bps, 1 ATR)` as the cost-aware challenger for a matched meta-layer
test; do not promote it from base economics alone.

## Causal 21-day admission replay

The strongest challenger scores were replayed through the frozen long-only
21-calendar-day pooled-parent/side-shrunk map. The map is an admission gate;
the raw OOF score remains the ranking variable after admission.

| Target | Admitted rows | Admission rate | Admitted raw-score top-5 net | Months | Worst month |
|---|---:|---:|---:|---:|---:|
| B25 | 11,125 | 0.831% | +209.6 bps | 6 | −363.7 |
| B75 | 11,209 | 0.837% | +117.7 bps | 6 | −183.8 |
| `max(150 bps, 1 ATR)` | 9,674 | 0.723% | **+47.2 bps** | 5 | **−75.4** |
| `150 bps + 0.5 ATR` | 10,048 | 0.751% | +5.1 bps | 6 | −12.5 |

The ATR-floor challenger materially reduces worst-month damage and gives a
more defensible long-only admitted tail, but it admits fewer rows and has a
lower rank-quality score. Its apparent stability is still based on only five
months of admitted top-5 support, so it is a challenger, not a frozen winner.

## Artifacts

Base OOF arms:

- `data_perp/artifacts/current_r3_long_target_b75_20260807_v1/`
- `data_perp/artifacts/current_r3_long_target_b100_20260807_v1/`
- `data_perp/artifacts/current_r3_long_target_b125_20260807_v1/`
- `data_perp/artifacts/current_r3_long_target_b150_20260807_v1/`
- `data_perp/artifacts/current_r3_long_target_max150_or_1atr_20260807_v1/`
- `data_perp/artifacts/current_r3_long_target_bps150_plus_half_atr_20260807_v1/`

Admission replays:

- `data_perp/artifacts/current_r3_21d_admission_20260807_v5/` (B25 control)
- `data_perp/artifacts/current_r3_long_target_b75_admission_20260807_v1/`
- `data_perp/artifacts/current_r3_long_target_max150_or_1atr_admission_20260807_v1/`
- `data_perp/artifacts/current_r3_long_target_bps150_plus_half_atr_admission_20260807_v1/`

## Next controlled test

Run a matched long-only base-plus-residual OOF comparison for B25 versus
`max(150 bps, 1 ATR)`, preserving the same feature contract, query grouping,
causal map, and admission gate. Select by: (1) no rank-quality or monotonicity
failure, (2) worst-month admitted net, (3) pooled top-5 net, and (4) coverage.
