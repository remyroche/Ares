# TP6/SL4 frozen GAM decision audit

## Decision

The no-GAM arm is now the exact model specified by
`TP6_SL4_BASE_CONSENSUS_CANONICAL_20260807.md`.  It is not a separate GAM
control and it does not use any GAM output.

The comparison is frozen to:

- R3 base score: `P(clear) - 0.5 * P(adverse)`;
- train-only monotonic base-to-bps map;
- eight fixed LambdaRank consensus residual heads (25/40/60/73-field caps,
  ordinary/equal-month weighting);
- `0.75 * base_rank + 0.25 * consensus_rank`;
- global pooled top-k ranking after monthly side normalization;
- TP `+6 ATR`, SL `-4 ATR`, H12 horizon, 100 bps cost applied once.

The GAM candidate is the one continuous field `gam_delta_bps`, fit with the
frozen one-month rolling zero-exposure GAM (`gamma = 0.25`).  It is added as an
input to the residual/consensus heads.  There is no production BaseEV
modulation.  If the month-ahead transport is invalid, the GAM score is now
exactly the canonical control score.

## 2025 matched comparison

The corrected run covers 10,224 long rows over January–December 2025.  All
arms use the same rows, labels, exits, and global ranking rule.

| Arm | Top-1 net | Top-2 net | Top-5 net | Top-10 net | Rank IC |
|---|---:|---:|---:|---:|---:|
| Canonical Base+Consensus | −1.93 | +15.36 | **+25.88** | +3.75 | 0.0705 |
| One-field GAM input (hard-gated) | +25.995 | +3.42 | **+29.11** | +3.26 | 0.0706 |
| GAM score modulation (diagnostic only) | +1.49 | −43.83 | −6.97 | −22.45 | 0.0590 |
| GAM input + modulation | +14.70 | −17.65 | +3.14 | −3.09 | 0.0677 |

The one-field input arm’s pooled Top-5 improvement is only +3.24 bps/trade.
Its monthly Top-5 mean is +28.22 versus +27.06 for the control, median +16.00
versus +9.55, and positive-month count 7/12 for both.  This is a small,
unstable improvement, not a promotion signal.  Direct modulation is rejected:
it lowers the monthly mean to −6.39 and the positive-month count to 5/12.

## Untouched later-period validation

The later replay uses the same frozen canonical contract and one-month June fit
to score the untouched 20–23 July 2026 population (7,200 long rows).  The
available strict-OOS downstream panel supplies 24,004 prior training rows.

| Arm | Top-1 net | Top-2 net | Top-5 net | Top-10 net | Rank IC |
|---|---:|---:|---:|---:|---:|
| Canonical control | −27.61 | −24.44 | **−49.24** | −69.70 | 0.0059 |
| Hard-gated one-field GAM input | −26.23 | −24.98 | **−52.46** | −73.32 | 0.0056 |

The later Top-5 delta is −3.21 bps/trade.  The control has a four-period Top-5
mean of −9.45 bps and one positive period; GAM has −15.84 bps and one positive
period.  Daily control/GAM Top-5 net (bps/trade) is:

| Date | Control | GAM input |
|---|---:|---:|
| 2026-07-20 | +191.12 | +170.81 |
| 2026-07-21 | −120.69 | −115.84 |
| 2026-07-22 | −3.98 | −18.99 |
| 2026-07-23 | −104.26 | −99.34 |

This is the decisive chronological result: the small 2025 gain does not
transport to the later period.

## Robustness diagnostics

The 10-seed later replay gives the GAM-minus-control Top-5 delta a mean of
−0.96 bps/trade, median −1.34, range −19.10 to +12.55, and a positive-seed
fraction of 50%.  It therefore does not show a stable positive effect.

The 200 within-period placebo permutations preserve the GAM field’s marginal
distribution and missingness.  Empirical fractions of placebo Top-k net at
least as high as the real GAM result are: Top-1 0.269, Top-2 0.323, Top-5
0.517, Top-10 0.552.  These are not evidence of a reliable causal uplift.

An abstention-on-invalid diagnostic was also materialized.  In 2025, 5,964 of
10,224 rows (58.3%) have valid transport.  Removing invalid rows and refilling
the Top-5 quota gives +39.14 bps/trade for the control and +44.73 for GAM, but
this is a reduced-exposure diagnostic and is not comparable to the full-
population global Top-5 result.  It is not promoted.

## Feature-contract caveat

The later canonical context has the required 73-field contract, with no target
imputation or aliases.  Three fields have zero later coverage because their
benchmark source is unavailable (`btc_ex_eth_oi_dominance_z_ratio`,
`btc_oi_dominance_z_ratio`, `ret48h_bench_resid`).  Three additional fields
have partial coverage (`fund_abs_z_mkt_resid` at 80.4%; the two negative-
funding interaction fields at 96.9%).  These values remain missing and are
not silently reconstructed.  The coverage and correctness audits are saved
alongside the later replay.

## Final status

**Keep the canonical TP6/SL4 Base+Consensus model as the baseline and do not
promote GAM.**  The GAM input is a valid, causal, reproducible diagnostic, but
its development uplift is small and fails the untouched later-period and
seed-robustness gates.  GAM modulation is actively harmful.  Any future GAM
work should first repair or replace the unavailable benchmark context and then
be re-evaluated on a materially longer untouched chronological window.

## Artifacts

- [Canonical baseline comparison](../data_perp/artifacts/tp6_sl4_gam_canonical_base_consensus_20260808_v1/TP6_SL4_GAM_CANONICAL_BASELINE_REPORT.md)
- [Later untouched replay](../data_perp/artifacts/tp6_sl4_gam_canonical_later_oos_20260808_v1/TP6_SL4_GAM_CANONICAL_LATER_OOS_REPORT.md)
- [Seed robustness](../data_perp/artifacts/tp6_sl4_gam_canonical_later_seed_robustness_20260808_v1/report.md)
- [Placebo distribution](../data_perp/artifacts/tp6_sl4_gam_canonical_later_placebo_20260808_v1/report.md)
- [Abstention diagnostic](../data_perp/artifacts/tp6_sl4_gam_canonical_abstention_20260808_v1/TP6_SL4_GAM_CANONICAL_ABSTENTION_REPORT.md)
