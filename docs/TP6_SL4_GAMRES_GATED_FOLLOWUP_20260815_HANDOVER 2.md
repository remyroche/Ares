# GAM residual decomposition, hard transport gate, and placebo follow-up

## Final mechanism

The production-style candidate is now:

```text
if transport_valid(target_month):
    use the GAM-residual-enhanced residual/meta score
else:
    use the exact control residual/meta score
```

This is a model-selection gate. It does not modify BaseEV and does not use
GAM anchor modulation.

The replay uses `feature_fraction=1.0`, native base-score ranking, the existing
4-hour UTC × side LambdaRank consensus/residual heads, and 12 long-side held
months in 2025.

## Field decomposition

| Arm | Enhanced Top-1 | Enhanced Top-5 | Hard-gated Top-1 | Hard-gated Top-5 |
|---|---:|---:|---:|---:|
| Control | +8.65 | -9.65 | +8.65 | -9.65 |
| `gam_delta_bps` only | -3.64 | -6.95 | -5.41 | -1.63 |
| `gam_residual_bps` only | +29.61 | -11.19 | +16.03 | -2.64 |
| delta + residual | +20.93 | **+13.32** | +17.87 | **+16.55** |
| delta + residual + validity | **+38.02** | +1.03 | **+56.36** | +6.59 |
| Saved current stack | +14.04 | -24.91 | — | — |

The explicit gate improves the broader tail for the delta+residual specialist:
Top-5 becomes +16.55 bps/trade, versus +13.32 ungated and −9.65 control.

## Redundancy audit

The two continuous fields are not independent. Across every nonconstant
month:

```text
gam_residual_bps = 4 × gam_delta_bps
```

Both Pearson and Spearman correlations are exactly 1.0, with zero numerical
deviation from the 4× identity. The apparent difference between single-field
and two-field LightGBM fits is therefore a model-regularization/representation
effect, not additional information.

The deployable semantic contract should use one canonical field, preferably
`gam_delta_bps`, with a separate transport-validity gate. The two-field result
is retained as a diagnostic because it currently scores better, but it should
not be interpreted as two independent signals.

## Hard-gated global metrics

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% |
|---|---:|---:|---:|---:|---:|
| Control | -20.09 | +8.65 | -10.06 | -9.65 | -26.00 |
| Ungated delta + residual | +48.51 | +20.93 | -8.79 | +13.32 | -24.79 |
| **Hard-gated delta + residual** | +5.93 | **+17.87** | **+5.58** | **+16.55** | -24.41 |
| Current saved stack | +37.59 | +14.04 | -8.00 | -24.91 | -15.81 |

The hard-gated specialist beats the saved current stack by approximately
41.5 bps/trade at Top-5, but remains negative at Top-10.

## Stability and leave-one-month-out tests

Full monthly Top-5 stability for the hard-gated arm:

- mean: **+16.19 bps/trade**;
- median: **+10.30 bps/trade**;
- MAD: 37.75 bps;
- worst month: −156.92 bps;
- positive months: 6/12.

Leave-one-month-out Top-5 results:

| Arm | Mean across exclusions | Median | Worst exclusion | Positive exclusions |
|---|---:|---:|---:|---:|
| Control | -9.89 | -9.35 | -19.68 | 1/12 |
| Hard-gated GAM residual | **+16.54** | **+16.54** | **-0.29** | **11/12** |
| Saved current stack | -25.16 | -25.22 | -35.70 | 0/12 |

Paired month bootstrap, gated minus control:

| Metric | Mean difference | 95% bootstrap interval | Positive months |
|---|---:|---:|---:|
| Top-1 | +6.85 bps | [-46.49, +56.38] | 3/12 |
| Top-5 | **+27.68 bps** | **[+11.26, +46.82]** | **7/12** |

The economically credible Top-5 improvement survives the leave-one-month-out
test and has a positive paired-month bootstrap interval. Top-1 is much noisier.

## Valid versus invalid months

The target-month gate is valid in 7 months and invalid in 5 months.

| Scope | Control Top-1 | Gated Top-1 | Control Top-5 | Gated Top-5 |
|---|---:|---:|---:|---:|
| Valid months | -13.20 | +5.93 | +4.23 | **+51.33** |
| Invalid months | +31.99 | +31.99 | -32.27 | -32.27 |

Invalid months revert exactly to control. This removes the prior failure mode in
which neutral held-month GAM fields still changed the ranker's learned
partitioning and hurt the invalid subset.

## Top-1 transitions

For the hard-gated score versus control:

- entered Top-1%: 12 rows, mean +137.05 bps, median +36.49 bps;
- exited Top-1%: 12 rows, mean +57.96 bps, median −33.24 bps.

The means are dominated by a small number of observations, so the Top-1
transition result is supportive but not a selection gate. The Top-5 result is
the stronger evidence.

## 50-seed placebo distribution

The placebo stage permutes `gam_delta_bps` and `gam_residual_bps` independently
within each training and held month, preserving marginal distributions and
missingness while destroying row-level information. It uses the same residual
LambdaRank head and `feature_fraction=1.0`; therefore its null is specifically
for the residual-head mechanism, not the complete eight-head stack.

| Metric | Real residual-head value | Placebo median | Placebo 5--95% | Empirical p(real or higher) |
|---|---:|---:|---:|---:|
| Top-1 net | +82.89 | +16.89 | [-28.56, +62.86] | **0.020** |
| Top-5 net | +17.94 | +6.37 | [-11.55, +17.47] | 0.078 |
| Mean monthly Top-5 | +2.78 | +7.83 | [-11.26, +21.88] | 0.588 |
| Q25 monthly Top-5 | -47.74 | -54.35 | [-78.47, -30.83] | 0.353 |

The Top-1 residual-head result is unlikely under this 50-seed null. Top-5 is
directionally better but does not clear a conventional 5% empirical threshold
in this residual-head-only placebo. This is why the hard-gated full-stack
Top-5 result should remain a promising research arm rather than a frozen
production contract until the placebo is extended or the sample is enlarged.

## Recommendation

1. Keep direct GAM anchor modulation disabled.
2. Use the explicit transport gate rather than asking LightGBM to infer it.
3. Continue with the hard-gated delta+residual arm for long-side research,
   using Top-5 and leave-one-month-out stability as the primary gates.
4. Collapse the two perfectly proportional fields into one canonical stored
   field in the eventual production contract, then repeat the hard-gated test
   with repeated placebo seeds for that simplified field.

## Artifacts

- [Follow-up runner](</Users/remyroche/Documents/Ares/scripts/run_tp6_sl4_gamres_gated_followup.py>)
- [Hard-gated field metrics](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_gamres_gated_followup_20260815_v1/metrics_hard_gated_variants.parquet>)
- [Decomposition/global metrics](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_gamres_gated_followup_20260815_v1/metrics_global.parquet>)
- [Leave-one-month-out metrics](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_gamres_gated_followup_20260815_v1/metrics_leave_one_month_out.parquet>)
- [Bootstrap confidence intervals](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_gamres_gated_followup_20260815_v1/metrics_bootstrap_ci.parquet>)
- [50-seed placebo report](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_gamres_placebo_distribution_20260815_v1/TP6_SL4_GAMRES_PLACEBO_REPORT.md>)
- [Placebo empirical p-values](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_gamres_placebo_distribution_20260815_v1/placebo_empirical_pvalues.parquet>)
