# One-month GAM attribution and structural-trust ablation

## Protocol

This diagnostic reruns the long-side 2025 residual/meta stack with
`feature_fraction = 1.0` for every arm. This removes the feature-universe
change as a LightGBM feature-subsampling confound. Direct GAM anchor
modulation is excluded, because the previous matched replay showed it was
harmful through the normal operating tails.

All arms use:

- the same native TP6/SL4 residual and consensus targets;
- the same 4-hour UTC × side LambdaRank query grouping;
- the same monthly expanding training/evaluation windows;
- the same native base-score ranking and pooled global ranking;
- 12 held months, 852 long candidates/month.

The only change is the set of GAM/transport fields exposed to both heads.

## Arms

| Arm | Fields exposed |
|---|---|
| Control | none |
| GAM residual only | `gam_delta_bps`, `gam_residual_bps` |
| Transport only | validity, matched/unmatched mass, archetype count, cluster count |
| GAM + valid | delta, residual, validity |
| GAM + transport | delta, residual plus all five transport diagnostics |
| Full current | all eight current GAM fields, including `gam_expected_bps` |
| Placebo | seven GAM + transport fields permuted independently within each training and held month |

The placebo preserves each field's monthly marginal distribution and
missingness but destroys row-level information.

## Dense global tail curve

Net bps/trade:

| Tail | Control | GAM residual | Transport | GAM + valid | GAM + transport | Full current | Placebo |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25% | -14.04 | +16.25 | +40.36 | +17.67 | -29.78 | -8.44 | **+46.00** |
| 0.50% | -20.09 | +48.51 | +14.07 | **+71.78** | -12.60 | -7.85 | +54.23 |
| 0.75% | +9.43 | +36.42 | +22.31 | +26.48 | -2.25 | **+29.64** | +24.73 |
| 1.00% | +8.65 | **+20.93** | +21.45 | **+38.02** | -8.16 | +14.54 | +3.24 |
| 1.25% | -7.41 | +1.68 | +17.12 | -4.47 | -6.79 | +3.40 | -2.42 |
| 1.50% | -1.61 | +1.20 | +10.56 | -15.27 | +0.21 | -9.62 | -15.05 |
| 2.00% | -10.06 | -8.79 | +0.11 | -10.83 | -14.26 | +5.90 | -7.63 |
| 3.00% | +0.33 | +5.97 | -12.99 | +8.23 | -5.72 | -2.45 | +6.32 |
| 4.00% | +12.38 | **+12.55** | -7.24 | -1.39 | +11.60 | -14.11 | -11.09 |
| 5.00% | -9.65 | **+13.32** | -6.17 | +1.03 | -12.51 | -10.43 | -23.24 |
| 7.50% | -21.96 | -19.35 | -22.33 | -21.37 | -36.12 | -23.46 | -30.78 |
| 10.00% | -26.00 | -24.79 | -32.46 | -26.20 | -39.56 | -31.91 | -24.25 |

The two-field GAM-residual arm is the strongest balanced arm: it is positive
at 0.25--1.5%, 3--5%, and has the best top-5 stability. The curve is still
not monotonic, so it is a research result rather than an execution-ready
policy.

## Stability

| Arm | Mean top-5 | Median | MAD | Worst month | Positive months |
|---|---:|---:|---:|---:|---:|
| Control | -11.49 | -13.31 | 45.69 | -156.92 | 5/12 |
| GAM residual only | **+12.69** | **+9.46** | 40.11 | **-135.79** | 6/12 |
| Transport only | -8.17 | -16.34 | **25.63** | -156.67 | 4/12 |
| GAM + valid | -0.48 | -6.21 | 54.72 | -199.76 | 6/12 |
| GAM + transport | -14.83 | -5.30 | 24.58 | -184.58 | 5/12 |
| Full current | -8.90 | +0.01 | 42.05 | -199.46 | 6/12 |
| Placebo | -27.69 | -29.49 | 39.59 | -186.56 | 3/12 |

## Valid versus invalid transport months

There are 7 valid and 5 invalid held months. At the 1% / 5% tails:

| Arm | Valid top 1% | Valid top 5% | Invalid top 1% | Invalid top 5% |
|---|---:|---:|---:|---:|
| Control | -13.20 | +4.23 | +31.99 | -32.27 |
| GAM residual only | +5.93 | **+51.33** | **+53.29** | -40.05 |
| Transport only | +8.55 | +25.38 | +39.45 | -48.94 |
| GAM + valid | **+55.87** | +34.35 | +13.13 | -50.69 |
| GAM + transport | -30.58 | +10.20 | +19.09 | -44.38 |
| Full current | +3.98 | +17.17 | +26.97 | -49.17 |
| Placebo | +6.36 | -6.43 | +13.56 | -47.33 |

The residual-only benefit is not confined to valid months. On invalid target
months the GAM fields are neutralized for held rows, so this is evidence of a
ranker interaction/regime effect rather than a clean numeric GAM correction.
The valid-month top-5 gain is much clearer for the real residual fields than
for the placebo.

## Top-1 transition analysis

The deterministic comparison uses global Top-1% membership, with candidate ID
tie-breaking. For the best residual-only arm:

| Transition vs control | Rows | Mean net | Median net |
|---|---:|---:|---:|
| Entered Top-1% | 19 | **+74.86** | +25.41 |
| Exited Top-1% | 19 | +8.28 | +7.30 |
| Stayed Top-1% | 84 | +8.73 | +29.67 |

For the full current bundle, entered rows average +7.62 bps and exited rows
−18.74 bps, but both medians are positive; the mean effect is therefore
outlier-sensitive. The residual-only transition is more supportive, though it
still contains only 19 promoted and 19 demoted rows.

## Attribution conclusion

1. The original full-bundle Top-1 uplift was not solely attributable to the
   numeric GAM prediction.
2. The strongest isolated signal is `gam_delta_bps + gam_residual_bps`, not
   `gam_expected_bps` itself.
3. Transport-only fields have some Top-1 value but do not improve top-5
   stability materially.
4. The placebo can also create a very high Top-0.5% result, so that tail is not
   reliable evidence of causality. Real residual fields beat placebo around
   Top-1 and Top-5, where sample counts are larger.
5. Keep direct GAM modulation disabled. If this line is continued, expose only
   the two residual fields first, with strict Top-1/top-5 stability gates and
   repeated placebo seeds.

## Artifacts

- [Attribution implementation](</Users/remyroche/Documents/Ares/scripts/run_tp6_sl4_rolling_gam_attribution_ff1.py>)
- [Transition materializer](</Users/remyroche/Documents/Ares/scripts/materialize_tp6_sl4_attribution_transitions.py>)
- [Dense metrics](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_rolling_gam_attribution_ff1_20260815_v1/metrics_global_dense.parquet>)
- [Valid/invalid metrics](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_rolling_gam_attribution_ff1_20260815_v1/metrics_valid_invalid.parquet>)
- [Transition summary](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_rolling_gam_attribution_ff1_20260815_v1/top1_transition_summary_all.parquet>)
