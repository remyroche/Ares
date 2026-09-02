# One-month gated GAM inside the residual/meta stack

## Scope

This is a long-side-only matched 2025 replay. It takes the one-month rolling
structural GAM selected previously:

`gated zero-exposure GAM, gamma = 0.25`

When the local structural contract is transport-invalid, the GAM score is the
base bps anchor. No broader specialist or archetype score is added directly to
the trading rank; only the saved one-month GAM output and its causal trust
fields are integrated into the existing residual/meta layer.

## Integration

The existing stack is refit before each held 2025 month:

1. Map the native base score to a train-only TP6/SL4 expected-net-bps anchor.
2. Fit the existing 4-hour UTC × side LambdaRank consensus heads.
3. Fit the existing per-row residual LambdaRank head.
4. Combine the normalized base, consensus, and residual ranks as
   `0.50 × base + 0.25 × consensus + 0.25 × residual`.
5. Normalize within month/side, then perform one pooled global ranking.

The GAM-input arms add these fields to both the consensus and residual heads:

- `gam_expected_bps` — gated one-month GAM bps score;
- `gam_delta_bps` — GAM score minus base bps anchor;
- `gam_residual_bps` — gated structural GAM residual;
- `gam_transport_valid`;
- `gam_matched_mass`, `gam_unmatched_mass`;
- `gam_archetype_count`, `gam_cluster_count`.

Rows predating the first rolling GAM output are retained in the training set
with an explicit neutral fallback: base anchor, zero GAM delta/residual,
`gam_transport_valid = 0`, and unmatched mass one. This keeps all arms on the
same training population.

## Ablation arms

| Arm | GAM fields in heads | Base anchor used by residual target | Final base component |
|---|---|---|---|
| Control | No | base expected bps | native base score rank |
| GAM input | Yes | base expected bps | native base score rank |
| GAM modulation | No | gated GAM bps | gated GAM bps rank |
| GAM input + modulation | Yes | gated GAM bps | gated GAM bps rank |

The control uses the native base score for ranking, matching the current stack;
the mapped bps anchor is used only as the residual/meta target reference.

## Global long-side metrics

Net bps/trade, 12 held months, global ranking:

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% |
|---|---:|---:|---:|---:|---:|
| Matched control | **+20.16** | −38.23 | **+9.78** | **−1.14** | **−18.26** |
| GAM inputs only | +1.46 | **+13.51** | −14.33 | −3.10 | −19.87 |
| GAM modulation only | −31.74 | −52.83 | −21.49 | −40.82 | −40.10 |
| GAM inputs + modulation | −11.91 | +7.59 | −35.30 | −37.89 | −39.74 |
| Existing saved current stack | +37.59 | **+14.04** | −8.00 | −24.91 | −15.81 |

Gross bps/trade for the GAM-input-only arm was 101.46 / 113.51 / 85.67 /
96.90 / 80.13 at the same tails; the corresponding net values are shown
above after the TP6/SL4 cost contract.

## Monthly top-5 net bps/trade

| Month | Control | GAM inputs | GAM modulation | Inputs + modulation |
|---|---:|---:|---:|---:|
| 2025-01 | +13.7 | -0.3 | +16.5 | **+46.0** |
| 2025-02 | +52.7 | +38.6 | -69.2 | -72.2 |
| 2025-03 | -157.3 | -182.8 | -248.0 | -256.6 |
| 2025-04 | +63.5 | +39.0 | **+91.0** | +48.8 |
| 2025-05 | -83.4 | -74.3 | -40.6 | -56.5 |
| 2025-06 | -17.6 | +4.2 | -81.2 | -80.1 |
| 2025-07 | -39.0 | -40.0 | -62.6 | +4.2 |
| 2025-08 | -23.9 | -67.4 | -45.6 | -60.8 |
| 2025-09 | -58.2 | **-13.4** | -75.3 | -40.9 |
| 2025-10 | +43.2 | +8.4 | **+53.6** | +27.4 |
| 2025-11 | +152.6 | **+245.6** | +51.8 | +101.4 |
| 2025-12 | +38.7 | -2.6 | -73.5 | -102.4 |

Stability summary at top 5%:

| Arm | Mean | Median | MAD | Worst month | Positive months |
|---|---:|---:|---:|---:|---:|
| Matched control | -1.26 | -1.94 | 49.85 | -157.35 | 6/12 |
| GAM inputs only | -3.75 | -1.42 | 39.28 | -182.80 | 5/12 |
| GAM modulation only | -40.27 | -54.09 | 24.15 | -248.03 | 4/12 |
| GAM inputs + modulation | -36.81 | -48.70 | 53.31 | -256.56 | 5/12 |

## GAM activation and training coverage

- 12 target months, 852 long candidates per month, 10,224 held rows per arm.
- Expanding residual/meta training rows: 7,816 to 17,173 per month.
- The one-month GAM transport gate was valid in 7/12 held months: January,
  April, May, September, October, November, and December.
- February, March, and June--August used the declared base fallback.
- Mean held-row GAM-valid fraction: 58.3%.
- Mean training-row GAM-valid fraction: 45.3%.

## Interpretation

1. Feeding GAM output and trust fields into the residual/meta heads is the only
   variant with a useful broad improvement relative to the matched control:
   top 1% improves by 51.74 bps and top 5% is approximately flat (−3.10 vs
   −1.14 bps). It is not a universal improvement: top 0.5% and top 2% are
   worse, and the worst month is more negative.
2. Directly replacing the residual target anchor with the GAM score is harmful
   at every normal operating tail. The GAM is better used as context/reliability
   information than as a hard score replacement.
3. Combining modulation with GAM inputs is also harmful at top 2--10% and has
   materially worse stability. Do not promote it.
4. The existing saved current stack remains better at top 0.5--1% than these
   matched refits, so the GAM-input result is an incremental diagnostic, not a
   new production winner.

## Artifacts

- [Integration implementation](</Users/remyroche/Documents/Ares/scripts/run_tp6_sl4_rolling_gam_residual_integration.py>)
- [Rank-correction materializer](</Users/remyroche/Documents/Ares/scripts/materialize_tp6_sl4_rolling_gam_integration_rankfix.py>)
- [Final predictions](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_rolling_gam_residual_integration_20260815_v3/predictions.parquet>)
- [Global metrics](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_rolling_gam_residual_integration_20260815_v3/metrics_global.parquet>)
- [Monthly metrics](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_rolling_gam_residual_integration_20260815_v3/metrics_monthly.parquet>)
- [Stability metrics](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_rolling_gam_residual_integration_20260815_v3/metrics_stability.parquet>)
