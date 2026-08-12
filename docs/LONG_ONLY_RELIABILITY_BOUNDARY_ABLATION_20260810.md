# Long-only reliability-boundary ablation

Date: 2026-08-10  
Scope: **long side only**. Short rows were excluded from fitting, score construction, selection, and every reported metric.

## Question

Does changing the reliability meta target boundary from the existing ordinal residual threshold improve the frozen ATR2/q4h stack for longs?

The ablation used the frozen specialist/residual inputs and the existing causal historical extension. It varied the three-class residual target boundary at ±50, ±75, and ±100 bps, and combined each boundary with correction strengths α ∈ {0.25, 0.50, 1.00}:

`adjusted_score = frozen_score + α × reliability_correction`

The learner, feature contract, expanding-month chronology, and ranking procedure were held fixed. The development selection rule was fixed before looking at November: highest long-only global top-5 net EV over July–October; November was untouched.

## Population and chronology

| Population | Long rows | Period |
|---|---:|---|
| Historical + transport source used for fitting | 509,010 total source rows | Sep 2023–Nov 2024 |
| Transport evaluation | 194,247 | Jul–Nov 2024 |
| Development selection | 157,298 | Jul–Oct 2024 |
| Untouched OOS | 36,949 | Nov 2024 |

The historical rows are used only as prior resolved support for the expanding learners. No November labels are used for model fitting or arm selection.

## Selection result

The frozen raw score remains the development winner:

| Arm | Jul–Oct top-5 gross (bps/trade) | Jul–Oct top-5 net (bps/trade) | Rank IC |
|---|---:|---:|---:|
| **Raw control** | **36.21** | **−63.79** | −0.0364 |
| ±75, α=1.00 | 36.17 | −63.83 | −0.0044 |
| ±100, α=1.00 | 35.36 | −64.64 | −0.0093 |
| ±100, α=0.25 | 32.16 | −67.84 | −0.0270 |
| ±100, α=0.50 | 32.07 | −67.93 | −0.0171 |
| ±50, α=0.50 | 31.78 | −68.22 | −0.0143 |
| ±50, α=1.00 | 30.77 | −69.23 | −0.0062 |
| ±50, α=0.25 | 29.84 | −70.16 | −0.0256 |
| ±75, α=0.50 | 29.30 | −70.70 | −0.0119 |
| ±75, α=0.25 | 28.63 | −71.37 | −0.0234 |

No reliability-boundary correction improves the predeclared development criterion. The closest challenger (±75, α=1) is 0.04 bps/trade worse, which is economically negligible but not an advance.

## Untouched November

November is useful as a genuine holdout, but it cannot be used to choose the arm.

| Arm | Nov top-1 net | Nov top-2 net | Nov top-5 net | Nov top-10 net (bps/trade) |
|---|---:|---:|---:|---:|
| ±50, α=1.00 | 124.25 | 115.43 | **119.33** | **105.17** |
| ±100, α=1.00 | **135.80** | 109.31 | 104.98 | 97.75 |
| ±50, α=0.50 | 28.79 | 82.21 | 101.32 | 89.59 |
| ±75, α=1.00 | 104.11 | 105.44 | 88.14 | 91.20 |
| ±100, α=0.50 | 50.94 | 76.84 | 84.82 | 78.94 |
| ±75, α=0.50 | 47.01 | 60.48 | 74.33 | 74.92 |
| ±50, α=0.25 | 13.39 | 31.27 | 64.63 | 69.97 |
| ±75, α=0.25 | 13.34 | 11.43 | 53.75 | 64.18 |
| ±100, α=0.25 | −6.57 | 24.10 | 44.87 | 65.46 |
| Raw control | 32.06 | −10.21 | −13.71 | 11.00 |

The holdout result is strongly positive for the corrected arms, especially ±50/α=1. However, this is a **regime-dependent holdout surprise**, not evidence that the correction is validated: every corrected arm remains negative over the July–October selection window, and the boundary was not selected using November.

## Monthly top-5 net EV

| Month | Raw | ±50 α=.25 | ±50 α=.50 | ±50 α=1 | ±75 α=.25 | ±75 α=.50 | ±75 α=1 | ±100 α=.25 | ±100 α=.50 | ±100 α=1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jul 2024 | −43.71 | −38.83 | −41.84 | −47.36 | −42.67 | −42.45 | −50.31 | −38.09 | −36.95 | −42.32 |
| Aug 2024 | −144.79 | −75.16 | −5.07 | **37.30** | −69.94 | 0.18 | 24.04 | −72.97 | −6.06 | 9.37 |
| Sep 2024 | −51.97 | −39.94 | −23.89 | **13.39** | −41.82 | −10.06 | 25.66 | −38.81 | 1.82 | 36.97 |
| Oct 2024 | −71.02 | −75.89 | −70.09 | −77.19 | −77.53 | −71.60 | −85.31 | −79.51 | −78.66 | −91.13 |
| Nov 2024 | −13.71 | 64.63 | 101.32 | **119.33** | 53.75 | 74.33 | 88.14 | 44.87 | 84.82 | 104.98 |

The correction improves August–September and November for some settings, but it does not repair July or October. This is consistent with a changing conversion relationship rather than a globally useful reliability correction.

## All-transport long-only tails

These are descriptive pooled transport metrics, not a selection criterion.

| Arm | Top-1 | Top-2 | Top-5 | Top-10 net bps/trade |
|---|---:|---:|---:|---:|
| ±50 α=.50 | 41.80 | **76.82** | **43.42** | 6.88 |
| ±75 α=.50 | 50.34 | 68.66 | 38.92 | 7.61 |
| ±100 α=.50 | 75.87 | 62.07 | 32.45 | 6.58 |
| ±50 α=.25 | 27.52 | 29.53 | 29.51 | **13.94** |
| ±50 α=1 | **111.69** | 71.23 | 26.51 | −5.50 |
| ±75 α=.25 | 13.59 | 27.75 | 26.40 | 12.46 |
| ±100 α=.25 | 15.92 | 23.39 | 24.27 | 10.53 |
| ±75 α=1 | 81.10 | 48.58 | 24.26 | −4.05 |
| ±100 α=1 | 82.02 | 46.25 | 21.80 | −11.64 |
| Raw control | −23.86 | −7.30 | 17.25 | 11.49 |

The pooled tails look better for several corrections, but this gain is driven disproportionately by November and does not survive the earlier-month development gate.

## Decision

**No boundary correction advances the long-only production candidate under the frozen selection rule.** Keep the raw frozen score as the controlled baseline. Retain ±50/α=1 and ±100/α=1 as research challengers because they produce large November gains and improve August–September, but do not promote them without another untouched period or a predeclared regime-conditional rule.

The main finding is not that the reliability layer is useless. It is that a single fixed residual boundary does not transport across the observed long-side regimes. The failure is concentrated in July and October, while November reverses strongly positive. This points to conversion/regime transport and admission calibration—not simply the choice between 50, 75, and 100 bps—as the next diagnostic target.

## Artifacts

- Predictions: `data_perp/artifacts/long_only_reliability_boundary_ablation_20260810_v1/long_only_predictions.parquet`
- Metrics: `data_perp/artifacts/long_only_reliability_boundary_ablation_20260810_v1/long_only_metrics.parquet`
- Manifest: `data_perp/artifacts/long_only_reliability_boundary_ablation_20260810_v1/manifest.json`
- Runner: `scripts/run_long_only_reliability_boundary_ablation.py`
