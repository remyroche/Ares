# Frozen conversion/reliability learner — long-only audit

Date: 2026-08-10  
Scope: **long side only; all short rows excluded**  
Transport rows: 194,247 (July–November 2024)

This is the historical-support learner using strict-OOS frozen ATR2/q4h scores
from September 2023–February 2024 as prior training support. The learner is
side-local, but only its long model and long predictions are included here.

## Long-only global transport tails

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% |
|---|---:|---:|---:|---:|---:|
| Raw control | −11.96 | −23.86 | −7.30 | **+17.25** | +11.49 |
| Regression α=.25 | −11.96 | −23.86 | −7.30 | +17.39 | +11.42 |
| Regression α=.50 | −11.96 | −23.86 | −7.30 | +15.55 | +10.20 |
| Regression α=1.00 | −11.96 | −23.86 | −7.30 | +10.25 | +7.37 |
| Ordinal α=.25 | −11.96 | −23.86 | −7.30 | **+26.40** | **+18.85** |
| Ordinal α=.50 | −11.96 | −23.86 | −7.30 | **+38.92** | +16.71 |
| Ordinal α=1.00 | −11.96 | −23.86 | −7.30 | +24.26 | +15.60 |

These pooled tails are descriptive only; they are not used to select a model
after seeing the full transport period.

## Predeclared selection and untouched November

Selection uses July–October long-only top-5 net. November is then untouched.

| Arm | Jul–Oct top-5 | November top-5 |
|---|---:|---:|
| Raw control | −63.79 | −13.71 |
| Regression α=.25 | −63.76 | −14.80 |
| Regression α=.50 | −62.62 | −10.62 |
| **Regression α=1.00 (selected)** | **−60.34** | **−1.68** |
| Ordinal α=.25 | −71.37 | +53.75 |
| Ordinal α=.50 | −70.70 | +74.33 |
| Ordinal α=1.00 | −63.83 | +88.14 |

Under the declared top-5-first rule, regression α=1.00 is the long-only
development winner and improves the untouched November result by 12.03 bps
versus raw. It remains negative, so it is not execution-ready.

## Monthly long-only top-5 net

| Arm | Jul | Aug | Sep | Oct | Nov | Mean Jul–Oct | Worst Jul–Oct |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw control | −43.71 | −144.79 | −51.97 | −71.02 | −13.71 | −77.87 | −144.79 |
| Regression α=.25 | −39.28 | −147.70 | −51.94 | −71.40 | −14.80 | −77.58 | −147.70 |
| Regression α=.50 | −36.30 | −152.52 | −50.84 | −73.01 | −10.62 | −78.17 | −152.52 |
| Regression α=1.00 | −32.77 | −165.78 | −51.12 | −71.45 | −1.68 | −80.28 | −165.78 |
| Ordinal α=.25 | −42.67 | −69.94 | −41.82 | −77.53 | **+53.75** | −57.99 | −77.53 |
| Ordinal α=.50 | −42.45 | **+0.18** | −10.06 | −71.60 | **+74.33** | **−30.98** | **−71.60** |
| Ordinal α=1.00 | −50.31 | **+24.04** | **+25.66** | −85.31 | **+88.14** | −21.48 | −85.31 |

The ordinal variants are more stable by month but lose the declared pooled
July–October top-5 selection criterion. They are research challengers, not
promoted models.

## Decision

For the long-only scope:

- selected development arm: regression α=1.00;
- untouched November: −1.68 net bps/trade versus raw −13.71;
- still no positive untouched-month execution result;
- ordinal α=.50 is the strongest stability challenger, but its July–October
  pooled top-5 is −70.70 and remains negative.

No long-only arm is execution-ready. Shorts were not used in any metric,
selection, or conclusion in this audit.

Artifacts: `data_perp/artifacts/frozen_conversion_reliability_learner_ablation_20260810_v2/long_only/`.
