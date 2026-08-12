# Family stability trust ablation

## Purpose

Test whether the residual correction should be attenuated or gated by the
cross-partition stability of the active semantic families. For each outer-test
row, the stability statistic is the represented absolute family mass whose
family rank IC was positive in both meta-train and pre-test calibration,
divided by represented mass.

The stability audit is computed only from the train/calibration labels. Outer
test outcomes are used only for reporting.

## Pooled global tails (net bps/trade)

| score | top 0.5% | top 1% | top 5% | top 10% |
|---|---:|---:|---:|---:|
| Cap-120 control | -55.24 | -56.66 | -61.58 | -68.86 |
| stability weighted | -34.90 | -44.32 | -70.13 | -74.17 |
| stability × mass | -35.08 | -44.00 | -69.86 | -74.28 |
| stability × assignment quality | -37.49 | -47.77 | -69.36 | -74.86 |
| stability gate ≥0.25 | **-29.78** | **-40.27** | -70.07 | -76.48 |
| stability gate ≥0.50 | -29.73 | -46.72 | -73.13 | -75.01 |
| stability gate ≥0.75 | -46.12 | -59.29 | -75.99 | -75.91 |

For reference, the best raw-family residual arm H was -22.24 / -35.58 /
-55.59 / -65.71 at the same tails, so stability gating does not beat the
existing residual correction either.

## Monthly top-5 stability

The ≥0.25 gate improved mean monthly top-5 from -67.27 to -59.35 bps/trade and
had one positive month, but its pooled top-5 was -70.07 bps/trade. The apparent
monthly improvement is therefore not a reliable global ranking improvement.

## Decision

Do not promote hard stability gating. It is useful as a diagnostic trust field,
especially for the narrow top 0.5–1%, but it sacrifices global top-5 ranking.
Keep it as a continuous residual-layer feature and retain the predeclared
global top-5 gate for promotion.

Artifact: [stability trust replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/family_stability_trust_ablation_semantic020_top64_20260808_v1)
