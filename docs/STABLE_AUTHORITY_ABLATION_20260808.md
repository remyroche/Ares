# Stable train/calibration authority ablation

## Purpose

The frozen 64-family contract meets the absolute contribution-mass gate, but
assignment quality is low. This ablation tests whether the residual learner
should use only families whose signed contribution has positive rank IC in both
the meta-train and pre-test calibration partitions.

## Contract and causality

- Same strict semantic 0.20/top-64 contract and exact 15-minute execution policy
  as the raw control.
- Eight families are selected independently within each fold.
- Selection score is the lower of train rank IC and calibration rank IC; a family
  must be positive in both when enough stable families exist.
- Calibration precedes the outer test partition and is not used in test
  execution labels, histories, or tail evaluation.

## Results (pooled net bps/trade)

| arm | top 0.5% | top 1% | top 5% | top 10% |
|---|---:|---:|---:|---:|
| raw all-family H control | **−22.24** | **−35.58** | **−55.59** | −65.71 |
| stable-authority H | −41.29 | −44.62 | −55.96 | **−65.65** |
| raw all-family G | −29.12 | −33.85 | **−54.99** | **−66.22** |
| stable-authority G | −38.59 | −44.84 | −55.91 | −66.85 |

Stable-authority H monthly top-5 net EV was −66.51 bps/trade on average,
median −72.63, worst −109.36, with one positive month. The raw all-family H
control averaged −63.20 bps/trade, median −69.51, worst −101.00, also with one
positive month.

## Decision

Do not promote stable train/calibration authority selection. Requiring positive
rank IC in both partitions reduces the correction layer's usable information
without improving top-5 or monthly portability. Keep all 64 families in the
residual input contract and expose train/calibration stability as a trust
feature rather than a hard family filter.

Artifact: [stable-authority replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/long_family_conditional_correctness_semantic020_top64_stable_authority8_20260808_v1)
