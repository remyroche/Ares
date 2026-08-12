# Train-only continuous family-stability feature ablation

## Setup

The residual MLP was retrained with three additional row-specific features:

- `family_train_positive_mass`
- `family_train_stability_score`
- `family_train_positive_active_fraction`

Each is computed from active family contribution shares and rank IC statistics
fit on the meta-train partition only. No calibration or outer-test outcomes enter
the feature values. The 64-family semantic contract, base score, target, and
execution policy are unchanged.

## Pooled OOS net bps/trade

| arm | top 0.5% | top 1% | top 5% | top 10% |
|---|---:|---:|---:|---:|
| raw all-family H control | **−22.24** | **−35.58** | **−55.59** | −65.71 |
| train-stability H | −44.36 | −54.82 | −58.30 | −66.42 |
| train-stability G | −45.78 | −52.51 | −59.31 | −66.90 |
| train-stability J | −47.22 | −51.36 | −58.89 | −68.69 |

Train-stability H monthly top-5 mean was −63.50 bps/trade, median −69.60,
worst −100.32, with one positive month. This is essentially unchanged from
raw H (−63.20 mean) and is materially worse at top 0.5–1%.

## Decision

Do not promote these features as a standalone residual input expansion. They
are causally valid and may remain available as diagnostic trust fields, but
they do not improve the global top-5 or narrow-tail promotion gates.

Artifact: [train-stability replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/long_family_conditional_correctness_semantic020_top64_train_stability_features_20260808_v1)
