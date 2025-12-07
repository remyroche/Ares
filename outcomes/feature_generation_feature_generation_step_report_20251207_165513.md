# Feature Generation Report

**Generated:** 2025-12-07 16:55:13
**Step:** feature_generation_feature_generation_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** full

## Summary

✅ **Successfully generated 309 features** from 140,354 rows of data.

## Artifacts

### generated_features

**Path:** `versioned_artifacts/ETHUSDT_binance_15m_long_analyst/generated_features_15m_20251207_165448_970.h5`
**Size:** 0.00 KB

### basic_feature_analysis_report

**Path:** `outcomes/basic_feature_analysis_20251207_165505.md`
**Size:** 2.01 KB

## Baseline Predictive Check

**Dataset:** 7324 samples, 313 features

### Top Single-Feature Signals

| Rank | Feature | Test R² | Pearson | AUC | Quality Score |
|------|---------|---------|---------|-----|---------------|
| 1 | `binary_label` | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | `vectorbt_enhanced_obv_10` | 0.185 | 0.412 | 0.737 | 0.737 |
| 3 | `vectorbt_enhanced_obv_20` | 0.183 | 0.410 | 0.736 | 0.736 |
| 4 | `vectorbt_enhanced_obv_50` | 0.182 | 0.408 | 0.736 | 0.736 |
| 5 | `volume_price_trend` | 0.172 | 0.394 | 0.730 | 0.730 |

### Small Multivariate LGBM Baseline

| Type | Features | Test R² |
|------|----------|---------|
| Pair | `binary_label`, `vectorbt_enhanced_obv_10` | 1.000 |
| Triplet | `binary_label`, `vectorbt_enhanced_obv_10`, `vectorbt_enhanced_obv_20` | 1.000 |

### Interpretation

**Quality Score:** 1.00/1.0

**Summary:** ✅ Strong individual signals detected

**Insights:**
- Best feature `binary_label` achieved Test R² = 1.000
- Positive Test R² features: 199 (63.6%)
- Median Test R² across evaluated features: 0.000
- LGBM best feature `binary_label` achieved Test R² = 1.000

**Recommendations:**
- Focus downstream modeling on the top-ranked features


### Baseline Metrics CSV Exports

- **Univariate baseline metrics CSV:** `outcomes/baseline_check_feature_generation_20251207_165513.csv`
- **Multivariate baseline metrics CSV:** `outcomes/multivariate_baseline_feature_generation_20251207_165513.csv`

These CSV files mirror the baseline diagnostics in a tabular format so you can track learnability across runs, symbols, or execution modes.

### How to Interpret Baseline Learnability Metrics

The baseline check fits simple models (linear regression and small LightGBM baselines) on each feature individually, and on a few 2–3 feature combinations. This provides an upper bound on how learnable the target is from the raw feature set alone, before any complex modeling.

- **Test R²** measures how much of the variance in the target is explained out-of-sample by a given feature (or feature combination). Values near 0 mean the feature carries very little predictive signal; values above roughly 0.3–0.4 indicate strong linear signal; negative values indicate that even a simple model fails to generalize.
- The **quality score** aggregates how many features achieve positive Test R², how strong the best feature(s) are, and how consistent performance is across evaluated features. Scores close to 1.0 mean that many features contain robust, learnable signal; scores near 0 indicate that almost all features behave like noise.

In practice, a low quality score, many negative Test R² values, or a best feature with weak Test R² suggests that labels/targets or feature definitions may need to be revisited before investing further in complex downstream models.

## Next Steps

- Features are ready for feature selection and interaction generation
- Consider running lookback optimization for optimal feature parameters
- Proceed to labeling step for profit-target generation

