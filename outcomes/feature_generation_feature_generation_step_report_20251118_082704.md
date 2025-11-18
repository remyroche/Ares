# Feature Generation Report

**Generated:** 2025-11-18 08:27:04
**Step:** feature_generation_feature_generation_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** blank

## Summary

✅ **Successfully generated 259 features** from 31,303 rows of data.

## Artifacts

### generated_features

**Path:** `versioned_artifacts/ETHUSDT_binance_15m_long_analyst/generated_features_15m_20251118_082627_996.h5`
**Size:** 0.00 KB

### basic_feature_analysis_report

**Path:** `outcomes/basic_feature_analysis_20251118_082632.md`
**Size:** 2.06 KB

## Baseline Predictive Check

**Dataset:** 31302 samples, 259 features

### Top Single-Feature Signals

| Rank | Feature | Test R² | Pearson | Quality Score |
|------|---------|---------|---------|---------------|
| 1 | `candlestick_doji_pattern` | 0.003 | -0.038 | 0.017 |
| 2 | `candlestick_harami_cross_pattern` | 0.001 | -0.025 | 0.010 |
| 3 | `enhanced_volatility_14` | 0.001 | 0.018 | 0.007 |
| 4 | `enhanced_volatility_10` | 0.001 | 0.018 | 0.007 |
| 5 | `volume_std_10` | 0.000 | 0.015 | 0.006 |

### Small Multivariate LGBM Baseline

| Type | Features | Test R² |
|------|----------|---------|
| Pair | `candlestick_doji_pattern`, `enhanced_volatility_10` | 0.004 |
| Triplet | `candlestick_doji_pattern`, `volume_std_10`, `vectorbt_adx_9` | 0.003 |

### Interpretation

**Quality Score:** 0.02/1.0

**Summary:** ⚠️ Moderate predictive signals detected

**Insights:**
- Best feature `candlestick_doji_pattern` achieved Test R² = 0.003
- Positive Test R² features: 69 (26.6%)
- Median Test R² across evaluated features: -0.000
- LGBM best feature `vectorbt_momentum_50_price_returns` achieved Test R² = 0.004

**Recommendations:**
- Focus downstream modeling on the top-ranked features


### Baseline Metrics CSV Exports

- **Univariate baseline metrics CSV:** `outcomes/baseline_check_feature_generation_20251118_082704.csv`
- **Multivariate baseline metrics CSV:** `outcomes/multivariate_baseline_feature_generation_20251118_082704.csv`

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

