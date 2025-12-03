# Final Feature Selection Report

**Generated:** 2025-12-03 00:02:40
**Step:** feature_generation_final_feature_selection_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** full
- **Feature Count Targets:** [60, 50, 40]
- **Selection Method:** permutation ✅
- **Importance Type:** Permutation (captures feature interactions, not just Gini splits) 📊
- **Optimization Enabled:** True

## Top IC Features (Meta-Label Overview)



## Feature Selection Methodology

✅ **Using Permutation Importance**
- Captures how features work together (feature interactions)
- More reliable than standard Gini importance for complex trading strategies
- Measures true impact on model predictions
- Better for identifying genuinely predictive features

## Feature Selection Results

- **60 Features Set:** 60 features selected
- **50 Features Set:** 50 features selected
- **40 Features Set:** 40 features selected
- **30 Features Set:** 30 features selected

- **Total Feature Sets:** 4

## Selected Features by Set

### 60 Features Set (60 features)

1. cci_20_returns_vwap
2. vectorbt_momentum_comprehensive_9
3. vectorbt_trend_consistency_10_price_returns
4. volume_std_10
5. vectorbt_volume_weighted_ad_line_20
6. vectorbt_acceleration_divergence_20_price_returns
7. cmf_20
8. vectorbt_trend_strength_50_price_returns
9. vectorbt_enhanced_obv_20
10. enhanced_volatility_14
11. volume_ema_5
12. vectorbt_trend_consistency_50_price_returns
13. volume_std_50
14. vectorbt_volatility_comprehensive_20
15. vectorbt_smoothed_obv_20
16. fibonacci_0.236_20_price_returns
17. returns_volatility_20_price_returns
18. vectorbt_atr_50
19. vectorbt_enhanced_ad_line_10
20. volume_accumulation_distribution
... and 40 more features

### 50 Features Set (50 features)

1. cci_20_returns_vwap
2. vectorbt_momentum_comprehensive_9
3. vectorbt_trend_consistency_10_price_returns
4. volume_std_10
5. vectorbt_volume_weighted_ad_line_20
6. vectorbt_acceleration_divergence_20_price_returns
7. cmf_20
8. vectorbt_trend_strength_50_price_returns
9. vectorbt_enhanced_obv_20
10. enhanced_volatility_14
11. volume_ema_5
12. vectorbt_trend_consistency_50_price_returns
13. volume_std_50
14. vectorbt_volatility_comprehensive_20
15. vectorbt_smoothed_obv_20
16. fibonacci_0.236_20_price_returns
17. returns_volatility_20_price_returns
18. vectorbt_atr_50
19. vectorbt_enhanced_ad_line_10
20. volume_accumulation_distribution
... and 30 more features

### 40 Features Set (40 features)

1. cci_20_returns_vwap
2. vectorbt_momentum_comprehensive_9
3. vectorbt_trend_consistency_10_price_returns
4. volume_std_10
5. vectorbt_volume_weighted_ad_line_20
6. vectorbt_acceleration_divergence_20_price_returns
7. cmf_20
8. vectorbt_trend_strength_50_price_returns
9. vectorbt_enhanced_obv_20
10. enhanced_volatility_14
11. volume_ema_5
12. vectorbt_trend_consistency_50_price_returns
13. volume_std_50
14. vectorbt_volatility_comprehensive_20
15. vectorbt_smoothed_obv_20
16. fibonacci_0.236_20_price_returns
17. returns_volatility_20_price_returns
18. vectorbt_atr_50
19. vectorbt_enhanced_ad_line_10
20. volume_accumulation_distribution
... and 20 more features

### 30 Features Set (30 features)

1. cci_20_returns_vwap
2. vectorbt_momentum_comprehensive_9
3. vectorbt_trend_consistency_10_price_returns
4. volume_std_10
5. vectorbt_volume_weighted_ad_line_20
6. vectorbt_acceleration_divergence_20_price_returns
7. cmf_20
8. vectorbt_trend_strength_50_price_returns
9. vectorbt_enhanced_obv_20
10. enhanced_volatility_14
11. volume_ema_5
12. vectorbt_trend_consistency_50_price_returns
13. volume_std_50
14. vectorbt_volatility_comprehensive_20
15. vectorbt_smoothed_obv_20
16. fibonacci_0.236_20_price_returns
17. returns_volatility_20_price_returns
18. vectorbt_atr_50
19. vectorbt_enhanced_ad_line_10
20. volume_accumulation_distribution
... and 10 more features


## Baseline Learnability of Selected Features

This baseline fits simple models (linear regression and small LightGBM baselines) using only the final selected features. It provides an upper bound on how much of the target variance is explainable by this feature set alone, before any complex downstream modeling.

## Baseline Predictive Check

**Dataset:** 97 samples, 60 features

### Top Single-Feature Signals

| Rank | Feature | Test R² | Pearson | AUC | Quality Score |
|------|---------|---------|---------|-----|---------------|
| 1 | `vectorbt_trend_consistency_10_price_returns` | 0.118 | 0.293 | N/A | 0.188 |
| 2 | `vectorbt_trend_consistency_50_price_returns` | 0.073 | 0.238 | N/A | 0.139 |
| 3 | `vectorbt_trend_consistency_20_price_returns` | 0.033 | 0.245 | N/A | 0.118 |
| 4 | `volume_std_50` | 0.030 | -0.249 | N/A | 0.118 |
| 5 | `vectorbt_acceleration_consistency_10_20_price_returns` | 0.049 | 0.205 | N/A | 0.111 |

### Small Multivariate LGBM Baseline

| Type | Features | Test R² |
|------|----------|---------|
| Pair | `vectorbt_trend_consistency_10_price_returns`, `vectorbt_momentum_comprehensive_9` | 0.479 |
| Triplet | `vectorbt_trend_consistency_10_price_returns`, `volume_std_50`, `vectorbt_momentum_comprehensive_9` | 0.533 |

### Interpretation

**Quality Score:** 0.19/1.0

**Summary:** ⚠️ Moderate predictive signals detected

**Insights:**
- Best feature `vectorbt_trend_consistency_10_price_returns` achieved Test R² = 0.118
- Positive Test R² features: 16 (26.7%)
- Median Test R² across evaluated features: -0.006
- LGBM best feature `vectorbt_trend_consistency_10_price_returns` achieved Test R² = 0.233

**Recommendations:**
- Focus downstream modeling on the top-ranked features


**Baseline learnability CSV:** `outcomes/baseline_check_final_feature_selection_20251203_000239.csv`

### How to Read These Learnability Metrics

- **Test R²** rows show, for each selected feature, how much of the target variance it explains out-of-sample in a simple regression. Values near 0 mean weak signal; values above roughly 0.3–0.4 indicate strong linear signal; negative values indicate that even a simple model fails to generalize.
- The **quality score** aggregates how many features achieve positive Test R², how strong the best feature(s) are, and how consistent performance is across evaluated features. Scores close to 1.0 mean that many features contain robust, learnable signal; scores near 0 indicate that this feature set behaves mostly like noise.
If the selected-feature quality score is low, or if most Test R² values are negative, it suggests that the final selection may be too aggressive or misaligned with the target. In that case, consider revisiting labeling, feature generation, or selection thresholds before relying on this set in production models.

## Performance Metrics

- **Execution Time:** N/A seconds
- **Optimization Enabled:** Yes
- **Hardware Optimization:** Yes

## Optimization Details

- **VectorBT Optimization:** Disabled
- **Rolling Optimizer:** Not Available
- **Hardware Manager:** Not Available

## Generated Artifacts

- **Feature Sets:** 4
- **Feature DataFrames:** 4
- **SHAP Analyses:** 0
- **Metadata Files:** 2
- **Total Artifacts:** 12

## Summary

Final feature selection completed successfully. Generated 4 optimized feature sets with comprehensive SHAP analysis and metadata. All artifacts saved in both pickle and markdown formats.

---
*Generated by Feature Generation Final Feature Selection Step at 2025-12-03 00:02:40*
