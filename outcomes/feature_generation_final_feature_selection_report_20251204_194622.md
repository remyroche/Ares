# Final Feature Selection Report

**Generated:** 2025-12-04 19:46:22
**Step:** feature_generation_final_feature_selection_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** light
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

1. volume_ratio_50
2. price_volume_oscillator_5_15
3. rsi_zscore_14_20
4. vectorbt_momentum_comprehensive_9
5. cmo_14_returns_vwap
6. momentum_30_price_returns
7. volume_sma_50
8. momentum_14_price_returns
9. pivot_point_5_price_returns
10. rsi_21_returns_vwap
11. volume_vwap_10
12. sma_10_returns_vwap
13. volume_accumulation_distribution
14. vectorbt_volume_weighted_ad_line_50
15. returns_volatility_20_price_returns
16. vectorbt_trend_strength_5_price_returns
17. volume_ema_5
18. rolling_zscore_returns_20
19. vectorbt_yang_zhang_volatility_20
20. pfe_12_returns_vwap
... and 40 more features

### 50 Features Set (50 features)

1. volume_ratio_50
2. price_volume_oscillator_5_15
3. rsi_zscore_14_20
4. vectorbt_momentum_comprehensive_9
5. cmo_14_returns_vwap
6. momentum_30_price_returns
7. volume_sma_50
8. momentum_14_price_returns
9. pivot_point_5_price_returns
10. rsi_21_returns_vwap
11. volume_vwap_10
12. sma_10_returns_vwap
13. volume_accumulation_distribution
14. vectorbt_volume_weighted_ad_line_50
15. returns_volatility_20_price_returns
16. vectorbt_trend_strength_5_price_returns
17. volume_ema_5
18. rolling_zscore_returns_20
19. vectorbt_yang_zhang_volatility_20
20. pfe_12_returns_vwap
... and 30 more features

### 40 Features Set (40 features)

1. volume_ratio_50
2. price_volume_oscillator_5_15
3. rsi_zscore_14_20
4. vectorbt_momentum_comprehensive_9
5. cmo_14_returns_vwap
6. momentum_30_price_returns
7. volume_sma_50
8. momentum_14_price_returns
9. pivot_point_5_price_returns
10. rsi_21_returns_vwap
11. volume_vwap_10
12. sma_10_returns_vwap
13. volume_accumulation_distribution
14. vectorbt_volume_weighted_ad_line_50
15. returns_volatility_20_price_returns
16. vectorbt_trend_strength_5_price_returns
17. volume_ema_5
18. rolling_zscore_returns_20
19. vectorbt_yang_zhang_volatility_20
20. pfe_12_returns_vwap
... and 20 more features

### 30 Features Set (30 features)

1. volume_ratio_50
2. price_volume_oscillator_5_15
3. rsi_zscore_14_20
4. vectorbt_momentum_comprehensive_9
5. cmo_14_returns_vwap
6. momentum_30_price_returns
7. volume_sma_50
8. momentum_14_price_returns
9. pivot_point_5_price_returns
10. rsi_21_returns_vwap
11. volume_vwap_10
12. sma_10_returns_vwap
13. volume_accumulation_distribution
14. vectorbt_volume_weighted_ad_line_50
15. returns_volatility_20_price_returns
16. vectorbt_trend_strength_5_price_returns
17. volume_ema_5
18. rolling_zscore_returns_20
19. vectorbt_yang_zhang_volatility_20
20. pfe_12_returns_vwap
... and 10 more features


## Baseline Learnability of Selected Features

This baseline fits simple models (linear regression and small LightGBM baselines) using only the final selected features. It provides an upper bound on how much of the target variance is explainable by this feature set alone, before any complex downstream modeling.

## Baseline Predictive Check

**Dataset:** 97 samples, 60 features

### Top Single-Feature Signals

| Rank | Feature | Test R² | Pearson | AUC | Quality Score |
|------|---------|---------|---------|-----|---------------|
| 1 | `cci_20_returns_vwap` | 0.211 | -0.354 | N/A | 0.269 |
| 2 | `rsi_30_returns_vwap` | 0.200 | -0.334 | N/A | 0.253 |
| 3 | `cmo_14_returns_vwap` | 0.172 | -0.317 | N/A | 0.230 |
| 4 | `vectorbt_smoothed_obv_20` | 0.149 | 0.344 | N/A | 0.227 |
| 5 | `volume_ratio_50` | 0.155 | -0.329 | N/A | 0.225 |

### Small Multivariate LGBM Baseline

| Type | Features | Test R² |
|------|----------|---------|
| Pair | `vectorbt_smoothed_obv_20`, `rsi_21_returns_vwap` | 0.293 |
| Triplet | `rsi_30_returns_vwap`, `vectorbt_smoothed_obv_20`, `rsi_21_returns_vwap` | 0.292 |

### Interpretation

**Quality Score:** 0.27/1.0

**Summary:** ⚠️ Moderate predictive signals detected

**Insights:**
- Best feature `cci_20_returns_vwap` achieved Test R² = 0.211
- Positive Test R² features: 48 (80.0%)
- Median Test R² across evaluated features: 0.054
- LGBM best feature `vectorbt_acceleration_volatility_10_10_price_returns` achieved Test R² = 0.261

**Recommendations:**
- Focus downstream modeling on the top-ranked features


### Backtest Implications (Expected Sharpe)

- **Best Test R²:** 0.2115
- **Assumed Trading Frequency:** 1 trade/day (252/year)
- **Expected Annualized Sharpe Ratio:** **7.30**
  *(Derived from Information Coefficient approximation: Sharpe ≈ IC * sqrt(N))*

**Baseline learnability CSV:** `outcomes/baseline_check_final_feature_selection_20251204_194621.csv`

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
*Generated by Feature Generation Final Feature Selection Step at 2025-12-04 19:46:22*
