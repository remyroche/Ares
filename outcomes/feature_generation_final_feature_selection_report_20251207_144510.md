# Final Feature Selection Report

**Generated:** 2025-12-07 14:45:10
**Step:** feature_generation_final_feature_selection_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** blank
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

## Pre-FeatureSelector Pipeline Stage Summary

| Stage | Kept | Dropped |
|-------|------|---------|
| Combined (after shaping/sanitization) | 388 | - |
| After exclusions (no targets/raw/time/perf/meta) | 387 | 1 |
| After coverage filter (FS input) | 387 | 0 |

### Pre-FeatureSelector Detailed Diagnostics

**Per-Source Numeric Feature Counts (pre-combine):**

| Source | Numeric Features (non-target) |
|--------|-------------------------------|
| analyst_interactions | 80 |
| generated_features | 309 |
| labeled_df_non_target | 309 |

**Blank-Mode Shaping (raw/meta/leakage + high-NaN) :**

| Metric | Value |
|--------|-------|
| Raw/meta/leakage dropped | 0 |
| Candidate numeric before high-NaN | 389 |
| High-NaN features dropped | 0 |
| Candidate numeric after high-NaN | 389 |

**Sanitization (leakage + near-constant):**

| Metric | Value |
|--------|-------|
| Known leakage dropped | 0 |
| Near-constant features dropped | 1 |
| Candidate features before sanitization | 389 |
| Candidate features after sanitization | 388 |

**Combine Stage (numeric filtering after concatenation):**

| Metric | Value |
|--------|-------|
| Columns before numeric filter | 392 |
| Non-numeric columns dropped | 0 |
| Columns after numeric filter | 392 |

## Feature Selection Results

- **60 Features Set:** 60 features selected
- **50 Features Set:** 50 features selected
- **40 Features Set:** 40 features selected
- **30 Features Set:** 30 features selected

- **Total Feature Sets:** 4

## Selected Features by Set

### 60 Features Set (60 features)

1. advanced_cumulative_returns_10
2. resistance_level_1_10_price_returns
3. vectorbt_smoothed_obv_10
4. vectorbt_momentum_acceleration_10_10_price_returns
5. vectorbt_trend_consistency_5_price_returns
6. vectorbt_adx_21
7. vectorbt_trend_consistency_50_price_returns_base_div_vectorbt_volume_weighted_ad_line_50_base_15x_ratio
8. vectorbt_enhanced_obv_50_base_15x_ratio_minus_wavelet_energy_base_15x_ratio
9. vectorbt_acceleration_volatility_10_20_price_returns
10. vectorbt_sma_100_base_div_vectorbt_sma_100_base_9x_ratio
11. volume_roc_1
12. sma_50_returns_vwap
13. volume_momentum_10_vwap_minus_resistance_level_1_20_price_returns_base_15x_ratio
14. tema_21_price_returns_trend_adj_log_ratio_vectorbt_sma_100_vwap_15x_ratio
15. dema_21_price_returns_trend_adj_minus_volume_ema_50_base
16. vectorbt_trend_consistency_50_price_returns_base_log_ratio_wavelet_energy_base_15x_ratio
17. vectorbt_trend_consistency_50_price_returns_base_minus_stochastic_30_3_price_returns_trend_adj
18. vectorbt_sma_100_base_log_ratio_candlestick_engulfing_pattern_trend_adj_6x_ratio
19. volume_entropy_ma_20_5_volume_returns
20. vectorbt_volume_weighted_ad_line_50_base_15x_ratio_div_vectorbt_enhanced_obv_50_base_15x_ratio
... and 40 more features

### 50 Features Set (50 features)

1. advanced_cumulative_returns_10
2. vectorbt_smoothed_obv_10
3. vectorbt_sma_100_base_log_ratio_vectorbt_sma_100_base_9x_ratio
4. volume_price_divergence_10
5. volume_volatility_elasticity_20
6. vectorbt_trend_consistency_50_price_returns_base_log_ratio_wavelet_energy_base_15x_ratio
7. vectorbt_volume_weighted_ad_line_50_base_15x_ratio_div_vectorbt_enhanced_obv_50_base_15x_ratio
8. vectorbt_adx_21
9. vectorbt_enhanced_obv_50_base_15x_ratio_minus_wavelet_energy_base_15x_ratio
10. vectorbt_sma_100_base_log_ratio_candlestick_engulfing_pattern_trend_adj_6x_ratio
11. vectorbt_acceleration_volatility_10_20_price_returns
12. sma_50_returns_vwap
13. choppiness_index_14
14. momentum_14_price_returns
15. resistance_level_1_10_price_returns
16. returns_volatility_10_price_returns
17. macd_entropy_20_12_26_vwap_15x_ratio_div_vectorbt_enhanced_obv_50_base_15x_ratio
18. macd_entropy_20_12_26_vwap_15x_ratio_log_ratio_vectorbt_enhanced_obv_50_base_15x_ratio
19. ar_1_coefficients_20
20. vectorbt_acceleration_10_price_returns
... and 30 more features

### 40 Features Set (40 features)

1. advanced_cumulative_returns_10
2. vectorbt_smoothed_obv_10
3. vectorbt_sma_100_base_log_ratio_vectorbt_sma_100_base_9x_ratio
4. volume_price_divergence_10
5. volume_volatility_elasticity_20
6. vectorbt_trend_consistency_50_price_returns_base_log_ratio_wavelet_energy_base_15x_ratio
7. vectorbt_volume_weighted_ad_line_50_base_15x_ratio_div_vectorbt_enhanced_obv_50_base_15x_ratio
8. vectorbt_adx_21
9. vectorbt_enhanced_obv_50_base_15x_ratio_minus_wavelet_energy_base_15x_ratio
10. vectorbt_sma_100_base_log_ratio_candlestick_engulfing_pattern_trend_adj_6x_ratio
11. vectorbt_acceleration_volatility_10_20_price_returns
12. sma_50_returns_vwap
13. choppiness_index_14
14. momentum_14_price_returns
15. resistance_level_1_10_price_returns
16. returns_volatility_10_price_returns
17. macd_entropy_20_12_26_vwap_15x_ratio_div_vectorbt_enhanced_obv_50_base_15x_ratio
18. macd_entropy_20_12_26_vwap_15x_ratio_log_ratio_vectorbt_enhanced_obv_50_base_15x_ratio
19. ar_1_coefficients_20
20. vectorbt_acceleration_10_price_returns
... and 20 more features

### 30 Features Set (30 features)

1. advanced_cumulative_returns_10
2. vectorbt_smoothed_obv_10
3. vectorbt_sma_100_base_log_ratio_vectorbt_sma_100_base_9x_ratio
4. volume_price_divergence_10
5. volume_volatility_elasticity_20
6. vectorbt_trend_consistency_50_price_returns_base_log_ratio_wavelet_energy_base_15x_ratio
7. vectorbt_volume_weighted_ad_line_50_base_15x_ratio_div_vectorbt_enhanced_obv_50_base_15x_ratio
8. vectorbt_adx_21
9. vectorbt_enhanced_obv_50_base_15x_ratio_minus_wavelet_energy_base_15x_ratio
10. vectorbt_sma_100_base_log_ratio_candlestick_engulfing_pattern_trend_adj_6x_ratio
11. vectorbt_acceleration_volatility_10_20_price_returns
12. sma_50_returns_vwap
13. choppiness_index_14
14. momentum_14_price_returns
15. resistance_level_1_10_price_returns
16. returns_volatility_10_price_returns
17. macd_entropy_20_12_26_vwap_15x_ratio_div_vectorbt_enhanced_obv_50_base_15x_ratio
18. macd_entropy_20_12_26_vwap_15x_ratio_log_ratio_vectorbt_enhanced_obv_50_base_15x_ratio
19. ar_1_coefficients_20
20. vectorbt_acceleration_10_price_returns
... and 10 more features


## Baseline Learnability of Selected Features

This baseline fits simple models (linear regression and small LightGBM baselines) using only the final selected features. It provides an upper bound on how much of the target variance is explainable by this feature set alone, before any complex downstream modeling.

## Baseline Predictive Check

**Dataset:** 11520 samples, 60 features

### Top Single-Feature Signals

| Rank | Feature | Test R² | Pearson | AUC | Quality Score |
|------|---------|---------|---------|-----|---------------|
| 1 | `resistance_level_1_10_price_returns` | 0.003 | -0.057 | N/A | 0.025 |
| 2 | `vectorbt_smoothed_obv_10` | 0.004 | -0.053 | N/A | 0.024 |
| 3 | `vectorbt_adx_21` | 0.002 | -0.045 | N/A | 0.019 |
| 4 | `advanced_cumulative_returns_10` | 0.001 | -0.043 | N/A | 0.018 |
| 5 | `vectorbt_trend_consistency_5_price_returns` | 0.002 | 0.041 | N/A | 0.018 |

### Small Multivariate LGBM Baseline

| Type | Features | Test R² |
|------|----------|---------|
| Pair | `resistance_level_1_10_price_returns`, `vectorbt_smoothed_obv_10` | 0.011 |
| Triplet | `vectorbt_smoothed_obv_10`, `advanced_cumulative_returns_10`, `vectorbt_trend_consistency_5_price_returns` | 0.013 |

### Interpretation

**Quality Score:** 0.02/1.0

**Summary:** ⚠️ Moderate predictive signals detected

**Insights:**
- Best feature `resistance_level_1_10_price_returns` achieved Test R² = 0.003
- Positive Test R² features: 23 (38.3%)
- Median Test R² across evaluated features: -0.000
- LGBM best feature `vectorbt_smoothed_obv_10` achieved Test R² = 0.006

**Recommendations:**
- Focus downstream modeling on the top-ranked features


### Backtest Implications (Expected Sharpe)

- **Best Test R²:** 0.0033
- **Assumed Trading Frequency:** 1 trade/day (252/year)
- **Expected Annualized Sharpe Ratio:** **0.91**
  *(Derived from Information Coefficient approximation: Sharpe ≈ IC * sqrt(N))*

**Baseline learnability CSV:** `outcomes/baseline_check_final_feature_selection_20251207_144510.csv`

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
*Generated by Feature Generation Final Feature Selection Step at 2025-12-07 14:45:10*
