# Final Feature Selection Report

**Generated:** 2025-11-08 19:10:44
**Step:** feature_generation_final_feature_selection_step

## Configuration

- **Symbol:** N/A
- **Exchange:** N/A
- **Timeframe:** N/A
- **Execution Mode:** analyst
- **Feature Count Targets:** N/A
- **Selection Method:** permutation ✅
- **Importance Type:** Permutation (captures feature interactions, not just Gini splits) 📊
- **Optimization Enabled:** True

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

- **Total Feature Sets:** 3

## Selected Features by Set

### 60 Features Set (60 features)

1. vectorbt_acceleration_momentum_5_10_price_returns
2. donchian_channel_20
3. volume_vwap_20_vwap_3x_ratio
4. enhanced_volatility_10
5. pivot_point_5_price_returns
6. vectorbt_momentum_comprehensive_9
7. price_range_pct
8. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_div_candlestick_piercing_line_pattern_vwap_6x_ratio
9. fibonacci_0.618_5_price_returns
10. fibonacci_0.786_20_price_returns
11. tema_21_price_returns
12. fibonacci_0.5_20_price_returns
13. wavelet_energy_base_6x_ratio
14. volume_accumulation_distribution
15. vectorbt_garman_klass_volatility_20
16. vectorbt_yang_zhang_volatility_50
17. fibonacci_0.5_20_price_returns_base
18. candlestick_engulfing_pattern_base_9x_ratio
19. candlestick_harami_cross_pattern_base_6x_ratio
20. vectorbt_enhanced_ad_line_20_base_27x_ratio_div_vectorbt_smoothed_obv_10_base_27x_ratio
... and 40 more features

### 50 Features Set (50 features)

1. enhanced_volatility_10
2. tema_21_price_returns
3. volume_accumulation_distribution
4. fibonacci_0.618_5_price_returns
5. price_range_pct
6. vectorbt_enhanced_obv_50_base
7. vectorbt_smoothed_obv_50
8. vectorbt_rogers_satchell_volatility_20
9. wavelet_energy_base_6x_ratio
10. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_minus_fibonacci_0.786_10_price_returns_vwap
11. fibonacci_0.5_20_price_returns_trend_adj
12. vectorbt_rogers_satchell_volatility_30_base_9x_ratio_x_wavelet_energy_base_9x_ratio
13. fibonacci_0.5_20_price_returns_base
14. vectorbt_yang_zhang_volatility_50
15. vectorbt_garman_klass_volatility_20
16. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_div_candlestick_piercing_line_pattern_vwap_6x_ratio
17. vectorbt_enhanced_ad_line_20_base_27x_ratio_div_vectorbt_smoothed_obv_10_base_27x_ratio
18. williams_r_14_price_returns
19. fractal_dimension_vwap_3x_ratio
20. volume_entropy_10_volume_returns
... and 30 more features

### 40 Features Set (40 features)

1. enhanced_volatility_10
2. fibonacci_0.618_5_price_returns
3. tema_21_price_returns
4. volume_accumulation_distribution
5. price_range_pct
6. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_minus_fibonacci_0.786_10_price_returns_vwap
7. vectorbt_rogers_satchell_volatility_50
8. fibonacci_0.5_20_price_returns_trend_adj
9. vectorbt_trend_consistency_50_price_returns
10. log_returns_5_price_returns
11. vectorbt_rogers_satchell_volatility_20
12. vectorbt_garman_klass_volatility_20
13. vectorbt_rogers_satchell_volatility_30_base_9x_ratio_x_wavelet_energy_base_9x_ratio
14. volume_vwap_20_base_27x_ratio
15. fibonacci_0.618_20_price_returns_vwap_27x_ratio_minus_cycle_length_vwap_3x_ratio
16. fractal_dimension_vwap_9x_ratio
17. vectorbt_yang_zhang_volatility_50
18. returns_skewness_20_price_returns
19. vectorbt_rogers_satchell_volatility_30_base_9x_ratio
20. vectorbt_parkinson_volatility_50
... and 20 more features


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.1566
- **Max Correlation:** 1.0000
- **Min Correlation:** 0.0000
- **High Correlation Pairs:** 38
- **Correlation Threshold:** 0.8

### Redundancy Detection

- **Redundancy Score:** 0.9802
- **Redundant Features:** 59
- **Total Features:** 60
- **Correlation Redundant Pairs:** 26
- **Mutual Info Redundant Pairs:** 1709
- **Low Variance Features:** 1

### Stability Analysis

- **Average Stability:** 0.0633
- **Stable Features:** 0
- **Stability Threshold:** 0.8
- **Time Windows:** 5

### Cross-Validation Analysis

- **Average Consistency:** 0.0467
- **Consistent Features:** 0
- **Consistency Threshold:** 0.6
- **CV Folds:** 5

### Baseline Comparison

- **Improvement Ratio:** 1.23x
- **Selected Features Avg Score:** 0.036480
- **Baseline Avg Score:** 0.029539
- **Baseline Trials:** 10
- **Features Compared:** 60

## Performance Metrics

- **Execution Time:** N/A seconds
- **Optimization Enabled:** Yes
- **Hardware Optimization:** No

## Optimization Details

- **VectorBT Optimization:** Enabled
- **Rolling Optimizer:** Available
- **Hardware Manager:** Available

## Generated Artifacts

- **Feature Sets:** 3
- **Feature DataFrames:** 3
- **SHAP Analyses:** 0
- **Metadata Files:** 2
- **Total Artifacts:** 10

## Summary

Final feature selection completed successfully. Generated 3 optimized feature sets with comprehensive SHAP analysis and metadata. All artifacts saved in both pickle and markdown formats.

---
*Generated by Feature Generation Final Feature Selection Step at 2025-11-08 19:10:44*
