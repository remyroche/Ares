# Final Feature Selection Report

**Generated:** 2025-11-08 19:02:30
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

1. candlestick_dark_cloud_cover_pattern_base_3x_ratio_minus_volume_entropy_ma_5_5_volume_returns_vwap
2. vectorbt_volume_weighted_ad_line_20
3. resistance_level_5_5_price_returns
4. rsi_21_returns_vwap
5. resistance_level_1_5_price_returns
6. vectorbt_enhanced_ad_line_20_base_27x_ratio_x_volume_vwap_20_base_3x_ratio
7. cycle_length_base_6x_ratio_log_ratio_candlestick_engulfing_pattern_base_3x_ratio
8. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio
9. resistance_level_4_5_price_returns
10. resistance_level_2_5_price_returns
11. candlestick_dark_cloud_cover_pattern_base_3x_ratio_x_volume_entropy_ma_5_5_volume_returns_vwap
12. resistance_level_3_5_price_returns
13. pivot_point_5_price_returns
14. wavelet_energy_vwap_6x_ratio
15. vectorbt_smoothed_obv_20
16. vectorbt_trend_consistency_50_price_returns
17. volume_vwap_20_base_6x_ratio
18. candlestick_doji_pattern
19. wavelet_energy_base_27x_ratio
20. cycle_length_base_27x_ratio_div_returns_volatility_20_price_returns_base
... and 40 more features

### 50 Features Set (50 features)

1. candlestick_dark_cloud_cover_pattern_base_3x_ratio_x_volume_entropy_ma_5_5_volume_returns_vwap
2. vectorbt_enhanced_ad_line_20_base_27x_ratio_x_volume_vwap_20_base_3x_ratio
3. resistance_level_3_5_price_returns
4. analyst_volume_pressure
5. cycle_length_base_27x_ratio_log_ratio_returns_volatility_20_price_returns_base
6. resistance_level_2_5_price_returns
7. pivot_point_5_price_returns
8. fractal_dimension_vwap_3x_ratio
9. resistance_level_1_5_price_returns
10. candlestick_dark_cloud_cover_pattern_base_3x_ratio
11. resistance_level_5_5_price_returns
12. rsi_21_returns_vwap
13. cycle_length_base_27x_ratio_div_returns_volatility_20_price_returns_base
14. resistance_level_4_5_price_returns
15. vectorbt_trend_consistency_50_price_returns
16. vectorbt_atr_50
17. fibonacci_0.786_5_price_returns
18. volume_vwap_20_vwap_9x_ratio
19. wavelet_energy_base_27x_ratio
20. volume_vwap_20_base_6x_ratio
... and 30 more features

### 40 Features Set (40 features)

1. candlestick_dark_cloud_cover_pattern_base_3x_ratio_minus_volume_entropy_ma_5_5_volume_returns_vwap
2. vectorbt_enhanced_ad_line_20_base_27x_ratio_x_volume_vwap_20_base_3x_ratio
3. resistance_level_1_5_price_returns
4. resistance_level_5_5_price_returns
5. resistance_level_3_5_price_returns
6. rsi_21_returns_vwap
7. resistance_level_4_5_price_returns
8. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio
9. vectorbt_trend_consistency_50_price_returns
10. resistance_level_2_5_price_returns
11. fibonacci_0.786_5_price_returns
12. analyst_volume_pressure
13. wavelet_energy_base_27x_ratio
14. pivot_point_5_price_returns
15. volume_vwap_20_base_6x_ratio
16. wavelet_energy_vwap_6x_ratio
17. cycle_length_base_27x_ratio_div_returns_volatility_20_price_returns_base
18. candlestick_harami_cross_pattern_base_6x_ratio
19. volume_vwap_20_vwap_9x_ratio
20. vectorbt_parabolic_sar_0.02_0.2
... and 20 more features


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.1937
- **Max Correlation:** 1.0000
- **Min Correlation:** 0.0000
- **High Correlation Pairs:** 63
- **Correlation Threshold:** 0.8

### Redundancy Detection

- **Redundancy Score:** 0.9542
- **Redundant Features:** 58
- **Total Features:** 60
- **Correlation Redundant Pairs:** 43
- **Mutual Info Redundant Pairs:** 1646
- **Low Variance Features:** 1

### Stability Analysis

- **Average Stability:** 0.0333
- **Stable Features:** 0
- **Stability Threshold:** 0.8
- **Time Windows:** 5

### Cross-Validation Analysis

- **Average Consistency:** 0.1433
- **Consistent Features:** 5
- **Consistency Threshold:** 0.6
- **CV Folds:** 5

### Baseline Comparison

- **Improvement Ratio:** 2.90x
- **Selected Features Avg Score:** 0.089288
- **Baseline Avg Score:** 0.030819
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
*Generated by Feature Generation Final Feature Selection Step at 2025-11-08 19:02:30*
