# Final Feature Selection Report

**Generated:** 2025-10-29 22:53:16
**Step:** feature_generation_final_feature_selection_step

## Configuration

- **Symbol:** N/A
- **Exchange:** N/A
- **Timeframe:** N/A
- **Execution Mode:** analyst
- **Feature Count Targets:** N/A
- **Selection Method:** N/A
- **Optimization Enabled:** True

## Feature Selection Results

- **60 Features Set:** 60 features selected
- **50 Features Set:** 50 features selected
- **40 Features Set:** 40 features selected

- **Total Feature Sets:** 3

## SHAP Analysis Summary

- **SHAP Analyses Generated:** 3

## Selected Features by Set

### 60 Features Set (60 features)

1. fractal_dimension_base_27x_ratio (SHAP: 0.0447)
2. candlestick_harami_cross_pattern_base_6x_ratio_log_ratio_candlestick_doji_pattern_base_3x_ratio (SHAP: 0.0075)
3. fibonacci_0.786_20_price_returns (SHAP: 0.0098)
4. volume_std_50 (SHAP: 0.0115)
5. volume_price_trend (SHAP: 0.0322)
6. fibonacci_0.236_5_price_returns (SHAP: 0.0019)
7. volume_price_trend_base (SHAP: 0.0017)
8. fibonacci_0.786_10_price_returns (SHAP: 0.0027)
9. fibonacci_0.5_5_price_returns (SHAP: 0.0044)
10. fibonacci_0.786_10_price_returns_base (SHAP: 0.0037)
11. vectorbt_smoothed_obv_50 (SHAP: 0.0026)
12. candlestick_harami_cross_pattern_base_6x_ratio_log_candlestick_doji_pattern_base_3x_ratio (SHAP: 0.0027)
13. fibonacci_0.618_20_price_returns_base (SHAP: 0.0035)
14. fibonacci_0.618_10_price_returns (SHAP: 0.0045)
15. fibonacci_0.382_20_price_returns (SHAP: 0.0127)
16. fibonacci_0.786_5_price_returns (SHAP: 0.0039)
17. fibonacci_0.618_20_price_returns (SHAP: 0.0050)
18. fibonacci_0.236_5_price_returns_base (SHAP: 0.0028)
19. fibonacci_0.5_20_price_returns (SHAP: 0.0040)
20. fibonacci_0.5_10_price_returns (SHAP: 0.0050)
... and 40 more features

**Average SHAP Importance:** 0.0076

### 50 Features Set (50 features)

1. fractal_dimension_base_27x_ratio (SHAP: 0.0563)
2. volume_price_trend (SHAP: 0.0059)
3. fibonacci_0.5_5_price_returns (SHAP: 0.0099)
4. fibonacci_0.786_20_price_returns (SHAP: 0.0162)
5. candlestick_harami_cross_pattern_base_6x_ratio_log_ratio_candlestick_doji_pattern_base_3x_ratio (SHAP: 0.0253)
6. fibonacci_0.618_20_price_returns_base (SHAP: 0.0029)
7. fibonacci_0.236_5_price_returns (SHAP: 0.0027)
8. fibonacci_0.786_10_price_returns (SHAP: 0.0037)
9. candlestick_harami_cross_pattern_base_6x_ratio_log_candlestick_doji_pattern_base_3x_ratio (SHAP: 0.0047)
10. fibonacci_0.786_10_price_returns_base (SHAP: 0.0089)
11. fibonacci_0.5_20_price_returns (SHAP: 0.0116)
12. fibonacci_0.5_20_price_returns_base (SHAP: 0.0190)
13. fibonacci_0.382_20_price_returns (SHAP: 0.0091)
14. fibonacci_0.618_10_price_returns_base (SHAP: 0.0152)
15. fibonacci_0.236_20_price_returns (SHAP: 0.0055)
16. max_drawdown (SHAP: 0.0074)
17. candlestick_doji_pattern_base_27x_ratio (SHAP: 0.0028)
18. candlestick_harami_cross_pattern_base_3x_ratio (SHAP: 0.0094)
19. resistance_level_2_20_price_returns (SHAP: 0.0030)
20. resistance_level_4_20_price_returns (SHAP: 0.0033)
... and 30 more features

**Average SHAP Importance:** 0.0086

### 40 Features Set (40 features)

1. candlestick_harami_cross_pattern_base_6x_ratio_log_ratio_candlestick_doji_pattern_base_3x_ratio (SHAP: 0.0533)
2. fibonacci_0.618_20_price_returns_base (SHAP: 0.0171)
3. fibonacci_0.786_20_price_returns (SHAP: 0.0117)
4. fibonacci_0.786_10_price_returns (SHAP: 0.0217)
5. fibonacci_0.786_10_price_returns_base (SHAP: 0.0300)
6. fibonacci_0.236_20_price_returns (SHAP: 0.0041)
7. max_drawdown (SHAP: 0.0086)
8. candlestick_harami_cross_pattern_base_3x_ratio (SHAP: 0.0060)
9. candlestick_doji_pattern_base_27x_ratio (SHAP: 0.0048)
10. support_level_4_5_price_returns (SHAP: 0.0038)
11. resistance_level_2_20_price_returns (SHAP: 0.0034)
12. resistance_level_1_20_price_returns (SHAP: 0.0165)
13. resistance_level_4_5_price_returns (SHAP: 0.0078)
14. support_level_2_5_price_returns (SHAP: 0.0023)
15. support_level_3_5_price_returns (SHAP: 0.0026)
16. support_level_1_5_price_returns (SHAP: 0.0025)
17. resistance_level_2_5_price_returns (SHAP: 0.0103)
18. resistance_level_5_20_price_returns (SHAP: 0.0029)
19. resistance_level_3_5_price_returns (SHAP: 0.0239)
20. resistance_level_3_20_price_returns (SHAP: 0.0209)
... and 20 more features

**Average SHAP Importance:** 0.0103


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.2856
- **Max Correlation:** 1.0000
- **Min Correlation:** 0.0001
- **High Correlation Pairs:** 74
- **Correlation Threshold:** 0.8

### Redundancy Detection

- **Redundancy Score:** 1.0356
- **Redundant Features:** 60
- **Total Features:** 60
- **Correlation Redundant Pairs:** 63
- **Mutual Info Redundant Pairs:** 1770
- **Low Variance Features:** 0

### Stability Analysis

- **Average Stability:** 0.2933
- **Stable Features:** 12
- **Stability Threshold:** 0.8
- **Time Windows:** 5

### Cross-Validation Analysis

- **Average Consistency:** 0.3100
- **Consistent Features:** 19
- **Consistency Threshold:** 0.6
- **CV Folds:** 5

### Baseline Comparison

- **Improvement Ratio:** 2.64x
- **Selected Features Avg Score:** 0.701123
- **Baseline Avg Score:** 0.265575
- **Baseline Trials:** 10
- **Features Compared:** 60

## Performance Metrics

- **Execution Time:** N/A seconds
- **Optimization Enabled:** Yes
- **Hardware Optimization:** Yes

## Optimization Details

- **VectorBT Optimization:** Enabled
- **Rolling Optimizer:** Available
- **Hardware Manager:** Available

## Generated Artifacts

- **Feature Sets:** 3
- **Feature DataFrames:** 3
- **SHAP Analyses:** 3
- **Metadata Files:** 2
- **Total Artifacts:** 13

## Summary

Final feature selection completed successfully. Generated 3 optimized feature sets with comprehensive SHAP analysis and metadata. All artifacts saved in both pickle and markdown formats.

---
*Generated by Feature Generation Final Feature Selection Step at 2025-10-29 22:53:16*
