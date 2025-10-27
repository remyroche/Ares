# Final Feature Selection Report

**Generated:** 2025-10-27 22:48:33
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

1. vectorbt_acceleration_consistency_10_20_price_returns (SHAP: 0.0081)
2. vectorbt_acceleration_trend_strength_10_20_price_returns (SHAP: 0.0208)
3. cmf_20_base (SHAP: 0.0134)
4. volume_volatility_elasticity_20 (SHAP: 0.0088)
5. fibonacci_0.786_10_price_returns (SHAP: 0.0057)
6. volume_volatility_elasticity_20_vwap (SHAP: 0.0052)
7. fibonacci_0.786_20_price_returns (SHAP: 0.0035)
8. vectorbt_acceleration_volatility_10_20_price_returns (SHAP: 0.0056)
9. cmo_14_returns_vwap (SHAP: 0.0049)
10. stochastic_21_3_price_returns (SHAP: 0.0026)
11. vectorbt_enhanced_ad_line_10 (SHAP: 0.0089)
12. price_volume_oscillator_5_15 (SHAP: 0.0053)
13. macd_delta_12_26_9 (SHAP: 0.0119)
14. volume_std_20 (SHAP: 0.0080)
15. vectorbt_volatility_comprehensive_30 (SHAP: 0.0051)
16. fibonacci_0.618_20_price_returns_vwap (SHAP: 0.0041)
17. log_returns_1_price_returns (SHAP: 0.0145)
18. vectorbt_enhanced_ad_line_20_base_x_vectorbt_smoothed_obv_10_base (SHAP: 0.0049)
19. vectorbt_trend_consistency_10_price_returns (SHAP: 0.0058)
20. spectral_entropy_20 (SHAP: 0.0089)
... and 40 more features

**Average SHAP Importance:** 0.0062

### 50 Features Set (50 features)

1. vectorbt_acceleration_consistency_10_20_price_returns (SHAP: 0.0041)
2. vectorbt_acceleration_trend_strength_10_20_price_returns (SHAP: 0.0221)
3. vectorbt_volume_acceleration_5_volume_returns (SHAP: 0.0202)
4. vectorbt_enhanced_ad_line_10 (SHAP: 0.0124)
5. fibonacci_0.786_10_price_returns (SHAP: 0.0152)
6. price_volume_oscillator_5_15 (SHAP: 0.0145)
7. vectorbt_acceleration_volatility_10_20_price_returns (SHAP: 0.0031)
8. vectorbt_acceleration_trend_strength_10_10_price_returns (SHAP: 0.0213)
9. macd_delta_12_26_9 (SHAP: 0.0082)
10. volume_volatility_elasticity_20 (SHAP: 0.0233)
11. macd_entropy_20_12_26 (SHAP: 0.0032)
12. fibonacci_0.618_20_price_returns_vwap (SHAP: 0.0031)
13. volume_std_20 (SHAP: 0.0032)
14. volume_volatility_elasticity_20_vwap (SHAP: 0.0162)
15. enhanced_volatility_30 (SHAP: 0.0039)
16. shannon_entropy_20_10 (SHAP: 0.0097)
17. spectral_entropy_20 (SHAP: 0.0090)
18. vectorbt_enhanced_ad_line_20_base_minus_returns_volatility_20_price_returns_base (SHAP: 0.0063)
19. vectorbt_volatility_comprehensive_30 (SHAP: 0.0096)
20. log_returns_1_price_returns (SHAP: 0.0028)
... and 30 more features

**Average SHAP Importance:** 0.0074

### 40 Features Set (40 features)

1. vectorbt_acceleration_consistency_10_20_price_returns (SHAP: 0.0053)
2. vectorbt_acceleration_trend_strength_10_20_price_returns (SHAP: 0.0275)
3. fibonacci_0.786_10_price_returns (SHAP: 0.0145)
4. vectorbt_enhanced_ad_line_20_base_x_vectorbt_smoothed_obv_10_base (SHAP: 0.0070)
5. aroon_25_returns_vwap (SHAP: 0.0031)
6. vectorbt_acceleration_volatility_10_20_price_returns (SHAP: 0.0069)
7. spectral_entropy_20 (SHAP: 0.0068)
8. order_flow_imbalance_20_base (SHAP: 0.0107)
9. fibonacci_0.786_20_price_returns (SHAP: 0.0064)
10. rsi_30_returns_vwap (SHAP: 0.0067)
11. volume_volatility_elasticity_20 (SHAP: 0.0049)
12. volume_volatility_elasticity_20_vwap (SHAP: 0.0076)
13. price_volume_oscillator_5_15 (SHAP: 0.0025)
14. support_level_1_5_price_returns (SHAP: 0.0089)
15. volume_entropy_ma_20_10_volume_returns (SHAP: 0.0134)
16. volume_std_20 (SHAP: 0.0054)
17. returns_skewness_20_price_returns (SHAP: 0.0056)
18. vectorbt_acceleration_trend_strength_10_10_price_returns (SHAP: 0.0078)
19. macd_delta_12_26_9 (SHAP: 0.0208)
20. fibonacci_0.618_20_price_returns_vwap (SHAP: 0.0046)
... and 20 more features

**Average SHAP Importance:** 0.0084


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.1788
- **Max Correlation:** 1.0000
- **Min Correlation:** 0.0002
- **High Correlation Pairs:** 56
- **Correlation Threshold:** 0.8

### Redundancy Detection

- **Redundancy Score:** 0.8655
- **Redundant Features:** 56
- **Total Features:** 60
- **Correlation Redundant Pairs:** 33
- **Mutual Info Redundant Pairs:** 1499
- **Low Variance Features:** 0

### Stability Analysis

- **Average Stability:** 0.0700
- **Stable Features:** 0
- **Stability Threshold:** 0.8
- **Time Windows:** 5

### Cross-Validation Analysis

- **Average Consistency:** 0.0500
- **Consistent Features:** 2
- **Consistency Threshold:** 0.6
- **CV Folds:** 5

### Baseline Comparison

- **Improvement Ratio:** 0.79x
- **Selected Features Avg Score:** 0.086100
- **Baseline Avg Score:** 0.109372
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
*Generated by Feature Generation Final Feature Selection Step at 2025-10-27 22:48:33*
