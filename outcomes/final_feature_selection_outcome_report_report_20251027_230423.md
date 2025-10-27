# Final Feature Selection Report

**Generated:** 2025-10-27 23:04:23
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

1. vectorbt_acceleration_consistency_10_20_price_returns (SHAP: 0.0077)
2. returns_kurtosis_20_price_returns (SHAP: 0.0233)
3. pfe_12_returns_vwap (SHAP: 0.0143)
4. vectorbt_volume_acceleration_5_volume_returns (SHAP: 0.0062)
5. volume_std_20 (SHAP: 0.0070)
6. shannon_entropy_20_10 (SHAP: 0.0034)
7. vectorbt_acceleration_trend_strength_10_10_price_returns (SHAP: 0.0049)
8. vectorbt_enhanced_ad_line_10 (SHAP: 0.0066)
9. spectral_entropy_20 (SHAP: 0.0045)
10. volume_volatility_elasticity_20_vwap (SHAP: 0.0023)
11. fibonacci_0.786_20_price_returns (SHAP: 0.0144)
12. fibonacci_0.786_10_price_returns (SHAP: 0.0058)
13. vectorbt_enhanced_ad_line_20_base_x_vectorbt_smoothed_obv_10_base (SHAP: 0.0132)
14. vectorbt_enhanced_ad_line_20_base_div_vectorbt_smoothed_obv_10_base (SHAP: 0.0089)
15. volume_volatility_elasticity_20 (SHAP: 0.0054)
16. order_flow_imbalance_20_base (SHAP: 0.0028)
17. enhanced_volatility_30 (SHAP: 0.0111)
18. vectorbt_acceleration_volatility_10_20_price_returns (SHAP: 0.0073)
19. macd_delta_12_26_9 (SHAP: 0.0069)
20. vectorbt_enhanced_ad_line_20_base_minus_returns_volatility_20_price_returns_base (SHAP: 0.0085)
... and 40 more features

**Average SHAP Importance:** 0.0067

### 50 Features Set (50 features)

1. vectorbt_acceleration_consistency_10_20_price_returns (SHAP: 0.0041)
2. vectorbt_acceleration_trend_strength_10_20_price_returns (SHAP: 0.0197)
3. price_volume_oscillator_5_15 (SHAP: 0.0185)
4. volume_volatility_elasticity_20 (SHAP: 0.0102)
5. vectorbt_enhanced_ad_line_10 (SHAP: 0.0144)
6. vectorbt_acceleration_volatility_10_20_price_returns (SHAP: 0.0064)
7. vectorbt_enhanced_ad_line_20_base_x_vectorbt_smoothed_obv_10_base (SHAP: 0.0029)
8. shannon_entropy_20_10 (SHAP: 0.0185)
9. cmo_14_returns_vwap (SHAP: 0.0069)
10. rsi_14_returns_vwap (SHAP: 0.0241)
11. vectorbt_acceleration_trend_strength_10_10_price_returns (SHAP: 0.0041)
12. volume_volatility_elasticity_20_vwap (SHAP: 0.0067)
13. order_flow_imbalance_20_base (SHAP: 0.0024)
14. returns_skewness_20_price_returns (SHAP: 0.0126)
15. spectral_entropy_20 (SHAP: 0.0054)
16. fibonacci_0.786_10_price_returns (SHAP: 0.0141)
17. log_returns_1_price_returns (SHAP: 0.0058)
18. fibonacci_0.618_20_price_returns_vwap (SHAP: 0.0065)
19. macd_entropy_20_12_26 (SHAP: 0.0124)
20. enhanced_volatility_30 (SHAP: 0.0022)
... and 30 more features

**Average SHAP Importance:** 0.0074

### 40 Features Set (40 features)

1. vectorbt_acceleration_consistency_10_20_price_returns (SHAP: 0.0053)
2. vectorbt_acceleration_trend_strength_10_20_price_returns (SHAP: 0.0239)
3. fibonacci_0.786_10_price_returns (SHAP: 0.0131)
4. price_volume_oscillator_5_15 (SHAP: 0.0083)
5. shannon_entropy_20_10 (SHAP: 0.0040)
6. fibonacci_0.786_20_price_returns (SHAP: 0.0052)
7. volume_volatility_elasticity_20 (SHAP: 0.0057)
8. rsi_14_returns_vwap (SHAP: 0.0100)
9. support_level_2_5_price_returns (SHAP: 0.0065)
10. vectorbt_acceleration_volatility_10_20_price_returns (SHAP: 0.0057)
11. enhanced_volatility_30 (SHAP: 0.0085)
12. vectorbt_acceleration_trend_strength_10_10_price_returns (SHAP: 0.0042)
13. order_flow_imbalance_20_base (SHAP: 0.0026)
14. volume_std_20 (SHAP: 0.0146)
15. vectorbt_enhanced_ad_line_20_base_x_vectorbt_smoothed_obv_10_base (SHAP: 0.0098)
16. vectorbt_rogers_satchell_volatility_30 (SHAP: 0.0050)
17. macd_entropy_20_12_26 (SHAP: 0.0053)
18. support_level_5_5_price_returns (SHAP: 0.0064)
19. cmo_14_returns_vwap (SHAP: 0.0162)
20. volume_price_correlation_10 (SHAP: 0.0045)
... and 20 more features

**Average SHAP Importance:** 0.0082


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.1588
- **Max Correlation:** 1.0000
- **Min Correlation:** 0.0000
- **High Correlation Pairs:** 41
- **Correlation Threshold:** 0.8

### Redundancy Detection

- **Redundancy Score:** 0.8554
- **Redundant Features:** 56
- **Total Features:** 60
- **Correlation Redundant Pairs:** 30
- **Mutual Info Redundant Pairs:** 1484
- **Low Variance Features:** 0

### Stability Analysis

- **Average Stability:** 0.0833
- **Stable Features:** 0
- **Stability Threshold:** 0.8
- **Time Windows:** 5

### Cross-Validation Analysis

- **Average Consistency:** 0.0767
- **Consistent Features:** 3
- **Consistency Threshold:** 0.6
- **CV Folds:** 5

### Baseline Comparison

- **Improvement Ratio:** 0.85x
- **Selected Features Avg Score:** 0.094970
- **Baseline Avg Score:** 0.111809
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
*Generated by Feature Generation Final Feature Selection Step at 2025-10-27 23:04:23*
