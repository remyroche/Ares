# Final Feature Selection Report

**Generated:** 2025-10-27 20:32:17
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

1. vectorbt_enhanced_ad_line_20_base_minus_vectorbt_smoothed_obv_10_base (SHAP: 0.0108)
2. volume_std_50 (SHAP: 0.0078)
3. ema_50_returns_vwap (SHAP: 0.0784)
4. volume_price_trend_base (SHAP: 0.0896)
5. vectorbt_volume_weighted_ad_line_50 (SHAP: 0.0102)
6. fibonacci_0.382_20_price_returns (SHAP: 0.0110)
7. vectorbt_smoothed_obv_20 (SHAP: 0.0092)
8. vectorbt_enhanced_ad_line_20_base_x_returns_volatility_20_price_returns_base (SHAP: 0.0025)
9. apo_12_26_returns_vwap (SHAP: 0.0013)
10. vectorbt_enhanced_ad_line_20_base_div_returns_volatility_20_price_returns_base (SHAP: 0.0018)
11. returns_volatility_20_price_returns (SHAP: 0.0059)
12. fibonacci_0.382_10_price_returns (SHAP: 0.0032)
13. volume_vwap_50 (SHAP: 0.0040)
14. max_drawdown (SHAP: 0.0042)
15. vectorbt_enhanced_ad_line_20_base_x_vectorbt_rogers_satchell_volatility_30_base (SHAP: 0.0085)
16. vectorbt_enhanced_ad_line_50 (SHAP: 0.0059)
17. vectorbt_enhanced_obv_10 (SHAP: 0.0036)
18. returns_kurtosis_20_price_returns (SHAP: 0.0026)
19. volume_price_trend (SHAP: 0.0037)
20. vectorbt_rogers_satchell_volatility_50 (SHAP: 0.0076)
... and 40 more features

**Average SHAP Importance:** 0.0080

### 50 Features Set (50 features)

1. vectorbt_enhanced_ad_line_20_base_minus_vectorbt_smoothed_obv_10_base (SHAP: 0.0120)
2. fibonacci_0.382_20_price_returns (SHAP: 0.0071)
3. vectorbt_volume_weighted_ad_line_50 (SHAP: 0.0800)
4. volume_std_50 (SHAP: 0.0808)
5. volume_price_trend_base (SHAP: 0.0042)
6. ema_50_returns_vwap (SHAP: 0.0127)
7. vectorbt_smoothed_obv_20 (SHAP: 0.0039)
8. vectorbt_enhanced_ad_line_20_base_x_returns_volatility_20_price_returns_base (SHAP: 0.0049)
9. max_drawdown (SHAP: 0.0059)
10. vectorbt_enhanced_ad_line_20_base_x_vectorbt_rogers_satchell_volatility_30_base (SHAP: 0.0034)
11. vectorbt_smoothed_obv_50 (SHAP: 0.0073)
12. vectorbt_enhanced_obv_10 (SHAP: 0.0037)
13. enhanced_volatility_10 (SHAP: 0.0056)
14. volume_entropy_ma_5_10_volume_returns (SHAP: 0.0167)
15. volume_price_trend_vwap (SHAP: 0.0116)
16. vectorbt_enhanced_obv_20 (SHAP: 0.0091)
17. vectorbt_enhanced_obv_10_base (SHAP: 0.0083)
18. vectorbt_parkinson_volatility_50_vwap (SHAP: 0.0051)
19. volume_price_trend (SHAP: 0.0147)
20. vectorbt_parkinson_volatility_50 (SHAP: 0.0047)
... and 30 more features

**Average SHAP Importance:** 0.0089

### 40 Features Set (40 features)

1. vectorbt_enhanced_ad_line_20_base_minus_vectorbt_smoothed_obv_10_base (SHAP: 0.0125)
2. volume_std_50 (SHAP: 0.0098)
3. vectorbt_volume_weighted_ad_line_50 (SHAP: 0.0819)
4. ema_50_returns_vwap (SHAP: 0.0874)
5. vectorbt_smoothed_obv_20 (SHAP: 0.0059)
6. volume_price_trend_base (SHAP: 0.0143)
7. max_drawdown (SHAP: 0.0036)
8. vectorbt_enhanced_ad_line_20_base_x_vectorbt_rogers_satchell_volatility_30_base (SHAP: 0.0078)
9. vectorbt_smoothed_obv_50 (SHAP: 0.0033)
10. vectorbt_enhanced_obv_10 (SHAP: 0.0045)
11. volume_price_trend_vwap (SHAP: 0.0054)
12. volume_price_trend (SHAP: 0.0044)
13. enhanced_volatility_10 (SHAP: 0.0043)
14. vectorbt_enhanced_obv_10_base (SHAP: 0.0069)
15. vectorbt_smoothed_obv_10 (SHAP: 0.0043)
16. vectorbt_smoothed_obv_10_base (SHAP: 0.0068)
17. cvar (SHAP: 0.0032)
18. support_level_5_20_price_returns (SHAP: 0.0072)
19. support_level_3_20_price_returns (SHAP: 0.0053)
20. support_level_5_10_price_returns (SHAP: 0.0046)
... and 20 more features

**Average SHAP Importance:** 0.0112


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.3669
- **Max Correlation:** 1.0000
- **Min Correlation:** 0.0000
- **High Correlation Pairs:** 152
- **Correlation Threshold:** 0.8

### Redundancy Detection

- **Redundancy Score:** 0.9887
- **Redundant Features:** 58
- **Total Features:** 60
- **Correlation Redundant Pairs:** 97
- **Mutual Info Redundant Pairs:** 1653
- **Low Variance Features:** 0

### Stability Analysis

- **Average Stability:** 0.1800
- **Stable Features:** 7
- **Stability Threshold:** 0.8
- **Time Windows:** 5

### Cross-Validation Analysis

- **Average Consistency:** 0.1900
- **Consistent Features:** 11
- **Consistency Threshold:** 0.6
- **CV Folds:** 5

### Baseline Comparison

- **Improvement Ratio:** 1.95x
- **Selected Features Avg Score:** 0.221920
- **Baseline Avg Score:** 0.113561
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
*Generated by Feature Generation Final Feature Selection Step at 2025-10-27 20:32:17*
