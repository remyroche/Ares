# Feature Generation Report

## Summary
- **Symbol**: ETHUSDT
- **Timeframe**: 15m
- **Direction**: DirectionType.LONGS
- **Generated At**: 2025-10-18 00:39:58
- **Status**: ✅ SUCCESS

## Feature Generation Results
- **Total Features Generated**: 276
- **Generation Time**: 2249.868 seconds
- **Memory Usage**: 1232.67 MB
- **Cache Hit**: No

## Feature Categories
- **Volatility**: Generated

## Feature Names
1. log_returns_1_price_returns_auto_optimized
2. simple_returns_1_price_returns_auto_optimized
3. log_returns_10_price_returns_auto_optimized
4. log_returns_5_price_returns_auto_optimized
5. simple_returns_5_price_returns_auto_optimized
6. simple_returns_10_price_returns_auto_optimized
7. cumulative_returns_10_price_returns_auto_optimized
8. cumulative_returns_20_price_returns_auto_optimized
9. returns_volatility_20_price_returns_auto_optimized
10. returns_skewness_20_price_returns_auto_optimized
11. rolling_returns_20_price_returns_auto_optimized
12. rolling_returns_10_price_returns_auto_optimized
13. advanced_cumulative_returns_10_auto_optimized
14. sharpe_ratio_20_0.0_price_returns_auto_optimized
15. returns_kurtosis_20_price_returns_auto_optimized
16. advanced_cumulative_returns_20_auto_optimized
17. ljung_box_pvalue_20_10_auto_optimized
18. ar_1_coefficients_20_auto_optimized
19. momentum_features_auto_optimized
20. rolling_zscore_returns_20_auto_optimized
21. williams_r_14_price_returns_auto_optimized
22. rsi_14_returns_vwap_auto_optimized
23. macd_12_26_9_returns_vwap_auto_optimized
24. stochastic_14_3_price_returns_auto_optimized
25. roc_14_price_returns_auto_optimized
26. momentum_14_price_returns_auto_optimized
27. stochastic_21_3_price_returns_auto_optimized
28. rsi_21_returns_vwap_auto_optimized
29. rsi_30_returns_vwap_auto_optimized
30. williams_r_21_price_returns_auto_optimized
31. momentum_21_price_returns_auto_optimized
32. roc_21_price_returns_auto_optimized
33. williams_r_30_price_returns_auto_optimized
34. momentum_30_price_returns_auto_optimized
35. stochastic_30_3_price_returns_auto_optimized
36. roc_30_price_returns_auto_optimized
37. macd_delta_12_26_9_auto_optimized
38. stochastic_kd_14_3_auto_optimized
39. rsi_zscore_14_20_auto_optimized
40. momentum_endpoints_sma_20_auto_optimized
41. analyst_momentum_5m_auto_optimized
42. donchian_channel_20_auto_optimized
43. advanced_momentum_5_20_auto_optimized
44. advanced_momentum_10_30_auto_optimized
45. analyst_momentum_alignment_auto_optimized
46. analyst_momentum_1h_auto_optimized
47. volume_sma_5_auto_optimized
48. analyst_momentum_15m_auto_optimized
49. volume_sma_10_auto_optimized
50. volume_ema_5_auto_optimized
... and 226 more features

## Optimization Statistics
- **total_generators**: 341
- **auto_optimized_generators**: 341
- **total_optimizations**: 276
- **total_optimization_time**: 4241.106933355331
- **memory_savings_mb**: 74040.12329864502
- **optimization_levels**: {'balanced': 341}

## Data Quality
- **Data Shape**: (1162368, 276)
- **Success**: Yes

## Technical Details
- **Feature Data Type**: DataFrame
- **Generated Features Type**: DataFrame
- **Metadata Available**: Yes

## Recommendations
- ✅ Feature generation completed successfully
- 📊 Consider analyzing feature importance for model training
- 🔍 Review feature categories for completeness
- 💾 Features are ready for model training pipeline
