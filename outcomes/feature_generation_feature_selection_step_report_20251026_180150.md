# Feature Selection Report

**Generated:** 2025-10-26 18:01:50
**Step:** feature_generation_feature_selection_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** light

## Feature Selection Results

- **Original Features:** 334
- **Selected Features:** 300
- **Selection Ratio:** 89.82%
- **Selection Method:** univariate_selection_optimized
- **Optimization Used:** True

## Performance Metrics

- **Total Execution Time:** 0.95s
- **Feature Loading Time:** 0.16s
- **Selection Time:** 0.79s

## Selected Features

The following features were selected for the model:

1. rsi_14_returns_vwap
2. rsi_21_returns_vwap
3. macd_12_26_9_returns_vwap
4. momentum_endpoints_sma_20
5. rsi_30_returns_vwap
6. volume_ema_5
7. volume_sma_10
8. macd_delta_12_26_9
9. volume_sma_5
10. rsi_zscore_14_20
11. volume_ema_10
12. volume_ema_20
13. volume_sma_20
14. volume_ema_50
15. volume_sma_50
16. sma_10_returns_vwap
17. sma_5_returns_vwap
18. sma_50_returns_vwap
19. sma_20_returns_vwap
20. sma_100_returns_vwap
21. ema_50_returns_vwap
22. ema_12_returns_vwap
23. ema_26_returns_vwap
24. dema_21_price_returns
25. tema_21_price_returns
26. rsi_entropy_20_14
27. macd_entropy_20_12_26
28. trend_persistence
29. log_returns_5_price_returns
30. log_returns_1_price_returns
31. simple_returns_5_price_returns
32. log_returns_10_price_returns
33. simple_returns_1_price_returns
34. cumulative_returns_20_price_returns
35. cumulative_returns_10_price_returns
36. rolling_returns_10_price_returns
37. simple_returns_10_price_returns
38. rolling_returns_20_price_returns
39. returns_kurtosis_20_price_returns
40. returns_skewness_20_price_returns
41. advanced_cumulative_returns_10
42. advanced_cumulative_returns_20
43. sharpe_ratio_20_0.0_price_returns
44. ar_1_coefficients_20
45. rolling_zscore_returns_20
46. williams_r_14_price_returns
47. ljung_box_pvalue_20_10
48. stochastic_14_3_price_returns
49. williams_r_21_price_returns
50. stochastic_21_3_price_returns

... and 250 more features

## Summary

Feature selection completed successfully using univariate_selection_optimized method. Selected 300 out of 334 features (89.8% selection ratio). Total execution time: 0.95 seconds.
