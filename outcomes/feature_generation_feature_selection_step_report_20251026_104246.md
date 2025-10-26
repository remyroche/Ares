# Feature Selection Report

**Generated:** 2025-10-26 10:42:46
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

- **Total Execution Time:** 2.22s
- **Feature Loading Time:** 0.89s
- **Selection Time:** 1.33s

## Selected Features

The following features were selected for the model:

1. rsi_14_returns_vwap
2. rsi_21_returns_vwap
3. macd_12_26_9_returns_vwap
4. rsi_30_returns_vwap
5. momentum_endpoints_sma_20
6. macd_delta_12_26_9
7. rsi_zscore_14_20
8. volume_ema_5
9. volume_sma_5
10. volume_sma_10
11. volume_sma_20
12. volume_sma_50
13. volume_ema_10
14. volume_ema_20
15. volume_ema_50
16. sma_5_returns_vwap
17. sma_10_returns_vwap
18. sma_20_returns_vwap
19. sma_50_returns_vwap
20. sma_100_returns_vwap
21. ema_12_returns_vwap
22. ema_26_returns_vwap
23. ema_50_returns_vwap
24. dema_21_price_returns
25. tema_21_price_returns
26. rsi_entropy_20_14
27. trend_persistence
28. macd_entropy_20_12_26
29. log_returns_10_price_returns
30. log_returns_1_price_returns
31. log_returns_5_price_returns
32. simple_returns_1_price_returns
33. simple_returns_5_price_returns
34. simple_returns_10_price_returns
35. rolling_returns_10_price_returns
36. cumulative_returns_20_price_returns
37. cumulative_returns_10_price_returns
38. rolling_returns_20_price_returns
39. advanced_cumulative_returns_10
40. returns_kurtosis_20_price_returns
41. returns_skewness_20_price_returns
42. advanced_cumulative_returns_20
43. sharpe_ratio_20_0.0_price_returns
44. rolling_zscore_returns_20
45. stochastic_14_3_price_returns
46. ljung_box_pvalue_20_10
47. ar_1_coefficients_20
48. williams_r_14_price_returns
49. roc_14_price_returns
50. stochastic_30_3_price_returns

... and 250 more features

## Summary

Feature selection completed successfully using univariate_selection_optimized method. Selected 300 out of 334 features (89.8% selection ratio). Total execution time: 2.22 seconds.
