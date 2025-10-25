# Feature Selection Report

**Generated:** 2025-10-26 00:03:28
**Step:** feature_generation_feature_selection_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** light

## Feature Selection Results

- **Original Features:** 334
- **Selected Features:** 10
- **Selection Ratio:** 2.99%
- **Selection Method:** univariate_selection_optimized
- **Optimization Used:** True

## Performance Metrics

- **Total Execution Time:** 0.59s
- **Feature Loading Time:** 0.18s
- **Selection Time:** 0.41s

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

## Summary

Feature selection completed successfully using univariate_selection_optimized method. Selected 10 out of 334 features (3.0% selection ratio). Total execution time: 0.59 seconds.
