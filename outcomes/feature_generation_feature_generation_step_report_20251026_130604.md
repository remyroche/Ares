# Feature Generation Report

**Generated:** 2025-10-26 13:06:04
**Step:** feature_generation_feature_generation_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 1h
- **Execution Mode:** light

## Summary

✅ **Successfully generated 333 features** from 480 rows of data.

## Feature Statistics

- **Total Features:** 333
- **Data Samples:** 480
- **Memory Usage:** 0.68 MB
- **Missing Values:** 0
- **Missing Value %:** 0.00%

## Comprehensive Feature Analysis

### Feature Quality Metrics

| Metric | Value |
|--------|-------|
| High Quality Features (>0.7 score) | 1 |
| Medium Quality Features (0.4-0.7) | 0 |
| Low Quality Features (<0.4) | 331 |
| Constant Features | 33 |
| Highly Correlated Pairs | 737 |
| Average Correlation | nan |
| Feature Stability Score | 0.005 |

### Constant Features (Zero Variance)

The following features have constant values across all data points and should be removed:

1. `candlestick_engulfing_pattern`
2. `candlestick_hammer_pattern`
3. `candlestick_hanging_man_pattern`
4. `candlestick_shooting_star_pattern`
5. `candlestick_harami_pattern`
6. `candlestick_dragonfly_doji_pattern`
7. `candlestick_gravestone_doji_pattern`
8. `candlestick_inverted_hammer_pattern`
9. `candlestick_abandoned_baby_pattern`
10. `price_entropy_5_price_returns`
11. `return_entropy_5_price_returns`
12. `price_entropy_ma_5_5_price_returns`
13. `return_entropy_ma_5_5_price_returns`
14. `price_entropy_ma_5_10_price_returns`
15. `return_entropy_ma_5_10_price_returns`
16. `price_entropy_10_price_returns`
17. `return_entropy_10_price_returns`
18. `price_entropy_ma_10_5_price_returns`
19. `return_entropy_ma_10_5_price_returns`
20. `price_entropy_ma_10_10_price_returns`
21. `price_entropy_20_price_returns`
22. `return_entropy_ma_10_10_price_returns`
23. `return_entropy_20_price_returns`
24. `price_entropy_ma_20_5_price_returns`
25. `price_entropy_ma_20_10_price_returns`
26. `return_entropy_ma_20_5_price_returns`
27. `return_entropy_ma_20_10_price_returns`
28. `sample_entropy_20_2_0.2`
29. `permutation_entropy_20_3_1`
30. `vectorbt_acceleration_regime_10_10_price_returns`
31. `vectorbt_acceleration_regime_10_20_price_returns`
32. `vectorbt_acceleration_correlation_20_price_returns`
33. `vectorbt_spectral_wavelet_batch`

### Top 10 Performing Features

| Rank | Feature | Quality Score | Correlation | Stability | Information |
|------|---------|---------------|-------------|-----------|-------------|
| 1 | `log_returns_1_price_returns` | 0.853 | 0.931 | 0.000 | 2.124 |
| 2 | `fractal_dimension` | 0.201 | 0.078 | 0.792 | 0.012 |
| 3 | `vectorbt_spectral_wavelet_batch` | 0.200 | 0.000 | 1.000 | 0.001 |
| 4 | `volume_roc_1` | 0.133 | 0.190 | 0.000 | 0.097 |
| 5 | `vectorbt_trend_strength_5_price_returns` | 0.115 | 0.254 | 0.000 | 0.081 |
| 6 | `sharpe_ratio_20_0.0_price_returns` | 0.095 | 0.181 | 0.000 | 0.035 |
| 7 | `candlestick_three_black_crows_pattern` | 0.093 | 0.224 | 0.000 | 0.051 |
| 8 | `candlestick_three_white_soldiers_pattern` | 0.088 | 0.208 | 0.000 | 0.075 |
| 9 | `vectorbt_momentum_50_price_returns` | 0.086 | 0.101 | 0.000 | 0.140 |
| 10 | `vectorbt_momentum_20_price_returns` | 0.077 | 0.133 | 0.000 | 0.167 |

### Feature Distribution Analysis

| Statistic | Value |
|-----------|-------|
| Mean Quality Score | 0.028 |
| Median Quality Score | 0.021 |
| Std Quality Score | 0.051 |
| Min Quality Score | 0.000 |
| Max Quality Score | 0.853 |

### Feature Redundancy Analysis

| Metric | Value |
|--------|-------|
| Redundant Feature Pairs | 737 |
| Redundancy Rate | 1.3% |
| Unique Features | -404 |
| Redundancy Score | 0.987 |

### Feature Stability Analysis

| Metric | Value |
|--------|-------|
| Stable Features (>0.8) | 1 |
| Moderately Stable (0.5-0.8) | 1 |
| Unstable Features (<0.5) | 330 |
| Average Stability | 0.005 |

### Feature Information Content

| Metric | Value |
|--------|-------|
| High Information (>0.7) | 1 |
| Medium Information (0.4-0.7) | 0 |
| Low Information (<0.4) | 331 |
| Average Information | 0.027 |

### Feature Recommendations

#### Features to Keep (High Quality)
- `log_returns_1_price_returns`

#### Features to Consider Removing (Low Quality)
- `fractal_dimension`
- `vectorbt_spectral_wavelet_batch`
- `volume_roc_1`
- `vectorbt_trend_strength_5_price_returns`
- `sharpe_ratio_20_0.0_price_returns`
- `candlestick_three_black_crows_pattern`
- `candlestick_three_white_soldiers_pattern`
- `vectorbt_momentum_50_price_returns`
- `vectorbt_momentum_20_price_returns`
- `rolling_returns_20_price_returns`
- `cumulative_returns_20_price_returns`
- `momentum_21_price_returns`
- `volume_ratio_50`
- `volume_percentile_100`
- `acceleration_features`
- `volume_ratio_10`
- `analyst_momentum_alignment`
- `volume_percentile_50`
- `volume_percentile_20`
- `returns_skewness_20_price_returns`
- `volume_ratio_20`
- `rolling_zscore_returns_20`
- `vectorbt_momentum_acceleration_5_20_price_returns`
- `vectorbt_acceleration_momentum_10_10_price_returns`
- `volume_ma_ratios_20_10`
- `spectral_entropy_20`
- `stochastic_30_3_price_returns`
- `volume_volatility_elasticity_20`
- `williams_r_30_price_returns`
- `vectorbt_trend_strength_10_price_returns`
- `williams_r_14_price_returns`
- `resistance_level_2_10_price_returns`
- `resistance_level_4_10_price_returns`
- `resistance_level_1_10_price_returns`
- `resistance_level_3_10_price_returns`
- `resistance_level_5_10_price_returns`
- `ar_1_coefficients_20`
- `stochastic_14_3_price_returns`
- `vectorbt_momentum_5_price_returns`
- `ljung_box_pvalue_20_10`
- `candlestick_harami_cross_pattern`
- `roc_21_price_returns`
- `vectorbt_volatility_comprehensive_10`
- `candlestick_doji_pattern`
- `vectorbt_trend_consistency_20_price_returns`
- `vectorbt_trend_consistency_10_price_returns`
- `simple_returns_5_price_returns`
- `roc_30_price_returns`
- `fibonacci_0.786_20_price_returns`
- `candlestick_long_legged_doji_pattern`
- `rsi_entropy_20_14`
- `aroon_25_returns_vwap`
- `volume_zscore_60_252`
- `volume_momentum_5`
- `williams_r_21_price_returns`
- `stochastic_21_3_price_returns`
- `candlestick_dark_cloud_cover_pattern`
- `vectorbt_acceleration_divergence_20_price_returns`
- `rsi_21_returns_vwap`
- `volume_trend_strength_20_50`
- `cci_20_returns_vwap`
- `vectorbt_acceleration_regime_5_20_price_returns`
- `vectorbt_volume_acceleration_5_volume_returns`
- `roc_14_price_returns`
- `simple_returns_10_price_returns`
- `enhanced_volatility_10`
- `vectorbt_atr_14`
- `fibonacci_0.236_10_price_returns`
- `volume_sma_5`
- `volume_ema_5`
- `vectorbt_atr_10`
- `donchian_channel_20`
- `cumulative_returns_10_price_returns`
- `rolling_returns_10_price_returns`
- `resistance_level_3_5_price_returns`
- `resistance_level_1_5_price_returns`
- `vectorbt_volatility_acceleration_5_20_price_returns`
- `resistance_level_5_5_price_returns`
- `resistance_level_4_5_price_returns`
- `resistance_level_2_5_price_returns`
- `vectorbt_acceleration_volatility_10_10_price_returns`
- `log_returns_10_price_returns`
- `vectorbt_momentum_10_price_returns`
- `volume_roc_5`
- `vectorbt_momentum_acceleration_10_10_price_returns`
- `fibonacci_0.5_20_price_returns`
- `vectorbt_volatility_comprehensive_14`
- `vectorbt_yang_zhang_volatility_14`
- `volume_momentum_10`
- `fibonacci_0.618_20_price_returns`
- `band_limited_volatility`
- `fibonacci_0.786_5_price_returns`
- `vectorbt_volatility_comprehensive_50`
- `fibonacci_0.786_10_price_returns`
- `sma_50_returns_vwap`
- `volume_roc_10`
- `volume_price_divergence_10`
- `vectorbt_acceleration_10_price_returns`
- `fibonacci_0.382_5_price_returns`
- `volume_entropy_ma_20_5_volume_returns`
- `analyst_volume_pressure`
- `rsi_14_returns_vwap`
- `cmo_14_returns_vwap`
- `vectorbt_volatility_comprehensive_20`
- `vectorbt_trend_strength_50_price_returns`
- `vectorbt_parkinson_volatility_14`
- `momentum_30_price_returns`
- `enhanced_volatility_30`
- `vectorbt_acceleration_momentum_10_20_price_returns`
- `fibonacci_0.618_5_price_returns`
- `vectorbt_rogers_satchell_volatility_10`
- `vectorbt_bbands_14_1.5`
- `vectorbt_enhanced_obv_10`
- `vectorbt_bbands_14_2.5`
- `vectorbt_trend_consistency_50_price_returns`
- `volume_entropy_10_volume_returns`
- `vectorbt_bbands_14_2.0`
- `vectorbt_acceleration_consistency_5_10_price_returns`
- `vectorbt_acceleration_trend_strength_5_20_price_returns`
- `volume_entropy_ma_20_10_volume_returns`
- `vectorbt_garman_klass_volatility_14`
- `volume_momentum_20`
- `vectorbt_rogers_satchell_volatility_14`
- `sma_20_returns_vwap`
- `vectorbt_smoothed_obv_10`
- `ema_50_returns_vwap`
- `vectorbt_jerk_5_price_returns`
- `vectorbt_yang_zhang_volatility_30`
- `vectorbt_acceleration_momentum_5_20_price_returns`
- `volume_roc_20`
- `vectorbt_volatility_comprehensive_30`
- `volume_std_20`
- `order_flow_imbalance_20`
- `returns_volatility_20_price_returns`
- `vectorbt_garman_klass_volatility_10`
- `vectorbt_yang_zhang_volatility_10`
- `vectorbt_garman_klass_volatility_30`
- `vectorbt_parkinson_volatility_30`
- `ultimate_oscillator_7_14_28_returns_vwap`
- `support_level_1_10_price_returns`
- `support_level_2_10_price_returns`
- `support_level_4_10_price_returns`
- `support_level_5_10_price_returns`
- `support_level_3_10_price_returns`
- `advanced_momentum_5_20`
- `vectorbt_yang_zhang_volatility_50`
- `vectorbt_parkinson_volatility_50`
- `kst_10_15_20_30_10_10_10_15_returns_vwap`
- `vectorbt_parkinson_volatility_10`
- `vectorbt_atr_20`
- `vectorbt_trend_consistency_5_price_returns`
- `price_volume_oscillator_5_15`
- `volume_entropy_20_volume_returns`
- `candlestick_piercing_line_pattern`
- `vectorbt_enhanced_ad_line_50`
- `fibonacci_0.618_10_price_returns`
- `vectorbt_atr_50`
- `vectorbt_bbands_10_1.5`
- `vectorbt_bbands_10_2.0`
- `vectorbt_bbands_10_2.5`
- `vectorbt_volume_weighted_ad_line_50`
- `returns_kurtosis_20_price_returns`
- `dfa_slopes`
- `vwap_deviations_20`
- `analyst_volume_trend`
- `stochastic_kd_14_3`
- `volume_sma_10`
- `volume_ema_10`
- `vectorbt_parkinson_volatility_20`
- `volume_sma_20`
- `volume_ema_20`
- `vectorbt_yang_zhang_volatility_20`
- `enhanced_volatility_50`
- `lempel_ziv_complexity_20`
- `pivot_point_5_price_returns`
- `volume_std_10`
- `enhanced_volatility_14`
- `cycle_length`
- `vectorbt_acceleration_volatility_5_10_price_returns`
- `log_returns_5_price_returns`
- `vectorbt_enhanced_obv_20`
- `vectorbt_momentum_acceleration_10_20_price_returns`
- `volume_entropy_ma_10_10_volume_returns`
- `vectorbt_smoothed_obv_50`
- `support_level_2_5_price_returns`
- `volume_std_50`
- `vectorbt_momentum_comprehensive_9`
- `directional_signal`
- `vectorbt_acceleration_volatility_5_20_price_returns`
- `macd_12_26_9_returns_vwap`
- `enhanced_volatility_20`
- `vectorbt_rogers_satchell_volatility_30`
- `vectorbt_acceleration_trend_strength_10_20_price_returns`
- `vectorbt_acceleration_regime_5_10_price_returns`
- `enhanced_volatility_100`
- `support_level_3_5_price_returns`
- `volume_price_correlation_10`
- `support_level_1_5_price_returns`
- `vectorbt_atr_30`
- `support_level_5_5_price_returns`
- `vectorbt_acceleration_trend_strength_10_10_price_returns`
- `support_level_4_5_price_returns`
- `fibonacci_0.5_5_price_returns`
- `volume_price_trend`
- `resistance_level_1_20_price_returns`
- `resistance_level_3_20_price_returns`
- `resistance_level_2_20_price_returns`
- `resistance_level_5_20_price_returns`
- `resistance_level_4_20_price_returns`
- `vectorbt_acceleration_consistency_10_10_price_returns`
- `vectorbt_trend_strength_20_price_returns`
- `vectorbt_jerk_10_price_returns`
- `momentum_14_price_returns`
- `advanced_cumulative_returns_10`
- `volume_price_divergence_20`
- `vectorbt_volume_weighted_ad_line_10`
- `volume_entropy_ma_10_5_volume_returns`
- `rsi_zscore_14_20`
- `vectorbt_enhanced_ad_line_10`
- `rsi_30_returns_vwap`
- `volume_accumulation_distribution`
- `vectorbt_acceleration_5_price_returns`
- `macd_entropy_20_12_26`
- `wma_20_price_returns`
- `pfe_12_returns_vwap`
- `fibonacci_0.382_10_price_returns`
- `volume_oscillator_5_15`
- `macd_delta_12_26_9`
- `volume_oscillator_10_20`
- `volume_price_correlation_20`
- `vectorbt_garman_klass_volatility_20`
- `vectorbt_momentum_comprehensive_30`
- `momentum_endpoints_sma_20`
- `pivot_point_10_price_returns`
- `vectorbt_multi_timeframe_acceleration_5_20_price_returns`
- `momentum_features`
- `volume_entropy_ma_5_10_volume_returns`
- `fibonacci_0.382_20_price_returns`
- `vectorbt_smoothed_obv_20`
- `vectorbt_volume_weighted_ad_line_20`
- `apo_12_26_returns_vwap`
- `natr_14_returns_vwap`
- `adx_14_returns_vwap`
- `vectorbt_enhanced_ad_line_20`
- `vectorbt_enhanced_obv_50`
- `trend_score_14`
- `wavelet_energy`
- `vectorbt_rogers_satchell_volatility_20`
- `vectorbt_garman_klass_volatility_50`
- `advanced_momentum_10_30`
- `volume_entropy_ma_5_5_volume_returns`
- `fibonacci_0.5_10_price_returns`
- `entropy_rate_20`
- `vwma_20_price_returns`
- `fibonacci_0.236_20_price_returns`
- `volume_vwap_50`
- `volume_sma_50`
- `volume_ema_50`
- `volume_vwap_10`
- `analyst_momentum_5m`
- `analyst_momentum_15m`
- `analyst_momentum_1h`
- `pivot_point_20_price_returns`
- `volume_entropy_5_volume_returns`
- `support_level_1_20_price_returns`
- `support_level_2_20_price_returns`
- `support_level_3_20_price_returns`
- `support_level_4_20_price_returns`
- `support_level_5_20_price_returns`
- `vectorbt_rogers_satchell_volatility_50`
- `price_volume_oscillator_10_20`
- `sma_5_returns_vwap`
- `cmf_20`
- `advanced_cumulative_returns_20`
- `t3_14_0.7_returns_vwap`
- `kama_30_2_30_returns_vwap`
- `vectorbt_acceleration_trend_strength_5_10_price_returns`
- `vectorbt_acceleration_consistency_10_20_price_returns`
- `volume_vwap_20`
- `tema_21_price_returns`
- `vectorbt_acceleration_consistency_5_20_price_returns`
- `ema_12_returns_vwap`
- `ema_26_returns_vwap`
- `sma_100_returns_vwap`
- `volume_trend_strength_10_30`
- `vectorbt_acceleration_volatility_10_20_price_returns`
- `sma_10_returns_vwap`
- `vectorbt_momentum_acceleration_5_10_price_returns`
- `keltner_channels_20_14_price_returns`
- `mama_21_0.05_price_returns`
- `dema_21_price_returns`
- `shannon_entropy_20_10`
- `vectorbt_bbands_20_2.0`
- `vectorbt_bbands_20_2.5`
- `vectorbt_bbands_20_1.5`
- `vectorbt_momentum_comprehensive_14`
- `return_entropy_5_price_returns`
- `candlestick_inverted_hammer_pattern`
- `return_entropy_ma_10_5_price_returns`
- `vectorbt_momentum_comprehensive_21`
- `price_entropy_10_price_returns`
- `candlestick_dragonfly_doji_pattern`
- `candlestick_shooting_star_pattern`
- `vectorbt_acceleration_momentum_5_10_price_returns`
- `return_entropy_ma_5_10_price_returns`
- `candlestick_hanging_man_pattern`
- `fibonacci_0.236_5_price_returns`
- `candlestick_hammer_pattern`
- `price_entropy_ma_20_10_price_returns`
- `return_entropy_ma_20_5_price_returns`
- `price_entropy_20_price_returns`
- `price_entropy_ma_20_5_price_returns`
- `candlestick_harami_pattern`
- `return_entropy_ma_5_5_price_returns`
- `price_entropy_5_price_returns`
- `return_entropy_10_price_returns`
- `candlestick_abandoned_baby_pattern`
- `return_entropy_20_price_returns`
- `candlestick_gravestone_doji_pattern`
- `candlestick_engulfing_pattern`
- `return_entropy_ma_20_10_price_returns`
- `price_entropy_ma_10_10_price_returns`
- `price_entropy_ma_5_10_price_returns`
- `sample_entropy_20_2_0.2`
- `price_entropy_ma_5_5_price_returns`
- `permutation_entropy_20_3_1`
- `return_entropy_ma_10_10_price_returns`
- `vectorbt_acceleration_correlation_20_price_returns`
- `vectorbt_acceleration_regime_10_10_price_returns`
- `price_entropy_ma_10_5_price_returns`
- `vectorbt_acceleration_regime_10_20_price_returns`

#### Features to Investigate (Medium Quality)

## Feature Categories

### Returns (178 features)

- `simple_returns_1_price_returns`
- `simple_returns_5_price_returns`
- `log_returns_5_price_returns`
- `simple_returns_10_price_returns`
- `cumulative_returns_10_price_returns`
- ... and 173 more

### Momentum (46 features)

- `momentum_features`
- `vectorbt_momentum_comprehensive_21`
- `vectorbt_momentum_comprehensive_30`
- `vectorbt_momentum_comprehensive_9`
- `vectorbt_momentum_comprehensive_14`
- ... and 41 more

### Volume (95 features)

- `returns_volatility_20_price_returns`
- `volume_sma_5`
- `volume_sma_10`
- `volume_ema_5`
- `volume_ema_10`
- ... and 90 more

### Volatility (57 features)

- `returns_volatility_20_price_returns`
- `volume_std_10`
- `volume_std_20`
- `volume_std_50`
- `volume_volatility_elasticity_20`
- ... and 52 more

### Trend (70 features)

- `macd_12_26_9_returns_vwap`
- `macd_delta_12_26_9`
- `momentum_endpoints_sma_20`
- `volume_sma_5`
- `volume_sma_10`
- ... and 65 more

### Oscillator (9 features)

- `macd_12_26_9_returns_vwap`
- `macd_delta_12_26_9`
- `volume_oscillator_10_20`
- `volume_oscillator_5_15`
- `price_volume_oscillator_5_15`
- ... and 4 more

### Support_resistance (33 features)

- `support_level_1_5_price_returns`
- `support_level_2_5_price_returns`
- `support_level_3_5_price_returns`
- `support_level_4_5_price_returns`
- `resistance_level_1_5_price_returns`
- ... and 28 more

### Candlestick (16 features)

- `candlestick_doji_pattern`
- `candlestick_engulfing_pattern`
- `candlestick_hammer_pattern`
- `candlestick_hanging_man_pattern`
- `candlestick_shooting_star_pattern`
- ... and 11 more

### Entropy (34 features)

- `rsi_entropy_20_14`
- `macd_entropy_20_12_26`
- `price_entropy_5_price_returns`
- `volume_entropy_5_volume_returns`
- `return_entropy_5_price_returns`
- ... and 29 more

### Acceleration (32 features)

- `vectorbt_acceleration_5_price_returns`
- `vectorbt_acceleration_10_price_returns`
- `vectorbt_volatility_acceleration_5_20_price_returns`
- `vectorbt_volume_acceleration_5_volume_returns`
- `vectorbt_momentum_acceleration_5_10_price_returns`
- ... and 27 more

## Data Quality

| Metric | Value |
|--------|-------|
| Total Columns | 333 |
| Total Rows | 480 |
| Non-Null Values | 159,840 |
| Null Values | 0 |
| Memory Usage (MB) | 0.68 |

## Artifacts

### generated_features

**Path:** `artifacts/pre_training/long/Analyst/feature_generation_feature_generation_step/feature_generation_feature_generation_step_generated_features_1h_long_Analyst_20251026_130537.parquet`
**Size:** 838.00 KB

## Next Steps

- Features are ready for feature selection and interaction generation
- Consider running lookback optimization for optimal feature parameters
- Proceed to labeling step for profit-target generation

