# Regime Models Training Report

**Symbol:** ETHUSDT
**Primary Model:** random_forest
**Generated:** 2025-11-11T15:58:44.670845
**Report Version:** 1.0

## Overall Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3252 |
| Number of Regimes | 6 |
| Mean Max Probability | 0.9702 |
| Std Max Probability | 0.1441 |
| Regime Balance | 15.4544 |
| Prediction Confidence | 0.9702 |
| Uncertainty Entropy | 0.0670 |

## Model Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.3077 |
| Precision (Weighted) | 0.3774 |
| Recall (Weighted) | 0.3077 |
| F1-Score (Weighted) | 0.1922 |

## Regime Statistics

| Regime | Sample Count | Percentage | Mean Prob | Std Prob | High Conf | Med Conf | Low Conf |
|--------|--------------|------------|-----------|----------|-----------|----------|----------|
| regime_0 | 458 | 14.1% | 0.116 | 0.312 | 359 | 3 | 2890 |
| regime_1 | 254 | 7.8% | 0.083 | 0.266 | 248 | 6 | 2998 |
| regime_2 | 1633 | 50.2% | 0.509 | 0.492 | 1623 | 10 | 1619 |
| regime_3 | 119 | 3.7% | 0.041 | 0.188 | 117 | 2 | 3133 |
| regime_4 | 473 | 14.5% | 0.150 | 0.349 | 471 | 2 | 2779 |
| regime_5 | 315 | 9.7% | 0.102 | 0.292 | 308 | 7 | 2937 |

## Feature Importance Analysis

### SHAP Feature Importance (Top 60)

| Rank | Feature | SHAP Value |
|------|---------|------------|
| 18 | oscillator_apo_12_26_returns_vwap_std3 | 0.015322 |
| 59 | microstructure_price_impact_std7 | 0.009917 |
| 7 | volume_vectorbt_smoothed_obv_20 | 0.009555 |
| 26 | momentum_advanced_momentum_10_30_std5 | 0.006310 |
| 30 | volume_cmf_20_ma7 | 0.005902 |
| 27 | oscillator_ultimate_oscillator_7_14_28_returns_vwap_std5 | 0.005628 |
| 38 | returns_returns_volatility_20_price_returns_ma7 | 0.004801 |
| 29 | volume_price_volume_oscillator_5_15_ma7 | 0.004763 |
| 19 | volume_cmf_20_ma5 | 0.004761 |
| 15 | volatility_vectorbt_rogers_satchell_volatility_20_std3 | 0.004638 |
| 6 | trend_sma_100_returns_vwap | 0.004632 |
| 45 | momentum_momentum_30_price_returns_std7 | 0.004477 |
| 60 | microstructure_amihud_illiquidity_vwap_distance_std7 | 0.004437 |
| 9 | volume_cmf_20_ma3 | 0.004340 |
| 25 | momentum_roc_30_price_returns_std5 | 0.004309 |
| 51 | oscillator_ultimate_oscillator_7_14_28_returns_vwap_std7 | 0.004189 |
| 47 | volume_volume_percentile_100_std7 | 0.004059 |
| 49 | trend_vectorbt_zigzag_5.0_2_std7 | 0.003995 |
| 31 | trend_vectorbt_zigzag_10.0_2_ma7 | 0.003882 |
| 2 | momentum_momentum_21_price_returns | 0.003837 |
| 55 | returns_rolling_returns_20_price_returns_std7 | 0.003725 |
| 8 | volume_volume_trend_strength_20_50_ma3 | 0.003707 |
| 46 | momentum_roc_30_price_returns_std7 | 0.003590 |
| 24 | momentum_momentum_30_price_returns_std5 | 0.003580 |
| 43 | returns_ljung_box_pvalue_20_10_ma7 | 0.003536 |
| 21 | returns_sharpe_ratio_20_0.0_price_returns_ma5 | 0.003336 |
| 44 | momentum_roc_21_price_returns_std7 | 0.003307 |
| 11 | returns_sharpe_ratio_20_0.0_price_returns_ma3 | 0.003283 |
| 42 | returns_ar_1_coefficients_20_ma7 | 0.003181 |
| 37 | returns_cumulative_returns_10_price_returns_ma7 | 0.003130 |
| 48 | volume_vectorbt_enhanced_ad_line_10_std7 | 0.003124 |
| 53 | returns_log_returns_5_price_returns_std7 | 0.003113 |
| 40 | returns_returns_kurtosis_20_price_returns_ma7 | 0.003099 |
| 32 | trend_vectorbt_zigzag_10.0_5_ma7 | 0.003098 |
| 41 | returns_sharpe_ratio_20_0.0_price_returns_ma7 | 0.003046 |
| 28 | returns_log_returns_1_price_returns_std5 | 0.003017 |
| 36 | returns_simple_returns_5_price_returns_ma7 | 0.002908 |
| 54 | returns_cumulative_returns_20_price_returns_std7 | 0.002832 |
| 16 | trend_sma_5_returns_vwap_std3 | 0.002798 |
| 39 | returns_returns_skewness_20_price_returns_ma7 | 0.002755 |
| 22 | returns_ar_1_coefficients_20_ma5 | 0.002712 |
| 4 | volume_volume_price_divergence_10 | 0.002696 |
| 13 | momentum_momentum_30_price_returns_std3 | 0.002679 |
| 23 | returns_ljung_box_pvalue_20_10_ma5 | 0.002638 |
| 5 | volume_cmf_20 | 0.002587 |
| 14 | volatility_enhanced_volatility_30_std3 | 0.002567 |
| 34 | returns_log_returns_10_price_returns_ma7 | 0.002556 |
| 20 | returns_simple_returns_5_price_returns_ma5 | 0.002427 |
| 56 | returns_returns_volatility_20_price_returns_std7 | 0.002324 |
| 58 | microstructure_trade_size_imbalance_std7 | 0.002228 |
| 52 | returns_log_returns_1_price_returns_std7 | 0.002196 |
| 33 | returns_log_returns_5_price_returns_ma7 | 0.002137 |
| 12 | returns_ljung_box_pvalue_20_10_ma3 | 0.002079 |
| 50 | trend_vectorbt_zigzag_10.0_3_std7 | 0.002065 |
| 57 | returns_returns_kurtosis_20_price_returns_std7 | 0.002048 |
| 17 | trend_trend_score_14_std3 | 0.001970 |
| 10 | returns_log_returns_5_price_returns_ma3 | 0.001891 |
| 35 | returns_simple_returns_1_price_returns_ma7 | 0.001821 |
| 1 | momentum_roc_14_price_returns | 0.001364 |
| 3 | momentum_roc_21_price_returns | 0.001078 |
