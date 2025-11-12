# Regime Models Training Report

**Symbol:** ETHUSDT
**Primary Model:** catboost
**Generated:** 2025-11-11T23:09:46.186522
**Report Version:** 1.0

## Overall Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3245 |
| Number of Regimes | 6 |
| Mean Max Probability | 0.7497 |
| Std Max Probability | 0.2019 |
| Regime Balance | 10.9558 |
| Prediction Confidence | 0.7497 |
| Uncertainty Entropy | 0.6631 |

## Model Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.3086 |
| Precision (Weighted) | 0.2821 |
| Recall (Weighted) | 0.3086 |
| F1-Score (Weighted) | 0.2713 |

## Regime Statistics

| Regime | Sample Count | Percentage | Mean Prob | Std Prob | High Conf | Med Conf | Low Conf |
|--------|--------------|------------|-----------|----------|-----------|----------|----------|
| regime_0 | 457 | 14.1% | 0.138 | 0.257 | 178 | 226 | 2841 |
| regime_1 | 349 | 10.8% | 0.094 | 0.252 | 210 | 81 | 2954 |
| regime_2 | 1242 | 38.3% | 0.337 | 0.345 | 546 | 504 | 2195 |
| regime_3 | 125 | 3.9% | 0.065 | 0.178 | 99 | 21 | 3125 |
| regime_4 | 694 | 21.4% | 0.234 | 0.302 | 327 | 260 | 2658 |
| regime_5 | 378 | 11.6% | 0.131 | 0.256 | 202 | 109 | 2934 |

## Feature Importance Analysis

### SHAP Feature Importance (Top 60)

| Rank | Feature | SHAP Value |
|------|---------|------------|
| 44 | volume_volume_trend_strength_20_50_std20 | 0.206581 |
| 43 | volume_volume_trend_strength_10_30_std20 | 0.134781 |
| 41 | volatility_vectorbt_bbands_10_1.5_std20 | 0.129651 |
| 26 | trend_sma_100_returns_vwap_std8 | 0.123677 |
| 39 | momentum_rsi_21_returns_vwap_std20 | 0.116568 |
| 22 | volume_vectorbt_volume_weighted_ad_line_50_std8 | 0.091825 |
| 40 | volatility_vectorbt_volatility_comprehensive_30_std20 | 0.085369 |
| 54 | trend_vectorbt_zigzag_5.0_3_ewm0.3 | 0.077418 |
| 15 | returns_returns_kurtosis_20_price_returns_ma8 | 0.077408 |
| 47 | trend_vectorbt_zigzag_7.0_3_std20 | 0.075850 |
| 60 | returns_ljung_box_pvalue_20_10_ewm0.3 | 0.073501 |
| 32 | volume_cmf_20_ma20 | 0.072553 |
| 50 | returns_simple_returns_10_price_returns_std20 | 0.071062 |
| 16 | returns_ar_1_coefficients_20_ma8 | 0.070843 |
| 21 | volume_cmf_20_std8 | 0.069941 |
| 8 | volume_volume_price_correlation_10 | 0.066994 |
| 36 | returns_ljung_box_pvalue_20_10_ma20 | 0.066192 |
| 35 | returns_sharpe_ratio_20_0.0_price_returns_ma20 | 0.065016 |
| 18 | momentum_roc_30_price_returns_std8 | 0.061821 |
| 31 | momentum_roc_21_price_returns_ma20 | 0.059358 |
| 51 | returns_returns_skewness_20_price_returns_std20 | 0.058564 |
| 42 | volume_volume_roc_1_std20 | 0.057406 |
| 25 | trend_vectorbt_zigzag_10.0_5_std8 | 0.057295 |
| 48 | trend_vectorbt_zigzag_10.0_2_std20 | 0.056395 |
| 45 | volume_cmf_20_std20 | 0.056294 |
| 28 | returns_simple_returns_5_price_returns_std8 | 0.055205 |
| 37 | microstructure_trade_intensity_ma20 | 0.052637 |
| 49 | returns_log_returns_5_price_returns_std20 | 0.050131 |
| 13 | trend_vectorbt_zigzag_10.0_3_ma8 | 0.048756 |
| 34 | returns_simple_returns_1_price_returns_ma20 | 0.047197 |
| 24 | trend_vectorbt_zigzag_10.0_2_std8 | 0.045107 |
| 30 | microstructure_amihud_illiquidity_vwap_distance_std8 | 0.043960 |
| 19 | volume_volume_trend_strength_10_30_std8 | 0.043606 |
| 33 | returns_log_returns_1_price_returns_ma20 | 0.043526 |
| 9 | returns_ar_1_coefficients_20 | 0.043095 |
| 57 | returns_log_returns_1_price_returns_ewm0.3 | 0.041417 |
| 59 | returns_simple_returns_5_price_returns_ewm0.3 | 0.040017 |
| 23 | trend_vectorbt_zigzag_5.0_2_std8 | 0.037381 |
| 12 | trend_vectorbt_zigzag_10.0_2_ma8 | 0.037306 |
| 29 | returns_returns_volatility_20_price_returns_std8 | 0.037270 |
| 27 | returns_log_returns_1_price_returns_std8 | 0.036695 |
| 11 | volume_price_volume_oscillator_10_20_ma8 | 0.030850 |
| 46 | volume_volume_volatility_elasticity_20_std20 | 0.030461 |
| 55 | trend_vectorbt_zigzag_10.0_2_ewm0.3 | 0.029704 |
| 52 | returns_ljung_box_pvalue_20_10_std20 | 0.029372 |
| 20 | volume_volume_price_correlation_10_std8 | 0.029218 |
| 56 | trend_vectorbt_zigzag_10.0_5_ewm0.3 | 0.028182 |
| 10 | momentum_roc_14_price_returns_ma8 | 0.027928 |
| 17 | momentum_roc_14_price_returns_std8 | 0.027906 |
| 58 | returns_log_returns_5_price_returns_ewm0.3 | 0.027639 |
| 14 | returns_simple_returns_1_price_returns_ma8 | 0.027408 |
| 38 | momentum_rsi_14_returns_vwap_std20 | 0.024774 |
| 4 | momentum_roc_30_price_returns | 0.021703 |
| 7 | volume_volume_roc_1 | 0.019108 |
| 5 | volatility_vectorbt_volatility_comprehensive_10 | 0.018059 |
| 3 | momentum_roc_21_price_returns | 0.016842 |
| 2 | momentum_momentum_21_price_returns | 0.013860 |
| 6 | volatility_vectorbt_volatility_comprehensive_20 | 0.012551 |
| 1 | momentum_roc_14_price_returns | 0.009303 |
| 53 | momentum_momentum_30_price_returns_ewm0.3 | 0.008973 |
