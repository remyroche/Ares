# Regime Models Training Report

**Symbol:** ETHUSDT
**Primary Model:** random_forest
**Generated:** 2025-11-12T20:20:51.805756
**Report Version:** 1.0

## Overall Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3223 |
| Number of Regimes | 7 |
| Mean Max Probability | 0.5202 |
| Std Max Probability | 0.3892 |
| Regime Balance | 19.2268 |
| Prediction Confidence | 0.5202 |
| Uncertainty Entropy | 1.0848 |

## Model Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.0000 |
| Precision (Weighted) | 0.0000 |
| Recall (Weighted) | 0.0000 |
| F1-Score (Weighted) | 0.0000 |

## Regime Statistics

| Regime | Sample Count | Percentage | Mean Prob | Std Prob | High Conf | Med Conf | Low Conf |
|--------|--------------|------------|-----------|----------|-----------|----------|----------|
| regime_0 | 1610 | 50.0% | 0.071 | 0.071 | 0 | 0 | 3223 |
| regime_1 | 0 | 0.0% | 0.071 | 0.071 | 0 | 0 | 3223 |
| regime_2 | 0 | 0.0% | 0.071 | 0.071 | 0 | 0 | 3223 |
| regime_3 | 0 | 0.0% | 0.071 | 0.071 | 0 | 0 | 3223 |
| regime_4 | 0 | 0.0% | 0.071 | 0.071 | 0 | 0 | 3223 |
| regime_5 | 1180 | 36.6% | 0.431 | 0.384 | 940 | 240 | 2043 |
| regime_6 | 433 | 13.4% | 0.213 | 0.263 | 248 | 185 | 2790 |

## Feature Importance Analysis

### SHAP Feature Importance (Top 60)

| Rank | Feature | SHAP Value |
|------|---------|------------|
| 44 | trend_ema_50_returns_vwap_ma20 | 0.030597 |
| 58 | trend_ema_50_returns_vwap_ewm0.3 | 0.024265 |
| 3 | volume_volume_accumulation_distribution | 0.015133 |
| 31 | volume_volume_accumulation_distribution_ma20 | 0.014666 |
| 53 | volume_volume_accumulation_distribution_ewm0.3 | 0.013340 |
| 16 | volume_volume_accumulation_distribution_ma8 | 0.013093 |
| 54 | volume_vectorbt_enhanced_obv_10_ewm0.3 | 0.012119 |
| 17 | volume_vectorbt_enhanced_obv_10_ma8 | 0.010402 |
| 15 | volume_volume_price_trend_ma8 | 0.009984 |
| 2 | volume_volume_price_trend | 0.009836 |
| 29 | volatility_vectorbt_rogers_satchell_volatility_50_ma20 | 0.009490 |
| 36 | volume_vectorbt_enhanced_ad_line_10_ma20 | 0.009444 |
| 52 | volume_volume_price_trend_ewm0.3 | 0.008726 |
| 38 | volume_vectorbt_enhanced_ad_line_50_ma20 | 0.008571 |
| 37 | volume_vectorbt_enhanced_ad_line_20_ma20 | 0.008353 |
| 4 | volume_vectorbt_enhanced_obv_10 | 0.008042 |
| 6 | volume_vectorbt_enhanced_ad_line_10 | 0.007990 |
| 40 | trend_vectorbt_trend_comprehensive_100_ma20 | 0.007705 |
| 39 | volume_vectorbt_volume_weighted_ad_line_50_ma20 | 0.007643 |
| 9 | trend_vectorbt_trend_comprehensive_50 | 0.007338 |
| 19 | volume_vectorbt_enhanced_ad_line_50_ma8 | 0.007250 |
| 25 | trend_ema_50_returns_vwap_std8 | 0.007223 |
| 30 | volume_volume_price_trend_ma20 | 0.007148 |
| 21 | volume_vectorbt_volume_weighted_ad_line_50_ma8 | 0.006425 |
| 7 | volume_vectorbt_enhanced_ad_line_20 | 0.006288 |
| 41 | trend_vectorbt_sma_100_ma20 | 0.006240 |
| 10 | trend_vectorbt_trend_comprehensive_100 | 0.006224 |
| 33 | volume_vectorbt_enhanced_obv_10_ma20 | 0.005977 |
| 8 | volume_vectorbt_enhanced_ad_line_50 | 0.005211 |
| 56 | volume_vectorbt_volume_weighted_ad_line_50_ewm0.3 | 0.004730 |
| 11 | trend_vectorbt_sma_100 | 0.004726 |
| 20 | volume_vectorbt_volume_weighted_ad_line_20_ma8 | 0.003830 |
| 55 | volume_vectorbt_enhanced_obv_50_ewm0.3 | 0.003779 |
| 13 | volume_vectorbt_smoothed_obv_50 | 0.003498 |
| 18 | volume_vectorbt_enhanced_obv_50_ma8 | 0.003478 |
| 35 | volume_vectorbt_enhanced_obv_50_ma20 | 0.003407 |
| 51 | volatility_vectorbt_atr_50_ewm0.3 | 0.003382 |
| 43 | trend_ema_26_returns_vwap_ma20 | 0.003339 |
| 34 | volume_vectorbt_enhanced_obv_20_ma20 | 0.003192 |
| 5 | volume_vectorbt_enhanced_obv_50 | 0.002790 |
| 42 | trend_sma_100_returns_vwap_ma20 | 0.002694 |
| 48 | momentum_momentum_30_price_returns_std20 | 0.002274 |
| 28 | volatility_vectorbt_atr_50_ma20 | 0.002174 |
| 60 | microstructure_liquidity_proxy_ewm0.3 | 0.001587 |
| 57 | trend_sma_100_returns_vwap_ewm0.3 | 0.001544 |
| 14 | volatility_enhanced_volatility_100_ma8 | 0.001278 |
| 23 | trend_sma_100_returns_vwap_ma8 | 0.001250 |
| 50 | volatility_enhanced_volatility_100_ewm0.3 | 0.001150 |
| 49 | volume_vectorbt_enhanced_ad_line_50_std20 | 0.001141 |
| 22 | trend_sma_50_returns_vwap_ma8 | 0.000976 |
| 27 | volatility_enhanced_volatility_100_ma20 | 0.000962 |
| 47 | microstructure_liquidity_proxy_ma20 | 0.000844 |
| 12 | trend_sma_100_returns_vwap | 0.000791 |
| 59 | microstructure_price_impact_ewm0.3 | 0.000752 |
| 1 | volatility_enhanced_volatility_100 | 0.000640 |
| 46 | microstructure_price_impact_ma20 | 0.000391 |
| 24 | microstructure_price_impact_ma8 | 0.000375 |
| 26 | momentum_rsi_30_returns_vwap_ma20 | 0.000136 |
| 45 | returns_returns_kurtosis_20_price_returns_ma20 | 0.000108 |
| 32 | volume_cmf_20_ma20 | 0.000097 |
