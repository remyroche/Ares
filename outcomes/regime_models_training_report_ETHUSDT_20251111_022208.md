# Regime Models Training Report

**Symbol:** ETHUSDT
**Primary Model:** random_forest
**Generated:** 2025-11-11T02:22:08.645158
**Report Version:** 1.0

## Overall Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3265 |
| Number of Regimes | 6 |
| Mean Max Probability | 0.9668 |
| Std Max Probability | 0.1297 |
| Regime Balance | 11.0993 |
| Prediction Confidence | 0.9668 |
| Uncertainty Entropy | 0.0785 |

## Model Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.2055 |
| Precision (Weighted) | 0.2000 |
| Recall (Weighted) | 0.2055 |
| F1-Score (Weighted) | 0.1530 |

## Regime Statistics

| Regime | Sample Count | Percentage | Mean Prob | Std Prob | High Conf | Med Conf | Low Conf |
|--------|--------------|------------|-----------|----------|-----------|----------|----------|
| regime_0 | 234 | 7.2% | 0.056 | 0.222 | 168 | 4 | 3093 |
| regime_1 | 574 | 17.6% | 0.182 | 0.374 | 566 | 8 | 2691 |
| regime_2 | 605 | 18.5% | 0.181 | 0.373 | 559 | 46 | 2660 |
| regime_3 | 115 | 3.5% | 0.038 | 0.184 | 113 | 2 | 3150 |
| regime_4 | 486 | 14.9% | 0.149 | 0.348 | 468 | 17 | 2780 |
| regime_5 | 1251 | 38.3% | 0.394 | 0.476 | 1238 | 13 | 2014 |

## Feature Importance Analysis

### SHAP Feature Importance (Top 60)

| Rank | Feature | SHAP Value |
|------|---------|------------|
| 26 | volatility_enhanced_volatility_100_ma7 | 0.042509 |
| 1 | volatility_enhanced_volatility_100 | 0.037392 |
| 22 | volume_vectorbt_volume_weighted_ad_line_10_std5 | 0.017920 |
| 3 | trend_sma_100_returns_vwap | 0.012017 |
| 42 | returns_ar_1_coefficients_20_ma7 | 0.010818 |
| 21 | volatility_enhanced_volatility_100_std5 | 0.010510 |
| 45 | volatility_enhanced_volatility_100_std7 | 0.010226 |
| 50 | volume_vectorbt_enhanced_ad_line_50_std7 | 0.009761 |
| 16 | oscillator_kst_10_15_20_30_10_10_10_15_returns_vwap_std3 | 0.007682 |
| 18 | returns_sharpe_ratio_20_0.0_price_returns_ma5 | 0.007626 |
| 47 | volume_volume_vwap_50_std7 | 0.007145 |
| 57 | oscillator_apo_12_26_returns_vwap_std7 | 0.007068 |
| 27 | volume_volume_std_50_ma7 | 0.006309 |
| 41 | returns_sharpe_ratio_20_0.0_price_returns_ma7 | 0.005769 |
| 35 | trend_sma_20_returns_vwap_ma7 | 0.005759 |
| 34 | trend_vectorbt_zigzag_10.0_5_ma7 | 0.005103 |
| 54 | trend_sma_100_returns_vwap_std7 | 0.005068 |
| 44 | volatility_enhanced_volatility_50_std7 | 0.005054 |
| 46 | volume_volume_trend_strength_20_50_std7 | 0.005019 |
| 56 | oscillator_aroon_25_returns_vwap_std7 | 0.004961 |
| 6 | returns_sharpe_ratio_20_0.0_price_returns | 0.004765 |
| 13 | returns_ljung_box_pvalue_20_10_ma3 | 0.004504 |
| 12 | returns_ar_1_coefficients_20_ma3 | 0.004239 |
| 5 | returns_returns_skewness_20_price_returns | 0.004079 |
| 4 | returns_returns_volatility_20_price_returns | 0.003821 |
| 48 | volume_volume_price_correlation_10_std7 | 0.003815 |
| 25 | momentum_roc_30_price_returns_ma7 | 0.003740 |
| 33 | trend_vectorbt_zigzag_5.0_3_ma7 | 0.003451 |
| 19 | returns_ljung_box_pvalue_20_10_ma5 | 0.003256 |
| 51 | trend_vectorbt_zigzag_5.0_2_std7 | 0.003073 |
| 58 | returns_simple_returns_5_price_returns_std7 | 0.003019 |
| 2 | volume_volume_price_correlation_20 | 0.003000 |
| 28 | volume_volume_trend_strength_10_30_ma7 | 0.003000 |
| 38 | returns_log_returns_10_price_returns_ma7 | 0.002972 |
| 10 | returns_simple_returns_1_price_returns_ma3 | 0.002898 |
| 8 | momentum_rsi_30_returns_vwap_ma3 | 0.002803 |
| 55 | oscillator_adx_14_returns_vwap_std7 | 0.002772 |
| 29 | volume_volume_trend_strength_20_50_ma7 | 0.002744 |
| 60 | microstructure_price_impact_std7 | 0.002724 |
| 30 | volume_volume_volatility_elasticity_20_ma7 | 0.002668 |
| 23 | trend_vectorbt_zigzag_10.0_5_std5 | 0.002608 |
| 15 | trend_vectorbt_zigzag_3.0_3_std3 | 0.002442 |
| 53 | trend_vectorbt_zigzag_10.0_3_std7 | 0.002426 |
| 43 | momentum_rsi_14_returns_vwap_std7 | 0.002333 |
| 49 | volume_vectorbt_enhanced_obv_10_std7 | 0.002309 |
| 20 | microstructure_trade_size_imbalance_ma5 | 0.002155 |
| 32 | trend_vectorbt_zigzag_5.0_2_ma7 | 0.002140 |
| 52 | trend_vectorbt_zigzag_10.0_2_std7 | 0.002139 |
| 7 | returns_ljung_box_pvalue_20_10 | 0.002131 |
| 24 | momentum_roc_21_price_returns_ma7 | 0.002020 |
| 14 | volume_analyst_volume_pressure_std3 | 0.001895 |
| 40 | returns_cumulative_returns_20_price_returns_ma7 | 0.001851 |
| 31 | trend_vectorbt_zigzag_3.0_2_ma7 | 0.001848 |
| 36 | oscillator_aroon_25_returns_vwap_ma7 | 0.001839 |
| 17 | returns_simple_returns_1_price_returns_ma5 | 0.001817 |
| 11 | returns_returns_kurtosis_20_price_returns_ma3 | 0.001778 |
| 9 | trend_vectorbt_zigzag_10.0_5_ma3 | 0.001602 |
| 37 | returns_log_returns_5_price_returns_ma7 | 0.001546 |
| 59 | returns_simple_returns_10_price_returns_std7 | 0.001443 |
| 39 | returns_simple_returns_5_price_returns_ma7 | 0.001332 |
