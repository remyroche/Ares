# Regime Models Training Report

**Symbol:** ETHUSDT
**Primary Model:** catboost
**Generated:** 2025-11-16T14:13:20.783035
**Report Version:** 1.0

## Top 3 Models Comparison

| Rank | Model Name | Accuracy | F1-Score | Combined Score |
|------|------------|----------|----------|----------------|
| 1 | catboost | 0.4251 | 0.2560 | 35.6468 |
| 2 | xgboost | 0.4251 | 0.2560 | 35.6468 |
| 3 | lightgbm | 0.4234 | 0.2785 | 0.6221 |

## Overall Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 7456 |
| Number of Regimes | 4 |
| Mean Max Probability | 0.3942 |
| Std Max Probability | 0.0066 |
| Regime Balance | 13.1422 |
| Prediction Confidence | 0.3942 |
| Uncertainty Entropy | 1.1878 |

## Model Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.4936 |
| Precision (Weighted) | 0.2437 |
| Recall (Weighted) | 0.4936 |
| F1-Score (Weighted) | 0.3263 |
| Cohen's Kappa | 0.0000 |

## Regime Statistics

| Regime | Sample Count | Percentage | Mean Prob | Std Prob | High Conf | Med Conf | Low Conf |
|--------|--------------|------------|-----------|----------|-----------|----------|----------|
| regime_0 | 2172 | 29.1% | 0.394 | 0.007 | 0 | 0 | 2172 |
| regime_1 | 414 | 5.6% | 0.394 | 0.008 | 0 | 0 | 414 |
| regime_2 | 3139 | 42.1% | 0.394 | 0.006 | 0 | 0 | 3139 |
| regime_3 | 1731 | 23.2% | 0.394 | 0.007 | 0 | 0 | 1731 |

## Feature Importance Analysis

### SHAP Feature Importance (Top 60)

| Rank | Feature | SHAP Value |
|------|---------|------------|
| 640 | trend_dema_21_price_returns_std8 | 0.034664 |
| 162 | trend_vectorbt_parabolic_sar_0.1_0.3 | 0.034537 |
| 1351 | microstructure_range_volume_shock_open_30_ewm0.3 | 0.033590 |
| 1290 | trend_vectorbt_parabolic_sar_0.05_0.3_ewm0.3 | 0.014422 |
| 1156 | momentum_momentum_30_price_returns_ewm0.3 | 0.013303 |
| 243 | momentum_stochastic_14_3_price_returns_ma8 | 0.007363 |
| 306 | volatility_vectorbt_rogers_satchell_volatility_30_ma8 | 0.003851 |
| 536 | volatility_vectorbt_rogers_satchell_volatility_50_std8 | 0.002555 |
| 900 | microstructure_price_impact_ma20 | 0.000000 |
| 905 | microstructure_analyst_tick_imbalance_ma20 | 0.000000 |
| 891 | returns_sharpe_ratio_20_0.0_price_returns_ma20 | 0.000000 |
| 904 | microstructure_market_depth_ma20 | 0.000000 |
| 903 | microstructure_liquidity_proxy_ma20 | 0.000000 |
| 902 | microstructure_trade_intensity_ma20 | 0.000000 |
| 901 | microstructure_volume_weighted_price_ma20 | 0.000000 |
| 894 | returns_rolling_zscore_returns_20_ma20 | 0.000000 |
| 895 | returns_ar_1_coefficients_20_ma20 | 0.000000 |
| 899 | microstructure_trade_size_imbalance_ma20 | 0.000000 |
| 906 | microstructure_corwin_schultz_spread_momentum_ma20 | 0.000000 |
| 892 | returns_advanced_cumulative_returns_10_ma20 | 0.000000 |
| 897 | microstructure_microstructure_features_ma20 | 0.000000 |
| 896 | returns_ljung_box_pvalue_20_10_ma20 | 0.000000 |
| 893 | returns_advanced_cumulative_returns_20_ma20 | 0.000000 |
| 898 | microstructure_order_flow_imbalance_ma20 | 0.000000 |
| 907 | microstructure_amihud_illiquidity_vwap_distance_ma20 | 0.000000 |
| 889 | returns_returns_skewness_20_price_returns_ma20 | 0.000000 |
| 918 | momentum_vectorbt_momentum_comprehensive_14_std20 | 0.000000 |
| 926 | momentum_roc_14_price_returns_std20 | 0.000000 |
| 925 | momentum_momentum_14_price_returns_std20 | 0.000000 |
| 924 | momentum_williams_r_14_price_returns_std20 | 0.000000 |
| 923 | momentum_stochastic_14_3_price_returns_std20 | 0.000000 |
| 922 | momentum_macd_12_26_9_returns_vwap_std20 | 0.000000 |
| 921 | momentum_rsi_14_returns_vwap_std20 | 0.000000 |
| 920 | momentum_vectorbt_momentum_comprehensive_30_std20 | 0.000000 |
| 919 | momentum_vectorbt_momentum_comprehensive_21_std20 | 0.000000 |
| 917 | momentum_vectorbt_momentum_comprehensive_9_std20 | 0.000000 |
| 908 | microstructure_roll_lambda_rv_short_ma20 | 0.000000 |
| 916 | momentum_momentum_features_std20 | 0.000000 |
| 915 | regime_regime_volume_30_std20 | 0.000000 |
| 914 | regime_regime_volume_20_std20 | 0.000000 |
| 913 | regime_regime_volume_14_std20 | 0.000000 |
| 912 | regime_regime_structural_trend_features_std20 | 0.000000 |
| 911 | regime_regime_statistical_features_std20 | 0.000000 |
| 910 | regime_regime_volatility_features_std20 | 0.000000 |
| 909 | microstructure_range_volume_shock_open_30_ma20 | 0.000000 |
| 890 | returns_returns_kurtosis_20_price_returns_ma20 | 0.000000 |
| 886 | returns_rolling_returns_10_price_returns_ma20 | 0.000000 |
| 888 | returns_returns_volatility_20_price_returns_ma20 | 0.000000 |
| 887 | returns_rolling_returns_20_price_returns_ma20 | 0.000000 |
| 866 | trend_sma_100_returns_vwap_ma20 | 0.000000 |
| 865 | trend_sma_50_returns_vwap_ma20 | 0.000000 |
| 864 | trend_sma_20_returns_vwap_ma20 | 0.000000 |
| 863 | trend_sma_10_returns_vwap_ma20 | 0.000000 |
| 862 | trend_sma_5_returns_vwap_ma20 | 0.000000 |
| 861 | trend_vectorbt_zigzag_10.0_5_ma20 | 0.000000 |
| 860 | trend_vectorbt_zigzag_10.0_3_ma20 | 0.000000 |
| 859 | trend_vectorbt_zigzag_10.0_2_ma20 | 0.000000 |
| 858 | trend_vectorbt_zigzag_7.0_5_ma20 | 0.000000 |
| 857 | trend_vectorbt_zigzag_7.0_3_ma20 | 0.000000 |
| 856 | trend_vectorbt_zigzag_7.0_2_ma20 | 0.000000 |
