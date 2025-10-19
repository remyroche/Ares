# LGBM/SHAP Feature Selection Report

Generated: 2025-10-19T14:59:33.822015

## OOF Settings

- Enabled: True
- Splits: 5
- Total validation rows: 968640

## Summary

- Features considered: 60
- Top-K for SHAP: 60
- Final selection target: 60
- Selection method: combined_shap_lgb_interaction

## Detailed Statistics

### SHAP Score Distribution

- Min: 0.000000
- Max: 626.152996
- Mean: 10.435883
- Median: 0.000000
- Std: 80.159540
- Q25: 0.000000
- Q75: 0.000000
- Q90: 0.000000
- Q95: 0.000000
- Q99: 256.722728
- Non-zero count: 1
- Zero count: 59

### LGB Score Distribution

- Min: 0.000000
- Max: 803298433531.365601
- Mean: 13388307225.522760
- Median: 0.000000
- Std: 102837539116.482758
- Q25: 0.000000
- Q75: 0.000000
- Q90: 0.000000
- Q95: 0.000000
- Q99: 329352357747.857178
- Non-zero count: 1
- Zero count: 59

### COMBINED Score Distribution

- Min: 0.000000
- Max: 0.800000
- Mean: 0.013333
- Median: 0.000000
- Std: 0.102415
- Q25: 0.000000
- Q75: 0.000000
- Q90: 0.000000
- Q95: 0.000000
- Q99: 0.328000
- Non-zero count: 1
- Zero count: 59

### Score Correlations

- SHAP ↔ LGB: 1.0000
- SHAP ↔ Combined: 1.0000
- LGB ↔ Combined: 1.0000

### Threshold Analysis

- Threshold value: 0.000100
- Features above threshold: 1
- Features below threshold: 59
- Threshold percentage: 1.67%
- Max score above threshold: 0.800000
- Min score above threshold: 0.800000
- Max score below threshold: 0.000000

## Methodology

- SHAP weight: 0.6
- LGB weight: 0.2
- Interaction weight: 0.2
- Normalization: max_normalization
- Threshold method: fixed_threshold
- Fallback method: top_n_selection

## Top Features (by combined score)

| Rank | Feature | Combined | SHAP | SHAP_Norm | LGB | LGB_Norm | Above_Thresh | SHAP_W | LGB_W | INTER_W |

|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | volume_momentum_5 | 0.800000 | 626.152996 | 1.000 | 803298433531.365601 | 1.000 | ✓ | 0.600000 | 0.200000 | -0.000000 |
| 2 | pfe_12_returns_vwap | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 3 | support_level_2_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 4 | lempel_ziv_complexity_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 5 | candlestick_three_white_soldiers_pattern | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 6 | momentum_endpoints_sma_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 7 | log_returns_1_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 8 | vectorbt_parkinson_volatility_50 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 9 | stochastic_30_3_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 10 | vectorbt_acceleration_volatility_10_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 11 | vectorbt_acceleration_trend_strength_5_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 12 | candlestick_engulfing_pattern | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 13 | resistance_level_4_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 14 | sma_100_returns_vwap | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 15 | resistance_level_3_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 16 | vectorbt_acceleration_correlation_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 17 | volume_sma_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 18 | vectorbt_enhanced_obv_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 19 | vectorbt_volatility_comprehensive_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 20 | shannon_entropy_20_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 21 | macd_delta_12_26_9 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 22 | vectorbt_acceleration_regime_5_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 23 | vectorbt_acceleration_consistency_5_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 24 | cmf_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 25 | vectorbt_bbands_14_1.5 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 26 | analyst_volume_trend | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 27 | price_entropy_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 28 | tema_21_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 29 | resistance_level_5_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 30 | support_level_1_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 31 | ctf_15m_trend_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 32 | resistance_level_4_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 33 | price_volume_oscillator_5_15 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 34 | ema_12_returns_vwap | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 35 | kama_30_2_30_returns_vwap | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 36 | fibonacci_0.236_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 37 | vectorbt_bbands_20_1.5 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 38 | mama_21_0.05_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 39 | fibonacci_0.236_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 40 | resistance_level_5_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 41 | support_level_1_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 42 | williams_r_30_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 43 | log_returns_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 44 | sample_entropy_20_2_0.2 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 45 | ema_50_returns_vwap | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 46 | volume_momentum_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 47 | volume_roc_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 48 | sma_20_returns_vwap | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 49 | ultimate_oscillator_7_14_28_returns_vwap | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 50 | fibonacci_0.236_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |

## Interaction Summary

- Total interactions: 1770
- Strong interactions: 0
- Avg strength: 0.000000
- Max strength: 0.000000
- Interaction coverage: 21 features
- Strong interaction ratio: 0.0000

## Top Interactions

| f1 | f2 | strength | imp1 | imp2 |
|---|---|---:|---:|---:|
| volume_momentum_5 | pfe_12_returns_vwap | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | support_level_2_20_price_returns | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | lempel_ziv_complexity_20 | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | candlestick_three_white_soldiers_pattern | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | momentum_endpoints_sma_20 | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | log_returns_1_price_returns | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | vectorbt_parkinson_volatility_50 | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | stochastic_30_3_price_returns | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | vectorbt_acceleration_volatility_10_10_price_returns | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | vectorbt_acceleration_trend_strength_5_20_price_returns | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | candlestick_engulfing_pattern | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | resistance_level_4_5_price_returns | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | sma_100_returns_vwap | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | resistance_level_3_5_price_returns | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | vectorbt_acceleration_correlation_20_price_returns | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | volume_sma_10 | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | vectorbt_enhanced_obv_10 | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | vectorbt_volatility_comprehensive_20 | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | shannon_entropy_20_10 | 0.000000 | 10.262252 | 0.000000 |
| volume_momentum_5 | macd_delta_12_26_9 | 0.000000 | 10.262252 | 0.000000 |
