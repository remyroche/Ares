# LGBM/SHAP Feature Selection Report

Generated: 2025-10-19T15:31:38.193006

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
| 1 | roc_14_price_returns | 0.800000 | 626.152996 | 1.000 | 803298433531.365601 | 1.000 | ✓ | 0.600000 | 0.200000 | -0.000000 |
| 2 | support_level_5_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 3 | fibonacci_0.382_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 4 | volume_price_correlation_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 5 | vectorbt_parkinson_volatility_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 6 | candlestick_shooting_star_pattern | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 7 | resistance_level_2_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 8 | vectorbt_acceleration_correlation_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 9 | ctf_ratio_momentum_5_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 10 | enhanced_volatility_100 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 11 | roc_30_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 12 | vectorbt_yang_zhang_volatility_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 13 | support_level_3_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 14 | ctf_5m_volatility_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 15 | volume_percentile_100 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 16 | cycle_length | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 17 | volume_ema_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 18 | return_entropy_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 19 | vectorbt_volatility_comprehensive_30 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 20 | volume_entropy_20_volume_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 21 | vectorbt_atr_14 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 22 | vectorbt_momentum_comprehensive_21 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 23 | resistance_level_4_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 24 | support_level_3_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 25 | fibonacci_0.236_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 26 | fibonacci_0.786_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 27 | fibonacci_0.382_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 28 | vectorbt_acceleration_consistency_10_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 29 | vectorbt_bbands_10_2.0 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 30 | ctf_30m_momentum_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 31 | momentum_21_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 32 | vectorbt_acceleration_regime_10_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 33 | volume_momentum_5 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 34 | sharpe_ratio_20_0.0_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 35 | vectorbt_volatility_comprehensive_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 36 | vectorbt_multi_timeframe_acceleration_5_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 37 | vectorbt_acceleration_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 38 | analyst_momentum_15m | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 39 | vectorbt_momentum_acceleration_10_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 40 | momentum_endpoints_sma_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 41 | volume_sma_5 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 42 | resistance_level_5_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 43 | vectorbt_acceleration_divergence_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 44 | candlestick_long_legged_doji_pattern | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 45 | apo_12_26_returns_vwap | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 46 | volume_oscillator_5_15 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 47 | vectorbt_enhanced_obv_50 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 48 | advanced_cumulative_returns_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 49 | ctf_5m_momentum_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 50 | vectorbt_enhanced_obv_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |

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
| roc_14_price_returns | support_level_5_5_price_returns | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | fibonacci_0.382_10_price_returns | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | volume_price_correlation_10 | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | vectorbt_parkinson_volatility_20 | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | candlestick_shooting_star_pattern | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | resistance_level_2_20_price_returns | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | vectorbt_acceleration_correlation_20_price_returns | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | ctf_ratio_momentum_5_20_price_returns | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | enhanced_volatility_100 | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | roc_30_price_returns | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | vectorbt_yang_zhang_volatility_10 | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | support_level_3_10_price_returns | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | ctf_5m_volatility_price_returns | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | volume_percentile_100 | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | cycle_length | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | volume_ema_20 | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | return_entropy_5_price_returns | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | vectorbt_volatility_comprehensive_30 | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | volume_entropy_20_volume_returns | 0.000000 | 10.995323 | 0.000000 |
| roc_14_price_returns | vectorbt_atr_14 | 0.000000 | 10.995323 | 0.000000 |
