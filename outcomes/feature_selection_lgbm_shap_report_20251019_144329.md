# LGBM/SHAP Feature Selection Report

Generated: 2025-10-19T14:43:29.664636

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
| 1 | pfe_12_returns_vwap | 0.800000 | 626.152996 | 1.000 | 803298433531.365601 | 1.000 | ✓ | 0.600000 | 0.200000 | -0.000000 |
| 2 | volume_ratio_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 3 | candlestick_inverted_hammer_pattern | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 4 | vwap_deviations_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 5 | vectorbt_acceleration_volatility_5_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 6 | volume_percentile_50 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 7 | cycle_length | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 8 | support_level_2_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 9 | support_level_3_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 10 | stochastic_14_3_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 11 | fibonacci_0.236_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 12 | fibonacci_0.786_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 13 | candlestick_shooting_star_pattern | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 14 | volume_price_divergence_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 15 | advanced_momentum_5_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 16 | price_entropy_ma_5_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 17 | resistance_level_3_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 18 | vectorbt_rogers_satchell_volatility_14 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 19 | resistance_level_1_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 20 | vectorbt_volatility_comprehensive_30 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 21 | vectorbt_momentum_acceleration_5_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 22 | momentum_30_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 23 | volume_entropy_10_volume_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 24 | volume_accumulation_distribution | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 25 | vectorbt_garman_klass_volatility_50 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 26 | cumulative_returns_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 27 | fibonacci_0.5_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 28 | volume_zscore_60_252 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 29 | volume_ratio_50 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 30 | pivot_point_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 31 | price_entropy_ma_20_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 32 | vectorbt_enhanced_ad_line_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 33 | volume_sma_50 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 34 | support_level_1_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 35 | ctf_30m_volume_volume_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 36 | fibonacci_0.382_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 37 | momentum_21_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 38 | vectorbt_garman_klass_volatility_30 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 39 | vectorbt_bbands_10_1.5 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 40 | dfa_slopes | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 41 | fractal_dimension | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 42 | rsi_zscore_14_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 43 | order_flow_imbalance_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 44 | price_entropy_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 45 | ctf_ratio_sma_10_50_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 46 | price_entropy_ma_5_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 47 | ctf_divergence_volatility_5_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 48 | vectorbt_parkinson_volatility_30 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 49 | fibonacci_0.5_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 50 | vectorbt_yang_zhang_volatility_50 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |

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
| pfe_12_returns_vwap | volume_ratio_20 | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | candlestick_inverted_hammer_pattern | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | vwap_deviations_20 | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | vectorbt_acceleration_volatility_5_10_price_returns | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | volume_percentile_50 | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | cycle_length | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | support_level_2_5_price_returns | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | support_level_3_5_price_returns | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | stochastic_14_3_price_returns | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | fibonacci_0.236_10_price_returns | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | fibonacci_0.786_5_price_returns | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | candlestick_shooting_star_pattern | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | volume_price_divergence_20 | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | advanced_momentum_5_20 | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | price_entropy_ma_5_10_price_returns | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | resistance_level_3_20_price_returns | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | vectorbt_rogers_satchell_volatility_14 | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | resistance_level_1_10_price_returns | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | vectorbt_volatility_comprehensive_30 | 0.000000 | 10.112659 | 0.000000 |
| pfe_12_returns_vwap | vectorbt_momentum_acceleration_5_20_price_returns | 0.000000 | 10.112659 | 0.000000 |
