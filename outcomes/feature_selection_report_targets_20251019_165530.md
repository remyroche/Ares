# 🚀 Feature Selection Comprehensive Report
**Generated:** 2025-10-19 16:55:31
**Execution Time:** 704.355 seconds

## 📊 Overall Summary
- **Initial Features:** 355
- **Final Features:** 60
- **Reduction Ratio:** 16.9%
- **Target Variable:** Provided
- **Target Distribution:** {0.0: 1152066, 1.0: 10302}

## 📈 Data Statistics
- **Total Samples:** 1,162,368
- **Total Features:** 355
- **Numeric Features:** 355
- **Categorical Features:** 0
- **Missing Values:** 0 (0.00%)

## 🎯 Target Statistics
- **Target Count:** 1,162,368
- **Missing Targets:** 0
- **Mean:** 0.008863
- **Std:** 0.093725
- **Min:** 0.000000
- **Max:** 1.000000
- **Median:** 0.000000

## 🏆 Feature Score Statistics
- **Total Features Selected:** 60

### Combined Score Distribution
- **Min:** 0.000000
- **Max:** 0.000000
- **Mean:** 0.000000
- **Median:** 0.000000
- **Std:** 0.000000
- **Q25:** 0.000000
- **Q75:** 0.000000
- **Q90:** 0.000000
- **Q95:** 0.000000

### SHAP Score Distribution
- **Min:** 0.000000
- **Max:** 0.000000
- **Mean:** 0.000000
- **Median:** 0.000000
- **Std:** 0.000000

### LGB Score Distribution
- **Min:** 0.000000
- **Max:** 0.000000
- **Mean:** 0.000000
- **Median:** 0.000000
- **Std:** 0.000000

### Stability Score Distribution
- **Min:** 3013.295843
- **Max:** 3013.295843
- **Mean:** 3013.295843
- **Median:** 3013.295843
- **Std:** 0.000000

### Uniqueness Score Distribution
- **Min:** 0.000000
- **Max:** 0.000000
- **Mean:** 0.000000
- **Median:** 0.000000
- **Std:** 0.000000

## 🔍 Stage 1: Lightweight Screening
- **Duration:** 412.583 seconds
- **Initial Features:** 355
- **Features After Screening:** 100
- **Features Removed:** 255
- **Reduction Ratio:** 28.2%
- **Methods Used:** correlation, stability, mutual_info

## 🚀 Stage 2: LGBM/SHAP Advanced Selection
### Model Configuration
- **Input Features:** 100
- **Target Selection Count:** 60
- **Duration:** 35.648 seconds
- **Features Selected:** 60

### SHAP Analysis
- **SHAP Sample Size:** 1000

## 🎯 Stage 3: Final Validation
- **Input Features:** 60
- **Final Features:** 60

## 🔗 Feature Interaction Analysis
- **Total Interactions Analyzed:** 1770
- **Strong Interactions:** 0
- **Average Interaction Strength:** 0.000000
- **Max Interaction Strength:** 0.000000

### Top Feature Interactions
1. **ctf_15m_hl_price_levels** ↔ **volume_price_correlation_10**
   - Interaction Strength: 0.000000
   - ctf_15m_hl_price_levels Importance: 0.000002
   - volume_price_correlation_10 Importance: 0.000000

2. **ctf_15m_hl_price_levels** ↔ **ar_1_coefficients_20**
   - Interaction Strength: 0.000000
   - ctf_15m_hl_price_levels Importance: 0.000002
   - ar_1_coefficients_20 Importance: 0.000000

3. **ctf_15m_hl_price_levels** ↔ **candlestick_abandoned_baby_pattern**
   - Interaction Strength: 0.000000
   - ctf_15m_hl_price_levels Importance: 0.000002
   - candlestick_abandoned_baby_pattern Importance: 0.000000

4. **ctf_15m_hl_price_levels** ↔ **ctf_divergence_volatility_5_20_price_returns**
   - Interaction Strength: 0.000000
   - ctf_15m_hl_price_levels Importance: 0.000002
   - ctf_divergence_volatility_5_20_price_returns Importance: 0.000000

5. **ctf_15m_hl_price_levels** ↔ **rolling_returns_20_price_returns**
   - Interaction Strength: 0.000000
   - ctf_15m_hl_price_levels Importance: 0.000002
   - rolling_returns_20_price_returns Importance: 0.000000

6. **ctf_15m_hl_price_levels** ↔ **support_level_4_10_price_returns**
   - Interaction Strength: 0.000000
   - ctf_15m_hl_price_levels Importance: 0.000002
   - support_level_4_10_price_returns Importance: 0.000000

7. **ctf_15m_hl_price_levels** ↔ **ctf_ratio_momentum_5_20_price_returns**
   - Interaction Strength: 0.000000
   - ctf_15m_hl_price_levels Importance: 0.000002
   - ctf_ratio_momentum_5_20_price_returns Importance: 0.000000

8. **ctf_15m_hl_price_levels** ↔ **vectorbt_enhanced_obv_20**
   - Interaction Strength: 0.000000
   - ctf_15m_hl_price_levels Importance: 0.000002
   - vectorbt_enhanced_obv_20 Importance: 0.000000

9. **ctf_15m_hl_price_levels** ↔ **return_entropy_ma_10_10_price_returns**
   - Interaction Strength: 0.000000
   - ctf_15m_hl_price_levels Importance: 0.000002
   - return_entropy_ma_10_10_price_returns Importance: 0.000000

10. **ctf_15m_hl_price_levels** ↔ **support_level_1_5_price_returns**
   - Interaction Strength: 0.000000
   - ctf_15m_hl_price_levels Importance: 0.000002
   - support_level_1_5_price_returns Importance: 0.000000

## 🏆 Final Selected Features
**Total:** 60 features selected

### 🥇 Top 20 Features (by combined score)
| Rank | Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|---:|
| 1 | ctf_15m_hl_price_levels | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 2 | volume_price_correlation_10 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 3 | ar_1_coefficients_20 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 4 | candlestick_abandoned_baby_pattern | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 5 | ctf_divergence_volatility_5_20_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 6 | rolling_returns_20_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 7 | support_level_4_10_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 8 | ctf_ratio_momentum_5_20_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 9 | vectorbt_enhanced_obv_20 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 10 | return_entropy_ma_10_10_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 11 | support_level_1_5_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 12 | fibonacci_0.5_5_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 13 | dema_21_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 14 | volume_momentum_10 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 15 | vectorbt_volatility_comprehensive_20 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 16 | tema_21_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 17 | vectorbt_bbands_14_2.5 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 18 | volume_percentile_20 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 19 | ctf_corr_momentum_5_15_20_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| 20 | keltner_channels_20_14_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

### 📊 Features by Category
#### CTF Features (6)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| ctf_15m_hl_price_levels | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| ctf_ratio_momentum_5_20_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| ctf_corr_momentum_5_15_20_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| ctf_30m_trend_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| ctf_30m_volatility_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### VOLUME Features (7)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| volume_price_correlation_10 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| volume_momentum_10 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| volume_percentile_20 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| volume_momentum_20 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| volume_std_50 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| volume_ratio_10 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| volume_entropy_ma_5_5_volume_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### AR Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| ar_1_coefficients_20 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### CANDLESTICK Features (4)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| candlestick_abandoned_baby_pattern | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| candlestick_piercing_line_pattern | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| candlestick_three_white_soldiers_pattern | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| candlestick_harami_pattern | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### ROLLING Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| rolling_returns_20_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### SUPPORT Features (3)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| support_level_4_10_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| support_level_1_5_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| support_level_1_20_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### VECTORBT Features (17)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| vectorbt_enhanced_obv_20 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_volatility_comprehensive_20 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_bbands_14_2.5 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_volatility_acceleration_5_20_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_atr_30 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_bbands_20_1.5 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_jerk_10_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_bbands_14_1.5 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_volume_acceleration_5_volume_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_acceleration_regime_5_10_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_yang_zhang_volatility_30 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_acceleration_trend_strength_10_10_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_atr_10 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_acceleration_volatility_5_10_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_trend_consistency_10_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_parkinson_volatility_50 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| vectorbt_volume_weighted_ad_line_50 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### RETURN Features (2)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| return_entropy_ma_10_10_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| return_entropy_10_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### FIBONACCI Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| fibonacci_0.5_5_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### DEMA Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| dema_21_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### TEMA Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| tema_21_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### KELTNER Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| keltner_channels_20_14_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### STOCHASTIC Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| stochastic_14_3_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### SHARPE Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| sharpe_ratio_20_0.0_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### MOMENTUM Features (2)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| momentum_30_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| momentum_21_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### RESISTANCE Features (2)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| resistance_level_5_10_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| resistance_level_5_5_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### DFA Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| dfa_slopes | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### ENHANCED Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| enhanced_volatility_50 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### LOG Features (2)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| log_returns_5_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |
| log_returns_10_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### CYCLE Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| cycle_length | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### T3 Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| t3_14_0.7_returns_vwap | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### SIMPLE Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| simple_returns_10_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### ROC Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| roc_14_price_returns | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

#### MACD Features (1)
| Feature | Combined | SHAP | LGB | Stability | Uniqueness |
|---|---:|---:|---:|---:|---:|
| macd_entropy_20_12_26 | 0.000000 | 0.000000 | 0.000000 | 3013.295843 | 0.000000 |

## 📈 Performance Summary
- **Total Execution Time:** 704.355 seconds
- **Memory Efficient:** Float32 optimization enabled
- **Algorithm:** LightGBM + TreeSHAP
- **Interaction Analysis:** Included

## 💡 Recommendations
✅ **Good reduction achieved** - Reasonable feature set maintained
🚀 **LightGBM optimization active** - 5-20x faster than traditional methods
🎯 **TreeSHAP importance scores** - More accurate than traditional methods
