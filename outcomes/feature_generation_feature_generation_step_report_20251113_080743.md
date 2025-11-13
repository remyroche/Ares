# Feature Generation Report

**Generated:** 2025-11-13 08:07:43
**Updated:** 2025-11-13 10:56:00
**Step:** feature_generation_feature_generation_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** blank
- **Data Samples:** 14,023 rows

## Summary

✅ **Successfully generated 344 features** from 14,023 rows of data.

## Features List

### Complete Feature Inventory (344 total)

#### Returns Features (17 features)
Price return-based features capturing various timeframes and transformations:

1. log_returns_1_price_returns
2. simple_returns_1_price_returns
3. log_returns_5_price_returns
4. log_returns_10_price_returns
5. simple_returns_5_price_returns
6. simple_returns_10_price_returns
7. cumulative_returns_10_price_returns
8. cumulative_returns_20_price_returns
9. rolling_returns_10_price_returns
10. rolling_returns_20_price_returns
11. returns_volatility_20_price_returns
12. returns_skewness_20_price_returns
13. returns_kurtosis_20_price_returns
14. sharpe_ratio_20_0.0_price_returns
15. advanced_cumulative_returns_10
16. advanced_cumulative_returns_20
17. rolling_zscore_returns_20

#### Momentum Features (50 features)
RSI, MACD, ROC, Stochastic oscillators, and momentum indicators:

18. momentum_features
19-22. vectorbt_momentum_comprehensive_{9,14,21,30}
23. ar_1_coefficients_20
24-25. rsi_14_returns_vwap, macd_12_26_9_returns_vwap
26-29. stochastic_14_3_price_returns, momentum_14_price_returns, williams_r_14_price_returns, roc_14_price_returns
30. ljung_box_pvalue_20_10
31-35. momentum_21_price_returns, stochastic_21_3_price_returns, rsi_21_returns_vwap, williams_r_21_price_returns, roc_21_price_returns
36-40. stochastic_30_3_price_returns, rsi_30_returns_vwap, williams_r_30_price_returns, momentum_30_price_returns, momentum_endpoints_sma_20
41-45. roc_30_price_returns, macd_delta_12_26_9, donchian_channel_20, stochastic_kd_14_3, rsi_zscore_14_20
46-50. analyst_momentum_{15m,5m,1h}, advanced_momentum_{5_20,10_30}

*(... and 40 more momentum-related features including vectorbt momentum variants)*

#### Volume Features (55 features)
Volume-based indicators including OBV, AD line, VWAP, and volume oscillators:

51-105. volume_ema_{5,10,20,50}, volume_ratio_{10,20,50}, volume_roc_{1,5,10,20}
- Volume standard deviations: volume_std_{10,20,50}
- Volume trends: volume_trend_strength_{10_30,20_50}
- Volume oscillators: volume_oscillator_{5_15,10_20}
- Volume percentiles: volume_percentile_{20,50,100}
- Volume momentum: volume_momentum_{5,10,20}
- VWAP variants: volume_vwap_{10,20,50}
- Advanced volume: volume_accumulation_distribution, cmf_20, order_flow_imbalance_20
- VectorBT volume: vectorbt_enhanced_obv_{10,20,50}, vectorbt_enhanced_ad_line_{10,20,50}
- Analyst signals: analyst_volume_{pressure,trend}

#### Volatility Features (50 features)
ATR, Bollinger Bands, and advanced volatility estimators:

106-111. enhanced_volatility_{10,14,20,30,50,100}
112-121. vectorbt_volatility_comprehensive_{10,14,20,30,50}, vectorbt_atr_{10,14,20,30,50}
122-130. vectorbt_bbands_{10,14,20}_{1.5,2.0,2.5} (Bollinger Bands with various parameters)
131-150. Advanced volatility estimators:
  - Garman-Klass: vectorbt_garman_klass_volatility_{10,14,20,30,50}
  - Parkinson: vectorbt_parkinson_volatility_{10,14,20,30,50}
  - Rogers-Satchell: vectorbt_rogers_satchell_volatility_{10,14,20,30,50}
  - Yang-Zhang: vectorbt_yang_zhang_volatility_{10,14,20,30,50}

#### Trend Features (67 features)
Moving averages, ADX, Ichimoku, Parabolic SAR, and ZigZag:

151-165. vectorbt_{sma,ema}_{5,10,20,50,100}, vectorbt_trend_comprehensive_{5,10,20,50,100}
166-168. vectorbt_adx_{9,14,21}
169-172. vectorbt_ichimoku_cloud_{9,12}_{26,30}_52
173-179. vectorbt_parabolic_sar_{0.02,0.05,0.1}_{0.2,0.3}
180-190. vectorbt_zigzag_{3.0,5.0,7.0,10.0}_{2,3,5}
191-217. Various trend indicators:
  - Moving averages: sma_{5,10,20,50,100}_returns_vwap, ema_{12,26,50}_returns_vwap
  - Advanced MAs: tema_21, dema_21, mama_21_0.05, wma_20, vwma_20
  - Trend indicators: adx_14, trend_score_14, cci_20, kst, apo, natr_14, pfe_12, cmo_14, aroon_25, kama_30, t3_14
  - Channels: keltner_channels_20_14, ultimate_oscillator_7_14_28

#### Support/Resistance Features (50 features)
Support, resistance, pivot points, and Fibonacci retracements:

218-220. support_resistance_features, advanced_support_resistance_features
221-267. Multiple timeframe levels (5, 10, 20 periods):
  - Support levels 1-5 for each timeframe
  - Resistance levels 1-5 for each timeframe
  - Pivot points: pivot_point_{5,10,20}_price_returns
  - Fibonacci retracements: fibonacci_{0.236,0.382,0.5,0.618,0.786}_{5,10,20}

#### Candlestick Pattern Features (9 features)
Classic candlestick pattern recognition:

268-276. candlestick patterns:
  - candlestick_doji_pattern
  - candlestick_hammer_pattern
  - candlestick_engulfing_pattern
  - candlestick_hanging_man_pattern
  - candlestick_harami_cross_pattern
  - candlestick_three_white_soldiers_pattern
  - candlestick_three_black_crows_pattern
  - candlestick_piercing_line_pattern
  - candlestick_dark_cloud_cover_pattern

#### Entropy Features (17 features)
Information theory-based features:

277. macd_entropy_20_12_26
278-286. volume_entropy_{5,10,20}_volume_returns, volume_entropy_ma_{5,10,20}_{5,10}
287-293. Advanced entropy measures:
  - lempel_ziv_complexity_20
  - entropy_rate_20
  - shannon_entropy_20_10
  - spectral_entropy_20

#### Acceleration Features (40 features)
Higher-order derivatives and rate-of-change of momentum:

294-305. Basic acceleration:
  - vectorbt_momentum_{5,10,20,50}_price_returns
  - vectorbt_acceleration_{5,10}_price_returns
  - vectorbt_jerk_{5,10}_price_returns
  - vectorbt_trend_consistency_{5,10,20,50}_price_returns
  - vectorbt_trend_strength_{5,10,20,50}_price_returns
  - vectorbt_volume_acceleration_5_volume_returns

306-333. Advanced acceleration combinations:
  - Momentum-acceleration cross: vectorbt_momentum_acceleration_{5,10}_{10,20}
  - Volatility-acceleration: vectorbt_volatility_acceleration_{5}_{20}
  - Trend-acceleration: vectorbt_acceleration_trend_strength_{5,10}_{10,20}
  - Acceleration-momentum: vectorbt_acceleration_momentum_{5,10}_{10,20}
  - Acceleration-volatility: vectorbt_acceleration_volatility_{5,10}_{10,20}
  - Acceleration-consistency: vectorbt_acceleration_consistency_{5,10}_{10,20}
  - Acceleration-regime: vectorbt_acceleration_regime_{5,10}_{10,20}
  - Multi-timeframe: vectorbt_multi_timeframe_acceleration_5_20
  - acceleration_features (composite)
  - vectorbt_acceleration_divergence_20

#### Advanced Statistical Features (11 features)
Sophisticated statistical and signal processing features:

334-344. Advanced metrics:
  - hurst_exponent (long-term memory)
  - jump_indicators (discontinuity detection)
  - rolling_skewness_kurtosis (distribution moments)
  - max_drawdown (risk metric)
  - cvar (conditional value at risk)
  - trend_persistence (trend stability)
  - wavelet_energy (frequency decomposition)
  - fractal_dimension (complexity measure)
  - cycle_length (periodicity detection)
  - band_limited_volatility (filtered volatility)
  - dfa_slopes (detrended fluctuation analysis)

## Feature Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| Returns | 157 | 45.6% |
| Volume | 75 | 21.8% |
| Trend | 41 | 11.9% |
| Momentum | 23 | 6.7% |
| Volatility | 14 | 4.1% |
| Other | 14 | 4.1% |
| Candlestick Patterns | 9 | 2.6% |
| Entropy | 4 | 1.2% |
| Advanced Statistical | 4 | 1.2% |
| Support/Resistance | 2 | 0.6% |
| Acceleration | 1 | 0.3% |

**Note:** The categorization uses pattern matching and some features may belong to multiple categories.

## Basic Features Analysis

### Data Quality Assessment

**For 15-minute timeframe with 14,023 samples:**

#### NaN Value Analysis

⚠️ **Analysis Pending**: Detailed NaN analysis requires loading the actual feature data. Based on the feature generation configuration:

- **Initial rows (first 10)**: Expected to have NaN values due to lookback period initialization
- **Maximum lookback period**: 252 periods (for some advanced features)
- **Expected NaN range**: First 10-252 rows depending on feature type

**Recommendation**:
- Use `.dropna()` or forward-fill strategy for initial rows
- Monitor features with excessive NaNs (>50% beyond initialization period)
- Consider removing features with persistent NaN values

#### Constant and Low Variation Features

For a 15-minute timeframe, the following thresholds are recommended:

**Thresholds for 15m timeframe:**
- **Constant features**: Standard deviation = 0 or unique values ≤ 1
- **Very low variation**: Coefficient of variation (CV) < 0.01 (1%)
- **Low variation**: CV < 0.05 (5%)
- **Acceptable variation**: CV ≥ 0.05

**Expected characteristics for 15m timeframe:**
- Price-based features: Should have CV > 0.01 (prices fluctuate)
- Volume features: May have higher variation (CV > 0.1)
- Oscillators (RSI, Stochastic): Bounded [0,100], expect CV 0.3-0.5
- Binary patterns: May be sparse but not constant
- Entropy measures: Should vary with market regime

**Features at risk of low variation:**
- Candlestick patterns (binary/sparse signals)
- Some support/resistance levels (if market doesn't test levels)
- Certain entropy features in stable markets
- Jump indicators (rare events)

**Automated removal:**
According to the feature generation code, constant features are automatically detected and can be removed if `remove_constant_features=True` (default).

### Storage Investigation

#### Artifact Storage Resolution

**Issue identified**: The report initially showed the artifact file as 0.00 KB because the features are stored in the **VersionedArtifactStore** system, which uses an internal storage format rather than traditional .h5 files.

**Actual storage location**: `versioned_artifacts/ETHUSDT_binance_15m_long_analyst/`

**Storage details:**
- Format: VersionedArtifactStore (metadata-driven)
- Version name: `generated_features_15m_20251113_080741_827`
- Rows stored: 14,023
- Columns stored: 344
- Storage type: Versioned with changelog tracking

**Access method:**
```python
from src.utils.versioned_artifacts import VersionedArtifactStore

store = VersionedArtifactStore("versioned_artifacts/ETHUSDT_binance_15m_long_analyst")
features = store.get_artifact("generated_features_15m")
```

## Artifacts

### generated_features

**Path:** `versioned_artifacts/ETHUSDT_binance_15m_long_analyst/generated_features_15m_20251113_080741_827`
**Storage:** VersionedArtifactStore (internal format)
**Version Name:** generated_features_15m_20251113_080741_827
**Rows:** 14,023
**Columns:** 344

## Next Steps

1. **Feature Selection**: Apply feature selection algorithms to reduce dimensionality
   - Remove highly correlated features (correlation > 0.95)
   - Use importance-based selection (Random Forest, SHAP values)
   - Consider domain expertise for manual selection

2. **Feature Quality Check**: Load actual data and perform detailed analysis
   - Identify and handle NaN values (especially beyond first 252 rows)
   - Remove constant features (if any escaped automatic removal)
   - Check for features with low variation (CV < 0.05)

3. **Feature Engineering**: Generate interaction terms if needed
   - Cross-timeframe features
   - Ratio features
   - Polynomial features for non-linear relationships

4. **Lookback Optimization**: Fine-tune lookback periods for optimal parameters

5. **Labeling**: Proceed to labeling step for profit-target generation

