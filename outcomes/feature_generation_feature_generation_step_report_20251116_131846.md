# Feature Generation Report

**Generated:** 2025-11-16 13:18:46
**Step:** feature_generation_feature_generation_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** blank

## Summary

✅ **Successfully generated 259 features** from 31,303 rows of data.

## Artifacts

### generated_features

**Path:** `versioned_artifacts/ETHUSDT_binance_15m_long_analyst/generated_features_15m_20251116_131743_506.h5`
**Size:** 0.00 KB

### basic_feature_analysis_report

**Path:** `outcomes/basic_feature_analysis_20251116_131750.md`
**Size:** 2.01 KB

## Baseline Predictive Check

**Dataset:** 31302 samples, 259 features

### Top Single-Feature Signals

| Rank | Feature | Test R² | Pearson | Quality Score |
|------|---------|---------|---------|---------------|
| 1 | `candlestick_doji_pattern` | 0.003 | -0.038 | 0.017 |
| 2 | `candlestick_harami_cross_pattern` | 0.001 | -0.025 | 0.010 |
| 3 | `enhanced_volatility_14` | 0.001 | 0.018 | 0.007 |
| 4 | `enhanced_volatility_10` | 0.001 | 0.018 | 0.007 |
| 5 | `volume_std_10` | 0.000 | 0.015 | 0.006 |

### Small Multivariate LGBM Baseline

| Type | Features | Test R² |
|------|----------|---------|
| Pair | `candlestick_doji_pattern`, `enhanced_volatility_10` | 0.004 |
| Triplet | `candlestick_doji_pattern`, `volume_std_10`, `vectorbt_adx_9` | 0.003 |

### Interpretation

**Quality Score:** 0.02/1.0

**Summary:** ⚠️ Moderate predictive signals detected

**Insights:**
- Best feature `candlestick_doji_pattern` achieved Test R² = 0.003
- Positive Test R² features: 69 (26.6%)
- Median Test R² across evaluated features: -0.000
- LGBM best feature `vectorbt_momentum_50_price_returns` achieved Test R² = 0.004

**Recommendations:**
- Focus downstream modeling on the top-ranked features

## Next Steps

- Features are ready for feature selection and interaction generation
- Consider running lookback optimization for optimal feature parameters
- Proceed to labeling step for profit-target generation

