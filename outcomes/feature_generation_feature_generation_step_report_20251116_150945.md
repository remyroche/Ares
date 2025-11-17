# Feature Generation Report

**Generated:** 2025-11-16 15:09:45
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

**Path:** `versioned_artifacts/ETHUSDT_binance_15m_long_analyst/generated_features_15m_20251116_150858_132.h5`
**Size:** 0.00 KB

### basic_feature_analysis_report

**Path:** `outcomes/basic_feature_analysis_20251116_150902.md`
**Size:** 1.99 KB

## Baseline Predictive Check

**Dataset:** 31302 samples, 259 features

### Top Single-Feature Signals

| Rank | Feature | Test R² | Pearson | Quality Score |
|------|---------|---------|---------|---------------|
| 1 | `candlestick_harami_cross_pattern` | 0.005 | 0.072 | 0.032 |
| 2 | `candlestick_doji_pattern` | 0.003 | 0.064 | 0.027 |
| 3 | `log_returns_1_price_returns` | 0.003 | -0.060 | 0.026 |
| 4 | `vectorbt_trend_strength_5_price_returns` | 0.002 | -0.046 | 0.020 |
| 5 | `volume_roc_1` | 0.002 | -0.046 | 0.020 |

### Small Multivariate LGBM Baseline

| Type | Features | Test R² |
|------|----------|---------|
| Pair | `log_returns_1_price_returns`, `dfa_slopes` | 0.023 |
| Triplet | `candlestick_doji_pattern`, `log_returns_1_price_returns`, `dfa_slopes` | 0.027 |

### Interpretation

**Quality Score:** 0.03/1.0

**Summary:** ⚠️ Weak predictive signals

**Insights:**
- Best feature `candlestick_harami_cross_pattern` achieved Test R² = 0.005
- Positive Test R² features: 36 (13.9%)
- Median Test R² across evaluated features: -0.000
- LGBM best feature `vectorbt_momentum_5_price_returns` achieved Test R² = 0.013

**Recommendations:**
- Consider revisiting labeling/target definitions; very few features carry signal

## Next Steps

- Features are ready for feature selection and interaction generation
- Consider running lookback optimization for optimal feature parameters
- Proceed to labeling step for profit-target generation

