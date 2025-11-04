# Advanced Economic Metrics - Sticky Finite HMM

## Overview

The Sticky Finite HMM now exports **comprehensive economic validation metrics** matching and exceeding those in HDP-HMM, with 13 advanced metrics calculated per regime.

## Metrics Exported Per Regime

### Basic Return Metrics (4)

1. **`regime_X_mean_return`**
   - Average forward return in the regime
   - Indicates regime's directional bias

2. **`regime_X_volatility`**
   - Standard deviation of returns
   - Measures regime risk/uncertainty

3. **`regime_X_sharpe`**
   - Mean return / volatility (risk-adjusted return)
   - Higher is better (>1.0 good, >2.0 excellent)

4. **`regime_X_skewness`**
   - Distribution asymmetry
   - Positive = more upside potential, Negative = more downside risk

### Risk Metrics (1)

5. **`regime_X_max_drawdown`**
   - Maximum peak-to-trough decline
   - Measures worst-case loss in regime
   - Lower (more negative) = riskier

### Target-Based Metrics (3)

6. **`regime_X_pct_above_target`**
   - Percentage of returns exceeding profit target
   - Measures upside capture

7. **`regime_X_pct_below_neg_target`**
   - Percentage of returns below loss threshold  
   - Measures downside risk

8. **`regime_X_pct_target_hits`**
   - Combined % of returns hitting either target
   - Measures regime's opportunity set

### Advanced Risk-Adjusted Metrics (5)

9. **`regime_X_risk_adj_target_hits`**
   - Target hits weighted by risk (volatility-adjusted)
   - Accounts for both opportunity and uncertainty

10. **`regime_X_win_rate`**
    - Percentage of positive returns
    - Simple hit rate metric
    - 50% = random, >55% = good edge

11. **`regime_X_return_per_vol`**
    - Return per unit of volatility
    - Similar to Sharpe but different scaling
    - Measures efficiency

12. **`regime_X_profit_factor`**
    - Gross profit / Gross loss
    - >1.0 = profitable, >2.0 = good, >3.0 = excellent
    - Trading system quality metric

13. **Plus Standard Metrics**:
    - `regime_X_size` - Number of observations
    - `regime_X_size_pct` - Percentage of total
    - `regime_X_duration_mean` - Average regime duration
    - `regime_X_duration_std` - Duration variability
    - `regime_X_silhouette_mean` - Cluster quality
    - `regime_X_silhouette_std` - Quality consistency

## Global Economic Metrics

Beyond per-regime metrics, the CSV also includes:

- **`economic_cv_ratio`**: Coefficient of variation ratio for economic metrics
  - Measures regime differentiation in economic terms
  - Higher = more distinct economic regimes

## Calculation Details

### Data Source
- **Forward Returns**: Calculated as `close.pct_change().shift(-1)`
- Passed from integration layer → clusterer → quality assessor
- Only calculated when `close` column is available

### Quality Assessor Integration
All metrics are calculated by `ClusterQualityAssessor._calculate_per_regime_metrics()`:
- Uses forward returns aligned with regime labels
- Handles NaN values gracefully
- Applies risk-free rate adjustments where appropriate
- Calculates profit factor from winning vs losing trades

### Thresholds Used
- **Target return**: 0.5% (configurable)
- **Negative target**: -0.5% (configurable)
- **Risk-free rate**: 0% (conservative)
- **Annualization**: 252 trading days for Sharpe

## Interpretation Guide

### Regime Quality Indicators

**Good Regime** (tradeable):
- Sharpe > 1.0
- Win rate > 52%
- Profit factor > 1.5
- Max drawdown < -15%
- Positive mean return

**Excellent Regime** (highly tradeable):
- Sharpe > 2.0
- Win rate > 55%
- Profit factor > 2.5
- Max drawdown < -10%
- Return per vol > 0.15

**Poor Regime** (avoid):
- Sharpe < 0
- Win rate < 48%
- Profit factor < 1.0
- Max drawdown < -25%

### Use Cases

1. **Regime Selection**: Choose regimes with best economic metrics for trading
2. **Risk Management**: Avoid regimes with high max drawdown
3. **Position Sizing**: Scale by Sharpe ratio or profit factor
4. **Entry Timing**: Trade when entering high win-rate regimes
5. **Strategy Validation**: Verify regimes have economic significance

## Comparison with HDP-HMM

| Metric Category | HDP-HMM | Sticky Finite HMM |
|-----------------|---------|-------------------|
| Per-regime economic metrics | 3 | 13 ✅ |
| Sharpe ratio | ✅ | ✅ |
| Max drawdown | ❌ | ✅ |
| Win rate | ❌ | ✅ |
| Profit factor | ❌ | ✅ |
| Target-based metrics | ❌ | ✅ (3 metrics) |
| Risk-adjusted metrics | ❌ | ✅ (5 metrics) |
| Skewness | ❌ | ✅ |

**Sticky Finite HMM provides 4x more economic metrics than HDP-HMM!**

## Example CSV Row

```csv
...,regime_0_mean_return,regime_0_volatility,regime_0_sharpe,regime_0_skewness,regime_0_max_drawdown,regime_0_pct_above_target,regime_0_pct_below_neg_target,regime_0_pct_target_hits,regime_0_risk_adj_target_hits,regime_0_win_rate,regime_0_return_per_vol,regime_0_profit_factor,...
...,0.0012,0.0156,1.85,0.23,-0.08,0.42,0.18,0.60,0.38,0.54,0.077,2.3,...
```

Interpretation:
- Regime 0: Mean return 0.12%, volatility 1.56%
- Sharpe 1.85 (excellent risk-adjusted return)
- Positive skew (0.23) = upside bias
- Max DD -8% (manageable risk)
- 42% returns above target, 18% below negative target
- 54% win rate (good edge)
- Profit factor 2.3 (profitable regime)

## Implementation Notes

### Files Modified
1. `sticky_finite_hmm_clusterer.py`: Added `forward_returns` parameter to `fit_predict()`
2. `enhanced_sticky_finite_hmm_clustering_integration.py`: Calculate forward returns from `close` prices
3. `sticky_finite_hmm_regime_discovery_step.py`: Export all 13 economic metrics to CSV

### Backward Compatibility
- All changes maintain HDP-HMM naming conventions
- Existing columns unchanged
- New metrics added as additional columns
- CSV format extensible for future metrics

## Future Enhancements

Potential additional metrics:
- **Sortino ratio**: Downside deviation-adjusted return
- **Calmar ratio**: Return / max drawdown
- **Omega ratio**: Probability-weighted gains vs losses
- **Information ratio**: Excess return vs tracking error
- **Tail risk metrics**: VaR, CVaR, expected shortfall
- **Regime-based Kelly criterion**: Optimal position sizing per regime

These can be added to the quality assessor and will automatically flow to the CSV export.

