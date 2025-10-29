# Regime Feature Selection Comprehensive Report

**Generated**: 2025-10-29T23:38:18.719993  
**Symbol**: ETHUSDT  
**Exchange**: binance  
**Timeframes**: 15m  
**Execution Mode**: light  
**Selection Method**: unknown  

---

## 📊 Executive Summary

This report provides a comprehensive analysis of the regime feature selection process, including detailed metrics for each selected feature, regime-specific analysis, and performance assessments.

### Key Results
- **Total Features**: 20
- **Selected Features**: 18
- **Selection Ratio**: 90.00%
- **Processing Time**: 0.00 seconds
- **Selection Method**: unknown
- **Regime-Specific Analysis**: ❌ Not Available

---

## 🔍 Feature Selection Analysis

### Selection Statistics

| Metric | Value |
|--------|-------|
| **Total Features** | 20 |
| **Selected Features** | 18 |
| **Selection Ratio** | 90.00% |
| **Min Importance Threshold** | 0.0100 |
| **Max Features Limit** | 50 |

### Selection Method Details

- **Primary Method**: unknown
- **TreeSHAP Available**: ✅ Yes
- **VectorBT Optimization**: ✅ Yes
- **Hardware Optimization**: ✅ Yes

---

## 📈 Per-Feature Analysis

### Top 20 Selected Features


| Rank | Feature Name | Importance Score | Category | Stability |
|------|--------------|------------------|----------|-----------|
| 1 | `roc_30_price_returns` | 1.0000 | Unknown | 0.000 |
| 2 | `vectorbt_momentum_50_price_returns` | 1.0000 | Unknown | 0.000 |
| 3 | `vectorbt_momentum_5_price_returns` | 1.0000 | Unknown | 0.000 |
| 4 | `vectorbt_momentum_10_price_returns` | 1.0000 | Unknown | 0.000 |
| 5 | `momentum_30_price_returns` | 0.9667 | Unknown | 0.000 |
| 6 | `volume_momentum_5` | 0.8333 | Unknown | 0.000 |
| 7 | `volume_roc_5` | 0.8333 | Unknown | 0.000 |
| 8 | `volume_roc_10` | 0.6667 | Unknown | 0.000 |
| 9 | `volume_momentum_10` | 0.6667 | Unknown | 0.000 |
| 10 | `vectorbt_acceleration_momentum_5_10_price_returns` | 0.5333 | Unknown | 0.000 |
| 11 | `vectorbt_acceleration_momentum_10_10_price_returns` | 0.3667 | Unknown | 0.000 |
| 12 | `analyst_momentum_5m` | 0.3333 | Unknown | 0.000 |
| 13 | `vectorbt_momentum_acceleration_5_10_price_returns` | 0.3333 | Unknown | 0.000 |
| 14 | `vectorbt_acceleration_momentum_5_20_price_returns` | 0.2000 | Unknown | 0.000 |
| 15 | `advanced_momentum_5_20` | 0.0667 | Unknown | 0.000 |
| 16 | `vectorbt_acceleration_momentum_10_20_price_returns` | 0.0333 | Unknown | 0.000 |
| 17 | `vectorbt_momentum_acceleration_5_20_price_returns` | 0.0000 | Unknown | 0.000 |
| 18 | `vectorbt_momentum_acceleration_10_10_price_returns` | 0.0000 | Unknown | 0.000 |

### Complete Feature List

The following features were selected for regime-based trading:

1. **roc_30_price_returns**
   - Importance: 1.0000
   - Category: Unknown

2. **vectorbt_momentum_50_price_returns**
   - Importance: 1.0000
   - Category: Unknown

3. **vectorbt_momentum_5_price_returns**
   - Importance: 1.0000
   - Category: Unknown

4. **vectorbt_momentum_10_price_returns**
   - Importance: 1.0000
   - Category: Unknown

5. **momentum_30_price_returns**
   - Importance: 0.9667
   - Category: Unknown

6. **volume_momentum_5**
   - Importance: 0.8333
   - Category: Unknown

7. **volume_roc_5**
   - Importance: 0.8333
   - Category: Unknown

8. **volume_roc_10**
   - Importance: 0.6667
   - Category: Unknown

9. **volume_momentum_10**
   - Importance: 0.6667
   - Category: Unknown

10. **vectorbt_acceleration_momentum_5_10_price_returns**
   - Importance: 0.5333
   - Category: Unknown

11. **vectorbt_acceleration_momentum_10_10_price_returns**
   - Importance: 0.3667
   - Category: Unknown

12. **analyst_momentum_5m**
   - Importance: 0.3333
   - Category: Unknown

13. **vectorbt_momentum_acceleration_5_10_price_returns**
   - Importance: 0.3333
   - Category: Unknown

14. **vectorbt_acceleration_momentum_5_20_price_returns**
   - Importance: 0.2000
   - Category: Unknown

15. **advanced_momentum_5_20**
   - Importance: 0.0667
   - Category: Unknown

16. **vectorbt_acceleration_momentum_10_20_price_returns**
   - Importance: 0.0333
   - Category: Unknown

17. **vectorbt_momentum_acceleration_5_20_price_returns**
   - Importance: 0.0000
   - Category: Unknown

18. **vectorbt_momentum_acceleration_10_10_price_returns**
   - Importance: 0.0000
   - Category: Unknown

---

## ⚡ Performance Metrics

### Execution Performance

- **Total Execution Time**: 0.00 seconds
- **Features Processed**: 20
- **Selection Efficiency**: 90.00%
- **Memory Usage**: N/A

### Component Status

- **TreeSHAP Integration**: ✅ Active
- **VectorBT Optimization**: ✅ Active
- **Hardware Optimization**: ✅ Active
- **ML Common Utilities**: ❌ Inactive

---

## ⚙️ Configuration Details

### Feature Selection Parameters

- **Max Features**: 50
- **Min Feature Importance**: 0.0100
- **Selection Method**: treeshap
- **Use HPO**: Yes
- **Use Explainability**: Yes
- **Use Data Leakage Detection**: Yes

---

## 🎯 Recommendations

### For Trading Strategy
- **Feature Count**: 18 features selected for regime-based trading
- **Selection Quality**: Low (lower is better)
- **Regime Coverage**: Basic regime-specific analysis

### For Further Analysis
- **Feature Validation**: Consider cross-validation with different time periods
- **Regime Profiling**: Analyze regime-specific feature importance patterns
- **Temporal Stability**: Monitor feature importance over time
- **Interaction Analysis**: Investigate feature interactions within regimes

---

## 📋 Artifact Summary

**Generated Artifacts:**
- `selected_features_ETHUSDT_binance`: Main selected features list
- `feature_importance_ETHUSDT_binance`: Feature importance scores
- `feature_selection_metrics_ETHUSDT_binance`: Performance metrics
- `feature_selection_report_ETHUSDT_binance`: This comprehensive report

**File Locations:**
- **Artifacts**: `artifacts/market_analysis/ETHUSDT/binance/regime_feature_selection/`
- **Report**: `outcomes/regime_feature_selection_report_ETHUSDT_binance_20251029_233818.md`

---

*Report generated by Ares Regime Feature Selector v1.0*
*Generated on: 2025-10-29T23:38:18.719993*
