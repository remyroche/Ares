# Improved MS-DR Clustering Report (Real Data)
**Generated:** 2025-10-30 21:41:14

## Data Source

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 1h
- **Samples:** 2000
- **Date Range:** 2025-08-08 14:35:47.971176 to 2025-10-30 21:35:47.971176

## Improvements Applied

1. **Enhanced Signal Construction**
   - Multi-scale indicators: 42 components
   - Adaptive weighting: 42 weighted features
   - Signal diversity score: 0.622

2. **Enhanced Burn-in Detection**
   - Burn-in detected: False

3. **MS-DR Configuration**
   - Model: AR(2) with powell optimization
   - Regime selection: BIC criterion (2-5 regimes)
   - Max iterations: 3000

---

## 🎯 Clustering Results

- **n_clusters:** 2
- **success:** True
- **processing_time:** 325.60s

### Regime Distribution

| Regime ID | Samples | Percentage |
|-----------|---------|------------|
| 0 | 337 | 16.9% |
| 1 | 1661 | 83.1% |

## 🎨 Quality Metrics

- **Silhouette Score:** None
- **Davies-Bouldin Index:** None
- **Balance Score:** 0.6014
- **Overall Quality:** 0.8007

## 🔍 Diagnostics

### Signal Quality

- **std:** 1.0000
- **range:** 6.2219
- **autocorr_lag1:** 0.5727
- **autocorr_lag10:** 0.0990
- **normality_pvalue:** 0.0771
- **transition_rate:** 0.2500
- **diversity_score:** 0.6222

### Burn-in Detection

- **regime_counts:** [337, 1661]
- **n_regimes:** 2
- **is_degenerate:** False
- **burn_in_detected:** False

---
*Report generated at 2025-10-30 21:41:14*
