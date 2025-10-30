# Improved MS-DR Clustering Report
**Generated:** 2025-10-30 20:41:38

## Improvements Applied

1. **Enhanced Signal Construction**
   - Multi-scale indicators: 42 components
   - Adaptive weighting: 42 weighted features
   - Signal diversity score: 0.616

2. **Enhanced Burn-in Detection**
   - Burn-in detected: False

3. **Improved MS-DR Configuration**
   - Model: AR(2) with powell optimization
   - Regime selection: BIC criterion (2-5 regimes)
   - Max iterations: 3000

---

## 🎯 Clustering Results

- **n_clusters:** 2
- **success:** True
- **processing_time:** 95.59s

### Regime Distribution

| Regime ID | Samples | Percentage |
|-----------|---------|------------|
| 0 | 274 | 27.5% |
| 1 | 724 | 72.5% |

## 🎨 Quality Metrics

- **Silhouette Score:** None
- **Davies-Bouldin Index:** None
- **Balance Score:** 0.6892
- **Overall Quality:** 0.8446

## 🔍 Diagnostics

### Signal Quality

- **std:** 1.0000
- **range:** 6.1642
- **autocorr_lag1:** 0.6113
- **autocorr_lag10:** 0.2228
- **normality_pvalue:** 0.7179
- **transition_rate:** 0.2500
- **diversity_score:** 0.6164

### Burn-in Detection

- **regime_counts:** [274, 724]
- **n_regimes:** 2
- **is_degenerate:** False
- **burn_in_detected:** False

---
*Report generated at 2025-10-30 20:41:38*
