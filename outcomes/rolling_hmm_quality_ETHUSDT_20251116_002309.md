# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-16T00:22:38.342159
**Data Points:** N/A
**Number of Regimes:** 4
**Report Version:** 1.3 (Enhanced with Financial Analysis)

This report provides a comprehensive assessment of cluster quality for ETHUSDT.

### Key Metrics


---

## Clustering Method Configuration

| Parameter | Value |
|---|---|
| rolling_hmm_params | {'ewma_config_idx': '1 (4+8)', 'n_components': 4, 'min_covar': 0.01, 'kappa': 12.0} |
| ewma_config | {'name': '4+8', 'short_window': 4, 'long_window': 8} |


## PCA Feature Analysis



---

## Top Configuration Analysis

### Clustering Configuration Parameters


---

## Clustering Metrics

### Silhouette Analysis

**Global Silhouette Score:** 0.1008

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.2586 | 0.1685 | -0.3820 | 0.4731 |
| 1 | 0.2305 | 0.1846 | -0.4014 | 0.4567 |
| 2 | -0.0903 | 0.1783 | -0.5903 | 0.1840 |
| 3 | 0.0043 | 0.2486 | -0.6252 | 0.3430 |


### Separation Metrics

- **Davies-Bouldin Index:** 1.6242 (lower is better)
- **Calinski-Harabasz Index:** 5676.03 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 2.9612 +/- 1.8289
- **Between-Regime CV:** 16.3365 +/- 15.9076

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 3.9942 |
| 1 | 1.0966 |
| 2 | 5.4373 |
| 3 | 1.3168 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 10.6458
- **Between-Regime CV (Mean Return):** 145.2555
- **CV Ratio (Between/Within):** 13.6444

| mean_return | 145.2555 |
| pct_above_target | 0.5343 |
| pct_below_neg_target | 0.6595 |
| pct_target_hits | 0.5274 |
| sharpe | 8.5257 |
| volatility | 0.3296 |


---

## Economic Gap Analysis

### Per-Regime Snapshot

| Regime | Type | Mean Return | Volatility | Sharpe | Max DD | Pct Target Hits |
|--------|------|-------------|------------|--------|--------|-----------------|
| 0 | stable | -0.000207 | 0.004553 | -0.0454 | -97.86% | 7.69% |
| 1 | stable | 0.000795 | 0.006196 | 0.1283 | -9.75% | 15.74% |
| 2 | mean_reverting | 0.001328 | 0.009065 | 0.1465 | -17.50% | 32.10% |
| 3 | mean_reverting | -0.001883 | 0.011173 | -0.1685 | -100.00% | 38.83% |

### Pairwise Economic Spreads

| Regime A | Regime B | Mean Return Spread | Sharpe Spread | Volatility Ratio | Max DD Spread |
|----------|----------|--------------------|---------------|------------------|---------------|
| 0 | 1 | -0.001002 | -0.1737 | 0.735 | -88.11% |
| 0 | 2 | -0.001535 | -0.1919 | 0.502 | -80.36% |
| 0 | 3 | 0.001676 | 0.1231 | 0.408 | 2.14% |
| 1 | 2 | -0.000533 | -0.0182 | 0.683 | 7.75% |
| 1 | 3 | 0.002678 | 0.2969 | 0.555 | 90.25% |
| 2 | 3 | 0.003211 | 0.3150 | 0.811 | 82.50% |

### Statistical Tests (ANOVA / t-tests)

- **ANOVA F-statistic:** 222.9095, p-value=0.0000 (significant)

**Pairwise t-tests:**

| Regime A | Regime B | t-stat | p-value | Cohen's d | Significant |
|----------|----------|--------|---------|-----------|-------------|
| 0 | 1 | -8.1975 | 0.0000 | -0.208 | Yes |
| 0 | 2 | -14.9298 | 0.0000 | -0.238 | Yes |
| 0 | 3 | 10.3990 | 0.0000 | 0.252 | Yes |
| 1 | 2 | -3.5062 | 0.0005 | -0.063 | Yes |
| 1 | 3 | 13.6450 | 0.0000 | 0.276 | Yes |
| 2 | 3 | 17.3750 | 0.0000 | 0.325 | Yes |


---

## Balance and Distribution

**Balance Score:** 0.6055 (0-1, higher is better)

- **Smallest Cluster:** 8.25% of total
- **Largest Cluster:** 51.02% of total
- **Cluster Size Std Dev:** 5521.44

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 51.02% |
| 1 | 8.25% |
| 2 | 25.88% |
| 3 | 14.86% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.7901 (0-1, higher = fewer transitions)
- **Regime Persistence:** 4.76 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 17294 samples (51.02%)

**Performance Metrics:**
- Mean Return: -0.00020663742907345295
- Volatility: 0.004553027916699648
- Sharpe Ratio: -0.045384607226092055
- Skewness: -1.3577638864517212
- Max Drawdown: -0.9785775412199372

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.03214987857060252
- Pct < -1.0% (Shorts): 0.044755406499363944
- Pct Target Hits: 0.07690528506996647

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 16.89101617340765
- Win Rate (Long Bias): 0.4180451073461112
- Return per Vol: 0.045384607226092055
- Profit Factor: 0.9121434092521667

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.00020663742907345295
- volatility: 0.004553027916699648
- stability_coefficient: 0.04341447783560744

### Regime 1 (stable)

**Size:** 2795 samples (8.25%)

**Performance Metrics:**
- Mean Return: 0.0007951018051244318
- Volatility: 0.006195724010467529
- Sharpe Ratio: 0.12833071251243822
- Skewness: 1.3732390403747559
- Max Drawdown: -0.09749533798996333

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.09230769230769231
- Pct < -1.0% (Shorts): 0.06511627906976744
- Pct Target Hits: 0.15742397137745975

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 25.40848264109458
- Win Rate (Long Bias): 0.5863636326388947
- Return per Vol: 0.12833071251243822
- Profit Factor: 1.1447006464004517

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0007951018051244318
- volatility: 0.006195724010467529
- stability_coefficient: 0.11373515981701587

### Regime 2 (mean_reverting)

**Size:** 8772 samples (25.88%)

**Performance Metrics:**
- Mean Return: 0.0013280127895995975
- Volatility: 0.00906473770737648
- Sharpe Ratio: 0.14650315165939862
- Skewness: 0.5652838349342346
- Max Drawdown: -0.1749533241173656

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2021203830369357
- Pct < -1.0% (Shorts): 0.11890104879160966
- Pct Target Hits: 0.32102143182854537

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 35.4143061583576
- Win Rate (Long Bias): 0.629616475311436
- Return per Vol: 0.14650315165939862
- Profit Factor: 1.2651392221450806

**Regime-Specific Characteristics:**

- reversion_center: 0.0013280127895995975
- reversion_speed: 157.7744938765648
- reversion_range: 0.0064801718108356

### Regime 3 (mean_reverting)

**Size:** 5037 samples (14.86%)

**Performance Metrics:**
- Mean Return: -0.0018827979220077395
- Volatility: 0.011172584258019924
- Sharpe Ratio: -0.1685194499327026
- Skewness: -0.7840157151222229
- Max Drawdown: -0.9999505216236043

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.14254516577327775
- Pct < -1.0% (Shorts): 0.24578121897955132
- Pct Target Hits: 0.3883263847528291

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 34.757075089141075
- Win Rate (Long Bias): 0.3670756636764007
- Return per Vol: 0.1685194499327026
- Profit Factor: 0.7326301336288452

**Regime-Specific Characteristics:**

- reversion_center: -0.0018827979220077395
- reversion_speed: 129.4418016428626
- reversion_range: 0.008070426061749458


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5044104198011623

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Economic Relevance Analysis

### Strategy Performance Summary

### Statistical Significance Tests

### Economic Regime Mapping

| Regime | Economic Interpretation | Recommended Position |
|---------|----------------------|----------------------|

### Economic Interpretation

**Detailed Economic Report:** /Users/remyroche/Documents/Ares/outcomes/regime_economic_relevance_report_20251116_002307.md


---

## Quality Assessment

**Overall Quality Score:** 0.6565 / 1.0
**Quality Level:** Good ✅
**Recommendation:** The clustering shows good quality. Suitable for most applications.

