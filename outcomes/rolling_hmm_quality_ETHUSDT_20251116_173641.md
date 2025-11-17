# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-16T17:36:26.221108
**Data Points:** N/A
**Number of Regimes:** 5
**Report Version:** 1.3 (Enhanced with Financial Analysis)

This report provides a comprehensive assessment of cluster quality for ETHUSDT.

### Key Metrics


---

## Clustering Method Configuration

| Parameter | Value |
|---|---|
| rolling_hmm_params | {'ewma_config_idx': '2 (6+12)', 'n_components': 5, 'min_covar': 0.01, 'kappa': 10.0} |
| ewma_config | {'name': '6+12', 'short_window': 6, 'long_window': 12} |


## PCA Feature Analysis



---

## Top Configuration Analysis

### Clustering Configuration Parameters


---

## Clustering Metrics

### Silhouette Analysis

**Global Silhouette Score:** 0.0441

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.0076 | 0.1804 | -0.4557 | 0.2524 |
| 1 | 0.3031 | 0.1157 | -0.1099 | 0.4774 |
| 2 | -0.1786 | 0.1654 | -0.6474 | 0.0916 |
| 3 | -0.0268 | 0.1678 | -0.4534 | 0.2094 |
| 4 | -0.0970 | 0.1288 | -0.5714 | 0.1482 |


### Separation Metrics

- **Davies-Bouldin Index:** 2.6961 (lower is better)
- **Calinski-Harabasz Index:** 556.84 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 23.7165 +/- 22.3443
- **Between-Regime CV:** 69.3974 +/- 141.2861

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 4.9649 |
| 1 | 15.1722 |
| 2 | 29.7412 |
| 3 | 4.2757 |
| 4 | 64.4286 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 76.2095
- **Between-Regime CV (Mean Return):** 3.0479
- **CV Ratio (Between/Within):** 0.0400

| mean_return | 3.0479 |
| pct_above_target | 0.5905 |
| pct_below_neg_target | 0.5639 |
| pct_target_hits | 0.5761 |
| sharpe | 2.6855 |
| volatility | 0.4894 |


---

## Economic Gap Analysis

### Per-Regime Snapshot

| Regime | Type | Mean Return | Volatility | Sharpe | Max DD | Pct Target Hits |
|--------|------|-------------|------------|--------|--------|-----------------|
| 0 | stable | 0.000480 | 0.006347 | 0.0756 | -11.86% | 21.37% |
| 1 | stable | 0.000118 | 0.005163 | 0.0228 | -21.50% | 8.32% |
| 2 | stable | -0.000081 | 0.016790 | -0.0048 | -33.63% | 55.16% |
| 3 | mean_reverting | -0.000225 | 0.006215 | -0.0363 | -37.28% | 19.71% |
| 4 | stable | 0.000098 | 0.008749 | 0.0112 | -45.67% | 33.84% |

### Pairwise Economic Spreads

| Regime A | Regime B | Mean Return Spread | Sharpe Spread | Volatility Ratio | Max DD Spread |
|----------|----------|--------------------|---------------|------------------|---------------|
| 0 | 1 | 0.000362 | 0.0528 | 1.229 | 9.63% |
| 0 | 2 | 0.000561 | 0.0804 | 0.378 | 21.77% |
| 0 | 3 | 0.000705 | 0.1119 | 1.021 | 25.41% |
| 0 | 4 | 0.000382 | 0.0644 | 0.726 | 33.80% |
| 1 | 2 | 0.000199 | 0.0276 | 0.308 | 12.13% |
| 1 | 3 | 0.000343 | 0.0591 | 0.831 | 15.78% |
| 1 | 4 | 0.000020 | 0.0116 | 0.590 | 24.17% |
| 2 | 3 | 0.000144 | 0.0315 | 2.701 | 3.65% |
| 2 | 4 | -0.000179 | -0.0160 | 1.919 | 12.04% |
| 3 | 4 | -0.000323 | -0.0475 | 0.710 | 8.39% |

### Statistical Tests (ANOVA / t-tests)

- **ANOVA F-statistic:** 1.2297, p-value=0.2958 (ns)

**Pairwise t-tests:**

| Regime A | Regime B | t-stat | p-value | Cohen's d | Significant |
|----------|----------|--------|---------|-----------|-------------|
| 0 | 1 | 1.6764 | 0.0938 | 0.065 | No |
| 0 | 2 | 0.7834 | 0.4337 | 0.051 | No |
| 0 | 3 | 2.7656 | 0.0057 | 0.112 | Yes |
| 0 | 4 | 1.4794 | 0.1391 | 0.047 | No |
| 1 | 2 | 0.2845 | 0.7761 | 0.023 | No |
| 1 | 3 | 1.6940 | 0.0904 | 0.062 | No |
| 1 | 4 | 0.0967 | 0.9229 | 0.003 | No |
| 2 | 3 | 0.2029 | 0.8393 | 0.013 | No |
| 2 | 4 | -0.2508 | 0.8020 | -0.017 | No |
| 3 | 4 | -1.3079 | 0.1910 | -0.041 | No |


---

## Balance and Distribution

**Balance Score:** 0.6851 (0-1, higher is better)

- **Smallest Cluster:** 7.55% of total
- **Largest Cluster:** 31.18% of total
- **Cluster Size Std Dev:** 719.70

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 14.59% |
| 1 | 30.08% |
| 2 | 7.55% |
| 3 | 16.59% |
| 4 | 31.18% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.8406 (0-1, higher = fewer transitions)
- **Regime Persistence:** 6.27 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 1142 samples (14.59%)

**Performance Metrics:**
- Mean Return: 0.0004796784487552941
- Volatility: 0.0063473014160990715
- Sharpe Ratio: 0.0755720174193457
- Skewness: 0.03761503845453262
- Max Drawdown: -0.11863660055749539

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.11821366024518389
- Pct < -1.0% (Shorts): 0.09544658493870403
- Pct Target Hits: 0.21366024518388793

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 33.6615826972365
- Win Rate (Long Bias): 0.5532786859350645
- Return per Vol: 0.0755720174193457
- Profit Factor: 1.0438801050186157

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0004796784487552941
- volatility: 0.0063473014160990715
- stability_coefficient: 0.07026231041963331

### Regime 1 (stable)

**Size:** 2355 samples (30.08%)

**Performance Metrics:**
- Mean Return: 0.00011777548934333026
- Volatility: 0.005162994377315044
- Sharpe Ratio: 0.02281146519340446
- Skewness: -2.875842571258545
- Max Drawdown: -0.21496554036769444

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.042887473460721866
- Pct < -1.0% (Shorts): 0.040339702760084924
- Pct Target Hits: 0.08322717622080679

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 16.119940100370204
- Win Rate (Long Bias): 0.5153061162574188
- Return per Vol: 0.02281146519340446
- Profit Factor: 1.0355287790298462

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.00011777548934333026
- volatility: 0.005162994377315044
- stability_coefficient: 0.02230289711809787

### Regime 2 (stable)

**Size:** 591 samples (7.55%)

**Performance Metrics:**
- Mean Return: -8.104436710709706e-05
- Volatility: 0.01678999327123165
- Sharpe Ratio: -0.004826944297769076
- Skewness: -0.36974167823791504
- Max Drawdown: -0.33629154716039916

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2893401015228426
- Pct < -1.0% (Shorts): 0.2622673434856176
- Pct Target Hits: 0.5516074450084603

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 32.85334325298709
- Win Rate (Long Bias): 0.5245398763496838
- Return per Vol: 0.004826944297769076
- Profit Factor: 1.0305601358413696

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -8.104436710709706e-05
- volatility: 0.01678999327123165
- stability_coefficient: 0.004803816104298687

### Regime 3 (mean_reverting)

**Size:** 1299 samples (16.59%)

**Performance Metrics:**
- Mean Return: -0.00022549359709955752
- Volatility: 0.006215289235115051
- Sharpe Ratio: -0.0362804613412206
- Skewness: -0.687630832195282
- Max Drawdown: -0.37275635673154833

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.09083910700538876
- Pct < -1.0% (Shorts): 0.10623556581986143
- Pct Target Hits: 0.1970746728252502

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 31.708040231463485
- Win Rate (Long Bias): 0.46093749766110226
- Return per Vol: 0.0362804613412206
- Profit Factor: 0.9379972219467163

**Regime-Specific Characteristics:**

- reversion_center: -0.00022549359709955752
- reversion_speed: 220.29822457353006
- reversion_range: 0.004243665840476751

### Regime 4 (stable)

**Size:** 2441 samples (31.18%)

**Performance Metrics:**
- Mean Return: 9.779063839232549e-05
- Volatility: 0.008748682215809822
- Sharpe Ratio: 0.011177755095259041
- Skewness: -0.33654308319091797
- Max Drawdown: -0.4566705718807541

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.1753379762392462
- Pct < -1.0% (Shorts): 0.16304793117574765
- Pct Target Hits: 0.33838590741499386

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 38.678495845351186
- Win Rate (Long Bias): 0.5181598047641306
- Return per Vol: 0.011177755095259041
- Profit Factor: 0.9308426976203918

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 9.779063839232549e-05
- volatility: 0.008748682215809822
- stability_coefficient: 0.011054307061097984


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5142455602401942

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Economic Relevance Analysis

### Strategy Performance Summary

### Statistical Significance Tests

### Economic Regime Mapping

| Regime | Economic Interpretation | Recommended Position |
|---------|----------------------|----------------------|

### Economic Interpretation

**Detailed Economic Report:** /Users/remyroche/Documents/Ares/outcomes/regime_economic_relevance_report_20251116_173635.md


---

## Quality Assessment

**Overall Quality Score:** 0.5468 / 1.0
**Quality Level:** Good ✅
**Recommendation:** The clustering shows good quality. Suitable for most applications.

