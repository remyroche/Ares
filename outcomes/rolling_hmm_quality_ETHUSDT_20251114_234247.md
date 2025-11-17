# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-14T23:41:59.213998
**Data Points:** N/A
**Number of Regimes:** 5
**Report Version:** 1.3 (Enhanced with Financial Analysis)

This report provides a comprehensive assessment of cluster quality for ETHUSDT.

### Key Metrics


---

## Clustering Method Configuration

| Parameter | Value |
|---|---|
| rolling_hmm_params | {'ewma_config_idx': '2 (6+12)', 'n_components': 5, 'min_covar': 0.006239651410114761, 'kappa': 20.0} |
| ewma_config | {'name': '6+12', 'short_window': 6, 'long_window': 12} |


## PCA Feature Analysis



---

## Top Configuration Analysis

### Clustering Configuration Parameters


---

## Clustering Metrics

### Silhouette Analysis

**Global Silhouette Score:** 0.1063

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.1817 | 0.0826 | -0.0494 | 0.3160 |
| 1 | -0.1486 | 0.1152 | -0.4438 | 0.0297 |
| 2 | 0.0029 | 0.1729 | -0.6027 | 0.2923 |
| 3 | 0.1363 | 0.2264 | -0.5588 | 0.4119 |
| 4 | 0.3591 | 0.0791 | 0.0983 | 0.4823 |


### Separation Metrics

- **Davies-Bouldin Index:** 2.4945 (lower is better)
- **Calinski-Harabasz Index:** 4566.79 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 7.4033 +/- 6.7894
- **Between-Regime CV:** 4.8568 +/- 1.7391

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 2.1957 |
| 1 | 15.8874 |
| 2 | 2.2565 |
| 3 | 1.1571 |
| 4 | 15.5199 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 60.2296
- **Between-Regime CV (Mean Return):** 5.3562
- **CV Ratio (Between/Within):** 0.0889

| mean_return | 5.3562 |
| pct_above_target | 0.5493 |
| pct_below_neg_target | 0.6520 |
| pct_target_hits | 0.5651 |
| sharpe | 6.9130 |
| volatility | 0.3797 |


---

## Economic Gap Analysis

### Per-Regime Snapshot

| Regime | Type | Mean Return | Volatility | Sharpe | Max DD | Pct Target Hits |
|--------|------|-------------|------------|--------|--------|-----------------|
| 0 | stable | -0.000100 | 0.007293 | -0.0137 | -45.09% | 21.18% |
| 1 | stable | 0.000090 | 0.005985 | 0.0150 | -33.80% | 15.67% |
| 2 | mean_reverting | 0.001137 | 0.010048 | 0.1132 | -18.97% | 33.43% |
| 3 | mean_reverting | -0.002076 | 0.011888 | -0.1746 | -99.94% | 42.29% |
| 4 | stable | -0.000024 | 0.003553 | -0.0068 | -39.70% | 4.58% |

### Pairwise Economic Spreads

| Regime A | Regime B | Mean Return Spread | Sharpe Spread | Volatility Ratio | Max DD Spread |
|----------|----------|--------------------|---------------|------------------|---------------|
| 0 | 1 | -0.000189 | -0.0287 | 1.219 | -11.29% |
| 0 | 2 | -0.001237 | -0.1269 | 0.726 | -26.12% |
| 0 | 3 | 0.001976 | 0.1609 | 0.613 | 54.84% |
| 0 | 4 | -0.000076 | -0.0069 | 2.052 | -5.40% |
| 1 | 2 | -0.001048 | -0.0982 | 0.596 | -14.83% |
| 1 | 3 | 0.002165 | 0.1896 | 0.503 | 66.13% |
| 1 | 4 | 0.000114 | 0.0218 | 1.684 | 5.89% |
| 2 | 3 | 0.003213 | 0.2878 | 0.845 | 80.96% |
| 2 | 4 | 0.001161 | 0.1200 | 2.828 | 20.72% |
| 3 | 4 | -0.002051 | -0.1678 | 3.346 | -60.24% |

### Statistical Tests (ANOVA / t-tests)

- **ANOVA F-statistic:** 105.6894, p-value=0.0000 (significant)

**Pairwise t-tests:**

| Regime A | Regime B | t-stat | p-value | Cohen's d | Significant |
|----------|----------|--------|---------|-----------|-------------|
| 0 | 1 | -1.1850 | 0.2361 | -0.031 | No |
| 0 | 2 | -6.2487 | 0.0000 | -0.132 | Yes |
| 0 | 3 | 7.7790 | 0.0000 | 0.192 | Yes |
| 0 | 4 | -0.4832 | 0.6290 | -0.016 | No |
| 1 | 2 | -7.6318 | 0.0000 | -0.140 | Yes |
| 1 | 3 | 10.3030 | 0.0000 | 0.288 | Yes |
| 1 | 4 | 1.7675 | 0.0772 | 0.022 | No |
| 2 | 3 | 13.3619 | 0.0000 | 0.299 | Yes |
| 2 | 4 | 8.7067 | 0.0000 | 0.163 | Yes |
| 3 | 4 | -9.8799 | 0.0000 | -0.289 | Yes |


---

## Balance and Distribution

**Balance Score:** 0.6249 (0-1, higher is better)

- **Smallest Cluster:** 6.84% of total
- **Largest Cluster:** 40.70% of total
- **Cluster Size Std Dev:** 4070.03

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 6.84% |
| 1 | 40.70% |
| 2 | 18.33% |
| 3 | 10.03% |
| 4 | 24.10% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.8340 (0-1, higher = fewer transitions)
- **Regime Persistence:** 6.03 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 2318 samples (6.84%)

**Performance Metrics:**
- Mean Return: -9.984081407310441e-05
- Volatility: 0.007292851805686951
- Sharpe Ratio: -0.013690227505380166
- Skewness: -0.04451394081115723
- Max Drawdown: -0.4509256327246526

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.10181190681622088
- Pct < -1.0% (Shorts): 0.1100086281276963
- Pct Target Hits: 0.21182053494391717

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 29.0449486075929
- Win Rate (Long Bias): 0.48065172889175006
- Return per Vol: 0.013690227505380166
- Profit Factor: 0.9683166742324829

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -9.984081407310441e-05
- volatility: 0.007292851805686951
- stability_coefficient: 0.013505471646523475

### Regime 1 (stable)

**Size:** 13797 samples (40.70%)

**Performance Metrics:**
- Mean Return: 8.953669748734683e-05
- Volatility: 0.005984972696751356
- Sharpe Ratio: 0.014960249121219588
- Skewness: -0.4299049377441406
- Max Drawdown: -0.33803786981799794

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.07958251793868232
- Pct < -1.0% (Shorts): 0.07711821410451547
- Pct Target Hits: 0.1567007320431978

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 26.182359352432066
- Win Rate (Long Bias): 0.5078630864907553
- Return per Vol: 0.014960249121219588
- Profit Factor: 0.9935219287872314

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 8.953669748734683e-05
- volatility: 0.005984972696751356
- stability_coefficient: 0.014739903576799834

### Regime 2 (mean_reverting)

**Size:** 6213 samples (18.33%)

**Performance Metrics:**
- Mean Return: 0.0011372604640200734
- Volatility: 0.010048044845461845
- Sharpe Ratio: 0.11318225269978362
- Skewness: 0.5950949788093567
- Max Drawdown: -0.18972496552730625

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.19893771125060358
- Pct < -1.0% (Shorts): 0.1353613391276356
- Pct Target Hits: 0.33429905037823915

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 33.270056239763655
- Win Rate (Long Bias): 0.5950890689950465
- Return per Vol: 0.11318225269978362
- Profit Factor: 1.2288508415222168

**Regime-Specific Characteristics:**

- reversion_center: 0.0011372604640200734
- reversion_speed: 147.6158586234903
- reversion_range: 0.0074205221608281136

### Regime 3 (mean_reverting)

**Size:** 3400 samples (10.03%)

**Performance Metrics:**
- Mean Return: -0.0020756337326020002
- Volatility: 0.011888078413903713
- Sharpe Ratio: -0.17459790268345962
- Skewness: -0.9877159595489502
- Max Drawdown: -0.9993645539009465

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.1623529411764706
- Pct < -1.0% (Shorts): 0.2605882352941176
- Pct Target Hits: 0.4229411764705882

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 35.57691379281476
- Win Rate (Long Bias): 0.38386648031631015
- Return per Vol: 0.17459790268345962
- Profit Factor: 0.7457900047302246

**Regime-Specific Characteristics:**

- reversion_center: -0.0020756337326020002
- reversion_speed: 120.53987176269199
- reversion_range: 0.008513658307492733

### Regime 4 (stable)

**Size:** 8170 samples (24.10%)

**Performance Metrics:**
- Mean Return: -2.421765930193942e-05
- Volatility: 0.0035533399786800146
- Sharpe Ratio: -0.006815461687252916
- Skewness: 0.5733290910720825
- Max Drawdown: -0.3969694597132064

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.02239902080783354
- Pct < -1.0% (Shorts): 0.023378212974296205
- Pct Target Hits: 0.04577723378212974

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 12.882871094215941
- Win Rate (Long Bias): 0.48930480214540045
- Return per Vol: 0.006815461687252916
- Profit Factor: 1.0061633586883545

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -2.421765930193942e-05
- volatility: 0.0035533399786800146
- stability_coefficient: 0.006769605128149848


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5033188777767944

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Economic Relevance Analysis

### Strategy Performance Summary

### Statistical Significance Tests

### Economic Regime Mapping

| Regime | Economic Interpretation | Recommended Position |
|---------|----------------------|----------------------|

### Economic Interpretation

**Detailed Economic Report:** /Users/remyroche/Documents/Ares/outcomes/regime_economic_relevance_report_20251114_234244.md


---

## Quality Assessment

**Overall Quality Score:** 0.3365 / 1.0
**Quality Level:** Moderate ⚠️
**Recommendation:** The clustering shows moderate quality. Consider parameter tuning.

