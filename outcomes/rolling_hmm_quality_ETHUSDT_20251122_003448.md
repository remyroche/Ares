# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-22T00:34:03.006438
**Data Points:** N/A
**Number of Regimes:** 5
**Report Version:** 1.3 (Enhanced with Financial Analysis)

This report provides a comprehensive assessment of cluster quality for ETHUSDT.

### Key Metrics


---

## Clustering Method Configuration

| Parameter | Value |
|---|---|
| rolling_hmm_params | {'ewma_config_idx': '2 (6+12)', 'n_components': 5, 'min_covar': 0.00967953177397244, 'kappa': 12.0} |
| ewma_config | {'name': '6+12', 'short_window': 6, 'long_window': 12} |


## PCA Feature Analysis



---

## Top Configuration Analysis

### Clustering Configuration Parameters


---

## Clustering Metrics

### Silhouette Analysis

**Global Silhouette Score:** 0.0434

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.0062 | 0.1802 | -0.4547 | 0.2511 |
| 1 | 0.3018 | 0.1162 | -0.1871 | 0.4766 |
| 2 | -0.1804 | 0.1673 | -0.6454 | 0.0903 |
| 3 | -0.0263 | 0.1681 | -0.4526 | 0.2105 |
| 4 | -0.0970 | 0.1284 | -0.5706 | 0.1475 |


### Separation Metrics

- **Davies-Bouldin Index:** 2.7070 (lower is better)
- **Calinski-Harabasz Index:** 552.85 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 25.0611 +/- 22.4636
- **Between-Regime CV:** 18.2580 +/- 18.1460

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 5.0049 |
| 1 | 15.1247 |
| 2 | 38.2611 |
| 3 | 4.2442 |
| 4 | 62.6708 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 99.3163
- **Between-Regime CV (Mean Return):** 2.9558
- **CV Ratio (Between/Within):** 0.0298

| max_drawdown | 0.4004 |
| mean_return | 2.9558 |
| pct_above_target | 0.5921 |
| pct_below_neg_target | 0.5640 |
| pct_target_hits | 0.5766 |
| sharpe | 2.7374 |
| volatility | 0.4886 |


---

## Economic Gap Analysis

### Per-Regime Snapshot

| Regime | Type | Mean Return | Volatility | Sharpe | Max DD | Pct Target Hits |
|--------|------|-------------|------------|--------|--------|-----------------|
| 0 | stable | 0.000493 | 0.006310 | 0.0781 | -12.31% | 21.27% |
| 1 | stable | 0.000123 | 0.005164 | 0.0239 | -20.44% | 8.35% |
| 2 | stable | -0.000053 | 0.016729 | -0.0032 | -33.63% | 55.02% |
| 3 | mean_reverting | -0.000242 | 0.006198 | -0.0390 | -36.58% | 19.52% |
| 4 | stable | 0.000088 | 0.008754 | 0.0100 | -45.79% | 33.89% |

### Pairwise Economic Spreads

| Regime A | Regime B | Mean Return Spread | Sharpe Spread | Volatility Ratio | Max DD Spread |
|----------|----------|--------------------|---------------|------------------|---------------|
| 0 | 1 | 0.000370 | 0.0543 | 1.222 | 8.13% |
| 0 | 2 | 0.000546 | 0.0813 | 0.377 | 21.31% |
| 0 | 3 | 0.000735 | 0.1171 | 1.018 | 24.26% |
| 0 | 4 | 0.000405 | 0.0681 | 0.721 | 33.48% |
| 1 | 2 | 0.000176 | 0.0270 | 0.309 | 13.19% |
| 1 | 3 | 0.000365 | 0.0628 | 0.833 | 16.13% |
| 1 | 4 | 0.000036 | 0.0138 | 0.590 | 25.35% |
| 2 | 3 | 0.000189 | 0.0358 | 2.699 | 2.95% |
| 2 | 4 | -0.000141 | -0.0132 | 1.911 | 12.16% |
| 3 | 4 | -0.000329 | -0.0490 | 0.708 | 9.22% |

### Statistical Tests (ANOVA / t-tests)

- **ANOVA F-statistic:** 1.3051, p-value=0.2655 (ns)

**Pairwise t-tests:**

| Regime A | Regime B | t-stat | p-value | Cohen's d | Significant |
|----------|----------|--------|---------|-----------|-------------|
| 0 | 1 | 1.7187 | 0.0858 | 0.066 | No |
| 0 | 2 | 0.7698 | 0.4417 | 0.049 | No |
| 0 | 3 | 2.8873 | 0.0039 | 0.118 | Yes |
| 0 | 4 | 1.5736 | 0.1157 | 0.050 | No |
| 1 | 2 | 0.2544 | 0.7993 | 0.020 | No |
| 1 | 3 | 1.8002 | 0.0720 | 0.066 | No |
| 1 | 4 | 0.1719 | 0.8635 | 0.005 | No |
| 2 | 3 | 0.2675 | 0.7892 | 0.018 | No |
| 2 | 4 | -0.1989 | 0.8424 | -0.013 | No |
| 3 | 4 | -1.3320 | 0.1830 | -0.041 | No |


---

## Balance and Distribution

**Balance Score:** 0.6850 (0-1, higher is better)

- **Smallest Cluster:** 7.64% of total
- **Largest Cluster:** 31.21% of total
- **Cluster Size Std Dev:** 720.03

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 14.54% |
| 1 | 30.12% |
| 2 | 7.64% |
| 3 | 16.49% |
| 4 | 31.21% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.8422 (0-1, higher = fewer transitions)
- **Regime Persistence:** 6.34 bars (average duration)

### Transition & Persistence Insights

- **Average Duration:** 6.33 bars
- **Max Duration:** 61.00 bars
- **Min Duration:** 1.00 bars
- **High-persistence regimes:** Regime 0 (p_self=0.75), Regime 1 (p_self=0.90), Regime 2 (p_self=0.81), Regime 3 (p_self=0.77), Regime 4 (p_self=0.87)
- **Flip-flop ratio:** 0.0111
- **Average regime persistence:** 6.34 bars
- **Transition entropy:** 0.6328
- **Regime stickiness:** 0.8208
- **Transition stability score:** 0.7138

**Dominant transition hotspots:**

| From | To | Probability |
|------|----|-------------|
| 2 | 4 | 0.191 |
| 0 | 3 | 0.103 |


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 1138 samples (14.54%)

**Performance Metrics:**
- Mean Return: 0.000493162777274847
- Volatility: 0.00631048996001482
- Sharpe Ratio: 0.07814966860735172
- Skewness: 0.00036295331665314734
- Max Drawdown: -0.12314991018995006

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.11862917398945519
- Pct < -1.0% (Shorts): 0.09402460456942004
- Pct Target Hits: 0.21265377855887524

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 33.69845229258926
- Win Rate (Long Bias): 0.5578512370461376
- Return per Vol: 0.07814966860735172
- Profit Factor: 1.043052315711975

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.000493162777274847
- volatility: 0.00631048996001482
- stability_coefficient: 0.07248513759186485

### Regime 1 (stable)

**Size:** 2358 samples (30.12%)

**Performance Metrics:**
- Mean Return: 0.00012330675963312387
- Volatility: 0.005164307076483965
- Sharpe Ratio: 0.02387672420137167
- Skewness: -2.8706891536712646
- Max Drawdown: -0.2044275662912146

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.043256997455470736
- Pct < -1.0% (Shorts): 0.04028837998303647
- Pct Target Hits: 0.0835453774385072

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 16.177458083675695
- Win Rate (Long Bias): 0.5177664912645006
- Return per Vol: 0.02387672420137167
- Profit Factor: 1.038193702697754

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.00012330675963312387
- volatility: 0.005164307076483965
- stability_coefficient: 0.02332011000326071

### Regime 2 (stable)

**Size:** 598 samples (7.64%)

**Performance Metrics:**
- Mean Return: -5.280273762764409e-05
- Volatility: 0.01672949828207493
- Sharpe Ratio: -0.0031562652735351365
- Skewness: -0.3698352575302124
- Max Drawdown: -0.33629154716039916

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.28929765886287623
- Pct < -1.0% (Shorts): 0.2608695652173913
- Pct Target Hits: 0.5501672240802675

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 32.88605443617505
- Win Rate (Long Bias): 0.5258358653056235
- Return per Vol: 0.0031562652735351365
- Profit Factor: 1.0454323291778564

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -5.280273762764409e-05
- volatility: 0.01672949828207493
- stability_coefficient: 0.0031463941934576096

### Regime 3 (mean_reverting)

**Size:** 1291 samples (16.49%)

**Performance Metrics:**
- Mean Return: -0.0002415231429040432
- Volatility: 0.006197901908308268
- Sharpe Ratio: -0.03896852636724629
- Skewness: -0.6908442378044128
- Max Drawdown: -0.3657615929946653

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.08907823392718822
- Pct < -1.0% (Shorts): 0.10611928737412858
- Pct Target Hits: 0.1951975213013168

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 31.494123768808183
- Win Rate (Long Bias): 0.4563492040113222
- Return per Vol: 0.03896852636724629
- Profit Factor: 0.931259274482727

**Regime-Specific Characteristics:**

- reversion_center: -0.0002415231429040432
- reversion_speed: 221.0216636232431
- reversion_range: 0.0042340923100709915

### Regime 4 (stable)

**Size:** 2443 samples (31.21%)

**Performance Metrics:**
- Mean Return: 8.778509800322354e-05
- Volatility: 0.008754085749387741
- Sharpe Ratio: 0.010027899027772739
- Skewness: -0.334512859582901
- Max Drawdown: -0.4579349789374108

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.17519443307408925
- Pct < -1.0% (Shorts): 0.1637331150225133
- Pct Target Hits: 0.3389275480966025

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 38.71649411291327
- Win Rate (Long Bias): 0.5169082110352575
- Return per Vol: 0.010027899027772739
- Profit Factor: 0.9294813275337219

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 8.778509800322354e-05
- volatility: 0.008754085749387741
- stability_coefficient: 0.00992845174849794


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5142455602401942

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Economic Relevance Analysis

### Strategy Performance Summary

#### Per-Regime Long/Short Strategies

- Regime 0 Only (Long): total_return=-0.0836, sharpe=-0.5542, max_dd=-0.2538
- Regime 0 Only (Short): total_return=-0.6717, sharpe=-1.3829, max_dd=-0.6731
- Regime 1 Only (Long): total_return=-0.1683, sharpe=-0.4956, max_dd=-0.3260
- Regime 1 Only (Short): total_return=-0.5511, sharpe=-0.8933, max_dd=-0.5835
- Regime 2 Only (Long): total_return=-0.2974, sharpe=-0.4431, max_dd=-0.3905
- Regime 2 Only (Short): total_return=-0.2118, sharpe=-0.3869, max_dd=-0.3591
- Regime 3 Only (Long): total_return=-0.5970, sharpe=-1.1589, max_dd=-0.6257
- Regime 3 Only (Short): total_return=-0.2807, sharpe=-0.7106, max_dd=-0.3361
- Regime 4 Only (Long): total_return=-0.3917, sharpe=-0.4106, max_dd=-0.5962
- Regime 4 Only (Short): total_return=-0.6100, sharpe=-0.5889, max_dd=-0.6398

### Statistical Significance Tests

### Economic Regime Mapping

| Regime | Economic Interpretation | Recommended Position |
|---------|----------------------|----------------------|

### Economic Interpretation

**Detailed Economic Report:** /Users/remyroche/Documents/Ares/outcomes/regime_economic_relevance_report_20251122_003445.md


---

## Quality Assessment

**Overall Quality Score:** 0.3430 / 1.0
**Quality Level:** Moderate ⚠️
**Recommendation:** The clustering shows moderate quality. Consider parameter tuning.

