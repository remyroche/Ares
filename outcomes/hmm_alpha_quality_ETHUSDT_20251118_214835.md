# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-18T21:48:29.009102
**Data Points:** N/A
**Number of Regimes:** 4
**Report Version:** 1.3 (Enhanced with Financial Analysis)

This report provides a comprehensive assessment of cluster quality for ETHUSDT.

### Key Metrics


---

## Clustering Method Configuration

| Parameter | Value |
|---|---|
| alpha_config | {'alpha_horizon_bars': 1, 'alpha_regime_bins': 5, 'alpha_target_type': 'regression'} |


## PCA Feature Analysis



---

## Top Configuration Analysis

### Clustering Configuration Parameters


---

## Clustering Metrics

### Silhouette Analysis

**Global Silhouette Score:** -0.0176

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | -0.0074 | 0.1016 | -0.3399 | 0.1389 |
| 1 | -0.0135 | 0.0402 | -0.1104 | 0.0526 |
| 2 | 0.0244 | 0.0406 | -0.1404 | 0.0983 |
| 3 | -0.0587 | 0.0584 | -0.1639 | 0.0704 |


### Separation Metrics

- **Davies-Bouldin Index:** 7.5776 (lower is better)
- **Calinski-Harabasz Index:** 250.04 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 68.1398 +/- 80.8686
- **Between-Regime CV:** 40.7790 +/- 53.2525

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 207.5999 |
| 1 | 9.6826 |
| 2 | 30.0671 |
| 3 | 25.2095 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 12.6024
- **Between-Regime CV (Mean Return):** 196.1945
- **CV Ratio (Between/Within):** 15.5680

| mean_return | 196.1945 |
| pct_above_target | 0.3502 |
| pct_below_neg_target | 0.4104 |
| pct_target_hits | 0.1561 |
| sharpe | 34.0711 |
| volatility | 0.1075 |


---

## Economic Gap Analysis

### Per-Regime Snapshot

| Regime | Type | Mean Return | Volatility | Sharpe | Max DD | Pct Target Hits |
|--------|------|-------------|------------|--------|--------|-----------------|
| 0 | stable | -0.001591 | 0.007881 | -0.2019 | -96.01% | 28.22% |
| 1 | stable | -0.000636 | 0.006318 | -0.1007 | -73.15% | 21.83% |
| 2 | stable | 0.000186 | 0.005976 | 0.0311 | -20.41% | 18.37% |
| 3 | stable | 0.002068 | 0.007054 | 0.2932 | -4.35% | 24.81% |

### Pairwise Economic Spreads

| Regime A | Regime B | Mean Return Spread | Sharpe Spread | Volatility Ratio | Max DD Spread |
|----------|----------|--------------------|---------------|------------------|---------------|
| 0 | 1 | -0.000955 | -0.1012 | 1.247 | -22.86% |
| 0 | 2 | -0.001777 | -0.2330 | 1.319 | -75.60% |
| 0 | 3 | -0.003659 | -0.4950 | 1.117 | -91.66% |
| 1 | 2 | -0.000822 | -0.1318 | 1.057 | -52.74% |
| 1 | 3 | -0.002704 | -0.3938 | 0.896 | -68.81% |
| 2 | 3 | -0.001882 | -0.2620 | 0.847 | -16.07% |

### Statistical Tests (ANOVA / t-tests)

- **ANOVA F-statistic:** 110.1510, p-value=0.0000 (significant)

**Pairwise t-tests:**

| Regime A | Regime B | t-stat | p-value | Cohen's d | Significant |
|----------|----------|--------|---------|-----------|-------------|
| 0 | 1 | -4.1473 | 0.0000 | -0.133 | Yes |
| 0 | 2 | -7.6836 | 0.0000 | -0.251 | Yes |
| 0 | 3 | -15.8987 | 0.0000 | -0.492 | Yes |
| 1 | 2 | -3.9649 | 0.0001 | -0.133 | Yes |
| 1 | 3 | -13.1218 | 0.0000 | -0.401 | Yes |
| 2 | 3 | -9.0830 | 0.0000 | -0.284 | Yes |


### Per-Category Coefficient of Variation


| Category | Within CV | Between CV | Ratio | # Features |
|----------|-----------|------------|-------|------------|
| momentum | 59.166 ± 92.976 | 13.027 ± 0.000 | 0.220 | 1 |
| other | 104.046 ± 137.516 | 44.760 ± 66.601 | 0.430 | 12 |
| price | 2.927 ± 1.541 | 53.522 ± 0.889 | 18.284 | 4 |
| volatility | 7.147 ± 4.150 | 12.653 ± 0.000 | 1.770 | 1 |
| volume | 18.111 ± 1.182 | 19.345 ± 9.108 | 1.068 | 2 |

**Interpretation:** Higher CV ratio indicates better regime separation for that feature category.


---

## Balance and Distribution

**Balance Score:** 0.8856 (0-1, higher is better)

- **Smallest Cluster:** 21.01% of total
- **Largest Cluster:** 29.98% of total
- **Cluster Size Std Dev:** 252.75

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 25.00% |
| 1 | 24.00% |
| 2 | 21.01% |
| 3 | 29.98% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.1178 (0-1, higher = fewer transitions)
- **Temporal Smoothness (Raw):** 0.1178
- **Flip-Flop Ratio:** 0.0205 (rapid back-and-forth transitions)
- **Regime Persistence:** 5.90 bars (average duration)


### Transition Probability Matrix

This matrix shows the probability of transitioning from one regime to another:


### Regime Duration Analysis

- **Duration Stability Score:** 0.269 (higher = more consistent durations)


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 1956 samples (25.00%)

**Performance Metrics:**
- Mean Return: -0.0015907798471602955
- Volatility: 0.00788069392928818
- Sharpe Ratio: -0.2018578134839667
- Skewness: -0.5471351230996443
- Max Drawdown: -0.9600953212418747

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.09406952965235174
- Pct < -1.0% (Shorts): 0.18813905930470348
- Pct Target Hits: 0.28220858895705525

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 35.810114652229736
- Win Rate (Long Bias): 0.33333333215217387
- Return per Vol: 0.2018578134839667
- Profit Factor: 0.6870945427549592

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.0015907798471602955
- volatility: 0.00788069392928818
- stability_coefficient: 0.16795492620810118

### Regime 1 (stable)

**Size:** 1878 samples (24.00%)

**Performance Metrics:**
- Mean Return: -0.0006359550326119869
- Volatility: 0.006317934920407778
- Sharpe Ratio: -0.10065867090512379
- Skewness: -0.17239859201316607
- Max Drawdown: -0.7315294233529808

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.08839190628328009
- Pct < -1.0% (Shorts): 0.1299254526091587
- Pct Target Hits: 0.21831735889243878

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 34.555171442503
- Win Rate (Long Bias): 0.4048780469259488
- Return per Vol: 0.10065867090512379
- Profit Factor: 0.8771797327979483

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.0006359550326119869
- volatility: 0.006317934920407778
- stability_coefficient: 0.09145326507253031

### Regime 2 (stable)

**Size:** 1644 samples (21.01%)

**Performance Metrics:**
- Mean Return: 0.00018603244591011597
- Volatility: 0.005976229135480662
- Sharpe Ratio: 0.031128728595247374
- Skewness: -0.19935037217267185
- Max Drawdown: -0.20412296862721874

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.09610705596107055
- Pct < -1.0% (Shorts): 0.08759124087591241
- Pct Target Hits: 0.18369829683698297

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 30.738156441863378
- Win Rate (Long Bias): 0.5231788050989868
- Return per Vol: 0.031128728595247374
- Profit Factor: 0.984071989564773

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.00018603244591011597
- volatility: 0.005976229135480662
- stability_coefficient: 0.030189146186647822

### Regime 3 (stable)

**Size:** 2346 samples (29.98%)

**Performance Metrics:**
- Mean Return: 0.0020681442023042927
- Volatility: 0.007054453068529357
- Sharpe Ratio: 0.2931685687104401
- Skewness: 0.3948930289146138
- Max Drawdown: -0.04345489888821561

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.1867007672634271
- Pct < -1.0% (Shorts): 0.061381074168797956
- Pct Target Hits: 0.24808184143222506

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 35.16669596573654
- Win Rate (Long Bias): 0.752577316554044
- Return per Vol: 0.2931685687104401
- Profit Factor: 1.5255777816937313

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0020681442023042927
- volatility: 0.007054453068529357
- stability_coefficient: 0.22670571923751934


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5433968578383038

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Economic Relevance Analysis

### Strategy Performance Summary

### Statistical Significance Tests

### Economic Regime Mapping

| Regime | Economic Interpretation | Recommended Position |
|---------|----------------------|----------------------|

### Economic Interpretation

**Detailed Economic Report:** /Users/remyroche/Documents/Ares/outcomes/regime_economic_relevance_report_20251118_214834.md


---

## Quality Assessment

**Overall Quality Score:** 0.3341 / 1.0
**Quality Level:** Moderate ⚠️
**Recommendation:** The clustering shows moderate quality. Consider parameter tuning.

