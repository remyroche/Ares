# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-18T01:06:05.877700
**Data Points:** N/A
**Number of Regimes:** 5
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

**Global Silhouette Score:** -0.0378

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | -0.0686 | 0.0779 | -0.3893 | 0.0670 |
| 1 | -0.0312 | 0.0798 | -0.2097 | 0.0887 |
| 2 | -0.0323 | 0.0310 | -0.1045 | 0.0406 |
| 3 | 0.0215 | 0.0503 | -0.1258 | 0.0951 |
| 4 | -0.0786 | 0.1032 | -0.2696 | 0.0944 |


### Separation Metrics

- **Davies-Bouldin Index:** 8.1667 (lower is better)
- **Calinski-Harabasz Index:** 170.53 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 690.4679 +/- 1340.3526
- **Between-Regime CV:** 42620.8607 +/- 76474.3738

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 3371.0031 |
| 1 | 48.9897 |
| 2 | 16.2680 |
| 3 | 8.1859 |
| 4 | 7.8929 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 28.5693
- **Between-Regime CV (Mean Return):** 27.5478
- **CV Ratio (Between/Within):** 0.9642

| mean_return | 27.5478 |
| pct_above_target | 0.4769 |
| pct_below_neg_target | 0.4485 |
| pct_target_hits | 0.2552 |
| sharpe | 11.9893 |
| volatility | 0.2539 |


---

## Economic Gap Analysis

### Per-Regime Snapshot

| Regime | Type | Mean Return | Volatility | Sharpe | Max DD | Pct Target Hits |
|--------|------|-------------|------------|--------|--------|-----------------|
| 0 | stable | -0.002144 | 0.011013 | -0.1947 | -96.83% | 30.93% |
| 1 | stable | -0.000801 | 0.006699 | -0.1195 | -76.68% | 22.36% |
| 2 | stable | -0.000056 | 0.006258 | -0.0089 | -29.22% | 17.90% |
| 3 | stable | 0.000394 | 0.005741 | 0.0686 | -13.73% | 16.55% |
| 4 | trending | 0.002909 | 0.008807 | 0.3303 | -3.85% | 30.22% |

### Pairwise Economic Spreads

| Regime A | Regime B | Mean Return Spread | Sharpe Spread | Volatility Ratio | Max DD Spread |
|----------|----------|--------------------|---------------|------------------|---------------|
| 0 | 1 | -0.001343 | -0.0751 | 1.644 | -20.15% |
| 0 | 2 | -0.002088 | -0.1857 | 1.760 | -67.61% |
| 0 | 3 | -0.002537 | -0.2632 | 1.919 | -83.10% |
| 0 | 4 | -0.005053 | -0.5249 | 1.251 | -92.98% |
| 1 | 2 | -0.000745 | -0.1106 | 1.070 | -47.46% |
| 1 | 3 | -0.001194 | -0.1881 | 1.167 | -62.95% |
| 1 | 4 | -0.003709 | -0.4498 | 0.761 | -72.83% |
| 2 | 3 | -0.000450 | -0.0775 | 1.090 | -15.49% |
| 2 | 4 | -0.002965 | -0.3392 | 0.711 | -25.37% |
| 3 | 4 | -0.002515 | -0.2617 | 0.652 | -9.88% |

### Statistical Tests (ANOVA / t-tests)

- **ANOVA F-statistic:** 85.6989, p-value=0.0000 (significant)

**Pairwise t-tests:**

| Regime A | Regime B | t-stat | p-value | Cohen's d | Significant |
|----------|----------|--------|---------|-----------|-------------|
| 0 | 1 | -4.1221 | 0.0000 | -0.147 | Yes |
| 0 | 2 | -6.5198 | 0.0000 | -0.233 | Yes |
| 0 | 3 | -8.0826 | 0.0000 | -0.289 | Yes |
| 0 | 4 | -14.1743 | 0.0000 | -0.507 | Yes |
| 1 | 2 | -3.2127 | 0.0013 | -0.115 | Yes |
| 1 | 3 | -5.3552 | 0.0000 | -0.191 | Yes |
| 1 | 4 | -13.2616 | 0.0000 | -0.474 | Yes |
| 2 | 3 | -2.0941 | 0.0363 | -0.075 | Yes |
| 2 | 4 | -10.8543 | 0.0000 | -0.388 | Yes |
| 3 | 4 | -9.4645 | 0.0000 | -0.338 | Yes |


### Per-Category Coefficient of Variation


| Category | Within CV | Between CV | Ratio | # Features |
|----------|-----------|------------|-------|------------|
| momentum | 19.749 ± 26.910 | 151532.476 ± 0.000 | 7672.957 | 1 |
| other | 1131.123 ± 2241.757 | 51123.335 ± 90261.107 | 45.197 | 12 |
| price | 3.426 ± 1.829 | 15995.150 ± 216.301 | 4669.123 | 4 |
| volatility | 4.048 ± 2.405 | 8524.008 ± 0.000 | 2105.733 | 1 |
| volume | 99.192 ± 163.331 | 7450.058 ± 195.870 | 75.107 | 2 |

**Interpretation:** Higher CV ratio indicates better regime separation for that feature category.


---

## Balance and Distribution

**Balance Score:** 0.9997 (0-1, higher is better)

- **Smallest Cluster:** 19.99% of total
- **Largest Cluster:** 20.00% of total
- **Cluster Size Std Dev:** 0.40

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 20.00% |
| 1 | 20.00% |
| 2 | 19.99% |
| 3 | 20.00% |
| 4 | 20.00% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.0911 (0-1, higher = fewer transitions)
- **Temporal Smoothness (Raw):** 0.0911
- **Flip-Flop Ratio:** 0.0259 (rapid back-and-forth transitions)
- **Regime Persistence:** 4.56 bars (average duration)


### Transition Probability Matrix

This matrix shows the probability of transitioning from one regime to another:


### Regime Duration Analysis

- **Duration Stability Score:** 0.447 (higher = more consistent durations)


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 1565 samples (20.00%)

**Performance Metrics:**
- Mean Return: -0.002143815219549954
- Volatility: 0.011013248898366256
- Sharpe Ratio: -0.19465782029226225
- Skewness: -2.5965208309112033
- Max Drawdown: -0.9682715401811225

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.10543130990415335
- Pct < -1.0% (Shorts): 0.20383386581469648
- Pct Target Hits: 0.3092651757188498

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 28.08119116272185
- Win Rate (Long Bias): 0.3409090898067712
- Return per Vol: 0.19465782029226225
- Profit Factor: 0.6179171508726852

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.002143815219549954
- volatility: 0.011013248898366256
- stability_coefficient: 0.16294030624129702

### Regime 1 (stable)

**Size:** 1565 samples (20.00%)

**Performance Metrics:**
- Mean Return: -0.0008006204300180484
- Volatility: 0.006699086435846351
- Sharpe Ratio: -0.11951186451664807
- Skewness: -0.22878064855651423
- Max Drawdown: -0.7668129107314438

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.08306709265175719
- Pct < -1.0% (Shorts): 0.14057507987220447
- Pct Target Hits: 0.22364217252396165

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 33.38397575276715
- Win Rate (Long Bias): 0.3714285697677551
- Return per Vol: 0.11951186451664807
- Profit Factor: 0.797174286643627

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.0008006204300180484
- volatility: 0.006699086435846351
- stability_coefficient: 0.10675368218836223

### Regime 2 (stable)

**Size:** 1564 samples (19.99%)

**Performance Metrics:**
- Mean Return: -5.599103127180812e-05
- Volatility: 0.006258493328409583
- Sharpe Ratio: -0.008946406009772104
- Skewness: -0.35732629462132864
- Max Drawdown: -0.2922161038440332

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.08567774936061381
- Pct < -1.0% (Shorts): 0.09335038363171355
- Pct Target Hits: 0.17902813299232737

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 28.60562358899225
- Win Rate (Long Bias): 0.47857142589826535
- Return per Vol: 0.008946406009772104
- Profit Factor: 0.9625485820288381

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -5.599103127180812e-05
- volatility: 0.006258493328409583
- stability_coefficient: 0.00886723590006602

### Regime 3 (stable)

**Size:** 1565 samples (20.00%)

**Performance Metrics:**
- Mean Return: 0.0003936392511605438
- Volatility: 0.005740535076591541
- Sharpe Ratio: 0.06857186261152602
- Skewness: 0.05382613376401954
- Max Drawdown: -0.1372881219176626

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.09520766773162939
- Pct < -1.0% (Shorts): 0.07028753993610223
- Pct Target Hits: 0.16549520766773163

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 28.82922526043854
- Win Rate (Long Bias): 0.5752895718134047
- Return per Vol: 0.06857186261152602
- Profit Factor: 1.0906896420737124

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0003936392511605438
- volatility: 0.005740535076591541
- stability_coefficient: 0.064171666137361

### Regime 4 (trending)

**Size:** 1565 samples (20.00%)

**Performance Metrics:**
- Mean Return: 0.0029087313335721095
- Volatility: 0.008806901185566051
- Sharpe Ratio: 0.33027860106580204
- Skewness: 1.9112104047532747
- Max Drawdown: -0.03847010803803857

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.23514376996805111
- Pct < -1.0% (Shorts): 0.0670926517571885
- Pct Target Hits: 0.3022364217252396

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 34.31813086565056
- Win Rate (Long Bias): 0.7780126824152434
- Return per Vol: 0.33027860106580204
- Profit Factor: 1.74033787518981

**Regime-Specific Characteristics:**

- trend_direction: bullish
- trend_consistency: 0.6146964856230032
- trend_acceleration: -0.1795056314699098


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5289535292350389

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Economic Relevance Analysis

### Strategy Performance Summary

### Statistical Significance Tests

### Economic Regime Mapping

| Regime | Economic Interpretation | Recommended Position |
|---------|----------------------|----------------------|

### Economic Interpretation

**Detailed Economic Report:** /Users/remyroche/Documents/Ares/outcomes/regime_economic_relevance_report_20251118_010612.md


---

## Quality Assessment

**Overall Quality Score:** 0.7502 / 1.0
**Quality Level:** Excellent ✅
**Recommendation:** The clustering shows excellent quality. Proceed with confidence.

