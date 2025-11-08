# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-08T17:23:21.515353
**Data Points:** N/A
**Number of Regimes:** 7
**Report Version:** 1.3 (Enhanced with Financial Analysis)

This report provides a comprehensive assessment of cluster quality for ETHUSDT.

### Key Metrics

# --- 5. START: NEW MODULAR SECTION ---
        # Dynamically add the method-specific configuration table if provided
        if method_specific_config:
            md += "
---

## Clustering Method Configuration

"
            md += "| Parameter | Value |
"
            md += "|---|---|
"
            for key, value in method_specific_config.items():
                # Format common values nicely
                if isinstance(value, float):
                    value_str = "{:.4f}".format(value)
                else:
                    value_str = str(value)
                md += "| {} | {} |
".format(key, value_str)
            md += "
"
        # --- END: NEW MODULAR SECTION ---

        md += 

---

## Top Configuration Analysis

### Clustering Configuration Parameters


---

## Clustering Metrics

### Silhouette Analysis

**Global Silhouette Score:** 0.0527

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.1860 | 0.0927 | 0.0232 | 0.3509 |
| 1 | 0.1462 | 0.0993 | -0.1376 | 0.3513 |
| 2 | 0.0183 | 0.1046 | -0.2192 | 0.2008 |
| 3 | 0.0772 | 0.1291 | -0.2849 | 0.2752 |
| 4 | 0.0140 | 0.1103 | -0.2825 | 0.2145 |
| 5 | 0.0324 | 0.1056 | -0.2422 | 0.2374 |
| 6 | -0.0848 | 0.1569 | -0.3826 | 0.1931 |


### Separation Metrics

- **Davies-Bouldin Index:** 2.3763 (lower is better)
- **Calinski-Harabasz Index:** 38.59 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 9.9078 +/- 5.8825
- **Between-Regime CV:** 14.0853 +/- 12.0670

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 9.4164 |
| 1 | 14.7464 |
| 2 | 4.0253 |
| 3 | 3.1922 |
| 4 | 21.1190 |
| 5 | 10.5242 |
| 6 | 6.3309 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 14.3095
- **Between-Regime CV (Mean Return):** 24.9896
- **CV Ratio (Between/Within):** 1.7464

| mean_return | 24.9896 |
| pct_above_target | 0.2134 |
| pct_below_neg_target | 0.2691 |
| pct_target_hits | 0.1395 |
| sharpe | 28.3081 |
| volatility | 0.1505 |


---

## Balance and Distribution

**Balance Score:** 0.7056 (0-1, higher is better)

- **Smallest Cluster:** 7.29% of total
- **Largest Cluster:** 24.17% of total
- **Cluster Size Std Dev:** 28.61

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 9.58% |
| 1 | 13.33% |
| 2 | 12.50% |
| 3 | 10.83% |
| 4 | 22.29% |
| 5 | 24.17% |
| 6 | 7.29% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.6806 (0-1, higher = fewer transitions)
- **Regime Persistence:** 3.13 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (mean_reverting)

**Size:** 46 samples (9.58%)

**Performance Metrics:**
- Mean Return: 0.003515258778365619
- Volatility: 0.014480384184169316
- Sharpe Ratio: 0.24276003253067305
- Skewness: 0.5485539309826213
- Max Drawdown: -0.03741214592899745

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.34782608695652173
- Pct < -1.0% (Shorts): 0.2608695652173913
- Pct Target Hits: 0.6086956521739131

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 42.03587435224926
- Win Rate (Long Bias): 0.5714285704897959
- Return per Vol: 0.24276003253067305
- Profit Factor: 1.6151682082402994

**Regime-Specific Characteristics:**

- reversion_center: 0.003515258778365619
- reversion_speed: 85.72588069873387
- reversion_range: 0.008397300674955758

### Regime 1 (stable)

**Size:** 64 samples (13.33%)

**Performance Metrics:**
- Mean Return: 0.0010182519817940033
- Volatility: 0.012853697875227657
- Sharpe Ratio: 0.0792185962716486
- Skewness: 0.10987436946672517
- Max Drawdown: -0.08893197297864963

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.234375
- Pct < -1.0% (Shorts): 0.21875
- Pct Target Hits: 0.453125

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 35.25249847522778
- Win Rate (Long Bias): 0.5172413781688466
- Return per Vol: 0.0792185962716486
- Profit Factor: 1.1106004472387925

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0010182519817940033
- volatility: 0.012853697875227657
- stability_coefficient: 0.07340373335294705

### Regime 2 (mean_reverting)

**Size:** 60 samples (12.50%)

**Performance Metrics:**
- Mean Return: -0.0007164714386092637
- Volatility: 0.011307460737067456
- Sharpe Ratio: -0.06336271174463223
- Skewness: -0.11063785254188914
- Max Drawdown: -0.06452548020127766

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.23333333333333334
- Pct < -1.0% (Shorts): 0.23333333333333334
- Pct Target Hits: 0.4666666666666667

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 41.270682803804455
- Win Rate (Long Bias): 0.4999999989285714
- Return per Vol: 0.06336271174463223
- Profit Factor: 0.9692743053192514

**Regime-Specific Characteristics:**

- reversion_center: -0.0007164714386092637
- reversion_speed: 115.10430893197625
- reversion_range: 0.007148568024808956

### Regime 3 (stable)

**Size:** 52 samples (10.83%)

**Performance Metrics:**
- Mean Return: -0.0003062730739097566
- Volatility: 0.017048065168641578
- Sharpe Ratio: -0.01796526778345799
- Skewness: -0.6210275148291156
- Max Drawdown: -0.0956162210313951

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.3269230769230769
- Pct < -1.0% (Shorts): 0.28846153846153844
- Pct Target Hits: 0.6153846153846154

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 36.0970334873853
- Win Rate (Long Bias): 0.5312499991367188
- Return per Vol: 0.01796526778345799
- Profit Factor: 0.8837055104827042

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.0003062730739097566
- volatility: 0.017048065168641578
- stability_coefficient: 0.01764827053506014

### Regime 4 (stable)

**Size:** 107 samples (22.29%)

**Performance Metrics:**
- Mean Return: 0.002273716061387393
- Volatility: 0.013794935690256301
- Sharpe Ratio: 0.16482250788387987
- Skewness: 0.15307543014607253
- Max Drawdown: -0.08790029345562053

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.37383177570093457
- Pct < -1.0% (Shorts): 0.2523364485981308
- Pct Target Hits: 0.6261682242990654

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 45.391163320187125
- Win Rate (Long Bias): 0.5970149244196926
- Return per Vol: 0.16482250788387987
- Profit Factor: 1.3364233546820654

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.002273716061387393
- volatility: 0.013794935690256301
- stability_coefficient: 0.14150016784417782

### Regime 5 (stable)

**Size:** 116 samples (24.17%)

**Performance Metrics:**
- Mean Return: 0.0024324099610847877
- Volatility: 0.012086063952204597
- Sharpe Ratio: 0.20125739607589122
- Skewness: -0.1241869562688246
- Max Drawdown: -0.0793217732982355

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.28448275862068967
- Pct < -1.0% (Shorts): 0.1896551724137931
- Pct Target Hits: 0.47413793103448276

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 39.23013262873423
- Win Rate (Long Bias): 0.5999999987345455
- Return per Vol: 0.20125739607589122
- Profit Factor: 1.150014503707337

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0024324099610847877
- volatility: 0.012086063952204597
- stability_coefficient: 0.16753901326497433

### Regime 6 (trending)

**Size:** 35 samples (7.29%)

**Performance Metrics:**
- Mean Return: -0.009337878193504616
- Volatility: 0.01712847174486615
- Sharpe Ratio: -0.5451670054064465
- Skewness: -0.7644114959533973
- Max Drawdown: -0.30703555253878356

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2
- Pct < -1.0% (Shorts): 0.42857142857142855
- Pct Target Hits: 0.6285714285714286

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 36.697459133350215
- Win Rate (Long Bias): 0.31818181767561987
- Return per Vol: 0.5451670054064465
- Profit Factor: 0.5183460959191963

**Regime-Specific Characteristics:**

- trend_direction: bearish
- trend_consistency: 0.7142857142857143
- trend_acceleration: 0.07628070035312388


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5365172955974843

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Quality Assessment

**Overall Quality Score:** 0.4267 / 1.0
**Quality Level:** Moderate ⚠️
**Recommendation:** The clustering shows moderate quality. Consider parameter tuning.

