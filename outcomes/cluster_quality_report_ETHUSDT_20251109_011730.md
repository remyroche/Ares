# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-09T01:17:30.025105
**Data Points:** N/A
**Number of Regimes:** 5
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

**Global Silhouette Score:** 0.1854

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.3470 | 0.1121 | -0.0537 | 0.5202 |
| 1 | 0.3032 | 0.1122 | -0.0673 | 0.5056 |
| 2 | 0.0672 | 0.1611 | -0.3834 | 0.3678 |
| 3 | 0.0145 | 0.1342 | -0.3376 | 0.3098 |
| 4 | -0.2213 | 0.1389 | -0.5214 | 0.1037 |


### Separation Metrics

- **Davies-Bouldin Index:** 2.0433 (lower is better)
- **Calinski-Harabasz Index:** 437.26 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 4.6480 +/- 2.1903
- **Between-Regime CV:** 12.7644 +/- 9.2718

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 2.0006 |
| 1 | 3.0495 |
| 2 | 7.2917 |
| 3 | 3.6955 |
| 4 | 7.2026 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 35.1690
- **Between-Regime CV (Mean Return):** 1.5312
- **CV Ratio (Between/Within):** 0.0435

| mean_return | 1.5312 |
| pct_above_target | 0.4035 |
| pct_below_neg_target | 0.3864 |
| pct_target_hits | 0.3793 |
| sharpe | 1.4794 |
| volatility | 0.2578 |


---

## Balance and Distribution

**Balance Score:** 0.7059 (0-1, higher is better)

- **Smallest Cluster:** 7.07% of total
- **Largest Cluster:** 29.48% of total
- **Cluster Size Std Dev:** 292.26

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 28.36% |
| 1 | 29.48% |
| 2 | 15.54% |
| 3 | 19.56% |
| 4 | 7.07% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.8899 (0-1, higher = fewer transitions)
- **Regime Persistence:** 9.09 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 995 samples (28.36%)

**Performance Metrics:**
- Mean Return: 0.00020971681806258857
- Volatility: 0.005585378501564264
- Sharpe Ratio: 0.037547460831238846
- Skewness: -0.9241966009140015
- Max Drawdown: -0.14834200134771786

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.07336683417085427
- Pct < -1.0% (Shorts): 0.0592964824120603
- Pct Target Hits: 0.13266331658291458

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 23.751889472462945
- Win Rate (Long Bias): 0.5530302988616276
- Return per Vol: 0.037547460831238846
- Profit Factor: 1.0492578744888306

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.00020971681806258857
- volatility: 0.005585378501564264
- stability_coefficient: 0.036188840788084146

### Regime 1 (stable)

**Size:** 1034 samples (29.48%)

**Performance Metrics:**
- Mean Return: 0.0001338523143203929
- Volatility: 0.006993517745286226
- Sharpe Ratio: 0.01913948030962412
- Skewness: -4.0984015464782715
- Max Drawdown: -0.1703469279929728

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.08027079303675048
- Pct < -1.0% (Shorts): 0.0735009671179884
- Pct Target Hits: 0.15377176015473887

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 21.987752625727094
- Win Rate (Long Bias): 0.5220125752216289
- Return per Vol: 0.01913948030962412
- Profit Factor: 0.9764066934585571

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0001338523143203929
- volatility: 0.006993517745286226
- stability_coefficient: 0.018780180406066955

### Regime 2 (stable)

**Size:** 545 samples (15.54%)

**Performance Metrics:**
- Mean Return: 0.0011243775952607393
- Volatility: 0.008787378668785095
- Sharpe Ratio: 0.12795368330957652
- Skewness: 0.4526984393596649
- Max Drawdown: -0.09270294415411688

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.1981651376146789
- Pct < -1.0% (Shorts): 0.12293577981651377
- Pct Target Hits: 0.3211009174311927

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 36.54114531682534
- Win Rate (Long Bias): 0.617142855220898
- Return per Vol: 0.12795368330957652
- Profit Factor: 1.224278450012207

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0011243775952607393
- volatility: 0.008787378668785095
- stability_coefficient: 0.11343887519717007

### Regime 3 (mean_reverting)

**Size:** 686 samples (19.56%)

**Performance Metrics:**
- Mean Return: -0.00022608407016377896
- Volatility: 0.00782767590135336
- Sharpe Ratio: -0.028882652287895324
- Skewness: -0.6963680982589722
- Max Drawdown: -0.27862547155401085

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.13702623906705538
- Pct < -1.0% (Shorts): 0.15451895043731778
- Pct Target Hits: 0.29154518950437314

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 37.24542966942934
- Win Rate (Long Bias): 0.4699999983879
- Return per Vol: 0.028882652287895324
- Profit Factor: 0.8464971780776978

**Regime-Specific Characteristics:**

- reversion_center: -0.00022608407016377896
- reversion_speed: 173.9200975700913
- reversion_range: 0.005307010840624571

### Regime 4 (mean_reverting)

**Size:** 248 samples (7.07%)

**Performance Metrics:**
- Mean Return: 0.00021745721460320055
- Volatility: 0.011896480806171894
- Sharpe Ratio: 0.018279119671362316
- Skewness: 0.5340057611465454
- Max Drawdown: -0.19250298421935588

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2056451612903226
- Pct < -1.0% (Shorts): 0.1774193548387097
- Pct Target Hits: 0.38306451612903225

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 32.199815236997075
- Win Rate (Long Bias): 0.5368421038617175
- Return per Vol: 0.018279119671362316
- Profit Factor: 1.1097261905670166

**Regime-Specific Characteristics:**

- reversion_center: 0.00021745721460320055
- reversion_speed: 127.94666112554359
- reversion_range: 0.008955048397183418


---

## Predictive Power

**Cross-Validation Accuracy:** 0.51981750784146

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Quality Assessment

**Overall Quality Score:** 0.5545 / 1.0
**Quality Level:** Good ✅
**Recommendation:** The clustering shows good quality. Suitable for most applications.

