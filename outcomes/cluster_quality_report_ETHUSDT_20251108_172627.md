# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-08T17:26:27.757486
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

**Global Silhouette Score:** 0.0474

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.2279 | 0.0666 | 0.0668 | 0.3492 |
| 1 | 0.0310 | 0.1263 | -0.3252 | 0.2646 |
| 2 | 0.0917 | 0.0903 | -0.1859 | 0.2630 |
| 3 | -0.1124 | 0.1092 | -0.3851 | 0.0801 |
| 4 | 0.1117 | 0.0756 | -0.0781 | 0.2677 |


### Separation Metrics

- **Davies-Bouldin Index:** 3.0009 (lower is better)
- **Calinski-Harabasz Index:** 44.09 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 7.0621 +/- 2.9020
- **Between-Regime CV:** 408.8490 +/- 948.3937

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 2.7892 |
| 1 | 6.9840 |
| 2 | 11.9293 |
| 3 | 6.6976 |
| 4 | 6.9104 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 14.1660
- **Between-Regime CV (Mean Return):** 5.6176
- **CV Ratio (Between/Within):** 0.3966

| mean_return | 5.6176 |
| pct_above_target | 0.1884 |
| pct_below_neg_target | 0.2579 |
| pct_target_hits | 0.1187 |
| sharpe | 3.3278 |
| volatility | 0.1745 |


---

## Balance and Distribution

**Balance Score:** 0.6994 (0-1, higher is better)

- **Smallest Cluster:** 7.08% of total
- **Largest Cluster:** 32.71% of total
- **Cluster Size Std Dev:** 41.26

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 7.08% |
| 1 | 32.71% |
| 2 | 25.21% |
| 3 | 18.33% |
| 4 | 16.67% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.8852 (0-1, higher = fewer transitions)
- **Regime Persistence:** 8.71 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (mean_reverting)

**Size:** 34 samples (7.08%)

**Performance Metrics:**
- Mean Return: 0.00035887794877213986
- Volatility: 0.015083097898280731
- Sharpe Ratio: 0.023793382990616542
- Skewness: 0.8589929986187771
- Max Drawdown: -0.06747737954904706

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.23529411764705882
- Pct < -1.0% (Shorts): 0.35294117647058826
- Pct Target Hits: 0.5882352941176471

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 38.99963118220341
- Win Rate (Long Bias): 0.39999999932
- Return per Vol: 0.023793382990616542
- Profit Factor: 1.2767454457428844

**Regime-Specific Characteristics:**

- reversion_center: 0.00035887794877213986
- reversion_speed: 86.09992320408584
- reversion_range: 0.009401587081638029

### Regime 1 (stable)

**Size:** 157 samples (32.71%)

**Performance Metrics:**
- Mean Return: 0.0022444716168557445
- Volatility: 0.014726282386054869
- Sharpe Ratio: 0.1524126324352252
- Skewness: -0.2962418695930791
- Max Drawdown: -0.09111242061336829

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.3630573248407643
- Pct < -1.0% (Shorts): 0.22929936305732485
- Pct Target Hits: 0.5923566878980892

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 40.22445259059896
- Win Rate (Long Bias): 0.6129032247717655
- Return per Vol: 0.1524126324352252
- Profit Factor: 1.1615566305202711

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0022444716168557445
- volatility: 0.014726282386054869
- stability_coefficient: 0.13225531901620172

### Regime 2 (mean_reverting)

**Size:** 121 samples (25.21%)

**Performance Metrics:**
- Mean Return: 0.002338534962868688
- Volatility: 0.009293364981345465
- Sharpe Ratio: 0.2516348723985281
- Skewness: 0.11371008423818257
- Max Drawdown: -0.04078282265161321

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2809917355371901
- Pct < -1.0% (Shorts): 0.15702479338842976
- Pct Target Hits: 0.4380165289256198

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 47.13217254166559
- Win Rate (Long Bias): 0.641509432497686
- Return per Vol: 0.2516348723985281
- Profit Factor: 1.2774209936551302

**Regime-Specific Characteristics:**

- reversion_center: 0.002338534962868688
- reversion_speed: 134.7908651252688
- reversion_range: 0.005555888541703477

### Regime 3 (mean_reverting)

**Size:** 88 samples (18.33%)

**Performance Metrics:**
- Mean Return: -0.0019944507020107287
- Volatility: 0.016372096413794033
- Sharpe Ratio: -0.12182010964156237
- Skewness: -0.5142096744951765
- Max Drawdown: -0.2457290286866827

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.29545454545454547
- Pct < -1.0% (Shorts): 0.3181818181818182
- Pct Target Hits: 0.6136363636363636

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 37.48062011403345
- Win Rate (Long Bias): 0.48148148069684504
- Return per Vol: 0.12182010964156237
- Profit Factor: 0.9961428250955701

**Regime-Specific Characteristics:**

- reversion_center: -0.0019944507020107287
- reversion_speed: 82.17815615901374
- reversion_range: 0.010875047885424396

### Regime 4 (mean_reverting)

**Size:** 80 samples (16.67%)

**Performance Metrics:**
- Mean Return: -0.0013576795237871114
- Volatility: 0.014732435241590833
- Sharpe Ratio: -0.09215580515829927
- Skewness: 0.04607449200771135
- Max Drawdown: -0.16224785527323993

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2125
- Pct < -1.0% (Shorts): 0.3
- Pct Target Hits: 0.5125

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 34.787186015655166
- Win Rate (Long Bias): 0.4146341455324212
- Return per Vol: 0.09215580515829927
- Profit Factor: 0.9021519211235882

**Regime-Specific Characteristics:**

- reversion_center: -0.0013576795237871114
- reversion_speed: 93.05801687607313
- reversion_range: 0.010005336817382497


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5260875262054507

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Quality Assessment

**Overall Quality Score:** 0.7369 / 1.0
**Quality Level:** Excellent ✅
**Recommendation:** The clustering shows excellent quality. Proceed with confidence.

