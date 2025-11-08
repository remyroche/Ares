# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-08T12:28:31.151123
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

**Global Silhouette Score:** 0.0968

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.3869 | 0.0689 | 0.2298 | 0.5132 |
| 1 | 0.0618 | 0.1239 | -0.3531 | 0.2855 |
| 2 | 0.1608 | 0.1330 | -0.2005 | 0.4168 |
| 3 | 0.1020 | 0.1034 | -0.1479 | 0.2821 |
| 4 | 0.0541 | 0.1229 | -0.3223 | 0.2665 |


### Separation Metrics

- **Davies-Bouldin Index:** 2.0957 (lower is better)
- **Calinski-Harabasz Index:** 124.15 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 10.8668 +/- 12.6218
- **Between-Regime CV:** 38.7213 +/- 67.9253

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 35.7455 |
| 1 | 6.2923 |
| 2 | 7.6777 |
| 3 | 2.4517 |
| 4 | 2.1669 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 9.1507
- **Between-Regime CV (Mean Return):** 9.3612
- **CV Ratio (Between/Within):** 1.0230

| mean_return | 9.3612 |
| pct_above_target | 0.0404 |
| pct_below_neg_target | 0.2682 |
| pct_target_hits | 0.1090 |
| sharpe | 3.2292 |
| volatility | 0.1970 |


---

## Balance and Distribution

**Balance Score:** 0.6755 (0-1, higher is better)

- **Smallest Cluster:** 2.90% of total
- **Largest Cluster:** 30.10% of total
- **Cluster Size Std Dev:** 96.06

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 2.90% |
| 1 | 30.10% |
| 2 | 20.60% |
| 3 | 18.40% |
| 4 | 28.00% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.8749 (0-1, higher = fewer transitions)
- **Regime Persistence:** 7.99 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 29 samples (2.90%)

**Performance Metrics:**
- Mean Return: -0.004313753357563719
- Volatility: 0.018346711525632872
- Sharpe Ratio: -0.23512405023716454
- Skewness: -0.7152095360371123
- Max Drawdown: -0.1559839685947006

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.27586206896551724
- Pct < -1.0% (Shorts): 0.3448275862068966
- Pct Target Hits: 0.6206896551724138

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 33.83111030410627
- Win Rate (Long Bias): 0.4444444437283951
- Return per Vol: 0.23512405023716454
- Profit Factor: 0.5636699959075652

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.004313753357563719
- volatility: 0.018346711525632872
- stability_coefficient: 0.190364769188726

### Regime 1 (mean_reverting)

**Size:** 301 samples (30.10%)

**Performance Metrics:**
- Mean Return: -0.0006594494278118306
- Volatility: 0.016117754985116743
- Sharpe Ratio: -0.04091446901297992
- Skewness: 0.04835144847083777
- Max Drawdown: -0.23475027606094742

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2857142857142857
- Pct < -1.0% (Shorts): 0.33222591362126247
- Pct Target Hits: 0.6179401993355482

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 38.33909632992079
- Win Rate (Long Bias): 0.46236559064961263
- Return per Vol: 0.04091446901297992
- Profit Factor: 0.9734428201802395

**Regime-Specific Characteristics:**

- reversion_center: -0.0006594494278118306
- reversion_speed: 81.11841564274754
- reversion_range: 0.010358699730146517

### Regime 2 (stable)

**Size:** 206 samples (20.60%)

**Performance Metrics:**
- Mean Return: 0.002437747028465544
- Volatility: 0.01116529674298543
- Sharpe Ratio: 0.21833246945851062
- Skewness: 0.5962578283392961
- Max Drawdown: -0.0640636938714647

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2912621359223301
- Pct < -1.0% (Shorts): 0.2087378640776699
- Pct Target Hits: 0.5

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 44.78160918853452
- Win Rate (Long Bias): 0.5825242706796117
- Return per Vol: 0.21833246945851062
- Profit Factor: 1.2285890205175094

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.002437747028465544
- volatility: 0.01116529674298543
- stability_coefficient: 0.1792060578659347

### Regime 3 (stable)

**Size:** 184 samples (18.40%)

**Performance Metrics:**
- Mean Return: 0.002142652645924622
- Volatility: 0.012421693525214083
- Sharpe Ratio: 0.1724927820093607
- Skewness: -0.36516204639876343
- Max Drawdown: -0.09280974090916638

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.29891304347826086
- Pct < -1.0% (Shorts): 0.1793478260869565
- Pct Target Hits: 0.4782608695652174

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 38.502063353306845
- Win Rate (Long Bias): 0.6249999986931818
- Return per Vol: 0.1724927820093607
- Profit Factor: 1.0886067913050081

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.002142652645924622
- volatility: 0.012421693525214083
- stability_coefficient: 0.14711635343124685

### Regime 4 (stable)

**Size:** 280 samples (28.00%)

**Performance Metrics:**
- Mean Return: 0.0017497239254685877
- Volatility: 0.011960733209012354
- Sharpe Ratio: 0.1462890065854134
- Skewness: 0.003496881205288253
- Max Drawdown: -0.09996051497342126

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.3107142857142857
- Pct < -1.0% (Shorts): 0.21428571428571427
- Pct Target Hits: 0.525

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 43.89362649697668
- Win Rate (Long Bias): 0.5918367335665695
- Return per Vol: 0.1462890065854134
- Profit Factor: 1.198245015119842

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0017497239254685877
- volatility: 0.011960733209012354
- stability_coefficient: 0.1276197270949057


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5175276381909548

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Quality Assessment

**Overall Quality Score:** 0.5887 / 1.0
**Quality Level:** Good ✅
**Recommendation:** The clustering shows good quality. Suitable for most applications.

