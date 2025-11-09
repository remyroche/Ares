# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-09T12:24:37.830401
**Data Points:** N/A
**Number of Regimes:** 6
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

**Global Silhouette Score:** 0.0895

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.1151 | 0.1702 | -0.3641 | 0.3761 |
| 1 | 0.2102 | 0.1273 | -0.2685 | 0.4163 |
| 2 | -0.2236 | 0.1457 | -0.6600 | 0.0935 |
| 3 | 0.4258 | 0.0622 | 0.2529 | 0.5423 |
| 4 | 0.3396 | 0.0993 | -0.0258 | 0.5091 |
| 5 | 0.1472 | 0.1478 | -0.4694 | 0.3498 |


### Separation Metrics

- **Davies-Bouldin Index:** 2.0890 (lower is better)
- **Calinski-Harabasz Index:** 488.01 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 5.5775 +/- 5.1878
- **Between-Regime CV:** 19.9262 +/- 19.3656

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 1.2459 |
| 1 | 1.6587 |
| 2 | 16.0689 |
| 3 | 7.9446 |
| 4 | 4.0040 |
| 5 | 2.5428 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 19.6685
- **Between-Regime CV (Mean Return):** 1.5963
- **CV Ratio (Between/Within):** 0.0812

| mean_return | 1.5963 |
| pct_above_target | 0.3524 |
| pct_below_neg_target | 0.4093 |
| pct_target_hits | 0.3449 |
| sharpe | 1.4959 |
| volatility | 0.2206 |


---

## Balance and Distribution

**Balance Score:** 0.6722 (0-1, higher is better)

- **Smallest Cluster:** 5.25% of total
- **Largest Cluster:** 31.19% of total
- **Cluster Size Std Dev:** 285.11

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 15.59% |
| 1 | 11.26% |
| 2 | 31.19% |
| 3 | 5.25% |
| 4 | 21.44% |
| 5 | 15.28% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.7542 (0-1, higher = fewer transitions)
- **Regime Persistence:** 4.07 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 547 samples (15.59%)

**Performance Metrics:**
- Mean Return: -0.00014634586113970727
- Volatility: 0.005827820394188166
- Sharpe Ratio: -0.02511158994777219
- Skewness: 0.07917364686727524
- Max Drawdown: -0.17763240665158117

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.06398537477148081
- Pct < -1.0% (Shorts): 0.10603290676416818
- Pct Target Hits: 0.170018281535649

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 29.173557327134436
- Win Rate (Long Bias): 0.3763440838079547
- Return per Vol: 0.02511158994777219
- Profit Factor: 0.9295470714569092

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.00014634586113970727
- volatility: 0.005827820394188166
- stability_coefficient: 0.02449661264658977

### Regime 1 (stable)

**Size:** 395 samples (11.26%)

**Performance Metrics:**
- Mean Return: 0.0008183162426576018
- Volatility: 0.008897121995687485
- Sharpe Ratio: 0.09197537710271592
- Skewness: -1.634596824645996
- Max Drawdown: -0.11100747015499181

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.16455696202531644
- Pct < -1.0% (Shorts): 0.12151898734177215
- Pct Target Hits: 0.2860759493670886

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 32.15375908658925
- Win Rate (Long Bias): 0.575221236927324
- Return per Vol: 0.09197537710271592
- Profit Factor: 1.0697526931762695

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0008183162426576018
- volatility: 0.008897121995687485
- stability_coefficient: 0.08422853795717761

### Regime 2 (stable)

**Size:** 1094 samples (31.19%)

**Performance Metrics:**
- Mean Return: 0.0003025209007319063
- Volatility: 0.008462544530630112
- Sharpe Ratio: 0.03574821543198019
- Skewness: -1.300912857055664
- Max Drawdown: -0.23067401730628537

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.1343692870201097
- Pct < -1.0% (Shorts): 0.10146252285191956
- Pct Target Hits: 0.23583180987202926

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 27.86771533676694
- Win Rate (Long Bias): 0.5697674394444745
- Return per Vol: 0.03574821543198019
- Profit Factor: 1.050703525543213

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0003025209007319063
- volatility: 0.008462544530630112
- stability_coefficient: 0.034514501755452974

### Regime 3 (mean_reverting)

**Size:** 184 samples (5.25%)

**Performance Metrics:**
- Mean Return: 0.0014233184047043324
- Volatility: 0.007324374280869961
- Sharpe Ratio: 0.1943262531101869
- Skewness: 2.28313946723938
- Max Drawdown: -0.04673637272827674

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.13043478260869565
- Pct < -1.0% (Shorts): 0.06521739130434782
- Pct Target Hits: 0.19565217391304346

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 26.7124725878061
- Win Rate (Long Bias): 0.6666666632592593
- Return per Vol: 0.1943262531101869
- Profit Factor: 1.4728699922561646

**Regime-Specific Characteristics:**

- reversion_center: 0.0014233184047043324
- reversion_speed: 201.00163866542363
- reversion_range: 0.005362812429666519

### Regime 4 (stable)

**Size:** 752 samples (21.44%)

**Performance Metrics:**
- Mean Return: 0.0003490505914669484
- Volatility: 0.0047690835781395435
- Sharpe Ratio: 0.07319027074229739
- Skewness: 0.9867442846298218
- Max Drawdown: -0.04596358196545729

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.06382978723404255
- Pct < -1.0% (Shorts): 0.041223404255319146
- Pct Target Hits: 0.1050531914893617

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 22.02795730880993
- Win Rate (Long Bias): 0.6075949309251722
- Return per Vol: 0.07319027074229739
- Profit Factor: 1.0876519680023193

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0003490505914669484
- volatility: 0.0047690835781395435
- stability_coefficient: 0.06819897871001036

### Regime 5 (mean_reverting)

**Size:** 536 samples (15.28%)

**Performance Metrics:**
- Mean Return: -0.0004464488010853529
- Volatility: 0.009214283898472786
- Sharpe Ratio: -0.04845181215954688
- Skewness: -0.836900532245636
- Max Drawdown: -0.29069329787404746

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.16791044776119404
- Pct < -1.0% (Shorts): 0.17164179104477612
- Pct Target Hits: 0.3395522388059702

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 36.850633830765204
- Win Rate (Long Bias): 0.4945054930491486
- Return per Vol: 0.04845181215954688
- Profit Factor: 0.8701431751251221

**Regime-Specific Characteristics:**

- reversion_center: -0.0004464488010853529
- reversion_speed: 151.04802637440937
- reversion_range: 0.006402443163096905


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5203877958368976

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Quality Assessment

**Overall Quality Score:** 0.5880 / 1.0
**Quality Level:** Good ✅
**Recommendation:** The clustering shows good quality. Suitable for most applications.

