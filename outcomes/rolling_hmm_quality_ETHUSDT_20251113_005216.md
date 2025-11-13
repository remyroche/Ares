# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-13T00:51:34.383559
**Data Points:** N/A
**Number of Regimes:** 4
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

**Global Silhouette Score:** 0.1304

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.2889 | 0.1169 | -0.1721 | 0.4770 |
| 1 | 0.1451 | 0.1531 | -0.3509 | 0.3796 |
| 2 | 0.1073 | 0.1775 | -0.4327 | 0.3861 |
| 3 | -0.0198 | 0.1365 | -0.4196 | 0.2340 |


### Separation Metrics

- **Davies-Bouldin Index:** 1.8325 (lower is better)
- **Calinski-Harabasz Index:** 6328.75 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 2.7497 +/- 0.7809
- **Between-Regime CV:** 143.6349 +/- 185.5682

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 3.8978 |
| 1 | 1.7095 |
| 2 | 2.8254 |
| 3 | 2.5663 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 20.4863
- **Between-Regime CV (Mean Return):** 20.9431
- **CV Ratio (Between/Within):** 1.0223

| mean_return | 20.9431 |
| pct_above_target | 0.4482 |
| pct_below_neg_target | 0.5456 |
| pct_target_hits | 0.4647 |
| sharpe | 7.5522 |
| volatility | 0.2986 |


---

## Balance and Distribution

**Balance Score:** 0.7667 (0-1, higher is better)

- **Smallest Cluster:** 14.94% of total
- **Largest Cluster:** 33.45% of total
- **Cluster Size Std Dev:** 2578.11

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 33.45% |
| 1 | 14.94% |
| 2 | 20.44% |
| 3 | 31.16% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.8427 (0-1, higher = fewer transitions)
- **Regime Persistence:** 6.36 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 11340 samples (33.45%)

**Performance Metrics:**
- Mean Return: -9.708449942991138e-05
- Volatility: 0.004168908577412367
- Sharpe Ratio: -0.02328774410362038
- Skewness: -1.2616243362426758
- Max Drawdown: -0.7446840953545603

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.028042328042328042
- Pct < -1.0% (Shorts): 0.03747795414462081
- Pct Target Hits: 0.06552028218694886

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 15.716407604987477
- Win Rate (Long Bias): 0.42799460988767307
- Return per Vol: 0.02328774410362038
- Profit Factor: 0.9444243907928467

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -9.708449942991138e-05
- volatility: 0.004168908577412367
- stability_coefficient: 0.02275800146018365

### Regime 1 (stable)

**Size:** 5066 samples (14.94%)

**Performance Metrics:**
- Mean Return: 0.0006616602186113596
- Volatility: 0.0061362613923847675
- Sharpe Ratio: 0.10782788875399038
- Skewness: 1.1201785802841187
- Max Drawdown: -0.12375013255333063

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.10461902881958153
- Pct < -1.0% (Shorts): 0.05823134622976708
- Pct Target Hits: 0.1628503750493486

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 26.53901750541981
- Win Rate (Long Bias): 0.6424242384793682
- Return per Vol: 0.10782788875399038
- Profit Factor: 1.272860050201416

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0006616602186113596
- volatility: 0.0061362613923847675
- stability_coefficient: 0.09733285541395767

### Regime 2 (stable)

**Size:** 6928 samples (20.44%)

**Performance Metrics:**
- Mean Return: -0.0009079216397367418
- Volatility: 0.009486570954322815
- Sharpe Ratio: -0.09570597726010212
- Skewness: -1.1875942945480347
- Max Drawdown: -0.9987556562295654

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.12485565819861431
- Pct < -1.0% (Shorts): 0.17234411085450346
- Pct Target Hits: 0.2971997690531178

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 31.32846833230318
- Win Rate (Long Bias): 0.420106846570908
- Return per Vol: 0.09570597726010212
- Profit Factor: 0.7678903937339783

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.0009079216397367418
- volatility: 0.009486570954322815
- stability_coefficient: 0.08734650048325768

### Regime 3 (stable)

**Size:** 10564 samples (31.16%)

**Performance Metrics:**
- Mean Return: 0.00045949561172164977
- Volatility: 0.008861735463142395
- Sharpe Ratio: 0.05185164483651452
- Skewness: 0.4367711544036865
- Max Drawdown: -0.4110374894156888

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.15070049223778872
- Pct < -1.0% (Shorts): 0.12523665278303672
- Pct Target Hits: 0.27593714502082545

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 31.13804457721116
- Win Rate (Long Bias): 0.5461406498218079
- Return per Vol: 0.05185164483651452
- Profit Factor: 1.116966724395752

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.00045949561172164977
- volatility: 0.008861735463142395
- stability_coefficient: 0.04929569482136853


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5003392630616279

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Quality Assessment

**Overall Quality Score:** 0.7526 / 1.0
**Quality Level:** Excellent ✅
**Recommendation:** The clustering shows excellent quality. Proceed with confidence.

