# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-08T23:59:42.947529
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

**Global Silhouette Score:** 0.2204

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.3369 | 0.1070 | -0.0816 | 0.5070 |
| 1 | 0.0499 | 0.1562 | -0.3512 | 0.3458 |
| 2 | -0.1499 | 0.1466 | -0.5296 | 0.1748 |
| 3 | 0.3170 | 0.1104 | -0.0077 | 0.5057 |


### Separation Metrics

- **Davies-Bouldin Index:** 1.8379 (lower is better)
- **Calinski-Harabasz Index:** 579.52 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 3.9333 +/- 1.3007
- **Between-Regime CV:** 167.0821 +/- 317.7062

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 2.3344 |
| 1 | 5.0834 |
| 2 | 2.9771 |
| 3 | 5.3381 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 75.4369
- **Between-Regime CV (Mean Return):** 1.9009
- **CV Ratio (Between/Within):** 0.0252

| mean_return | 1.9009 |
| pct_above_target | 0.4615 |
| pct_below_neg_target | 0.4053 |
| pct_target_hits | 0.4242 |
| sharpe | 1.7463 |
| volatility | 0.2517 |


---

## Balance and Distribution

**Balance Score:** 0.7063 (0-1, higher is better)

- **Smallest Cluster:** 12.49% of total
- **Largest Cluster:** 35.35% of total
- **Cluster Size Std Dev:** 364.75

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 35.35% |
| 1 | 16.96% |
| 2 | 12.49% |
| 3 | 35.21% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.9250 (0-1, higher = fewer transitions)
- **Regime Persistence:** 13.33 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 1240 samples (35.35%)

**Performance Metrics:**
- Mean Return: 5.373579915612936e-05
- Volatility: 0.005809396971017122
- Sharpe Ratio: 0.009249805130276033
- Skewness: -1.0679324865341187
- Max Drawdown: -0.19320084389237807

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.07419354838709677
- Pct < -1.0% (Shorts): 0.07258064516129033
- Pct Target Hits: 0.1467741935483871

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 25.264957622224934
- Win Rate (Long Bias): 0.505494502050477
- Return per Vol: 0.009249805130276033
- Profit Factor: 0.952166736125946

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 5.373579915612936e-05
- volatility: 0.005809396971017122
- stability_coefficient: 0.009165200942454614

### Regime 1 (stable)

**Size:** 595 samples (16.96%)

**Performance Metrics:**
- Mean Return: 0.0013576741330325603
- Volatility: 0.008728346787393093
- Sharpe Ratio: 0.15554766676386741
- Skewness: 0.6207973957061768
- Max Drawdown: -0.08135399441072287

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.20840336134453782
- Pct < -1.0% (Shorts): 0.12941176470588237
- Pct Target Hits: 0.3378151260504202

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 38.70321557744847
- Win Rate (Long Bias): 0.6169154210593797
- Return per Vol: 0.15554766676386741
- Profit Factor: 1.3188079595565796

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0013576741330325603
- volatility: 0.008728346787393093
- stability_coefficient: 0.13460957588076145

### Regime 2 (mean_reverting)

**Size:** 438 samples (12.49%)

**Performance Metrics:**
- Mean Return: -0.0001871532731456682
- Volatility: 0.011211730539798737
- Sharpe Ratio: -0.016692628830910078
- Skewness: 0.10244131833314896
- Max Drawdown: -0.2143896538341125

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.21232876712328766
- Pct < -1.0% (Shorts): 0.19406392694063926
- Pct Target Hits: 0.4063926940639269

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 36.247094627742236
- Win Rate (Long Bias): 0.5224719088267263
- Return per Vol: 0.016692628830910078
- Profit Factor: 0.9530674815177917

**Regime-Specific Characteristics:**

- reversion_center: -0.0001871532731456682
- reversion_speed: 128.36221175776174
- reversion_range: 0.008054368197917938

### Regime 3 (stable)

**Size:** 1235 samples (35.21%)

**Performance Metrics:**
- Mean Return: 5.3844247304368764e-05
- Volatility: 0.006863559130579233
- Sharpe Ratio: 0.007844944355404768
- Skewness: -3.7402095794677734
- Max Drawdown: -0.16514621846065897

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.08097165991902834
- Pct < -1.0% (Shorts): 0.08097165991902834
- Pct Target Hits: 0.16194331983805668

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 23.594653030946766
- Win Rate (Long Bias): 0.4999999969125
- Return per Vol: 0.007844944355404768
- Profit Factor: 0.9460631608963013

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 5.3844247304368764e-05
- volatility: 0.006863559130579233
- stability_coefficient: 0.007784024810884753


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5203877958368976

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Quality Assessment

**Overall Quality Score:** 0.7647 / 1.0
**Quality Level:** Excellent ✅
**Recommendation:** The clustering shows excellent quality. Proceed with confidence.

