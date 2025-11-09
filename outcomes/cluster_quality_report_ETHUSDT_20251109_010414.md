# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-09T01:04:14.178458
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

**Global Silhouette Score:** 0.2034

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.3378 | 0.1150 | -0.0857 | 0.5114 |
| 1 | -0.2183 | 0.1351 | -0.4798 | 0.1038 |
| 2 | 0.0715 | 0.1507 | -0.3350 | 0.3627 |
| 3 | 0.3040 | 0.1129 | -0.0371 | 0.5075 |
| 4 | 0.0153 | 0.1341 | -0.3944 | 0.3109 |


### Separation Metrics

- **Davies-Bouldin Index:** 2.0235 (lower is better)
- **Calinski-Harabasz Index:** 440.77 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 7.5748 +/- 4.9983
- **Between-Regime CV:** 502.7388 +/- 1072.7575

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 2.8922 |
| 1 | 11.2752 |
| 2 | 1.7188 |
| 3 | 7.0175 |
| 4 | 14.9703 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 36.1013
- **Between-Regime CV (Mean Return):** 1.6074
- **CV Ratio (Between/Within):** 0.0445

| mean_return | 1.6074 |
| pct_above_target | 0.4171 |
| pct_below_neg_target | 0.3889 |
| pct_target_hits | 0.3861 |
| sharpe | 1.5464 |
| volatility | 0.2691 |


---

## Balance and Distribution

**Balance Score:** 0.6613 (0-1, higher is better)

- **Smallest Cluster:** 6.41% of total
- **Largest Cluster:** 32.10% of total
- **Cluster Size Std Dev:** 359.33

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 32.10% |
| 1 | 6.41% |
| 2 | 13.91% |
| 3 | 31.78% |
| 4 | 15.79% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.8905 (0-1, higher = fewer transitions)
- **Regime Persistence:** 9.13 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 1126 samples (32.10%)

**Performance Metrics:**
- Mean Return: 0.00020198219863232225
- Volatility: 0.0057546887546777725
- Sharpe Ratio: 0.0350987120492705
- Skewness: -0.8705664277076721
- Max Drawdown: -0.1928750334286469

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.07282415630550622
- Pct < -1.0% (Shorts): 0.0674955595026643
- Pct Target Hits: 0.14031971580817051

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 24.3835413879801
- Win Rate (Long Bias): 0.5189873380735459
- Return per Vol: 0.0350987120492705
- Profit Factor: 1.0313225984573364

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.00020198219863232225
- volatility: 0.0057546887546777725
- stability_coefficient: 0.03390873296624655

### Regime 1 (mean_reverting)

**Size:** 225 samples (6.41%)

**Performance Metrics:**
- Mean Return: 0.0001932782615767792
- Volatility: 0.01243848167359829
- Sharpe Ratio: 0.015538733031081704
- Skewness: 0.5247756838798523
- Max Drawdown: -0.18029735015115525

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2222222222222222
- Pct < -1.0% (Shorts): 0.19111111111111112
- Pct Target Hits: 0.41333333333333333

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 33.23020533771918
- Win Rate (Long Bias): 0.5376344073014221
- Return per Vol: 0.015538733031081704
- Profit Factor: 1.1150200366973877

**Regime-Specific Characteristics:**

- reversion_center: 0.0001932782615767792
- reversion_speed: 120.49013517607547
- reversion_range: 0.009248120710253716

### Regime 2 (stable)

**Size:** 488 samples (13.91%)

**Performance Metrics:**
- Mean Return: 0.0011575252283364534
- Volatility: 0.008870244026184082
- Sharpe Ratio: 0.13049529352566405
- Skewness: 0.39624130725860596
- Max Drawdown: -0.10291027926659486

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2069672131147541
- Pct < -1.0% (Shorts): 0.11885245901639344
- Pct Target Hits: 0.32581967213114754

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 36.731755568122445
- Win Rate (Long Bias): 0.6352201238365571
- Return per Vol: 0.13049529352566405
- Profit Factor: 1.2354670763015747

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0011575252283364534
- volatility: 0.008870244026184082
- stability_coefficient: 0.11543206505101555

### Regime 3 (stable)

**Size:** 1115 samples (31.78%)

**Performance Metrics:**
- Mean Return: 0.00014536599337588996
- Volatility: 0.006943891756236553
- Sharpe Ratio: 0.020934366137111155
- Skewness: -3.8804712295532227
- Max Drawdown: -0.13136009421932712

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.08340807174887892
- Pct < -1.0% (Shorts): 0.07623318385650224
- Pct Target Hits: 0.15964125560538117

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 22.990167217372672
- Win Rate (Long Bias): 0.522471906839572
- Return per Vol: 0.020934366137111155
- Profit Factor: 0.966275155544281

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.00014536599337588996
- volatility: 0.006943891756236553
- stability_coefficient: 0.020505245824725595

### Regime 4 (mean_reverting)

**Size:** 554 samples (15.79%)

**Performance Metrics:**
- Mean Return: -0.0002505603770259768
- Volatility: 0.008128904737532139
- Sharpe Ratio: -0.030823383259214004
- Skewness: -0.5757036209106445
- Max Drawdown: -0.23191464847719515

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.14981949458483754
- Pct < -1.0% (Shorts): 0.1624548736462094
- Pct Target Hits: 0.31227436823104693

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 38.41530192547793
- Win Rate (Long Bias): 0.4797687845907982
- Return per Vol: 0.030823383259214004
- Profit Factor: 0.8748026490211487

**Regime-Specific Characteristics:**

- reversion_center: -0.0002505603770259768
- reversion_speed: 165.6173470184678
- reversion_range: 0.005436501931399107


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5203877958368976

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Quality Assessment

**Overall Quality Score:** 0.7533 / 1.0
**Quality Level:** Excellent ✅
**Recommendation:** The clustering shows excellent quality. Proceed with confidence.

