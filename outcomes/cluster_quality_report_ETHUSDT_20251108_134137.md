# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-08T13:41:36.710428
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

**Global Silhouette Score:** 0.0834

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.0844 | 0.1038 | -0.2155 | 0.2912 |
| 1 | 0.2101 | 0.0779 | -0.0728 | 0.3851 |
| 2 | 0.0785 | 0.1399 | -0.2832 | 0.3105 |
| 3 | 0.1072 | 0.1044 | -0.1085 | 0.3077 |
| 4 | 0.0241 | 0.1124 | -0.2499 | 0.2322 |
| 5 | 0.0474 | 0.1167 | -0.2777 | 0.2690 |
| 6 | 0.0862 | 0.2536 | -0.3102 | 0.3896 |


### Separation Metrics

- **Davies-Bouldin Index:** 2.0068 (lower is better)
- **Calinski-Harabasz Index:** 100.12 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 9.2485 +/- 7.1355
- **Between-Regime CV:** 235.7520 +/- 501.1920

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 6.8751 |
| 1 | 3.4217 |
| 2 | 23.3589 |
| 3 | 16.6432 |
| 4 | 5.1333 |
| 5 | 6.0463 |
| 6 | 3.2606 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 72.8308
- **Between-Regime CV (Mean Return):** 2.9083
- **CV Ratio (Between/Within):** 0.0399

| mean_return | 2.9083 |
| pct_above_target | 0.1589 |
| pct_below_neg_target | 0.3002 |
| pct_target_hits | 0.1558 |
| sharpe | 2.2753 |
| volatility | 0.1372 |


---

## Balance and Distribution

**Balance Score:** 0.6605 (0-1, higher is better)

- **Smallest Cluster:** 1.00% of total
- **Largest Cluster:** 22.10% of total
- **Cluster Size Std Dev:** 73.42

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 22.10% |
| 1 | 10.40% |
| 2 | 8.20% |
| 3 | 20.00% |
| 4 | 17.20% |
| 5 | 21.10% |
| 6 | 1.00% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.7818 (0-1, higher = fewer transitions)
- **Regime Persistence:** 4.58 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (mean_reverting)

**Size:** 221 samples (22.10%)

**Performance Metrics:**
- Mean Return: 0.0025795781103426334
- Volatility: 0.011510570724109223
- Sharpe Ratio: 0.2241051245907823
- Skewness: -0.2624083014460792
- Max Drawdown: -0.0634545388039792

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2895927601809955
- Pct < -1.0% (Shorts): 0.15384615384615385
- Pct Target Hits: 0.44343891402714936

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 38.52449075994669
- Win Rate (Long Bias): 0.6530612230170761
- Return per Vol: 0.2241051245907823
- Profit Factor: 1.1883083862357084

**Regime-Specific Characteristics:**

- reversion_center: 0.0025795781103426334
- reversion_speed: 120.3913753217234
- reversion_range: 0.007948960766803315

### Regime 1 (mean_reverting)

**Size:** 104 samples (10.40%)

**Performance Metrics:**
- Mean Return: 0.0008272987283905932
- Volatility: 0.01570449260634271
- Sharpe Ratio: 0.05267910886706115
- Skewness: 0.03977438495656974
- Max Drawdown: -0.07447642931792639

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.34615384615384615
- Pct < -1.0% (Shorts): 0.28846153846153844
- Pct Target Hits: 0.6346153846153846

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 40.409796108234715
- Win Rate (Long Bias): 0.5454545445950414
- Return per Vol: 0.05267910886706115
- Profit Factor: 1.2143515880397049

**Regime-Specific Characteristics:**

- reversion_center: 0.0008272987283905932
- reversion_speed: 81.87164022440432
- reversion_range: 0.009796978539193734

### Regime 2 (stable)

**Size:** 82 samples (8.20%)

**Performance Metrics:**
- Mean Return: -0.0026092172557134813
- Volatility: 0.01656181835834714
- Sharpe Ratio: -0.15754411995795853
- Skewness: -0.016716924392315254
- Max Drawdown: -0.2674657998275884

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2926829268292683
- Pct < -1.0% (Shorts): 0.4268292682926829
- Pct Target Hits: 0.7195121951219512

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 43.44403109066142
- Win Rate (Long Bias): 0.40677966045159436
- Return per Vol: 0.15754411995795853
- Profit Factor: 0.7831088143256949

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.0026092172557134813
- volatility: 0.01656181835834714
- stability_coefficient: 0.136102095480836

### Regime 3 (stable)

**Size:** 200 samples (20.00%)

**Performance Metrics:**
- Mean Return: 0.0023620311540573967
- Volatility: 0.010932592069311867
- Sharpe Ratio: 0.21605406321101375
- Skewness: 0.5617332794129754
- Max Drawdown: -0.0640636938714647

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.29
- Pct < -1.0% (Shorts): 0.2
- Pct Target Hits: 0.49

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 44.82010780913866
- Win Rate (Long Bias): 0.5918367334860474
- Return per Vol: 0.21605406321101375
- Profit Factor: 1.2066427486883977

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0023620311540573967
- volatility: 0.010932592069311867
- stability_coefficient: 0.17766821493949572

### Regime 4 (stable)

**Size:** 172 samples (17.20%)

**Performance Metrics:**
- Mean Return: 0.0020100191375743482
- Volatility: 0.014566407089298855
- Sharpe Ratio: 0.13799003331857768
- Skewness: -0.22347867629902674
- Max Drawdown: -0.09669219242752558

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.3372093023255814
- Pct < -1.0% (Shorts): 0.23255813953488372
- Pct Target Hits: 0.5697674418604651

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 39.11516403821215
- Win Rate (Long Bias): 0.5918367336551437
- Return per Vol: 0.13799003331857768
- Profit Factor: 1.1340119355980882

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0020100191375743482
- volatility: 0.014566407089298855
- stability_coefficient: 0.12125774209751088

### Regime 5 (stable)

**Size:** 211 samples (21.10%)

**Performance Metrics:**
- Mean Return: -0.0009306479316110172
- Volatility: 0.014104581804799204
- Sharpe Ratio: -0.0659819538437079
- Skewness: -0.11079191954219794
- Max Drawdown: -0.2695309509529548

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2559241706161137
- Pct < -1.0% (Shorts): 0.3033175355450237
- Pct Target Hits: 0.5592417061611374

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 39.64964536000673
- Win Rate (Long Bias): 0.4576271178257684
- Return per Vol: 0.0659819538437079
- Profit Factor: 0.9135076124633748

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.0009306479316110172
- volatility: 0.014104581804799204
- stability_coefficient: 0.06189788157738755

### Regime 6 (mean_reverting)

**Size:** 10 samples (1.00%)

**Performance Metrics:**
- Mean Return: 3.074727865497451e-05
- Volatility: 0.014688486373487757
- Sharpe Ratio: 0.002093291015824567
- Skewness: 0.7895322155231448
- Max Drawdown: -0.04698455006677323

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2
- Pct < -1.0% (Shorts): 0.3
- Pct Target Hits: 0.5

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 34.040264820085135
- Win Rate (Long Bias): 0.39999999920000007
- Return per Vol: 0.002093291015824567
- Profit Factor: 1.508742338547293

**Regime-Specific Characteristics:**

- reversion_center: 3.074727865497451e-05
- reversion_speed: 94.44600461054132
- reversion_range: 0.009549256352409148


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5425425425425425

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Quality Assessment

**Overall Quality Score:** 0.7382 / 1.0
**Quality Level:** Excellent ✅
**Recommendation:** The clustering shows excellent quality. Proceed with confidence.

