# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-09T14:40:39.328830
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

**Global Silhouette Score:** 0.2813

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.3849 | 0.0935 | 0.0957 | 0.4944 |
| 1 | 0.0886 | 0.1424 | -0.2187 | 0.3061 |
| 2 | 0.1310 | 0.3257 | -0.4327 | 0.3337 |
| 3 | 0.6592 | 0.0504 | 0.5977 | 0.7258 |
| 4 | 0.1624 | 0.2348 | -0.4051 | 0.3965 |


### Separation Metrics

- **Davies-Bouldin Index:** 1.3263 (lower is better)
- **Calinski-Harabasz Index:** 17.70 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 1.2607 +/- 0.7965
- **Between-Regime CV:** 26.8561 +/- 67.3397

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 1.9192 |
| 1 | 2.2061 |
| 2 | 0.2630 |
| 3 | 0.3807 |
| 4 | 1.5342 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 5.3370
- **Between-Regime CV (Mean Return):** 5.0910
- **CV Ratio (Between/Within):** 0.9539

| mean_return | 5.0910 |
| pct_above_target | 0.5139 |
| pct_below_neg_target | 0.9282 |
| pct_target_hits | 0.2956 |
| sharpe | 1.6501 |
| volatility | 0.6476 |


---

## Balance and Distribution

**Balance Score:** 0.6277 (0-1, higher is better)

- **Smallest Cluster:** 6.35% of total
- **Largest Cluster:** 36.51% of total
- **Cluster Size Std Dev:** 7.47

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 36.51% |
| 1 | 31.75% |
| 2 | 6.35% |
| 3 | 12.70% |
| 4 | 12.70% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.8710 (0-1, higher = fewer transitions)
- **Regime Persistence:** 7.75 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (mean_reverting)

**Size:** 23 samples (36.51%)

**Performance Metrics:**
- Mean Return: 0.0014239602023735642
- Volatility: 0.011170221492648125
- Sharpe Ratio: 0.12747823092250565
- Skewness: 2.402024030685425
- Max Drawdown: -0.02819408176776575

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.13043478260869565
- Pct < -1.0% (Shorts): 0.08695652173913043
- Pct Target Hits: 0.21739130434782608

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 19.461680775911667
- Win Rate (Long Bias): 0.59999999724
- Return per Vol: 0.12747823092250565
- Profit Factor: 1.7104867696762085

**Regime-Specific Characteristics:**

- reversion_center: 0.0014239602023735642
- reversion_speed: 148.98162916088714
- reversion_range: 0.008813162334263325

### Regime 1 (stable)

**Size:** 20 samples (31.75%)

**Performance Metrics:**
- Mean Return: 0.0007991539896465838
- Volatility: 0.008615434169769287
- Sharpe Ratio: 0.09275840092799217
- Skewness: -0.3575800657272339
- Max Drawdown: -0.02965059222223015

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.15
- Pct < -1.0% (Shorts): 0.2
- Pct Target Hits: 0.35

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 40.62476161716305
- Win Rate (Long Bias): 0.42857142734693876
- Return per Vol: 0.09275840092799217
- Profit Factor: 0.7370503544807434

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0007991539896465838
- volatility: 0.008615434169769287
- stability_coefficient: 0.08488474389212407

### Regime 2 (trending)

**Size:** 4 samples (6.35%)

**Performance Metrics:**
- Mean Return: 0.008379757404327393
- Volatility: 0.010090074501931667
- Sharpe Ratio: 0.8304950149008452
- Skewness: -0.6686567068099976
- Max Drawdown: -0.004445422545316951

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.5
- Pct < -1.0% (Shorts): 0.0
- Pct Target Hits: 0.5

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 49.55364307276778
- Win Rate (Long Bias): 0.9999999980000001
- Return per Vol: 0.8304950149008452
- Profit Factor: 2.846701145172119

**Regime-Specific Characteristics:**

- trend_direction: bullish
- trend_consistency: 0.75

### Regime 3 (mean_reverting)

**Size:** 8 samples (12.70%)

**Performance Metrics:**
- Mean Return: 0.0014710351824760437
- Volatility: 0.00574547378346324
- Sharpe Ratio: 0.25603370268198056
- Skewness: 0.590836226940155
- Max Drawdown: -0.008126513912722759

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.25
- Pct < -1.0% (Shorts): 0.0
- Pct Target Hits: 0.25

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 43.5125049577375
- Win Rate (Long Bias): 0.9999999959999999
- Return per Vol: 0.25603370268198056
- Profit Factor: 1.9513344764709473

**Regime-Specific Characteristics:**

- reversion_center: 0.0014710351824760437
- reversion_speed: 203.45610061196953
- reversion_range: 0.002324109897017479

### Regime 4 (mean_reverting)

**Size:** 8 samples (12.70%)

**Performance Metrics:**
- Mean Return: -0.0072149112820625305
- Volatility: 0.029492143541574478
- Sharpe Ratio: -0.24463840775945664
- Skewness: -2.7155849933624268
- Max Drawdown: -0.08063603611223832

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.25
- Pct < -1.0% (Shorts): 0.125
- Pct Target Hits: 0.375

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 12.715250309158431
- Win Rate (Long Bias): 0.6666666648888888
- Return per Vol: 0.24463840775945664
- Profit Factor: 0.2886444628238678

**Regime-Specific Characteristics:**

- reversion_center: -0.0072149112820625305
- reversion_speed: 55.35331717587443
- reversion_range: 0.022288791835308075


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5317460317460317

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Quality Assessment

**Overall Quality Score:** 0.7600 / 1.0
**Quality Level:** Excellent ✅
**Recommendation:** The clustering shows excellent quality. Proceed with confidence.

