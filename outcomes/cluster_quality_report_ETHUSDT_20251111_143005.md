# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-11T14:29:51.520869
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

**Global Silhouette Score:** 0.0887

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.1134 | 0.1711 | -0.3645 | 0.3749 |
| 1 | 0.2087 | 0.1280 | -0.2684 | 0.4154 |
| 2 | -0.2235 | 0.1455 | -0.6602 | 0.0937 |
| 3 | 0.4262 | 0.0621 | 0.2538 | 0.5425 |
| 4 | 0.3383 | 0.0995 | -0.0250 | 0.5080 |
| 5 | 0.1471 | 0.1481 | -0.4699 | 0.3498 |


### Separation Metrics

- **Davies-Bouldin Index:** 2.0896 (lower is better)
- **Calinski-Harabasz Index:** 487.12 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 5.3253 +/- 4.6330
- **Between-Regime CV:** 19.4616 +/- 18.6709

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 1.2534 |
| 1 | 1.6616 |
| 2 | 14.4443 |
| 3 | 7.9446 |
| 4 | 3.9389 |
| 5 | 2.7088 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 17.4315
- **Between-Regime CV (Mean Return):** 1.6643
- **CV Ratio (Between/Within):** 0.0955

| mean_return | 1.6643 |
| pct_above_target | 0.3529 |
| pct_below_neg_target | 0.4180 |
| pct_target_hits | 0.3446 |
| sharpe | 1.5529 |
| volatility | 0.2212 |


---

## Balance and Distribution

**Balance Score:** 0.6721 (0-1, higher is better)

- **Smallest Cluster:** 5.25% of total
- **Largest Cluster:** 31.21% of total
- **Cluster Size Std Dev:** 285.24

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 15.65% |
| 1 | 11.29% |
| 2 | 31.21% |
| 3 | 5.25% |
| 4 | 21.41% |
| 5 | 15.19% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.7588 (0-1, higher = fewer transitions)
- **Regime Persistence:** 4.15 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 549 samples (15.65%)

**Performance Metrics:**
- Mean Return: -0.00019092224829364568
- Volatility: 0.005848823115229607
- Sharpe Ratio: -0.0326428431650916
- Skewness: 0.084588922560215
- Max Drawdown: -0.1933677816507379

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.06375227686703097
- Pct < -1.0% (Shorts): 0.1092896174863388
- Pct Target Hits: 0.17304189435336975

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 29.585757913764287
- Win Rate (Long Bias): 0.36842105050249313
- Return per Vol: 0.0326428431650916
- Profit Factor: 0.9240183234214783

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.00019092224829364568
- volatility: 0.005848823115229607
- stability_coefficient: 0.03161113676009925

### Regime 1 (mean_reverting)

**Size:** 396 samples (11.29%)

**Performance Metrics:**
- Mean Return: 0.0008768485859036446
- Volatility: 0.00890154018998146
- Sharpe Ratio: 0.09850525512262116
- Skewness: -1.64199960231781
- Max Drawdown: -0.10430133073670628

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.1691919191919192
- Pct < -1.0% (Shorts): 0.12121212121212122
- Pct Target Hits: 0.29040404040404044

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 32.62401804430063
- Win Rate (Long Bias): 0.5826086936459735
- Return per Vol: 0.09850525512262116
- Profit Factor: 1.0743521451950073

**Regime-Specific Characteristics:**

- reversion_center: 0.0008768485859036446
- reversion_speed: 168.29694211167626
- reversion_range: 0.006621339358389378

### Regime 2 (stable)

**Size:** 1095 samples (31.21%)

**Performance Metrics:**
- Mean Return: 0.0003030422085430473
- Volatility: 0.008462845347821712
- Sharpe Ratio: 0.03580854432280324
- Skewness: -1.2994571924209595
- Max Drawdown: -0.2306742097545971

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.13515981735159818
- Pct < -1.0% (Shorts): 0.10136986301369863
- Pct Target Hits: 0.2365296803652968

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 27.949187618912422
- Win Rate (Long Bias): 0.5714285690126862
- Return per Vol: 0.03580854432280324
- Profit Factor: 1.0489754676818848

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0003030422085430473
- volatility: 0.008462845347821712
- stability_coefficient: 0.03457073479710313

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

**Size:** 751 samples (21.41%)

**Performance Metrics:**
- Mean Return: 0.00038440932985395193
- Volatility: 0.004741550888866186
- Sharpe Ratio: 0.08107247138992567
- Skewness: 1.0096105337142944
- Max Drawdown: -0.04596367382494726

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.06391478029294274
- Pct < -1.0% (Shorts): 0.03861517976031957
- Pct Target Hits: 0.10252996005326231

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 21.62371359765496
- Win Rate (Long Bias): 0.6233766172966775
- Return per Vol: 0.08107247138992567
- Profit Factor: 1.1041570901870728

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.00038440932985395193
- volatility: 0.004741550888866186
- stability_coefficient: 0.07499282835969882

### Regime 5 (mean_reverting)

**Size:** 533 samples (15.19%)

**Performance Metrics:**
- Mean Return: -0.0004983241087757051
- Volatility: 0.009213320910930634
- Sharpe Ratio: -0.0540873436957083
- Skewness: -0.8311605453491211
- Max Drawdown: -0.30381737536245484

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.16322701688555347
- Pct < -1.0% (Shorts): 0.1726078799249531
- Pct Target Hits: 0.3358348968105066

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 36.45101083595853
- Win Rate (Long Bias): 0.48603351810583306
- Return per Vol: 0.0540873436957083
- Profit Factor: 0.8659695386886597

**Regime-Specific Characteristics:**

- reversion_center: -0.0004983241087757051
- reversion_speed: 151.31808930358312
- reversion_range: 0.006413241382688284


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5149700598802396

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Economic Relevance Analysis

### Strategy Performance Summary

### Statistical Significance Tests

### Economic Regime Mapping

| Regime | Economic Interpretation | Recommended Position |
|---------|----------------------|----------------------|

### Economic Interpretation

**Detailed Economic Report:** /Users/remyroche/Documents/Ares/outcomes/regime_economic_relevance_report_20251111_142957.md


---

## Quality Assessment

**Overall Quality Score:** 0.5920 / 1.0
**Quality Level:** Good ✅
**Recommendation:** The clustering shows good quality. Suitable for most applications.

