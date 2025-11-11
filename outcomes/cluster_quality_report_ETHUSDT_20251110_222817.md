# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-10T22:27:17.748248
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

**Global Silhouette Score:** -0.0280

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.1428 | 0.0823 | -0.0994 | 0.3225 |
| 1 | 0.1750 | 0.1438 | -0.2927 | 0.4022 |
| 2 | 0.2459 | 0.1284 | -0.1967 | 0.4420 |
| 3 | 0.1144 | 0.0864 | -0.1419 | 0.3068 |
| 4 | -0.2498 | 0.1472 | -0.5819 | 0.0441 |
| 5 | -0.2676 | 0.1206 | -0.5568 | -0.0021 |


### Separation Metrics

- **Davies-Bouldin Index:** 3.2601 (lower is better)
- **Calinski-Harabasz Index:** 1836.06 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 3.4562 +/- 2.6851
- **Between-Regime CV:** 31.2403 +/- 31.2500

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 1.6738 |
| 1 | 1.5783 |
| 2 | 1.4436 |
| 3 | 2.2514 |
| 4 | 4.9709 |
| 5 | 8.8190 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 21.3737
- **Between-Regime CV (Mean Return):** 21.9283
- **CV Ratio (Between/Within):** 1.0259

| mean_return | 21.9283 |
| pct_above_target | 0.6253 |
| pct_below_neg_target | 0.6031 |
| pct_target_hits | 0.6010 |
| sharpe | 11.9456 |
| volatility | 0.4034 |


---

## Balance and Distribution

**Balance Score:** 0.7726 (0-1, higher is better)

- **Smallest Cluster:** 9.04% of total
- **Largest Cluster:** 23.84% of total
- **Cluster Size Std Dev:** 1662.71

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 17.74% |
| 1 | 12.79% |
| 2 | 9.04% |
| 3 | 15.73% |
| 4 | 20.86% |
| 5 | 23.84% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.7980 (0-1, higher = fewer transitions)
- **Regime Persistence:** 4.95 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (mean_reverting)

**Size:** 6014 samples (17.74%)

**Performance Metrics:**
- Mean Return: 0.00014904270938131958
- Volatility: 0.003920302726328373
- Sharpe Ratio: 0.03801815363956712
- Skewness: 0.6956257224082947
- Max Drawdown: -0.10454028663629926

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.037412703691386766
- Pct < -1.0% (Shorts): 0.025440638510143
- Pct Target Hits: 0.06285334220152977

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 16.032773629097388
- Win Rate (Long Bias): 0.5952380857678258
- Return per Vol: 0.03801815363956712
- Profit Factor: 1.1264740228652954

**Regime-Specific Characteristics:**

- reversion_center: 0.00014904270938131958
- reversion_speed: 392.924485658387
- reversion_range: 0.0029817079193890095

### Regime 1 (stable)

**Size:** 4335 samples (12.79%)

**Performance Metrics:**
- Mean Return: 0.0005402241367846727
- Volatility: 0.01042721513658762
- Sharpe Ratio: 0.05180904756439286
- Skewness: 0.3549477458000183
- Max Drawdown: -0.46229283000559124

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.18500576701268742
- Pct < -1.0% (Shorts): 0.1594002306805075
- Pct Target Hits: 0.3444059976931949

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 33.029525156261336
- Win Rate (Long Bias): 0.5371734746626611
- Return per Vol: 0.05180904756439286
- Profit Factor: 1.1089143753051758

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0005402241367846727
- volatility: 0.01042721513658762
- stability_coefficient: 0.04925717608841494

### Regime 2 (stable)

**Size:** 3064 samples (9.04%)

**Performance Metrics:**
- Mean Return: -0.00047788367373868823
- Volatility: 0.012078015133738518
- Sharpe Ratio: -0.03956640465181831
- Skewness: -0.8225947022438049
- Max Drawdown: -0.8361879979312783

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.20398172323759792
- Pct < -1.0% (Shorts): 0.20169712793733682
- Pct Target Hits: 0.40567885117493474

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 33.5882024566699
- Win Rate (Long Bias): 0.5028157670630511
- Return per Vol: 0.03956640465181831
- Profit Factor: 0.8957412242889404

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.00047788367373868823
- volatility: 0.012078015133738518
- stability_coefficient: 0.038060567626869844

### Regime 3 (mean_reverting)

**Size:** 5331 samples (15.73%)

**Performance Metrics:**
- Mean Return: -0.00012790822074748576
- Volatility: 0.004187982063740492
- Sharpe Ratio: -0.030541723498099506
- Skewness: -0.34677252173423767
- Max Drawdown: -0.5862390183296122

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.023635340461451885
- Pct < -1.0% (Shorts): 0.04164321890827237
- Pct Target Hits: 0.06527855936972425

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 15.58711159433887
- Win Rate (Long Bias): 0.3620689599707195
- Return per Vol: 0.030541723498099506
- Profit Factor: 0.8966936469078064

**Regime-Specific Characteristics:**

- reversion_center: -0.00012790822074748576
- reversion_speed: 374.1287664052144
- reversion_range: 0.0032239090651273727

### Regime 4 (stable)

**Size:** 7072 samples (20.86%)

**Performance Metrics:**
- Mean Return: 0.0005869581364095211
- Volatility: 0.00682439049705863
- Sharpe Ratio: 0.08600886052075192
- Skewness: 0.3359541594982147
- Max Drawdown: -0.1409777393679906

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.11835407239819004
- Pct < -1.0% (Shorts): 0.07480203619909502
- Pct Target Hits: 0.19315610859728505

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 28.30378484008969
- Win Rate (Long Bias): 0.6127379177648005
- Return per Vol: 0.08600886052075192
- Profit Factor: 1.2222471237182617

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0005869581364095211
- volatility: 0.00682439049705863
- stability_coefficient: 0.07919733455279626

### Regime 5 (stable)

**Size:** 8082 samples (23.84%)

**Performance Metrics:**
- Mean Return: -0.0005480095278471708
- Volatility: 0.007128407247364521
- Sharpe Ratio: -0.07687684386619946
- Skewness: -0.6784741878509521
- Max Drawdown: -0.9909399195259476

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.08537490720118783
- Pct < -1.0% (Shorts): 0.1267013115565454
- Pct Target Hits: 0.21207621875773325

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 29.750851999272207
- Win Rate (Long Bias): 0.4025670926175337
- Return per Vol: 0.07687684386619946
- Profit Factor: 0.8121045827865601

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.0005480095278471708
- volatility: 0.007128407247364521
- stability_coefficient: 0.07138883576878519


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5024633448387763

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Quality Assessment

**Overall Quality Score:** 0.7087 / 1.0
**Quality Level:** Excellent ✅
**Recommendation:** The clustering shows excellent quality. Proceed with confidence.

