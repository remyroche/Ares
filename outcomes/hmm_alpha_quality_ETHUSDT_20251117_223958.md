# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-17T22:39:37.375493
**Data Points:** N/A
**Number of Regimes:** 5
**Report Version:** 1.3 (Enhanced with Financial Analysis)

This report provides a comprehensive assessment of cluster quality for ETHUSDT.

### Key Metrics


---

## Clustering Method Configuration

| Parameter | Value |
|---|---|
| alpha_config | {'alpha_horizon_bars': 1, 'alpha_regime_bins': 5, 'alpha_target_type': 'regression'} |


## PCA Feature Analysis



---

## Top Configuration Analysis

### Clustering Configuration Parameters


---

## Clustering Metrics

### Silhouette Analysis

**Global Silhouette Score:** -0.0220

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | -0.0443 | 0.1208 | -0.4237 | 0.1620 |
| 1 | -0.0361 | 0.0471 | -0.1451 | 0.0603 |
| 2 | 0.0411 | 0.0479 | -0.1352 | 0.1224 |
| 3 | -0.0323 | 0.0385 | -0.1439 | 0.0602 |
| 4 | -0.0385 | 0.1023 | -0.3757 | 0.1447 |


### Separation Metrics

- **Davies-Bouldin Index:** 7.1056 (lower is better)
- **Calinski-Harabasz Index:** 348.58 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 57.4355 +/- 57.2625
- **Between-Regime CV:** 35463.0773 +/- 54074.7904

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 5.3343 |
| 1 | 9.8894 |
| 2 | 18.2154 |
| 3 | 137.8119 |
| 4 | 115.9267 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 12.2799
- **Between-Regime CV (Mean Return):** 48.0413
- **CV Ratio (Between/Within):** 3.9122

| mean_return | 48.0413 |
| pct_above_target | 0.8029 |
| pct_below_neg_target | 0.9423 |
| pct_target_hits | 0.5425 |
| sharpe | 9.5252 |
| volatility | 0.3251 |


---

## Economic Gap Analysis

### Per-Regime Snapshot

| Regime | Type | Mean Return | Volatility | Sharpe | Max DD | Pct Target Hits |
|--------|------|-------------|------------|--------|--------|-----------------|
| 0 | mean_reverting | -0.004031 | 0.010010 | -0.4027 | -99.92% | 35.31% |
| 1 | mean_reverting | -0.000708 | 0.005883 | -0.1204 | -71.90% | 12.91% |
| 2 | mean_reverting | 0.000099 | 0.004300 | 0.0229 | -8.98% | 8.76% |
| 3 | mean_reverting | 0.001053 | 0.005015 | 0.2101 | -8.67% | 12.33% |
| 4 | mean_reverting | 0.003853 | 0.008746 | 0.4406 | -2.91% | 31.16% |

### Pairwise Economic Spreads

| Regime A | Regime B | Mean Return Spread | Sharpe Spread | Volatility Ratio | Max DD Spread |
|----------|----------|--------------------|---------------|------------------|---------------|
| 0 | 1 | -0.003322 | -0.2823 | 1.701 | -28.01% |
| 0 | 2 | -0.004129 | -0.4256 | 2.328 | -90.94% |
| 0 | 3 | -0.005084 | -0.6128 | 1.996 | -91.25% |
| 0 | 4 | -0.007884 | -0.8433 | 1.144 | -97.01% |
| 1 | 2 | -0.000807 | -0.1433 | 1.368 | -62.93% |
| 1 | 3 | -0.001762 | -0.3305 | 1.173 | -63.24% |
| 1 | 4 | -0.004562 | -0.5610 | 0.673 | -68.99% |
| 2 | 3 | -0.000955 | -0.1871 | 0.858 | -0.31% |
| 2 | 4 | -0.003755 | -0.4177 | 0.492 | -6.07% |
| 3 | 4 | -0.002800 | -0.2306 | 0.573 | -5.76% |

### Statistical Tests (ANOVA / t-tests)

- **ANOVA F-statistic:** 278.3182, p-value=0.0000 (significant)

**Pairwise t-tests:**

| Regime A | Regime B | t-stat | p-value | Cohen's d | Significant |
|----------|----------|--------|---------|-----------|-------------|
| 0 | 1 | -11.9217 | 0.0000 | -0.405 | Yes |
| 0 | 2 | -15.7926 | 0.0000 | -0.536 | Yes |
| 0 | 3 | -18.9204 | 0.0000 | -0.642 | Yes |
| 0 | 4 | -24.7141 | 0.0000 | -0.839 | Yes |
| 1 | 2 | -4.6130 | 0.0000 | -0.157 | Yes |
| 1 | 3 | -9.4927 | 0.0000 | -0.322 | Yes |
| 1 | 4 | -18.0309 | 0.0000 | -0.612 | Yes |
| 2 | 3 | -6.0198 | 0.0000 | -0.204 | Yes |
| 2 | 4 | -16.0517 | 0.0000 | -0.545 | Yes |
| 3 | 4 | -11.5715 | 0.0000 | -0.393 | Yes |


### Per-Category Coefficient of Variation


| Category | Within CV | Between CV | Ratio | # Features |
|----------|-----------|------------|-------|------------|
| momentum | 8.681 ± 13.940 | 95204.554 ± 0.000 | 10966.856 | 1 |
| other | 6.212 ± 3.938 | 48680.281 ± 61916.090 | 7836.125 | 12 |
| price | 263.874 ± 291.465 | 4751.738 ± 117.902 | 18.008 | 4 |
| volatility | 2.682 ± 0.498 | 3631.599 ± 0.000 | 1353.885 | 1 |
| volume | 3.652 ± 0.980 | 3627.533 ± 60.303 | 993.370 | 2 |

**Interpretation:** Higher CV ratio indicates better regime separation for that feature category.


---

## Balance and Distribution

**Balance Score:** 0.9997 (0-1, higher is better)

- **Smallest Cluster:** 20.00% of total
- **Largest Cluster:** 20.01% of total
- **Cluster Size Std Dev:** 0.49

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 20.01% |
| 1 | 20.00% |
| 2 | 20.00% |
| 3 | 20.00% |
| 4 | 20.01% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.0183 (0-1, higher = fewer transitions)
- **Temporal Smoothness (Raw):** 0.0471
- **Flip-Flop Ratio:** 0.0683 (rapid back-and-forth transitions)
- **Regime Persistence:** 2.36 bars (average duration)


### Transition Probability Matrix

This matrix shows the probability of transitioning from one regime to another:


### Regime Duration Analysis

- **Duration Stability Score:** 0.511 (higher = more consistent durations)


---

## Per-Regime Analysis

### Regime 0 (mean_reverting)

**Size:** 1736 samples (20.01%)

**Performance Metrics:**
- Mean Return: -0.004030842679872107
- Volatility: 0.010009549622249815
- Sharpe Ratio: -0.40269966474939584
- Skewness: -2.309145996077598
- Max Drawdown: -0.9991732793658754

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.06566820276497695
- Pct < -1.0% (Shorts): 0.28744239631336405
- Pct Target Hits: 0.35311059907834097

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 35.2773678264263
- Win Rate (Long Bias): 0.18597063568867042
- Return per Vol: 0.40269966474939584
- Profit Factor: 0.564424112829445

**Regime-Specific Characteristics:**

- reversion_center: -0.004030842679872107
- reversion_speed: 147.47445290753473
- reversion_range: 0.007361037044012965

### Regime 1 (mean_reverting)

**Size:** 1735 samples (20.00%)

**Performance Metrics:**
- Mean Return: -0.0007084427396005283
- Volatility: 0.005883472157436263
- Sharpe Ratio: -0.1204123347966856
- Skewness: -2.9368974086008075
- Max Drawdown: -0.7190459707660541

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.04956772334293948
- Pct < -1.0% (Shorts): 0.07953890489913544
- Pct Target Hits: 0.12910662824207492

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 21.943947866727996
- Win Rate (Long Bias): 0.383928568454839
- Return per Vol: 0.1204123347966856
- Profit Factor: 0.8801394158467715

**Regime-Specific Characteristics:**

- reversion_center: -0.0007084427396005283
- reversion_speed: 284.19563853226083
- reversion_range: 0.004714535153015548

### Regime 2 (mean_reverting)

**Size:** 1735 samples (20.00%)

**Performance Metrics:**
- Mean Return: 9.86344675866021e-05
- Volatility: 0.004300349392027903
- Sharpe Ratio: 0.02293637927026903
- Skewness: 0.21347414416465518
- Max Drawdown: -0.08977522255425359

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.04438040345821326
- Pct < -1.0% (Shorts): 0.043227665706051875
- Pct Target Hits: 0.08760806916426514

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 20.37230950451739
- Win Rate (Long Bias): 0.506578941586089
- Return per Vol: 0.02293637927026903
- Profit Factor: 0.9730247711256117

**Regime-Specific Characteristics:**

- reversion_center: 9.86344675866021e-05
- reversion_speed: 335.58955740672246
- reversion_range: 0.003099758098365694

### Regime 3 (mean_reverting)

**Size:** 1735 samples (20.00%)

**Performance Metrics:**
- Mean Return: 0.0010533651471669758
- Volatility: 0.005014797876483059
- Sharpe Ratio: 0.21005132471149346
- Skewness: -1.5425616851198733
- Max Drawdown: -0.08667712988207744

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.08357348703170028
- Pct < -1.0% (Shorts): 0.03976945244956772
- Pct Target Hits: 0.12334293948126801

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 24.595789884951465
- Win Rate (Long Bias): 0.6775700879645602
- Return per Vol: 0.21005132471149346
- Profit Factor: 1.1773488967637846

**Regime-Specific Characteristics:**

- reversion_center: 0.0010533651471669758
- reversion_speed: 298.963480461203
- reversion_range: 0.003735433313744533

### Regime 4 (mean_reverting)

**Size:** 1736 samples (20.01%)

**Performance Metrics:**
- Mean Return: 0.00385346220004139
- Volatility: 0.008745798871691511
- Sharpe Ratio: 0.4406071779111152
- Skewness: 2.383645615597611
- Max Drawdown: -0.02911967527747179

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.2597926267281106
- Pct < -1.0% (Shorts): 0.05184331797235023
- Pct Target Hits: 0.31163594470046085

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 35.6326407272555
- Win Rate (Long Bias): 0.8336414021308659
- Return per Vol: 0.4406071779111152
- Profit Factor: 1.8603319423287656

**Regime-Specific Characteristics:**

- reversion_center: 0.00385346220004139
- reversion_speed: 168.42075448162305
- reversion_range: 0.006419864437183335


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5318118948824343

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Economic Relevance Analysis

### Strategy Performance Summary

### Statistical Significance Tests

### Economic Regime Mapping

| Regime | Economic Interpretation | Recommended Position |
|---------|----------------------|----------------------|

### Economic Interpretation

**Detailed Economic Report:** /Users/remyroche/Documents/Ares/outcomes/regime_economic_relevance_report_20251117_223956.md


---

## Quality Assessment

**Overall Quality Score:** 0.7522 / 1.0
**Quality Level:** Excellent ✅
**Recommendation:** The clustering shows excellent quality. Proceed with confidence.

