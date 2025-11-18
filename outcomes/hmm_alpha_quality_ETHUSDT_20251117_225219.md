# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-17T22:52:06.548463
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

**Global Silhouette Score:** -0.0256

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | -0.0504 | 0.1145 | -0.4237 | 0.1433 |
| 1 | -0.0354 | 0.0556 | -0.1543 | 0.0666 |
| 2 | 0.0342 | 0.0518 | -0.1420 | 0.1225 |
| 3 | -0.0323 | 0.0415 | -0.1382 | 0.0629 |
| 4 | -0.0438 | 0.1004 | -0.3830 | 0.1310 |


### Separation Metrics

- **Davies-Bouldin Index:** 7.0653 (lower is better)
- **Calinski-Harabasz Index:** 308.14 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 10.6874 +/- 4.3826
- **Between-Regime CV:** 31740.4695 +/- 43420.0058

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 5.5079 |
| 1 | 8.5918 |
| 2 | 18.4768 |
| 3 | 11.8463 |
| 4 | 9.0141 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 47.8045
- **Between-Regime CV (Mean Return):** 43.8836
- **CV Ratio (Between/Within):** 0.9180

| mean_return | 43.8836 |
| pct_above_target | 0.7417 |
| pct_below_neg_target | 0.8652 |
| pct_target_hits | 0.5204 |
| sharpe | 12.5505 |
| volatility | 0.3091 |


---

## Economic Gap Analysis

### Per-Regime Snapshot

| Regime | Type | Mean Return | Volatility | Sharpe | Max DD | Pct Target Hits |
|--------|------|-------------|------------|--------|--------|-----------------|
| 0 | mean_reverting | -0.003602 | 0.010166 | -0.3543 | -99.82% | 35.08% |
| 1 | mean_reverting | -0.000679 | 0.005100 | -0.1332 | -69.75% | 12.45% |
| 2 | mean_reverting | -0.000023 | 0.005183 | -0.0045 | -17.84% | 9.34% |
| 3 | mean_reverting | 0.000980 | 0.005305 | 0.1848 | -8.25% | 13.37% |
| 4 | mean_reverting | 0.003590 | 0.008718 | 0.4118 | -2.73% | 30.24% |

### Pairwise Economic Spreads

| Regime A | Regime B | Mean Return Spread | Sharpe Spread | Volatility Ratio | Max DD Spread |
|----------|----------|--------------------|---------------|------------------|---------------|
| 0 | 1 | -0.002923 | -0.2211 | 1.993 | -30.07% |
| 0 | 2 | -0.003578 | -0.3497 | 1.961 | -81.99% |
| 0 | 3 | -0.004582 | -0.5391 | 1.916 | -91.57% |
| 0 | 4 | -0.007192 | -0.7661 | 1.166 | -97.09% |
| 1 | 2 | -0.000656 | -0.1286 | 0.984 | -51.92% |
| 1 | 3 | -0.001660 | -0.3180 | 0.961 | -61.50% |
| 1 | 4 | -0.004269 | -0.5450 | 0.585 | -67.02% |
| 2 | 3 | -0.001004 | -0.1893 | 0.977 | -9.58% |
| 2 | 4 | -0.003613 | -0.4163 | 0.595 | -15.10% |
| 3 | 4 | -0.002610 | -0.2270 | 0.609 | -5.52% |

### Statistical Tests (ANOVA / t-tests)

- **ANOVA F-statistic:** 227.2141, p-value=0.0000 (significant)

**Pairwise t-tests:**

| Regime A | Regime B | t-stat | p-value | Cohen's d | Significant |
|----------|----------|--------|---------|-----------|-------------|
| 0 | 1 | -10.7055 | 0.0000 | -0.363 | Yes |
| 0 | 2 | -13.0641 | 0.0000 | -0.443 | Yes |
| 0 | 3 | -16.6474 | 0.0000 | -0.565 | Yes |
| 0 | 4 | -22.3739 | 0.0000 | -0.759 | Yes |
| 1 | 2 | -3.7559 | 0.0002 | -0.128 | Yes |
| 1 | 3 | -9.3933 | 0.0000 | -0.319 | Yes |
| 1 | 4 | -17.6097 | 0.0000 | -0.598 | Yes |
| 2 | 3 | -5.6377 | 0.0000 | -0.191 | Yes |
| 2 | 4 | -14.8430 | 0.0000 | -0.504 | Yes |
| 3 | 4 | -10.6532 | 0.0000 | -0.362 | Yes |


### Per-Category Coefficient of Variation


| Category | Within CV | Between CV | Ratio | # Features |
|----------|-----------|------------|-------|------------|
| momentum | 9.269 ± 14.124 | 121235.750 ± 0.000 | 13079.357 | 1 |
| other | 7.656 ± 5.528 | 40293.756 ± 44293.462 | 5262.697 | 12 |
| price | 25.084 ± 10.499 | 4752.030 ± 78.925 | 189.443 | 4 |
| volatility | 2.451 ± 0.522 | 3674.684 ± 0.000 | 1499.519 | 1 |
| volume | 4.907 ± 1.791 | 3682.881 ± 111.949 | 750.572 | 2 |

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

- **Temporal Smoothness (Penalized):** 0.0210 (0-1, higher = fewer transitions)
- **Temporal Smoothness (Raw):** 0.0564
- **Flip-Flop Ratio:** 0.0497 (rapid back-and-forth transitions)
- **Regime Persistence:** 2.82 bars (average duration)


### Transition Probability Matrix

This matrix shows the probability of transitioning from one regime to another:


### Regime Duration Analysis

- **Duration Stability Score:** 0.492 (higher = more consistent durations)


---

## Per-Regime Analysis

### Regime 0 (mean_reverting)

**Size:** 1736 samples (20.01%)

**Performance Metrics:**
- Mean Return: -0.003601675940412185
- Volatility: 0.01016635553769492
- Sharpe Ratio: -0.35427401420143534
- Skewness: -2.2428580905454867
- Max Drawdown: -0.9982469841835573

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.07891705069124424
- Pct < -1.0% (Shorts): 0.271889400921659
- Pct Target Hits: 0.35080645161290325

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 34.506605224022934
- Win Rate (Long Bias): 0.2249589484556178
- Return per Vol: 0.35427401420143534
- Profit Factor: 0.5872629602825087

**Regime-Specific Characteristics:**

- reversion_center: -0.003601675940412185
- reversion_speed: 145.51771011466417
- reversion_range: 0.007490192188244504

### Regime 1 (mean_reverting)

**Size:** 1735 samples (20.00%)

**Performance Metrics:**
- Mean Return: -0.0006791287148876573
- Volatility: 0.005099853849437092
- Sharpe Ratio: -0.13316628314678752
- Skewness: 2.120436036801912
- Max Drawdown: -0.6975245399820666

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.04207492795389049
- Pct < -1.0% (Shorts): 0.08242074927953891
- Pct Target Hits: 0.1244956772334294

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 24.4116118809089
- Win Rate (Long Bias): 0.3379629602483068
- Return per Vol: 0.13316628314678752
- Profit Factor: 0.8607325786211163

**Regime-Specific Characteristics:**

- reversion_center: -0.0006791287148876573
- reversion_speed: 291.04692617469783
- reversion_range: 0.0037678231523611636

### Regime 2 (mean_reverting)

**Size:** 1735 samples (20.00%)

**Performance Metrics:**
- Mean Return: -2.346131099108765e-05
- Volatility: 0.005183065469853123
- Sharpe Ratio: -0.004526530988469552
- Skewness: -6.185410189957669
- Max Drawdown: -0.17835866159211117

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.04726224783861672
- Pct < -1.0% (Shorts): 0.04610951008645533
- Pct Target Hits: 0.09337175792507205

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 18.0147714616745
- Win Rate (Long Bias): 0.5061728340851244
- Return per Vol: 0.004526530988469552
- Profit Factor: 0.9628572050483336

**Regime-Specific Characteristics:**

- reversion_center: -2.346131099108765e-05
- reversion_speed: 329.2511967460746
- reversion_range: 0.004199321623390039

### Regime 3 (mean_reverting)

**Size:** 1735 samples (20.00%)

**Performance Metrics:**
- Mean Return: 0.0009804012807676777
- Volatility: 0.005305315839674821
- Sharpe Ratio: 0.18479599058739146
- Skewness: -1.7403321346993397
- Max Drawdown: -0.0825377479841404

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.0893371757925072
- Pct < -1.0% (Shorts): 0.04438040345821326
- Pct Target Hits: 0.13371757925072048

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 25.204447404674973
- Win Rate (Long Bias): 0.6681034432794849
- Return per Vol: 0.18479599058739146
- Profit Factor: 1.1075048870374766

**Regime-Specific Characteristics:**

- reversion_center: 0.0009804012807676777
- reversion_speed: 289.2773472348814
- reversion_range: 0.004023605625437428

### Regime 4 (mean_reverting)

**Size:** 1736 samples (20.01%)

**Performance Metrics:**
- Mean Return: 0.003589945605257875
- Volatility: 0.008717948798390062
- Sharpe Ratio: 0.41178782721607643
- Skewness: 2.484626910972045
- Max Drawdown: -0.0273456952843523

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.24539170506912442
- Pct < -1.0% (Shorts): 0.057027649769585256
- Pct Target Hits: 0.3024193548387097

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 34.689274638236306
- Win Rate (Long Bias): 0.8114285687454476
- Return per Vol: 0.41178782721607643
- Profit Factor: 1.8499733551256272

**Regime-Specific Characteristics:**

- reversion_center: 0.003589945605257875
- reversion_speed: 169.78846987135356
- reversion_range: 0.006426064290843259


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

**Detailed Economic Report:** /Users/remyroche/Documents/Ares/outcomes/regime_economic_relevance_report_20251117_225217.md


---

## Quality Assessment

**Overall Quality Score:** 0.7518 / 1.0
**Quality Level:** Excellent ✅
**Recommendation:** The clustering shows excellent quality. Proceed with confidence.

