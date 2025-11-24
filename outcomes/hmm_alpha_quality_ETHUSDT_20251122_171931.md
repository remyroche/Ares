# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-22T17:19:14.567663
**Data Points:** N/A
**Number of Regimes:** 4
**Report Version:** 1.3 (Enhanced with Financial Analysis)

This report provides a comprehensive assessment of cluster quality for ETHUSDT.

### Key Metrics


---

## Clustering Method Configuration

| Parameter | Value |
|---|---|
| alpha_config | {'alpha_horizon_bars': 1, 'alpha_regime_bins': 5, 'alpha_target_type': 'regression'} |
| alpha_calibration | {'target_type': 'regression', 'regression_calibration_enabled': True, 'regression_calibration_used': True, 'regression_calibration_method': 'isotonic_regression', 'val_rmse_uncalibrated': 0.2193790509199266, 'val_rmse_calibrated': 0.1976915784437165} |
| alpha_auto_prune | {'enabled': True, 'adopted': False, 'baseline_val_r2': 0.972483031157958, 'best_val_r2': 0.972483031157958, 'best_quantile': None, 'n_best_dropped_features': 0, 'best_dropped_features': []} |
| alpha_score_diagnostics | {'distribution': {'mean': 0.5211789616838057, 'std': 0.2845157710866367, 'min': 0.0005016606961838626, 'max': 1.0, 'q05': 0.06809446818296426, 'q50': 0.523623130088157, 'q95': 0.9564690160064782, 'n': 7805}, 'decile_forward_returns': [{'decile': 0, 'n': 781, 'mean_forward_return': -0.007047428855566763, 'vol_forward_return': 0.006912813207950347, 'sharpe_forward_return': -1.019471877634829}, {'decile': 1, 'n': 780, 'mean_forward_return': -0.005604399878831621, 'vol_forward_return': 0.006229708101073626, 'sharpe_forward_return': -0.8996233517959283}, {'decile': 2, 'n': 781, 'mean_forward_return': -0.0037516684953430887, 'vol_forward_return': 0.0051476331352282075, 'sharpe_forward_return': -0.7288128560560693}, {'decile': 3, 'n': 780, 'mean_forward_return': -0.0014190889808131505, 'vol_forward_return': 0.005262727350554526, 'sharpe_forward_return': -0.26964845218118727}, {'decile': 4, 'n': 781, 'mean_forward_return': 0.0005026297929444191, 'vol_forward_return': 0.005295743178159122, 'sharpe_forward_return': 0.09491186164366147}, {'decile': 5, 'n': 780, 'mean_forward_return': 0.0001503969782098325, 'vol_forward_return': 0.0050364812261068734, 'sharpe_forward_return': 0.02986145938867979}, {'decile': 6, 'n': 780, 'mean_forward_return': 0.0030321242482626827, 'vol_forward_return': 0.004693759442130417, 'sharpe_forward_return': 0.6459891747231744}, {'decile': 7, 'n': 781, 'mean_forward_return': 0.0042117991450432955, 'vol_forward_return': 0.005203202809532101, 'sharpe_forward_return': 0.8094612500429407}, {'decile': 8, 'n': 780, 'mean_forward_return': 0.00502121475105183, 'vol_forward_return': 0.005486688210168347, 'sharpe_forward_return': 0.9151614611764413}, {'decile': 9, 'n': 781, 'mean_forward_return': 0.005876188009089425, 'vol_forward_return': 0.006134950583531715, 'sharpe_forward_return': 0.9578200102643002}], 'ic_pearson': 0.9929059938924285, 'ic_spearman': 0.9933599111530315} |


## PCA Feature Analysis



---

## Top Configuration Analysis

### Clustering Configuration Parameters


---

## Clustering Metrics

### Silhouette Analysis

**Global Silhouette Score:** 0.0108

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | -0.0096 | 0.0625 | -0.2355 | 0.1400 |
| 1 | 0.0216 | 0.0437 | -0.1317 | 0.1176 |
| 2 | 0.0245 | 0.0412 | -0.1023 | 0.1175 |
| 3 | 0.0039 | 0.0566 | -0.2291 | 0.1240 |


### Separation Metrics

- **Davies-Bouldin Index:** 4.8090 (lower is better)
- **Calinski-Harabasz Index:** 455.51 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 121.4394 +/- 66.5651
- **Between-Regime CV:** 44.9654 +/- 59.8433

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 163.8166 |
| 1 | 103.2136 |
| 2 | 196.8393 |
| 3 | 21.8878 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 3.7591
- **Between-Regime CV (Mean Return):** 433.5432
- **CV Ratio (Between/Within):** 115.3324

| max_drawdown | 0.9419 |
| mean_return | 433.5432 |
| pct_above_target | 0.7928 |
| pct_below_neg_target | 1.0498 |
| pct_target_hits | 0.4296 |
| sharpe | 12.8322 |
| volatility | 0.1828 |


---

## Economic Gap Analysis

### Per-Regime Snapshot

| Regime | Type | Mean Return | Volatility | Sharpe | Max DD | Pct Target Hits |
|--------|------|-------------|------------|--------|--------|-----------------|
| 0 | mean_reverting | -0.004982 | 0.007802 | -0.6386 | -99.99% | 37.99% |
| 1 | stable | -0.000682 | 0.005288 | -0.1290 | -75.96% | 14.49% |
| 2 | stable | 0.001163 | 0.004955 | 0.2348 | -4.72% | 13.75% |
| 3 | mean_reverting | 0.004469 | 0.006534 | 0.6840 | -2.30% | 30.02% |

### Pairwise Economic Spreads

| Regime A | Regime B | Mean Return Spread | Sharpe Spread | Volatility Ratio | Max DD Spread |
|----------|----------|--------------------|---------------|------------------|---------------|
| 0 | 1 | -0.004300 | -0.5096 | 1.475 | -24.03% |
| 0 | 2 | -0.006145 | -0.8734 | 1.575 | -95.26% |
| 0 | 3 | -0.009451 | -1.3226 | 1.194 | -97.68% |
| 1 | 2 | -0.001845 | -0.3638 | 1.067 | -71.24% |
| 1 | 3 | -0.005151 | -0.8130 | 0.809 | -73.66% |
| 2 | 3 | -0.003306 | -0.4493 | 0.758 | -2.42% |

### Statistical Tests (ANOVA / t-tests)

- **ANOVA F-statistic:** 758.0975, p-value=0.0000 (significant)

**Pairwise t-tests:**

| Regime A | Regime B | t-stat | p-value | Cohen's d | Significant |
|----------|----------|--------|---------|-----------|-------------|
| 0 | 1 | -19.6912 | 0.0000 | -0.653 | Yes |
| 0 | 2 | -28.6507 | 0.0000 | -0.953 | Yes |
| 0 | 3 | -40.0192 | 0.0000 | -1.318 | Yes |
| 1 | 2 | -11.4709 | 0.0000 | -0.360 | Yes |
| 1 | 3 | -27.2831 | 0.0000 | -0.868 | Yes |
| 2 | 3 | -17.9372 | 0.0000 | -0.572 | Yes |


### Per-Category Coefficient of Variation


| Category | Within CV | Between CV | Ratio | # Features |
|----------|-----------|------------|-------|------------|
| momentum | 3.811 ± 2.173 | 33.653 ± 0.000 | 8.831 | 1 |
| other | 107.824 ± 95.930 | 33.962 ± 11.628 | 0.315 | 18 |
| price | 207.370 ± 149.076 | 79.771 ± 106.895 | 0.385 | 8 |
| volatility | 30.061 ± 35.793 | 20.978 ± 0.000 | 0.698 | 1 |
| volume | 4.760 ± 1.234 | 22.425 ± 0.362 | 4.711 | 2 |

**Interpretation:** Higher CV ratio indicates better regime separation for that feature category.


---

## Balance and Distribution

**Balance Score:** 0.9533 (0-1, higher is better)

- **Smallest Cluster:** 23.00% of total
- **Largest Cluster:** 26.00% of total
- **Cluster Size Std Dev:** 95.53

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 23.00% |
| 1 | 26.00% |
| 2 | 26.00% |
| 3 | 25.01% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.0414 (0-1, higher = fewer transitions)
- **Temporal Smoothness (Raw):** 0.0626
- **Flip-Flop Ratio:** 0.0834 (rapid back-and-forth transitions)
- **Regime Persistence:** 3.13 bars (average duration)

### Transition & Persistence Insights

- **Average Duration:** 3.13 bars
- **Max Duration:** 37.00 bars
- **Min Duration:** 1.00 bars
- **High-persistence regimes:** Regime 0 (p_self=0.78), Regime 3 (p_self=0.79)
- **Flip-flop ratio:** 0.0834
- **Average regime persistence:** 3.13 bars
- **Transition entropy:** 0.8094
- **Regime stickiness:** 0.6844
- **Transition stability score:** 0.5503

**Dominant transition hotspots:**

| From | To | Probability |
|------|----|-------------|
| 2 | 1 | 0.219 |
| 1 | 2 | 0.215 |
| 0 | 1 | 0.203 |
| 3 | 2 | 0.193 |
| 2 | 3 | 0.179 |


### Transition Probability Matrix

This matrix shows the probability of transitioning from one regime to another:


### Regime Duration Analysis

**Average Regime Durations:**

| Regime | Mean Duration | Std Duration | Min Duration | Max Duration |
|--------|---------------|--------------|--------------|--------------|
| 0 | 4.5 | 4.1 | 1 | 33 |
| 1 | 2.4 | 2.0 | 1 | 14 |
| 2 | 2.4 | 2.0 | 1 | 18 |
| 3 | 4.7 | 4.4 | 1 | 37 |

- **Duration Stability Score:** 0.503 (higher = more consistent durations)


---

## Per-Regime Analysis

### Regime 0 (mean_reverting)

**Size:** 1795 samples (23.00%)

**Performance Metrics:**
- Mean Return: -0.0049820851523092536
- Volatility: 0.007801582307867565
- Sharpe Ratio: -0.6385992375784839
- Skewness: -0.2639853037753482
- Max Drawdown: -0.9998770717340193

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.043454038997214485
- Pct < -1.0% (Shorts): 0.33649025069637883
- Pct Target Hits: 0.37994428969359334

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 48.70092065932766
- Win Rate (Long Bias): 0.11436950116525914
- Return per Vol: 0.6385992375784839
- Profit Factor: 0.5349557210202535

**Regime-Specific Characteristics:**

- reversion_center: -0.0049820851523092536
- reversion_speed: 167.56528128668396
- reversion_range: 0.005022940429002451

### Regime 1 (stable)

**Size:** 2029 samples (26.00%)

**Performance Metrics:**
- Mean Return: -0.0006820105663797779
- Volatility: 0.005287670570268008
- Sharpe Ratio: -0.12898126468645438
- Skewness: 0.12114443714483114
- Max Drawdown: -0.7596242606549692

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.05963528831936915
- Pct < -1.0% (Shorts): 0.08526367668802366
- Pct Target Hits: 0.1448989650073928

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 27.403170390185284
- Win Rate (Long Bias): 0.4115646230099843
- Return per Vol: 0.12898126468645438
- Profit Factor: 0.9281011831084801

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.0006820105663797779
- volatility: 0.005287670570268008
- stability_coefficient: 0.11424587620719695

### Regime 2 (stable)

**Size:** 2029 samples (26.00%)

**Performance Metrics:**
- Mean Return: 0.0011633384832820067
- Volatility: 0.004954884813563567
- Sharpe Ratio: 0.23478613373843435
- Skewness: 0.2892146031362501
- Max Drawdown: -0.04724219893577611

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.09512074913750616
- Pct < -1.0% (Shorts): 0.04238541153277477
- Pct Target Hits: 0.13750616067028093

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 27.75163058124767
- Win Rate (Long Bias): 0.6917562673707044
- Return per Vol: 0.23478613373843435
- Profit Factor: 1.3051002125819873

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0011633384832820067
- volatility: 0.004954884813563567
- stability_coefficient: 0.19014332048627244

### Regime 3 (mean_reverting)

**Size:** 1952 samples (25.01%)

**Performance Metrics:**
- Mean Return: 0.004469261302025208
- Volatility: 0.006533567614860959
- Sharpe Ratio: 0.6840459732617836
- Skewness: 0.5415115602135061
- Max Drawdown: -0.023041064509271077

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.28125
- Pct < -1.0% (Shorts): 0.018954918032786885
- Pct Target Hits: 0.30020491803278687

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 45.94807764778874
- Win Rate (Long Bias): 0.9368600651386504
- Return per Vol: 0.6840459732617836
- Profit Factor: 2.287949132622469

**Regime-Specific Characteristics:**

- reversion_center: 0.004469261302025208
- reversion_speed: 201.54902452169455
- reversion_range: 0.004249435385948458


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5762410996460398

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Economic Relevance Analysis

### Strategy Performance Summary

#### Per-Regime Long/Short Strategies

- Regime 0 Only (Long): total_return=-0.9942, sharpe=-2.8147, max_dd=-0.9942
- Regime 0 Only (Short): total_return=29.2963, sharpe=1.3700, max_dd=-0.0412
- Regime 1 Only (Long): total_return=-0.9513, sharpe=-2.3142, max_dd=-0.9523
- Regime 1 Only (Short): total_return=-0.3410, sharpe=-0.6407, max_dd=-0.3766
- Regime 2 Only (Long): total_return=-0.1944, sharpe=-0.5571, max_dd=-0.2338
- Regime 2 Only (Short): total_return=-0.9603, sharpe=-2.6392, max_dd=-0.9605
- Regime 3 Only (Long): total_return=51.4485, sharpe=1.8741, max_dd=-0.0561
- Regime 3 Only (Short): total_return=-0.9967, sharpe=-3.5808, max_dd=-0.9967

### Statistical Significance Tests

### Economic Regime Mapping

| Regime | Economic Interpretation | Recommended Position |
|---------|----------------------|----------------------|

### Economic Interpretation

**Detailed Economic Report:** /Users/remyroche/Documents/Ares/outcomes/regime_economic_relevance_report_20251122_171928.md


---

## Quality Assessment

**Overall Quality Score:** 0.3107 / 1.0
**Quality Level:** Moderate ⚠️
**Recommendation:** The clustering shows moderate quality. Consider parameter tuning.

