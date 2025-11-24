# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-22T16:44:59.009217
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
| alpha_calibration | {'target_type': 'regression', 'regression_calibration_enabled': True, 'regression_calibration_used': True, 'regression_calibration_method': 'isotonic_regression', 'val_rmse_uncalibrated': 0.6007159903922079, 'val_rmse_calibrated': 0.5066388765795858} |
| alpha_auto_prune | {'enabled': True, 'adopted': False, 'baseline_val_r2': 0.8113188559963782, 'best_val_r2': 0.8113188559963782, 'best_quantile': None, 'n_best_dropped_features': 0, 'best_dropped_features': []} |
| alpha_score_diagnostics | {'distribution': {'mean': 0.5209168163857893, 'std': 0.27073688922259087, 'min': 0.0, 'max': 0.9994037114430943, 'q05': 0.08929118345019739, 'q50': 0.5210948039163894, 'q95': 0.9482615345640351, 'n': 697}, 'decile_forward_returns': [{'decile': 0, 'n': 70, 'mean_forward_return': -0.007328787824539361, 'vol_forward_return': 0.007742820815554495, 'sharpe_forward_return': -0.9465256311447015}, {'decile': 1, 'n': 70, 'mean_forward_return': -0.004147814239973047, 'vol_forward_return': 0.005789576747105135, 'sharpe_forward_return': -0.7164266503212868}, {'decile': 2, 'n': 69, 'mean_forward_return': -0.002539103313779545, 'vol_forward_return': 0.004010897144687252, 'sharpe_forward_return': -0.6330496374474234}, {'decile': 3, 'n': 70, 'mean_forward_return': -0.0012187771722568174, 'vol_forward_return': 0.004500883132156916, 'sharpe_forward_return': -0.2707856277566749}, {'decile': 4, 'n': 70, 'mean_forward_return': 0.0004975401517836808, 'vol_forward_return': 0.004447301167439793, 'sharpe_forward_return': 0.11187437376236085}, {'decile': 5, 'n': 69, 'mean_forward_return': 0.0003167361210380071, 'vol_forward_return': 0.00347406047279881, 'sharpe_forward_return': 0.09117147263360939}, {'decile': 6, 'n': 70, 'mean_forward_return': 0.002592752461306957, 'vol_forward_return': 0.004038019459702245, 'sharpe_forward_return': 0.6420835923020087}, {'decile': 7, 'n': 69, 'mean_forward_return': 0.00267633943575102, 'vol_forward_return': 0.0034859776482868057, 'sharpe_forward_return': 0.7677420879750712}, {'decile': 8, 'n': 70, 'mean_forward_return': 0.003928783066267868, 'vol_forward_return': 0.004491863539243407, 'sharpe_forward_return': 0.8746424029848391}, {'decile': 9, 'n': 70, 'mean_forward_return': 0.005258864745794535, 'vol_forward_return': 0.0054338949173360915, 'sharpe_forward_return': 0.9677874062567573}], 'ic_pearson': 0.9809793846808993, 'ic_spearman': 0.9877967149259292} |


## PCA Feature Analysis



---

## Top Configuration Analysis

### Clustering Configuration Parameters


---

## Clustering Metrics

### Silhouette Analysis

**Global Silhouette Score:** -0.0319

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | -0.1133 | 0.1016 | -0.3427 | 0.0758 |
| 1 | -0.0345 | 0.0534 | -0.2220 | 0.0697 |
| 2 | -0.0088 | 0.0450 | -0.1025 | 0.0876 |
| 3 | 0.0365 | 0.0410 | -0.0790 | 0.1116 |
| 4 | -0.0418 | 0.0648 | -0.1848 | 0.0751 |


### Separation Metrics

- **Davies-Bouldin Index:** 5.3273 (lower is better)
- **Calinski-Harabasz Index:** 28.28 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 11.2296 +/- 4.5869
- **Between-Regime CV:** 83.9261 +/- 105.1434

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 20.1237 |
| 1 | 10.8609 |
| 2 | 9.4544 |
| 3 | 7.9668 |
| 4 | 7.7424 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 3.6181
- **Between-Regime CV (Mean Return):** 30.2990
- **CV Ratio (Between/Within):** 8.3743

| max_drawdown | 1.3330 |
| mean_return | 30.2990 |
| pct_above_target | 1.0618 |
| pct_below_neg_target | 1.2788 |
| pct_target_hits | 0.6891 |
| sharpe | 6.7979 |
| volatility | 0.3424 |


---

## Economic Gap Analysis

### Per-Regime Snapshot

| Regime | Type | Mean Return | Volatility | Sharpe | Max DD | Pct Target Hits |
|--------|------|-------------|------------|--------|--------|-----------------|
| 0 | mean_reverting | -0.004861 | 0.008350 | -0.5821 | -47.94% | 36.84% |
| 1 | mean_reverting | -0.000992 | 0.004736 | -0.2096 | -14.03% | 13.53% |
| 2 | stable | 0.000710 | 0.003928 | 0.1809 | -2.13% | 7.24% |
| 3 | stable | 0.000687 | 0.003265 | 0.2103 | -2.23% | 4.51% |
| 4 | mean_reverting | 0.003981 | 0.005506 | 0.7230 | -0.83% | 27.40% |

### Pairwise Economic Spreads

| Regime A | Regime B | Mean Return Spread | Sharpe Spread | Volatility Ratio | Max DD Spread |
|----------|----------|--------------------|---------------|------------------|---------------|
| 0 | 1 | -0.003869 | -0.3726 | 1.763 | -33.91% |
| 0 | 2 | -0.005571 | -0.7630 | 2.126 | -45.81% |
| 0 | 3 | -0.005548 | -0.7925 | 2.557 | -45.71% |
| 0 | 4 | -0.008842 | -1.3052 | 1.516 | -47.11% |
| 1 | 2 | -0.001703 | -0.3904 | 1.206 | -11.89% |
| 1 | 3 | -0.001679 | -0.4199 | 1.450 | -11.79% |
| 1 | 4 | -0.004974 | -0.9326 | 0.860 | -13.20% |
| 2 | 3 | 0.000024 | -0.0295 | 1.203 | 0.10% |
| 2 | 4 | -0.003271 | -0.5422 | 0.713 | -1.31% |
| 3 | 4 | -0.003295 | -0.5127 | 0.593 | -1.41% |

### Statistical Tests (ANOVA / t-tests)

- **ANOVA F-statistic:** 48.8134, p-value=0.0000 (significant)

**Pairwise t-tests:**

| Regime A | Regime B | t-stat | p-value | Cohen's d | Significant |
|----------|----------|--------|---------|-----------|-------------|
| 0 | 1 | -4.6474 | 0.0000 | -0.570 | Yes |
| 0 | 2 | -7.0430 | 0.0000 | -0.873 | Yes |
| 0 | 3 | -7.1359 | 0.0000 | -0.875 | Yes |
| 0 | 4 | -10.3353 | 0.0000 | -1.262 | Yes |
| 1 | 2 | -3.2763 | 0.0012 | -0.394 | Yes |
| 1 | 3 | -3.3666 | 0.0009 | -0.413 | Yes |
| 1 | 4 | -8.1079 | 0.0000 | -0.965 | Yes |
| 2 | 3 | 0.0555 | 0.9558 | 0.007 | No |
| 2 | 4 | -5.8824 | 0.0000 | -0.686 | Yes |
| 3 | 4 | -6.1407 | 0.0000 | -0.720 | Yes |


### Per-Category Coefficient of Variation


| Category | Within CV | Between CV | Ratio | # Features |
|----------|-----------|------------|-------|------------|
| momentum | 20.104 ± 26.585 | 28.550 ± 0.000 | 1.420 | 1 |
| other | 7.363 ± 2.427 | 53.291 ± 38.237 | 7.238 | 16 |
| price | 20.227 ± 20.444 | 152.580 ± 167.849 | 7.543 | 8 |
| volatility | 4.909 ± 4.305 | 42.511 ± 0.000 | 8.659 | 1 |
| volume | 4.898 ± 1.490 | 102.785 ± 35.688 | 20.983 | 2 |

**Interpretation:** Higher CV ratio indicates better regime separation for that feature category.


---

## Balance and Distribution

**Balance Score:** 0.9453 (0-1, higher is better)

- **Smallest Cluster:** 19.08% of total
- **Largest Cluster:** 21.81% of total
- **Cluster Size Std Dev:** 8.06

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 19.08% |
| 1 | 19.08% |
| 2 | 21.81% |
| 3 | 19.08% |
| 4 | 20.95% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.0141 (0-1, higher = fewer transitions)
- **Temporal Smoothness (Raw):** 0.0564
- **Flip-Flop Ratio:** 0.0819 (rapid back-and-forth transitions)
- **Regime Persistence:** 2.83 bars (average duration)

### Transition & Persistence Insights

- **Average Duration:** 2.82 bars
- **Max Duration:** 18.00 bars
- **Min Duration:** 1.00 bars
- **High-persistence regimes:** Regime 0 (p_self=0.77), Regime 4 (p_self=0.78)
- **Flip-flop ratio:** 0.0819
- **Average regime persistence:** 2.83 bars
- **Transition entropy:** 0.9271
- **Regime stickiness:** 0.6464
- **Transition stability score:** 0.5352

**Dominant transition hotspots:**

| From | To | Probability |
|------|----|-------------|
| 1 | 2 | 0.280 |
| 0 | 1 | 0.211 |
| 2 | 1 | 0.197 |
| 3 | 4 | 0.195 |
| 3 | 2 | 0.188 |


### Transition Probability Matrix

This matrix shows the probability of transitioning from one regime to another:


### Regime Duration Analysis

**Average Regime Durations:**

| Regime | Mean Duration | Std Duration | Min Duration | Max Duration |
|--------|---------------|--------------|--------------|--------------|
| 0 | 4.4 | 3.9 | 1 | 18 |
| 1 | 2.1 | 1.5 | 1 | 8 |
| 2 | 2.2 | 2.2 | 1 | 14 |
| 3 | 2.4 | 1.6 | 1 | 8 |
| 4 | 4.6 | 3.7 | 1 | 16 |

- **Duration Stability Score:** 0.514 (higher = more consistent durations)


---

## Per-Regime Analysis

### Regime 0 (mean_reverting)

**Size:** 133 samples (19.08%)

**Performance Metrics:**
- Mean Return: -0.004861011279725808
- Volatility: 0.008350405169276818
- Sharpe Ratio: -0.5821287229848358
- Skewness: -0.41340633143848077
- Max Drawdown: -0.4793966335591081

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.05263157894736842
- Pct < -1.0% (Shorts): 0.3157894736842105
- Pct Target Hits: 0.3684210526315789

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 44.120135615329694
- Win Rate (Long Bias): 0.14285714246938774
- Return per Vol: 0.5821287229848358
- Profit Factor: 0.5315009381992465

**Regime-Specific Characteristics:**

- reversion_center: -0.004861011279725808
- reversion_speed: 157.91525633400477
- reversion_range: 0.005415237616398919

### Regime 1 (mean_reverting)

**Size:** 133 samples (19.08%)

**Performance Metrics:**
- Mean Return: -0.0009924295615513982
- Volatility: 0.004735999846210828
- Sharpe Ratio: -0.20955012335891357
- Skewness: 0.3255680322490148
- Max Drawdown: -0.14026583432929635

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.045112781954887216
- Pct < -1.0% (Shorts): 0.09022556390977443
- Pct Target Hits: 0.13533834586466165

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 28.576503733723552
- Win Rate (Long Bias): 0.33333333087037037
- Return per Vol: 0.20955012335891357
- Profit Factor: 0.8319490005388099

**Regime-Specific Characteristics:**

- reversion_center: -0.0009924295615513982
- reversion_speed: 286.69875575585996
- reversion_range: 0.0031892818584816182

### Regime 2 (stable)

**Size:** 152 samples (21.81%)

**Performance Metrics:**
- Mean Return: 0.0007104591113683391
- Volatility: 0.003927956815241608
- Sharpe Ratio: 0.1808723883468282
- Skewness: 0.5366015567033443
- Max Drawdown: -0.02134101205793409

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.046052631578947366
- Pct < -1.0% (Shorts): 0.02631578947368421
- Pct Target Hits: 0.07236842105263158

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 18.423930311018424
- Win Rate (Long Bias): 0.636363627570248
- Return per Vol: 0.1808723883468282
- Profit Factor: 1.3442329503901733

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0007104591113683391
- volatility: 0.003927956815241608
- stability_coefficient: 0.15316866133626952

### Regime 3 (stable)

**Size:** 133 samples (19.08%)

**Performance Metrics:**
- Mean Return: 0.0006868207804609794
- Volatility: 0.0032651767877126215
- Sharpe Ratio: 0.21034713118703496
- Skewness: -0.24656662725073322
- Max Drawdown: -0.0223452222803424

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.022556390977443608
- Pct < -1.0% (Shorts): 0.022556390977443608
- Pct Target Hits: 0.045112781954887216

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 13.816332490271655
- Win Rate (Long Bias): 0.4999999889166669
- Return per Vol: 0.21034713118703496
- Profit Factor: 1.1268643403925809

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0006868207804609794
- volatility: 0.0032651767877126215
- stability_coefficient: 0.17379099931668018

### Regime 4 (mean_reverting)

**Size:** 146 samples (20.95%)

**Performance Metrics:**
- Mean Return: 0.003981329117226656
- Volatility: 0.005506417024417763
- Sharpe Ratio: 0.7230343028029792
- Skewness: 0.6061274466921421
- Max Drawdown: -0.008259756571428353

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.273972602739726
- Pct < -1.0% (Shorts): 0.0
- Pct Target Hits: 0.273972602739726

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 49.755140551410506
- Win Rate (Long Bias): 0.9999999963499999
- Return per Vol: 0.7230343028029792
- Profit Factor: 2.5279012737974065

**Regime-Specific Characteristics:**

- reversion_center: 0.003981329117226656
- reversion_speed: 227.37342390238737
- reversion_range: 0.0032930809868442286


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5617816091954023

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Economic Relevance Analysis

### Strategy Performance Summary

#### Per-Regime Long/Short Strategies

- Regime 0 Only (Long): total_return=-0.3616, sharpe=-2.8002, max_dd=-0.3627
- Regime 0 Only (Short): total_return=0.3756, sharpe=1.4803, max_dd=-0.0221
- Regime 1 Only (Long): total_return=-0.1708, sharpe=-2.5199, max_dd=-0.1960
- Regime 1 Only (Short): total_return=-0.0602, sharpe=-1.1802, max_dd=-0.0742
- Regime 2 Only (Long): total_return=-0.1216, sharpe=-2.1597, max_dd=-0.1242
- Regime 2 Only (Short): total_return=-0.1349, sharpe=-2.4115, max_dd=-0.1422
- Regime 3 Only (Long): total_return=-0.0148, sharpe=-0.7894, max_dd=-0.0438
- Regime 3 Only (Short): total_return=-0.1877, sharpe=-2.9102, max_dd=-0.1889
- Regime 4 Only (Long): total_return=0.3253, sharpe=1.8155, max_dd=-0.0139
- Regime 4 Only (Short): total_return=-0.3399, sharpe=-3.8343, max_dd=-0.3456

### Statistical Significance Tests

### Economic Regime Mapping

| Regime | Economic Interpretation | Recommended Position |
|---------|----------------------|----------------------|

### Economic Interpretation

**Detailed Economic Report:** /Users/remyroche/Documents/Ares/outcomes/regime_economic_relevance_report_20251122_164500.md


---

## Quality Assessment

**Overall Quality Score:** 0.7052 / 1.0
**Quality Level:** Excellent ✅
**Recommendation:** The clustering shows excellent quality. Proceed with confidence.

