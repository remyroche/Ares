# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-22T15:18:31.463802
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
| alpha_calibration | {'target_type': 'regression', 'regression_calibration_enabled': True, 'regression_calibration_used': True, 'regression_calibration_method': 'isotonic_regression', 'val_rmse_uncalibrated': 0.008421651527753453, 'val_rmse_calibrated': 0.008102753965790523} |
| alpha_auto_prune | {'enabled': True, 'adopted': True, 'baseline_val_r2': -0.06417334017384713, 'best_val_r2': -0.0611361302470963, 'best_quantile': 0.35, 'n_best_dropped_features': 6, 'best_dropped_features': ['high', 'low', 'hl_range', 'trend_ema_fast', 'trend_ema_slow', 'trend_price_slope']} |
| alpha_score_diagnostics | {'distribution': {'mean': -0.00023618321946039126, 'std': 0.0016000749881215747, 'min': -0.009741561239810721, 'max': 0.011934691381980828, 'q05': -0.0026840284529780606, 'q50': -9.921361929577224e-05, 'q95': 0.002039140430213554, 'n': 7825}, 'decile_forward_returns': [{'decile': 0, 'n': 783, 'mean_forward_return': -0.0016405809003917735, 'vol_forward_return': 0.00781212380621904, 'sharpe_forward_return': -0.21000419873578574}, {'decile': 1, 'n': 782, 'mean_forward_return': -0.00231917770181857, 'vol_forward_return': 0.007616419800297689, 'sharpe_forward_return': -0.304496694990601}, {'decile': 2, 'n': 783, 'mean_forward_return': -0.0015740907080461434, 'vol_forward_return': 0.007164701719168252, 'sharpe_forward_return': -0.21970049455512203}, {'decile': 3, 'n': 782, 'mean_forward_return': -0.0007734660504598549, 'vol_forward_return': 0.006482580491125152, 'sharpe_forward_return': -0.11931434686777634}, {'decile': 4, 'n': 783, 'mean_forward_return': -9.397695684203652e-05, 'vol_forward_return': 0.006211285206337445, 'sharpe_forward_return': -0.015130009719414224}, {'decile': 5, 'n': 782, 'mean_forward_return': 7.979113337495322e-05, 'vol_forward_return': 0.005362858437644272, 'sharpe_forward_return': 0.014878443188138843}, {'decile': 6, 'n': 782, 'mean_forward_return': 0.00036737805330020146, 'vol_forward_return': 0.005357154517439784, 'sharpe_forward_return': 0.06857695934187462}, {'decile': 7, 'n': 783, 'mean_forward_return': 0.0008057093210135376, 'vol_forward_return': 0.005595648445520929, 'sharpe_forward_return': 0.14398829536468072}, {'decile': 8, 'n': 782, 'mean_forward_return': 0.0016922025756687374, 'vol_forward_return': 0.0061936232500633, 'sharpe_forward_return': 0.27321646396344873}, {'decile': 9, 'n': 783, 'mean_forward_return': 0.004534614130164607, 'vol_forward_return': 0.009074138986182341, 'sharpe_forward_return': 0.49972886020162216}], 'ic_pearson': 0.4048831015752791, 'ic_spearman': 0.3391463800305277} |


## PCA Feature Analysis



---

## Top Configuration Analysis

### Clustering Configuration Parameters


---

## Clustering Metrics

### Silhouette Analysis

**Global Silhouette Score:** 0.0060

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.0759 | 0.1478 | -0.3586 | 0.2505 |
| 1 | -0.0307 | 0.0683 | -0.2146 | 0.0773 |
| 2 | 0.0870 | 0.0810 | -0.1802 | 0.2191 |
| 3 | -0.0732 | 0.0911 | -0.2801 | 0.1006 |


### Separation Metrics

- **Davies-Bouldin Index:** 4.8033 (lower is better)
- **Calinski-Harabasz Index:** 523.56 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 38.8720 +/- 35.2059
- **Between-Regime CV:** 16.7568 +/- 8.0750

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 95.8165 |
| 1 | 9.6169 |
| 2 | 40.4534 |
| 3 | 9.6012 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 12.0543
- **Between-Regime CV (Mean Return):** 195.3624
- **CV Ratio (Between/Within):** 16.2069

| max_drawdown | 0.7442 |
| mean_return | 195.3624 |
| pct_above_target | 0.3828 |
| pct_below_neg_target | 0.3515 |
| pct_target_hits | 0.1828 |
| sharpe | 58.6437 |
| volatility | 0.1041 |


---

## Economic Gap Analysis

### Per-Regime Snapshot

| Regime | Type | Mean Return | Volatility | Sharpe | Max DD | Pct Target Hits |
|--------|------|-------------|------------|--------|--------|-----------------|
| 0 | stable | -0.001435 | 0.007625 | -0.1882 | -92.88% | 26.72% |
| 1 | stable | -0.000516 | 0.006635 | -0.0778 | -74.88% | 22.52% |
| 2 | stable | 0.000224 | 0.005787 | 0.0387 | -22.11% | 16.55% |
| 3 | stable | 0.001752 | 0.007355 | 0.2381 | -4.97% | 27.06% |

### Pairwise Economic Spreads

| Regime A | Regime B | Mean Return Spread | Sharpe Spread | Volatility Ratio | Max DD Spread |
|----------|----------|--------------------|---------------|------------------|---------------|
| 0 | 1 | -0.000919 | -0.1104 | 1.149 | -18.00% |
| 0 | 2 | -0.001659 | -0.2268 | 1.318 | -70.78% |
| 0 | 3 | -0.003186 | -0.4263 | 1.037 | -87.91% |
| 1 | 2 | -0.000740 | -0.1165 | 1.147 | -52.77% |
| 1 | 3 | -0.002268 | -0.3160 | 0.902 | -69.91% |
| 2 | 3 | -0.001528 | -0.1995 | 0.787 | -17.13% |

### Statistical Tests (ANOVA / t-tests)

- **ANOVA F-statistic:** 79.3112, p-value=0.0000 (significant)

**Pairwise t-tests:**

| Regime A | Regime B | t-stat | p-value | Cohen's d | Significant |
|----------|----------|--------|---------|-----------|-------------|
| 0 | 1 | -3.9549 | 0.0001 | -0.129 | Yes |
| 0 | 2 | -7.2272 | 0.0000 | -0.244 | Yes |
| 0 | 3 | -13.5436 | 0.0000 | -0.426 | Yes |
| 1 | 2 | -3.6108 | 0.0003 | -0.118 | Yes |
| 1 | 3 | -10.7276 | 0.0000 | -0.323 | Yes |
| 2 | 3 | -7.3321 | 0.0000 | -0.226 | Yes |


### Per-Category Coefficient of Variation


| Category | Within CV | Between CV | Ratio | # Features |
|----------|-----------|------------|-------|------------|
| momentum | 22.742 ± 25.495 | 10.371 ± 0.000 | 0.456 | 1 |
| other | 26.172 ± 22.763 | 17.722 ± 10.319 | 0.677 | 14 |
| price | 2.966 ± 1.366 | 17.917 ± 2.476 | 6.040 | 8 |
| volatility | 8.329 ± 5.325 | 11.333 ± 0.000 | 1.361 | 1 |
| volume | 294.732 ± 480.966 | 11.267 ± 0.789 | 0.038 | 2 |

**Interpretation:** Higher CV ratio indicates better regime separation for that feature category.


---

## Balance and Distribution

**Balance Score:** 0.8807 (0-1, higher is better)

- **Smallest Cluster:** 21.01% of total
- **Largest Cluster:** 29.99% of total
- **Cluster Size Std Dev:** 264.88

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 23.00% |
| 1 | 25.99% |
| 2 | 21.01% |
| 3 | 29.99% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.1211 (0-1, higher = fewer transitions)
- **Temporal Smoothness (Raw):** 0.1211
- **Flip-Flop Ratio:** 0.0372 (rapid back-and-forth transitions)
- **Regime Persistence:** 6.06 bars (average duration)

### Transition & Persistence Insights

- **Average Duration:** 6.06 bars
- **Max Duration:** 720.00 bars
- **Min Duration:** 1.00 bars
- **High-persistence regimes:** Regime 0 (p_self=0.93), Regime 1 (p_self=0.80), Regime 2 (p_self=0.70), Regime 3 (p_self=0.88)
- **Flip-flop ratio:** 0.0372
- **Average regime persistence:** 6.06 bars
- **Transition entropy:** 0.5460
- **Regime stickiness:** 0.8296
- **Transition stability score:** 0.7179

**Dominant transition hotspots:**

| From | To | Probability |
|------|----|-------------|
| 2 | 1 | 0.165 |
| 2 | 3 | 0.128 |
| 1 | 2 | 0.116 |
| 3 | 2 | 0.106 |


### Transition Probability Matrix

This matrix shows the probability of transitioning from one regime to another:


### Regime Duration Analysis

**Average Regime Durations:**

| Regime | Mean Duration | Std Duration | Min Duration | Max Duration |
|--------|---------------|--------------|--------------|--------------|
| 0 | 14.4 | 67.0 | 1 | 720 |
| 1 | 5.0 | 4.8 | 1 | 35 |
| 2 | 3.4 | 4.0 | 1 | 36 |
| 3 | 8.6 | 9.7 | 1 | 56 |

- **Duration Stability Score:** 0.217 (higher = more consistent durations)


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 1800 samples (23.00%)

**Performance Metrics:**
- Mean Return: -0.0014349356575819189
- Volatility: 0.007625139011937958
- Sharpe Ratio: -0.18818482747010234
- Skewness: -0.6274809099722166
- Max Drawdown: -0.9288182320377699

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.09055555555555556
- Pct < -1.0% (Shorts): 0.17666666666666667
- Pct Target Hits: 0.26722222222222225

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 35.04489383852071
- Win Rate (Long Bias): 0.33887733760919075
- Return per Vol: 0.18818482747010234
- Profit Factor: 0.6916560862263145

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.0014349356575819189
- volatility: 0.007625139011937958
- stability_coefficient: 0.1583802067359511

### Regime 1 (stable)

**Size:** 2034 samples (25.99%)

**Performance Metrics:**
- Mean Return: -0.0005163863068037869
- Volatility: 0.00663469771050693
- Sharpe Ratio: -0.07783116149434491
- Skewness: -0.24126096344776252
- Max Drawdown: -0.7487900387952203

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.09341199606686332
- Pct < -1.0% (Shorts): 0.13176007866273354
- Pct Target Hits: 0.22517207472959685

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 33.93855313625724
- Win Rate (Long Bias): 0.4148471597296962
- Return per Vol: 0.07783116149434491
- Profit Factor: 0.8491222369071555

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.0005163863068037869
- volatility: 0.00663469771050693
- stability_coefficient: 0.07221104287723638

### Regime 2 (stable)

**Size:** 1644 samples (21.01%)

**Performance Metrics:**
- Mean Return: 0.0002236898293008782
- Volatility: 0.00578658039822271
- Sharpe Ratio: 0.038656646110531205
- Skewness: -0.06915890262026392
- Max Drawdown: -0.221060406432096

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.08637469586374696
- Pct < -1.0% (Shorts): 0.07907542579075426
- Pct Target Hits: 0.16545012165450124

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 28.592032198029315
- Win Rate (Long Bias): 0.5220588203740268
- Return per Vol: 0.038656646110531205
- Profit Factor: 1.038185937417699

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0002236898293008782
- volatility: 0.00578658039822271
- stability_coefficient: 0.0372180923011432

### Regime 3 (stable)

**Size:** 2347 samples (29.99%)

**Performance Metrics:**
- Mean Return: 0.0017515170821666769
- Volatility: 0.0073554630654084005
- Sharpe Ratio: 0.23812461954695532
- Skewness: 0.2573790999704213
- Max Drawdown: -0.04971573870984182

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.19216020451640392
- Pct < -1.0% (Shorts): 0.0783979548359608
- Pct Target Hits: 0.2705581593523647

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 36.783288851176295
- Win Rate (Long Bias): 0.7102362178473631
- Return per Vol: 0.23812461954695532
- Profit Factor: 1.4180761307984375

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0017515170821666769
- volatility: 0.0073554630654084005
- stability_coefficient: 0.1923269691442213


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5218558282208589

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Economic Relevance Analysis

### Strategy Performance Summary

#### Per-Regime Long/Short Strategies

- Regime 0 Only (Long): total_return=-0.8601, sharpe=-1.3962, max_dd=-0.8615
- Regime 0 Only (Short): total_return=2.9036, sharpe=0.4353, max_dd=-0.1931
- Regime 1 Only (Long): total_return=-0.8089, sharpe=-1.3102, max_dd=-0.8490
- Regime 1 Only (Short): total_return=-0.0644, sharpe=-0.3762, max_dd=-0.3618
- Regime 2 Only (Long): total_return=-0.5488, sharpe=-1.0300, max_dd=-0.5704
- Regime 2 Only (Short): total_return=-0.7032, sharpe=-1.3426, max_dd=-0.7152
- Regime 3 Only (Long): total_return=10.7556, sharpe=0.9477, max_dd=-0.0845
- Regime 3 Only (Short): total_return=-0.9748, sharpe=-2.1239, max_dd=-0.9749

### Statistical Significance Tests

### Economic Regime Mapping

| Regime | Economic Interpretation | Recommended Position |
|---------|----------------------|----------------------|

### Economic Interpretation

**Detailed Economic Report:** /Users/remyroche/Documents/Ares/outcomes/regime_economic_relevance_report_20251122_151838.md


---

## Quality Assessment

**Overall Quality Score:** 0.3131 / 1.0
**Quality Level:** Moderate ⚠️
**Recommendation:** The clustering shows moderate quality. Consider parameter tuning.

