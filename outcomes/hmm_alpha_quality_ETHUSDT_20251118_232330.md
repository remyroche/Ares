# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-18T23:23:11.296686
**Data Points:** N/A
**Number of Regimes:** 6
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

**Global Silhouette Score:** -0.0275

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.0160 | 0.0975 | -0.3233 | 0.1607 |
| 1 | -0.0478 | 0.0726 | -0.2788 | 0.0817 |
| 2 | -0.0242 | 0.0340 | -0.1251 | 0.0306 |
| 3 | -0.0291 | 0.0200 | -0.1360 | 0.0177 |
| 4 | 0.0097 | 0.0436 | -0.1185 | 0.0701 |
| 5 | -0.0793 | 0.0960 | -0.2678 | 0.0826 |


### Separation Metrics

- **Davies-Bouldin Index:** 9.5753 (lower is better)
- **Calinski-Harabasz Index:** 188.03 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 11.0404 +/- 3.8885
- **Between-Regime CV:** 155.2261 +/- 446.3189

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 11.5674 |
| 1 | 12.1464 |
| 2 | 8.9397 |
| 3 | 18.5006 |
| 4 | 9.0788 |
| 5 | 6.0097 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 11.1097
- **Between-Regime CV (Mean Return):** 85.7355
- **CV Ratio (Between/Within):** 7.7172

| mean_return | 85.7355 |
| pct_above_target | 0.4442 |
| pct_below_neg_target | 0.4311 |
| pct_target_hits | 0.2439 |
| sharpe | 35.2805 |
| volatility | 0.1356 |


---

## Economic Gap Analysis

### Per-Regime Snapshot

| Regime | Type | Mean Return | Volatility | Sharpe | Max DD | Pct Target Hits |
|--------|------|-------------|------------|--------|--------|-----------------|
| 0 | stable | -0.001345 | 0.008049 | -0.1671 | -84.51% | 28.30% |
| 1 | mean_reverting | -0.001567 | 0.007360 | -0.2129 | -86.09% | 28.30% |
| 2 | stable | -0.000348 | 0.006109 | -0.0569 | -44.12% | 18.92% |
| 3 | stable | 0.000256 | 0.005939 | 0.0430 | -18.42% | 18.38% |
| 4 | stable | 0.000463 | 0.005684 | 0.0814 | -8.42% | 16.11% |
| 5 | mean_reverting | 0.002639 | 0.007676 | 0.3438 | -4.33% | 30.40% |

### Pairwise Economic Spreads

| Regime A | Regime B | Mean Return Spread | Sharpe Spread | Volatility Ratio | Max DD Spread |
|----------|----------|--------------------|---------------|------------------|---------------|
| 0 | 1 | 0.000222 | 0.0458 | 1.094 | 1.58% |
| 0 | 2 | -0.000997 | -0.1102 | 1.318 | -40.40% |
| 0 | 3 | -0.001601 | -0.2101 | 1.355 | -66.09% |
| 0 | 4 | -0.001808 | -0.2485 | 1.416 | -76.10% |
| 0 | 5 | -0.003984 | -0.5109 | 1.049 | -80.18% |
| 1 | 2 | -0.001219 | -0.1559 | 1.205 | -41.98% |
| 1 | 3 | -0.001822 | -0.2559 | 1.239 | -67.67% |
| 1 | 4 | -0.002029 | -0.2943 | 1.295 | -77.67% |
| 1 | 5 | -0.004206 | -0.5567 | 0.959 | -81.76% |
| 2 | 3 | -0.000603 | -0.1000 | 1.029 | -25.69% |
| 2 | 4 | -0.000810 | -0.1383 | 1.075 | -35.70% |
| 2 | 5 | -0.002987 | -0.4007 | 0.796 | -39.79% |
| 3 | 4 | -0.000207 | -0.0383 | 1.045 | -10.01% |
| 3 | 5 | -0.002383 | -0.3008 | 0.774 | -14.09% |
| 4 | 5 | -0.002176 | -0.2624 | 0.741 | -4.09% |

### Statistical Tests (ANOVA / t-tests)

- **ANOVA F-statistic:** 68.1328, p-value=0.0000 (significant)

**Pairwise t-tests:**

| Regime A | Regime B | t-stat | p-value | Cohen's d | Significant |
|----------|----------|--------|---------|-----------|-------------|
| 0 | 1 | 0.7161 | 0.4740 | 0.029 | No |
| 0 | 2 | -3.5511 | 0.0004 | -0.139 | Yes |
| 0 | 3 | -5.8365 | 0.0000 | -0.227 | Yes |
| 0 | 4 | -6.5045 | 0.0000 | -0.257 | Yes |
| 0 | 5 | -13.3825 | 0.0000 | -0.507 | Yes |
| 1 | 2 | -4.4404 | 0.0000 | -0.181 | Yes |
| 1 | 3 | -6.8053 | 0.0000 | -0.275 | Yes |
| 1 | 4 | -7.4738 | 0.0000 | -0.309 | Yes |
| 1 | 5 | -14.4150 | 0.0000 | -0.558 | Yes |
| 2 | 3 | -2.5797 | 0.0099 | -0.100 | Yes |
| 2 | 4 | -3.4025 | 0.0007 | -0.137 | Yes |
| 2 | 5 | -11.4433 | 0.0000 | -0.427 | Yes |
| 3 | 4 | -0.8985 | 0.3690 | -0.036 | No |
| 3 | 5 | -9.3867 | 0.0000 | -0.345 | Yes |
| 4 | 5 | -8.4401 | 0.0000 | -0.316 | Yes |


### Per-Category Coefficient of Variation


| Category | Within CV | Between CV | Ratio | # Features |
|----------|-----------|------------|-------|------------|
| momentum | 13.247 ± 11.849 | 14.859 ± 0.000 | 1.122 | 1 |
| other | 11.336 ± 3.572 | 236.110 ± 561.716 | 20.829 | 12 |
| price | 2.791 ± 1.354 | 46.814 ± 0.445 | 16.774 | 4 |
| volatility | 4.236 ± 1.340 | 28.136 ± 0.000 | 6.642 | 1 |
| volume | 28.066 ± 15.131 | 20.473 ± 3.370 | 0.729 | 2 |

**Interpretation:** Higher CV ratio indicates better regime separation for that feature category.


---

## Balance and Distribution

**Balance Score:** 0.9165 (0-1, higher is better)

- **Smallest Cluster:** 14.99% of total
- **Largest Cluster:** 19.34% of total
- **Cluster Size Std Dev:** 118.82

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 16.67% |
| 1 | 14.99% |
| 2 | 16.35% |
| 3 | 17.66% |
| 4 | 14.99% |
| 5 | 19.34% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.0900 (0-1, higher = fewer transitions)
- **Temporal Smoothness (Raw):** 0.0904
- **Flip-Flop Ratio:** 0.0400 (rapid back-and-forth transitions)
- **Regime Persistence:** 4.52 bars (average duration)

### Transition & Persistence Insights

- **Average Duration:** 4.52 bars
- **Max Duration:** 149.00 bars
- **Min Duration:** 1.00 bars
- **High-persistence regimes:** Regime 0 (p_self=0.92), Regime 1 (p_self=0.77), Regime 2 (p_self=0.71), Regime 3 (p_self=0.69), Regime 4 (p_self=0.68), Regime 5 (p_self=0.87)
- **Flip-flop ratio:** 0.0400
- **Average regime persistence:** 4.52 bars
- **Transition entropy:** 0.6960
- **Regime stickiness:** 0.7753
- **Transition stability score:** 0.6935

**Dominant transition hotspots:**

| From | To | Probability |
|------|----|-------------|
| 4 | 3 | 0.176 |
| 2 | 3 | 0.152 |
| 3 | 2 | 0.144 |
| 3 | 4 | 0.140 |
| 4 | 5 | 0.133 |


### Transition Probability Matrix

This matrix shows the probability of transitioning from one regime to another:


### Regime Duration Analysis

**Average Regime Durations:**

| Regime | Mean Duration | Std Duration | Min Duration | Max Duration |
|--------|---------------|--------------|--------------|--------------|
| 0 | 12.5 | 19.5 | 1 | 149 |
| 1 | 4.4 | 4.2 | 1 | 23 |
| 2 | 3.5 | 4.7 | 1 | 41 |
| 3 | 3.3 | 2.7 | 1 | 18 |
| 4 | 3.1 | 2.8 | 1 | 21 |
| 5 | 7.8 | 7.6 | 1 | 48 |

- **Duration Stability Score:** 0.399 (higher = more consistent durations)


---

## Per-Regime Analysis

### Regime 0 (stable)

**Size:** 1304 samples (16.67%)

**Performance Metrics:**
- Mean Return: -0.0013449987424793137
- Volatility: 0.008048742131268738
- Sharpe Ratio: -0.1671066799552962
- Skewness: -0.5831449376245756
- Max Drawdown: -0.8451266052770086

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.10506134969325154
- Pct < -1.0% (Shorts): 0.17791411042944785
- Pct Target Hits: 0.28297546012269936

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 35.157720343113176
- Win Rate (Long Bias): 0.37127371142509236
- Return per Vol: 0.1671066799552962
- Profit Factor: 0.6996796943225082

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.0013449987424793137
- volatility: 0.008048742131268738
- stability_coefficient: 0.14318040250159292

### Regime 1 (mean_reverting)

**Size:** 1173 samples (14.99%)

**Performance Metrics:**
- Mean Return: -0.001566721927437893
- Volatility: 0.007359779079438082
- Sharpe Ratio: -0.2128761879468431
- Skewness: -0.3425199086572055
- Max Drawdown: -0.8609055136688493

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.08951406649616368
- Pct < -1.0% (Shorts): 0.19352088661551578
- Pct Target Hits: 0.2830349531116795

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 38.45698513498098
- Win Rate (Long Bias): 0.31626505912355746
- Return per Vol: 0.2128761879468431
- Profit Factor: 0.7333553514793323

**Regime-Specific Characteristics:**

- reversion_center: -0.001566721927437893
- reversion_speed: 180.35949429484043
- reversion_range: 0.0048372354624111455

### Regime 2 (stable)

**Size:** 1279 samples (16.35%)

**Performance Metrics:**
- Mean Return: -0.00034779726209085136
- Volatility: 0.00610889219276634
- Sharpe Ratio: -0.05693294204303406
- Skewness: -0.13133496048775695
- Max Drawdown: -0.44115462192168564

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.08522283033620015
- Pct < -1.0% (Shorts): 0.10398749022673964
- Pct Target Hits: 0.1892103205629398

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 30.97292988965455
- Win Rate (Long Bias): 0.4504132207600061
- Return per Vol: 0.05693294204303406
- Profit Factor: 0.9433248543685703

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.00034779726209085136
- volatility: 0.00610889219276634
- stability_coefficient: 0.053866336712674885

### Regime 3 (stable)

**Size:** 1382 samples (17.66%)

**Performance Metrics:**
- Mean Return: 0.0002555631616434746
- Volatility: 0.005939272867369201
- Sharpe Ratio: 0.04302936139172787
- Skewness: -0.20011031451528485
- Max Drawdown: -0.18423450032995153

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.09479015918958032
- Pct < -1.0% (Shorts): 0.08900144717800289
- Pct Target Hits: 0.1837916063675832

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 30.945130746933145
- Win Rate (Long Bias): 0.5157480286899064
- Return per Vol: 0.04302936139172787
- Profit Factor: 0.9935206626837174

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0002555631616434746
- volatility: 0.005939272867369201
- stability_coefficient: 0.04125438013083712

### Regime 4 (stable)

**Size:** 1173 samples (14.99%)

**Performance Metrics:**
- Mean Return: 0.00046255646248155244
- Volatility: 0.0056842385431798755
- Sharpe Ratio: 0.08137525854914784
- Skewness: 0.05227761207171872
- Max Drawdown: -0.08415746497949574

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.09548167092924126
- Pct < -1.0% (Shorts): 0.06564364876385337
- Pct Target Hits: 0.16112531969309463

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 28.34597635604884
- Win Rate (Long Bias): 0.5925925889147561
- Return per Vol: 0.08137525854914784
- Profit Factor: 1.1085652986827128

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.00046255646248155244
- volatility: 0.0056842385431798755
- stability_coefficient: 0.07525179980847255

### Regime 5 (mean_reverting)

**Size:** 1513 samples (19.34%)

**Performance Metrics:**
- Mean Return: 0.002638812668077317
- Volatility: 0.0076756152033088
- Sharpe Ratio: 0.3437916381149671
- Skewness: 0.3088532082785846
- Max Drawdown: -0.04330100342788538

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.232650363516193
- Pct < -1.0% (Shorts): 0.0713813615333774
- Pct Target Hits: 0.30403172504957043

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 39.61007390110368
- Win Rate (Long Bias): 0.7652173887874478
- Return per Vol: 0.3437916381149671
- Profit Factor: 1.5766019854843314

**Regime-Specific Characteristics:**

- reversion_center: 0.002638812668077317
- reversion_speed: 173.0024558162939
- reversion_range: 0.005047922544090231


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5378997221575074

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Economic Relevance Analysis

### Strategy Performance Summary

### Statistical Significance Tests

### Economic Regime Mapping

| Regime | Economic Interpretation | Recommended Position |
|---------|----------------------|----------------------|

### Economic Interpretation

**Detailed Economic Report:** /Users/remyroche/Documents/Ares/outcomes/regime_economic_relevance_report_20251118_232328.md


---

## Quality Assessment

**Overall Quality Score:** 0.7403 / 1.0
**Quality Level:** Excellent ✅
**Recommendation:** The clustering shows excellent quality. Proceed with confidence.

