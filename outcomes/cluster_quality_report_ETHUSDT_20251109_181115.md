# Cluster Quality Assessment Report

**Symbol:** ETHUSDT  
**Generated:** 2025-11-09T18:10:30.515723
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

**Global Silhouette Score:** -0.0183

#### Per-Cluster Silhouette Scores

| Cluster | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 0 | 0.0961 | 0.0918 | -0.1970 | 0.2930 |
| 1 | -0.2560 | 0.1284 | -0.5407 | 0.0231 |
| 2 | 0.2287 | 0.1338 | -0.2558 | 0.4405 |
| 3 | 0.1554 | 0.0803 | -0.0916 | 0.3315 |
| 4 | -0.2615 | 0.1378 | -0.5985 | 0.0096 |
| 5 | 0.1982 | 0.1373 | -0.2656 | 0.4116 |


### Separation Metrics

- **Davies-Bouldin Index:** 3.4266 (lower is better)
- **Calinski-Harabasz Index:** 1834.71 (higher is better)

<!-- *** REVERTED: Kept original section name *** -->
### Coefficient of Variation

- **Within-Regime CV:** 15.5198 +/- 17.8817
- **Between-Regime CV:** 61.4952 +/- 92.9269

#### Per-Regime CV Values

| Regime | CV |
|--------|----|
| 0 | 3.5666 |
| 1 | 6.9554 |
| 2 | 49.1228 |
| 3 | 2.0983 |
| 4 | 29.7046 |
| 5 | 1.6708 |


### Economic Coefficient of Variation

- **Average Within-Regime CV (Forward Returns):** 33.7081
- **Between-Regime CV (Mean Return):** 7.2318
- **CV Ratio (Between/Within):** 0.2145

| mean_return | 7.2318 |
| pct_above_target | 0.5957 |
| pct_below_neg_target | 0.5887 |
| pct_target_hits | 0.5798 |
| sharpe | 5.7218 |
| volatility | 0.3889 |


---

## Balance and Distribution

**Balance Score:** 0.7894 (0-1, higher is better)

- **Smallest Cluster:** 10.55% of total
- **Largest Cluster:** 24.40% of total
- **Cluster Size Std Dev:** 1506.87

### Cluster Size Distribution

| Cluster Index | Size (%) |
|---------------|----------|
| 0 | 15.92% |
| 1 | 24.40% |
| 2 | 10.55% |
| 3 | 17.96% |
| 4 | 18.48% |
| 5 | 12.69% |


---

## Temporal Analysis

- **Temporal Smoothness (Penalized):** 0.8356 (0-1, higher = fewer transitions)
- **Regime Persistence:** 6.08 bars (average duration)


---

## Per-Regime Analysis

### Regime 0 (mean_reverting)

**Size:** 5397 samples (15.92%)

**Performance Metrics:**
- Mean Return: 0.00017335344455204904
- Volatility: 0.004420712124556303
- Sharpe Ratio: 0.03921390953624663
- Skewness: 0.6168468594551086
- Max Drawdown: -0.17368869610761542

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.048360200111172875
- Pct < -1.0% (Shorts): 0.03372243839169909
- Pct Target Hits: 0.08208263850287197

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 18.56773696689726
- Win Rate (Long Bias): 0.5891647783753448
- Return per Vol: 0.03921390953624663
- Profit Factor: 1.143881916999817

**Regime-Specific Characteristics:**

- reversion_center: 0.00017335344455204904
- reversion_speed: 342.86498854455306
- reversion_range: 0.0033218322787433863

### Regime 1 (stable)

**Size:** 8271 samples (24.40%)

**Performance Metrics:**
- Mean Return: -0.0005018180818296969
- Volatility: 0.0070312488824129105
- Sharpe Ratio: -0.07136968394266315
- Skewness: -0.8253779411315918
- Max Drawdown: -0.9878047581031943

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.08269858541893363
- Pct < -1.0% (Shorts): 0.12102526901221133
- Pct Target Hits: 0.20372385443114496

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 28.974059781421555
- Win Rate (Long Bias): 0.40593471610831694
- Return per Vol: 0.07136968394266315
- Profit Factor: 0.8069528937339783

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.0005018180818296969
- volatility: 0.0070312488824129105
- stability_coefficient: 0.06661549905187282

### Regime 2 (stable)

**Size:** 3577 samples (10.55%)

**Performance Metrics:**
- Mean Return: -0.0003510032838676125
- Volatility: 0.011773470789194107
- Sharpe Ratio: -0.029813065351697643
- Skewness: -0.7538506984710693
- Max Drawdown: -0.8300425865810136

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.19849035504612805
- Pct < -1.0% (Shorts): 0.19765166340508805
- Pct Target Hits: 0.3961420184512161

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 33.647001117784306
- Win Rate (Long Bias): 0.501058573188224
- Return per Vol: 0.029813065351697643
- Profit Factor: 0.9095267057418823

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -0.0003510032838676125
- volatility: 0.011773470789194107
- stability_coefficient: 0.028950060250235273

### Regime 3 (stable)

**Size:** 6087 samples (17.96%)

**Performance Metrics:**
- Mean Return: -3.819819539785385e-05
- Volatility: 0.003680651308968663
- Sharpe Ratio: -0.010378104800819172
- Skewness: -0.051908254623413086
- Max Drawdown: -0.3882288289674672

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.020699852143913258
- Pct < -1.0% (Shorts): 0.03170691637916872
- Pct Target Hits: 0.05240676852308197

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 14.238445830751834
- Win Rate (Long Bias): 0.3949843184819137
- Return per Vol: 0.010378104800819172
- Profit Factor: 0.9436030387878418

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: -3.819819539785385e-05
- volatility: 0.003680651308968663
- stability_coefficient: 0.010271774935024135

### Regime 4 (stable)

**Size:** 6266 samples (18.48%)

**Performance Metrics:**
- Mean Return: 0.0005203087348490953
- Volatility: 0.007140425033867359
- Sharpe Ratio: 0.07286802389398725
- Skewness: 0.3643336594104767
- Max Drawdown: -0.1737900696574725

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.12208745611235237
- Pct < -1.0% (Shorts): 0.08107245451643792
- Pct Target Hits: 0.2031599106287903

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 28.452071300114834
- Win Rate (Long Bias): 0.6009426521873474
- Return per Vol: 0.07286802389398725
- Profit Factor: 1.219534158706665

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.0005203087348490953
- volatility: 0.007140425033867359
- stability_coefficient: 0.06791903786746979

### Regime 5 (stable)

**Size:** 4300 samples (12.69%)

**Performance Metrics:**
- Mean Return: 0.000524848117493093
- Volatility: 0.010041678324341774
- Sharpe Ratio: 0.0522669665641306
- Skewness: 0.41448938846588135
- Max Drawdown: -0.38019964420186686

**Target-Based Metrics:**
- Pct > 1.0% (Longs): 0.17651162790697675
- Pct < -1.0% (Shorts): 0.15023255813953487
- Pct Target Hits: 0.3267441860465116

**Risk-Adjusted Metrics:**
- Risk-Adj Target Hits: 32.53879908856077
- Win Rate (Long Bias): 0.5402135214783501
- Return per Vol: 0.0522669665641306
- Profit Factor: 1.1024351119995117

**Regime-Specific Characteristics:**

- stability_regime: low_volatility
- mean_return: 0.000524848117493093
- volatility: 0.010041678324341774
- stability_coefficient: 0.049670917941793895


---

## Predictive Power

**Cross-Validation Accuracy:** 0.5015783107649644

This metric indicates how well the clustering can predict regime assignments on unseen data.

---

## Quality Assessment

**Overall Quality Score:** 0.6024 / 1.0
**Quality Level:** Good ✅
**Recommendation:** The clustering shows good quality. Suitable for most applications.

