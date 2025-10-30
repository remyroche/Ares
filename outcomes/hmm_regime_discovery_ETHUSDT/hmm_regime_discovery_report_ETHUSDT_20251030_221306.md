# HMM Regime Discovery Comprehensive Report

**Generated**: 2025-10-30T22:13:06.279520  
**Report ID**: `hmm_regime_discovery_ETHUSDT_1h_20251030_221306`
**Model**: Hidden Markov Model (HMM) with 4 States

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Symbol** | ETHUSDT |
| **Exchange** | binance |
| **Timeframe** | 1h |
| **Processing Time** | 1.40 seconds |
| **Success Status** | ✅ TARGETS MET |
| **Regimes Discovered** | 4 |
| **Quality Score** | 0.656 |
| **Noise Ratio** | 0.0% |

### Optimization Targets Achievement

**Targets Met** (3/3):
- ✅ Silhouette Score (0.127)
- ✅ Temporal Smoothness (0.967)
- ✅ Cluster Count (4)

---

## 🔍 Regime Discovery Results

### Cluster Statistics
- **Total Regimes**: 4
- **Noise Points**: 0.0% of total samples (GMM: 0.0% - no noise)
- **Average Regime Size**: 120 samples per regime
- **Balance Score**: 0.500 (higher is better)

### Regime Distribution
| Regime ID | Sample Count | Percentage |
|-----------|--------------|------------|
| **Regime 0** | 351 | 73.1% |
| **Regime 2** | 129 | 26.9% |


---

## 🔄 HMM Transition Matrix

The transition matrix shows the probability of moving from one regime to another.
Higher diagonal values indicate regime persistence (states tend to stay).

| From \ To | State 0 | State 1 | State 2 | State 3 |
|-----------|---------|---------|---------|---------|
| **State 0** | **0.981** | 0.000 | 0.019 | 0.000 |
| **State 1** | 0.000 | **0.950** | 0.050 | 0.000 |
| **State 2** | 0.054 | 0.000 | **0.939** | 0.008 |
| **State 3** | 0.000 | 0.000 | 0.056 | **0.944** |


### Transition Analysis
- **Average Regime Persistence**: 95.3% (probability of staying in same state)
- **Regime-Specific Persistence**:
  - State 0: 98.1% persistence
  - State 1: 95.0% persistence
  - State 2: 93.9% persistence
  - State 3: 94.4% persistence


---

## 💰 Economic Performance Per Regime

This section evaluates trading performance within each regime.

| Regime | Sharpe Ratio | Win Rate | Expected Return | Max Drawdown | Volatility Clustering |
|--------|--------------|----------|-----------------|--------------|----------------------|
| 🔴 **Regime 0** 🟢 | -0.971 | 53.3% | -0.0001 | -16.72% | 0.206 |
| 🟢 **Regime 2** 🟢 | 3.497 | 48.1% | 0.0002 | -3.32% | 0.275 |


**Reliability Legend**:
- 🟢 RELIABLE: N ≥ 100 samples (statistics are trustworthy)
- 🟡 MARGINAL: 50 ≤ N < 100 samples (use with caution)
- 🔴 UNRELIABLE: N < 50 samples (DO NOT TRADE on these stats)


### Bootstrap Confidence Intervals (95% CI)

Statistical validation using block bootstrap (1000 iterations):

| Regime | Sharpe CI | Mean Return CI | Samples | Reliability |
|--------|-----------|----------------|---------|-------------|
| **Regime 0** | [-10.56, 7.78] | [-0.000805, 0.000471] | 351 | N=351 ≥ 100 (RELIABLE) |
| **Regime 2** | [-4.47, 15.09] | [-0.000238, 0.000926] | 129 | N=129 ≥ 100 (RELIABLE) |


**Trading Rule**: Only act on regimes where the **lower bound of Sharpe CI > 0.5**



### Return Distribution Details

#### Regime 0 Return Statistics
- **Mean Return**: -0.000067 (-0.0067%)
- **Median Return**: 0.000428
- **Std Dev**: 0.006503
- **Skewness**: -0.9829 (left-tailed)
- **Kurtosis**: 6.2578 (fat-tailed)
- **Range**: [-0.0381, 0.0237]
- **IQR**: [-0.0029, 0.0032]
- **Total Return**: -0.0237
- **Samples**: 351

#### Regime 2 Return Statistics
- **Mean Return**: 0.000204 (0.0204%)
- **Median Return**: -0.000163
- **Std Dev**: 0.005466
- **Skewness**: 0.3278 (right-tailed)
- **Kurtosis**: 2.5359 (fat-tailed)
- **Range**: [-0.0179, 0.0214]
- **IQR**: [-0.0026, 0.0029]
- **Total Return**: 0.0263
- **Samples**: 129


---

## 📈 Comprehensive Quality Metrics (from cluster_quality_assessor.py)

### Overall Quality Score: 0.656

**Quality Score Breakdown:**

| Metric | Normalized Value | Weight | Contribution |
|--------|------------------|--------|--------------|
| **CV Ratio** | 0.3459 | 30.00% | 0.1038 |
| **Silhouette Score** | 0.5633 | 20.00% | 0.1127 |
| **Temporal Smoothness** | 0.9666 | 30.00% | 0.2900 |
| **Balance Score** | 0.4997 | 10.00% | 0.0500 |
| **Noise Ratio (inverted)** | 1.0000 | 10.00% | 0.1000 |

**Total Weight**: 100.00%  
**Weighted Score**: 0.6564

---

### Core Clustering Metrics
- **Silhouette Score**: 0.1267 (range: [-1, 1], higher is better)
  - *Interpretation*: Fair cluster separation
- **Calinski-Harabasz Score**: 21.99 (higher is better)
- **Davies-Bouldin Score**: 2.9181 (lower is better)

### Coefficient of Variation (CV) Metrics

**CV Ratio**: 0.3608

- **Within-Regime CV**: 18.210890 (lower = more cohesive regimes)
- **Between-Regime CV**: 6.570094 (higher = better separation)
- **CV Ratio Interpretation**: Fair separation

### Temporal Metrics

- **Temporal Smoothness**: 0.9666 (range: [0, 1], higher = more stable over time)
  - *Interpretation*: Excellent temporal stability

- **Regime Persistence**: 29.94 periods (average duration)

### Balance Metrics

- **Balance Score**: 0.4997 (range: [0, 1], higher = more balanced)
- **Min Cluster Size**: 375.0% of total samples
- **Max Cluster Size**: 6520.8% of total samples
- **Cluster Size Std Dev**: 120.14

### Per-Regime Details

#### 🎯 Regime 0
- **Size**: 313
- **Percentage**: 65.2083
- **Feature Coefficient Of Variation**: {'PC_1': 3.3954104788687816, 'PC_2': 11.824218075985657, 'PC_3': 4.2025755227336745, 'PC_4': 12.331869806823049, 'PC_5': 12.088012954487496, 'PC_6': 3.2111906773594927, 'PC_7': 1.7427858127200482, 'PC_8': 14.775041889221956, 'PC_9': 5.06545378652464, 'PC_10': 205.08187386047467, 'PC_11': 4.631254716003879, 'PC_12': 285.6876794419423, 'PC_13': 8.197934539733145, 'PC_14': 12.195048664394125, 'PC_15': 225.00392906188137, 'PC_16': 36.23116363797585, 'PC_17': 57.25570195778294, 'PC_18': 108.19613295350925, 'PC_19': 10.725466589896376, 'PC_20': 27.989785019486217}
- **Mean Cv**: 52.4916
- **Std Cv**: 82.8713
- **Balance Contribution**: 2.6083
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 1
- **Size**: 20
- **Percentage**: 4.1667
- **Feature Coefficient Of Variation**: {'PC_1': 0.4559620237279592, 'PC_2': 2.3166362815967227, 'PC_3': 0.7923711064635417, 'PC_4': 0.7140898598385061, 'PC_5': 8.416181161802823, 'PC_6': 0.7870629449645874, 'PC_7': 0.616528834340589, 'PC_8': 0.27558695226010727, 'PC_9': 0.4564116840628508, 'PC_10': 0.21439709335734777, 'PC_11': 0.3571741655099636, 'PC_12': 0.5490176401905887, 'PC_13': 0.3389636195388783, 'PC_14': 0.367065940237558, 'PC_15': 0.7859709347813473, 'PC_16': 2.2982133886801797, 'PC_17': 0.7880724103653397, 'PC_18': 1.7302333580048823, 'PC_19': 2.8075740742969884, 'PC_20': 8.56418743539043}
- **Mean Cv**: 1.6816
- **Std Cv**: 2.3847
- **Balance Contribution**: 0.1667
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 2
- **Size**: 129
- **Percentage**: 26.8750
- **Feature Coefficient Of Variation**: {'PC_1': 10.345947444622832, 'PC_2': 2.8530635337491685, 'PC_3': 1.566399133163586, 'PC_4': 7.763131460247731, 'PC_5': 3.726819439965714, 'PC_6': 1.2386708309404275, 'PC_7': 1.5217219555858534, 'PC_8': 143.55859430812066, 'PC_9': 4.518333698576203, 'PC_10': 18.302826586801412, 'PC_11': 40.352334940538945, 'PC_12': 15.441411505131333, 'PC_13': 3.162528620322908, 'PC_14': 3.6854697848767084, 'PC_15': 10.834971322229805, 'PC_16': 20.0049425847493, 'PC_17': 7.958912385663559, 'PC_18': 9.457370815534738, 'PC_19': 6.755340573191931, 'PC_20': 16.244599047866796}
- **Mean Cv**: 16.4647
- **Std Cv**: 30.5030
- **Balance Contribution**: 1.0750
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 3
- **Size**: 18
- **Percentage**: 3.7500
- **Feature Coefficient Of Variation**: {'PC_1': 0.3105186568623912, 'PC_2': 1.0120412316915552, 'PC_3': 7.294134520222398, 'PC_4': 0.7069184913779121, 'PC_5': 3.0788928552825485, 'PC_6': 0.8029121544038009, 'PC_7': 0.8252865940572226, 'PC_8': 0.20519431746468425, 'PC_9': 1.4324932633420373, 'PC_10': 0.48827423137498493, 'PC_11': 0.7343129671161766, 'PC_12': 2.936961892475477, 'PC_13': 6.371399674432772, 'PC_14': 0.9366881961415553, 'PC_15': 1.8614129487854065, 'PC_16': 2.95705153201155, 'PC_17': 1.6071513540496616, 'PC_18': 2.2597446959206815, 'PC_19': 3.396677822025088, 'PC_20': 4.895529974133134}
- **Mean Cv**: 2.2057
- **Std Cv**: 1.9583
- **Balance Contribution**: 0.1500
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

---

## 🔧 Feature Engineering Details

### Correlation-Based Feature Reduction
- **Original Features**: 300
- **Reduced Features**: 171
- **Features Removed**: 129 (43.0%)
- **Correlation Threshold**: 0.85

### Dimensionality Reduction
- **PCA Applied**: ✅ Yes
- **Total Variance Explained**: 62.1%
- **Number of Principal Components**: 20
- **Top 5 Components Variance**: 12.4%, 7.3%, 5.2%, 4.2%, 3.6%

### HMM Model Parameters
- **Number of Hidden States**: 4
- **Covariance Type**: full
- **Maximum Iterations**: 100
- **Random State**: 42
- **Converged**: ✅ Yes

---

## 🎯 Optimization Goals & Targets

This GMM regime discovery run was guided by the following optimization goals from `clustering_optimization_goals.py`:

### Cluster Configuration Targets
- **Target Cluster Count**: 4-5 clusters
- **Minimum Cluster Size**: 2.0% of total samples
- **Maximum Cluster Size**: 20.0% of total samples

### Quality Targets
- **Minimum Silhouette Score**: 0.10
- **Target Silhouette Score**: 0.30
- **Minimum Temporal Smoothness**: 0.20
- **Target Temporal Smoothness**: 0.40
- **Minimum CV Score**: 1.20
- **Target CV Score**: 2.00

### Economic Targets (for future integration)
- **Minimum Sharpe Ratio**: 0.50
- **Target Sharpe Ratio**: 1.50
- **Max Drawdown Threshold**: 30.0%

---

## 📊 Quality Score Interpretation

**Score: 0.656**

| Score Range | Interpretation |
|-------------|----------------|
| 0.70 - 1.00 | Excellent: Highly distinct regimes with strong temporal stability |
| 0.50 - 0.70 | Good: Clear regime separation with reasonable stability |
| 0.30 - 0.50 | Moderate: Some regime distinction, room for improvement |
| 0.00 - 0.30 | Poor: Weak regime separation, consider parameter tuning |

**Current Status**: Good

---

*Generated by HMM Regime Discovery at 2025-10-30T22:13:06.279520*

