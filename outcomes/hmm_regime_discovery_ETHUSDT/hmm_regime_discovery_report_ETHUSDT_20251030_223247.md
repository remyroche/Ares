# HMM Regime Discovery Comprehensive Report

**Generated**: 2025-10-30T22:32:47.173278  
**Report ID**: `hmm_regime_discovery_ETHUSDT_1h_20251030_223247`
**Model**: Hidden Markov Model (HMM) with 4 States

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Symbol** | ETHUSDT |
| **Exchange** | binance |
| **Timeframe** | 1h |
| **Processing Time** | 314.69 seconds |
| **Success Status** | ⚠️ PARTIAL SUCCESS |
| **Regimes Discovered** | 4 |
| **Quality Score** | 0.586 |
| **Noise Ratio** | 0.0% |

### Optimization Targets Achievement

**Targets Met** (2/3):
- ✅ Temporal Smoothness (0.892)
- ✅ Cluster Count (4)

**Targets Not Met**:
- ❌ Silhouette Score (0.042 < 0.10)

---

## 🔍 Regime Discovery Results

### Cluster Statistics
- **Total Regimes**: 4
- **Noise Points**: 0.0% of total samples (GMM: 0.0% - no noise)
- **Average Regime Size**: 6544 samples per regime
- **Balance Score**: 0.760 (higher is better)

### Regime Distribution
| Regime ID | Sample Count | Percentage |
|-----------|--------------|------------|
| **Regime 0** | 5,886 | 22.5% |
| **Regime 1** | 7,694 | 29.4% |
| **Regime 2** | 9,048 | 34.6% |
| **Regime 3** | 3,549 | 13.6% |


---

## 🔄 HMM Transition Matrix

The transition matrix shows the probability of moving from one regime to another.
Higher diagonal values indicate regime persistence (states tend to stay).

| From \ To | State 0 | State 1 | State 2 | State 3 |
|-----------|---------|---------|---------|---------|
| **State 0** | **0.940** | 0.000 | 0.039 | 0.021 |
| **State 1** | 0.000 | **0.888** | 0.047 | 0.065 |
| **State 2** | 0.037 | 0.002 | **0.918** | 0.043 |
| **State 3** | 0.006 | *0.234* | 0.040 | **0.720** |


### Transition Analysis
- **Average Regime Persistence**: 86.7% (probability of staying in same state)
- **Regime-Specific Persistence**:
  - State 0: 94.0% persistence
  - State 1: 88.8% persistence
  - State 2: 91.8% persistence
  - State 3: 72.0% persistence


---

## 💰 Economic Performance Per Regime

This section evaluates trading performance within each regime.

| Regime | Sharpe Ratio | Win Rate | Expected Return | Max Drawdown | Volatility Clustering |
|--------|--------------|----------|-----------------|--------------|----------------------|
| 🔴 **Regime 0** 🟢 | 0.445 | 50.2% | 0.0000 | -14.09% | 0.039 |
| 🟢 **Regime 1** 🟢 | 2.123 | 51.0% | 0.0001 | -16.48% | 0.073 |
| 🟢 **Regime 2** 🟢 | 1.477 | 51.0% | 0.0001 | -19.98% | 0.044 |
| 🔴 **Regime 3** 🟢 | 0.379 | 50.0% | 0.0001 | -76.32% | 0.091 |


**Reliability Legend**:
- 🟢 RELIABLE: N ≥ 100 samples (statistics are trustworthy)
- 🟡 MARGINAL: 50 ≤ N < 100 samples (use with caution)
- 🔴 UNRELIABLE: N < 50 samples (DO NOT TRADE on these stats)


### Bootstrap Confidence Intervals (95% CI)

Statistical validation using block bootstrap (500 iterations):

| Regime | Sharpe CI | Mean Return CI | Samples | Reliability |
|--------|-----------|----------------|---------|-------------|
| **Regime 0** | [-1.02, 2.03] | [-0.000025, 0.000049] | 5886 | N=5886 ≥ 100 (🟢 SUFFICIENT SAMPLE SIZE) |
| **Regime 1** | [0.33, 3.54] | [0.000018, 0.000190] | 7694 | N=7694 ≥ 100 (🟢 SUFFICIENT SAMPLE SIZE) |
| **Regime 2** | [-0.01, 3.02] | [-0.000000, 0.000137] | 9048 | N=9048 ≥ 100 (🟢 SUFFICIENT SAMPLE SIZE) |
| **Regime 3** | [-3.07, 3.69] | [-0.000480, 0.000577] | 3549 | N=3549 ≥ 100 (🟢 SUFFICIENT SAMPLE SIZE) |


**Trading Rule**: Only act on regimes where the **lower bound of Sharpe CI > 0.5** OR **(Sharpe ≥ 1.0 AND mean return CI lower > 0)**



### 🎯 Production-Ready Trading Status

Conservative evaluation for live trading based on strict statistical criteria:

| Regime | Status | Samples | Sharpe (CI Lower) | Mean Return (CI Lower) | Decision |
|--------|--------|---------|-------------------|------------------------|----------|
| **Regime 0** | 🔴 NO TRADE | 5886 | 0.45 (-1.02) | -0.000025 | Do NOT trade (unreliable) |
| **Regime 1** | 🟢 LONG | 7694 | 2.12 (0.33) | 0.000018 | **Trade with 0.5x size** (scale by vol) |
| **Regime 2** | 🔴 NO TRADE | 9048 | 1.48 (-0.01) | -0.000000 | Do NOT trade (unreliable) |
| **Regime 3** | 🔴 NO TRADE | 3549 | 0.38 (-3.07) | -0.000480 | Do NOT trade (unreliable) |


**Production Rules**:
1. ✅ **N ≥ 100**: Sufficient sample size for statistical reliability
2. ✅ **Sharpe CI lower ≥ 0.5** OR **(Sharpe ≥ 1.0 AND Mean Return CI lower > 0)**: Edge survives conservative CI
3. ⚠️ **Conservative sizing**: Use 0.5x max position, scale by volatility
4. ⚠️ **Do NOT short**: Small negative regimes are unreliable — stay flat instead



### Return Distribution Details

#### Regime 0 Return Statistics
- **Mean Return**: 0.000011 (0.0011%)
- **Median Return**: 0.000011
- **Std Dev**: 0.002255
- **Skewness**: -0.0047 (left-tailed)
- **Kurtosis**: 0.5898 (fat-tailed)
- **Range**: [-0.0081, 0.0083]
- **IQR**: [-0.0013, 0.0014]
- **Total Return**: 0.0631
- **Samples**: 5886

#### Regime 1 Return Statistics
- **Mean Return**: 0.000113 (0.0113%)
- **Median Return**: 0.000125
- **Std Dev**: 0.005001
- **Skewness**: -0.0726 (left-tailed)
- **Kurtosis**: 0.2641 (fat-tailed)
- **Range**: [-0.0184, 0.0169]
- **IQR**: [-0.0029, 0.0033]
- **Total Return**: 0.8728
- **Samples**: 7694

#### Regime 2 Return Statistics
- **Mean Return**: 0.000067 (0.0067%)
- **Median Return**: 0.000095
- **Std Dev**: 0.004256
- **Skewness**: -0.0918 (left-tailed)
- **Kurtosis**: 0.8262 (fat-tailed)
- **Range**: [-0.0182, 0.0159]
- **IQR**: [-0.0024, 0.0026]
- **Total Return**: 0.6075
- **Samples**: 9048

#### Regime 3 Return Statistics
- **Mean Return**: 0.000059 (0.0059%)
- **Median Return**: 0.000005
- **Std Dev**: 0.014680
- **Skewness**: -0.1670 (left-tailed)
- **Kurtosis**: 3.9607 (fat-tailed)
- **Range**: [-0.1174, 0.0956]
- **IQR**: [-0.0084, 0.0093]
- **Total Return**: 0.2108
- **Samples**: 3549


---

## 📈 Comprehensive Quality Metrics (from cluster_quality_assessor.py)

### Overall Quality Score: 0.586

**Quality Score Breakdown:**

| Metric | Normalized Value | Weight | Contribution |
|--------|------------------|--------|--------------|
| **CV Ratio** | 0.1271 | 30.00% | 0.0381 |
| **Silhouette Score** | 0.5209 | 20.00% | 0.1042 |
| **Temporal Smoothness** | 0.8923 | 30.00% | 0.2677 |
| **Balance Score** | 0.7605 | 10.00% | 0.0760 |
| **Noise Ratio (inverted)** | 1.0000 | 10.00% | 0.1000 |

**Total Weight**: 100.00%  
**Weighted Score**: 0.5860

---

### Core Clustering Metrics
- **Silhouette Score**: 0.0418 (range: [-1, 1], higher is better)
  - *Interpretation*: Fair cluster separation
- **Calinski-Harabasz Score**: 3903.53 (higher is better)
- **Davies-Bouldin Score**: 2.4434 (lower is better)

### Coefficient of Variation (CV) Metrics

**CV Ratio**: 0.1278

- **Within-Regime CV**: 164.877448 (lower = more cohesive regimes)
- **Between-Regime CV**: 21.070828 (higher = better separation)
- **CV Ratio Interpretation**: Fair separation

### Temporal Metrics

- **Temporal Smoothness**: 0.8923 (range: [0, 1], higher = more stable over time)
  - *Interpretation*: Excellent temporal stability

- **Regime Persistence**: 9.28 periods (average duration)

### Balance Metrics

- **Balance Score**: 0.7605 (range: [0, 1], higher = more balanced)
- **Min Cluster Size**: 1355.8% of total samples
- **Max Cluster Size**: 3456.5% of total samples
- **Cluster Size Std Dev**: 2061.28

### Per-Regime Details

#### 🎯 Regime 0
- **Size**: 5886
- **Percentage**: 22.4854
- **Feature Coefficient Of Variation**: {'returns': 39.57977541256544, 'sma_5': 0.5318858667393915, 'volatility_5': 0.20945928344298098, 'volatility_10': 0.1646928162058729, 'volatility_20': 0.16002684975792816, 'hl_ratio': 0.4604586944434478, 'price_position': 33.92659615723703, 'volume_ma_5': 0.3282173692440941, 'volume_ma_20': 0.4600736854517023, 'volume_ratio': 3.14527806673294}
- **Mean Cv**: 7.8966
- **Std Cv**: 14.5079
- **Balance Contribution**: 0.8994
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 1
- **Size**: 7694
- **Percentage**: 29.3922
- **Feature Coefficient Of Variation**: {'returns': 108.26009266392917, 'sma_5': 2.0168884731553183, 'volatility_5': 3.6856285759391603, 'volatility_10': 1.93972411984075, 'volatility_20': 1.2927107580643795, 'hl_ratio': 19.28101628904204, 'price_position': 82.72497947025802, 'volume_ma_5': 3.427972130234598, 'volume_ma_20': 1.6992372917735288, 'volume_ratio': 1.6513408936835627}
- **Mean Cv**: 22.5980
- **Std Cv**: 37.2414
- **Balance Contribution**: 1.1757
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 2
- **Size**: 9048
- **Percentage**: 34.5647
- **Feature Coefficient Of Variation**: {'returns': 4107.230531462804, 'sma_5': 18.96907103135456, 'volatility_5': 1.2266389437172374, 'volatility_10': 0.7873439019032296, 'volatility_20': 0.6200980892379525, 'hl_ratio': 2.0907069407545857, 'price_position': 328.57788918436023, 'volume_ma_5': 1.232505483075574, 'volume_ma_20': 1.1317812861773546, 'volume_ratio': 13.034532097483432}
- **Mean Cv**: 447.4901
- **Std Cv**: 1223.7390
- **Balance Contribution**: 1.3826
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 3
- **Size**: 3549
- **Percentage**: 13.5577
- **Feature Coefficient Of Variation**: {'returns': 1770.262361190468, 'sma_5': 3.507344383360521, 'volatility_5': 1.1503399214443188, 'volatility_10': 1.2008439947027967, 'volatility_20': 1.3773610056861298, 'hl_ratio': 1.1392645479527685, 'price_position': 32.602529154343785, 'volume_ma_5': 1.078471163153671, 'volume_ma_20': 1.3643809753774347, 'volume_ratio': 1.5678628207982723}
- **Mean Cv**: 181.5251
- **Std Cv**: 529.6605
- **Balance Contribution**: 0.5423
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

---

## 🔧 Feature Engineering Details

### Correlation-Based Feature Reduction
- **Original Features**: 19
- **Reduced Features**: 10
- **Features Removed**: 9 (47.4%)
- **Correlation Threshold**: 0.85

### Dimensionality Reduction
- **PCA Applied**: ❌ No

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

**Score: 0.586**

| Score Range | Interpretation |
|-------------|----------------|
| 0.70 - 1.00 | Excellent: Highly distinct regimes with strong temporal stability |
| 0.50 - 0.70 | Good: Clear regime separation with reasonable stability |
| 0.30 - 0.50 | Moderate: Some regime distinction, room for improvement |
| 0.00 - 0.30 | Poor: Weak regime separation, consider parameter tuning |

**Current Status**: Good

---

*Generated by HMM Regime Discovery at 2025-10-30T22:32:47.173278*

