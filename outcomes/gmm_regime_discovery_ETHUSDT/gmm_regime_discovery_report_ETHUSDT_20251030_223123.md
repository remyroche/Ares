# GMM Regime Discovery Comprehensive Report

**Generated**: 2025-10-30T22:31:23.913016  
**Report ID**: `gmm_regime_discovery_ETHUSDT_1h_20251030_223123`

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Symbol** | ETHUSDT |
| **Exchange** | binance |
| **Timeframe** | 1h |
| **Processing Time** | 18.08 seconds |
| **Success Status** | ⚠️ PARTIAL SUCCESS |
| **Regimes Discovered** | 8 |
| **Quality Score** | 0.836 |
| **Noise Ratio** | 0.0% |

### Optimization Targets Achievement

**Targets Met** (1/3):
- ✅ Temporal Smoothness (0.906)

**Targets Not Met**:
- ❌ Silhouette Score (0.066 < 0.10)
- ❌ Cluster Count (8 outside (4, 5))

---

## 🔍 Regime Discovery Results

### Cluster Statistics
- **Total Regimes**: 8
- **Noise Points**: 0.0% of total samples (GMM: 0.0% - no noise)
- **Average Regime Size**: 60 samples per regime
- **Balance Score**: 0.669 (higher is better)

### Regime Distribution
| Regime ID | Sample Count | Percentage |
|-----------|--------------|------------|
| **Regime 0** | 68 | 14.2% |
| **Regime 1** | 100 | 20.8% |
| **Regime 2** | 52 | 10.8% |
| **Regime 3** | 56 | 11.7% |
| **Regime 4** | 20 | 4.2% |
| **Regime 5** | 96 | 20.0% |
| **Regime 6** | 13 | 2.7% |
| **Regime 7** | 75 | 15.6% |

---

## 📈 Comprehensive Quality Metrics (from cluster_quality_assessor.py)

### Overall Quality Score: 0.836

**Quality Score Breakdown:**

| Metric | Normalized Value | Weight | Contribution |
|--------|------------------|--------|--------------|
| **CV Ratio** | 0.9687 | 30.00% | 0.2906 |
| **Silhouette Score** | 0.5332 | 20.00% | 0.1066 |
| **Temporal Smoothness** | 0.9061 | 30.00% | 0.2718 |
| **Balance Score** | 0.6686 | 10.00% | 0.0669 |
| **Noise Ratio (inverted)** | 1.0000 | 10.00% | 0.1000 |

**Total Weight**: 100.00%  
**Weighted Score**: 0.8359

---

### Core Clustering Metrics
- **Silhouette Score**: 0.0664 (range: [-1, 1], higher is better)
  - *Interpretation*: Fair cluster separation
- **Calinski-Harabasz Score**: 18.63 (higher is better)
- **Davies-Bouldin Score**: 2.7497 (lower is better)

### Coefficient of Variation (CV) Metrics

**CV Ratio**: 2.0707

- **Within-Regime CV**: 8.907011 (lower = more cohesive regimes)
- **Between-Regime CV**: 18.443731 (higher = better separation)
- **CV Ratio Interpretation**: Excellent separation

### Temporal Metrics

- **Temporal Smoothness**: 0.9061 (range: [0, 1], higher = more stable over time)
  - *Interpretation*: Excellent temporal stability

- **Regime Persistence**: 10.64 periods (average duration)

### Balance Metrics

- **Balance Score**: 0.6686 (range: [0, 1], higher = more balanced)
- **Min Cluster Size**: 270.8% of total samples
- **Max Cluster Size**: 2083.3% of total samples
- **Cluster Size Std Dev**: 29.74

### Per-Regime Details

#### 🎯 Regime 0
- **Size**: 68
- **Percentage**: 14.1667
- **Feature Coefficient Of Variation**: {'PC_1': 3.009390963376553, 'PC_2': 2.00136839120186, 'PC_3': 6.824000329113171, 'PC_4': 9.905968429006462, 'PC_5': 1.8762797231651491, 'PC_6': 8.237341239838269, 'PC_7': 0.8672654253612712, 'PC_8': 0.981470937951701, 'PC_9': 2.816867348901613, 'PC_10': 8.276071905399785, 'PC_11': 30.891292050096798, 'PC_12': 4.077305431731986, 'PC_13': 0.728646568309127, 'PC_14': 1.5209143965686351, 'PC_15': 0.7976332520118838, 'PC_16': 1.0002251004924254, 'PC_17': 3.6364361189014516, 'PC_18': 3.1891341053103135, 'PC_19': 6.509626771687193, 'PC_20': 11.243976902002695}
- **Mean Cv**: 5.4196
- **Std Cv**: 6.6611
- **Balance Contribution**: 1.1333
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 1
- **Size**: 100
- **Percentage**: 20.8333
- **Feature Coefficient Of Variation**: {'PC_1': 14.870768936785943, 'PC_2': 2.3603719419733995, 'PC_3': 1.8560234044386175, 'PC_4': 3.001956517122029, 'PC_5': 2.194987669626286, 'PC_6': 3.8040659786976425, 'PC_7': 7.384892413735612, 'PC_8': 48.85910078533761, 'PC_9': 0.8651154872924992, 'PC_10': 3.45439325972344, 'PC_11': 9.880762430356768, 'PC_12': 4.90857963430608, 'PC_13': 10.03687722006269, 'PC_14': 19.70005069135741, 'PC_15': 12.95707424030676, 'PC_16': 9.689891668104908, 'PC_17': 3.324295359972753, 'PC_18': 15.113050312515893, 'PC_19': 4.359263664415947, 'PC_20': 3.659449754789801}
- **Mean Cv**: 9.1140
- **Std Cv**: 10.4780
- **Balance Contribution**: 1.6667
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 2
- **Size**: 52
- **Percentage**: 10.8333
- **Feature Coefficient Of Variation**: {'PC_1': 0.7474898953224743, 'PC_2': 0.5076621921772433, 'PC_3': 3.2689244032177296, 'PC_4': 7.472949462063408, 'PC_5': 1.8472272852648408, 'PC_6': 7.485188974778046, 'PC_7': 1.2216368628290302, 'PC_8': 2.6738269728618373, 'PC_9': 47.27654108788193, 'PC_10': 6.079372070689156, 'PC_11': 4.609350834600416, 'PC_12': 1.6176420812186965, 'PC_13': 2.213117342901199, 'PC_14': 1.8043511025703554, 'PC_15': 4.5641856900936775, 'PC_16': 1.536240332604746, 'PC_17': 1.0710005186326854, 'PC_18': 15.897327942268483, 'PC_19': 19.76356372458007, 'PC_20': 46.15813554388603}
- **Mean Cv**: 8.8908
- **Std Cv**: 13.5218
- **Balance Contribution**: 0.8667
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 3
- **Size**: 56
- **Percentage**: 11.6667
- **Feature Coefficient Of Variation**: {'PC_1': 1.9277106944520987, 'PC_2': 5.464793661508341, 'PC_3': 1.677304670930902, 'PC_4': 1.8489591685437274, 'PC_5': 1.06573158603633, 'PC_6': 21.658722628135184, 'PC_7': 3.666640010364495, 'PC_8': 17.609903354034923, 'PC_9': 2.3755036998380277, 'PC_10': 3.292013320879579, 'PC_11': 1.0995404450941435, 'PC_12': 5.8640942534154235, 'PC_13': 2.976968380435746, 'PC_14': 5.77088703881069, 'PC_15': 1.0564012656463861, 'PC_16': 3.218034730533227, 'PC_17': 2.855427763422751, 'PC_18': 4.053759283747585, 'PC_19': 1.7986340017536142, 'PC_20': 4.263773236479457}
- **Mean Cv**: 4.6772
- **Std Cv**: 5.2335
- **Balance Contribution**: 0.9333
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 4
- **Size**: 20
- **Percentage**: 4.1667
- **Feature Coefficient Of Variation**: {'PC_1': 0.3616456786732503, 'PC_2': 0.4238997609896491, 'PC_3': 0.31774727678380305, 'PC_4': 5.455239119725578, 'PC_5': 1.650857483867954, 'PC_6': 0.6012658387901543, 'PC_7': 3.157423343672624, 'PC_8': 1.1146300089321959, 'PC_9': 0.5041452936986787, 'PC_10': 2.16987769836259, 'PC_11': 0.5668790300526223, 'PC_12': 2.2657790935215436, 'PC_13': 19.72929344673993, 'PC_14': 1.2886113797423795, 'PC_15': 0.49407009166562116, 'PC_16': 10.104731692602108, 'PC_17': 1.0408788582220492, 'PC_18': 8.148144070416805, 'PC_19': 2.633087452195016, 'PC_20': 8.742791310772454}
- **Mean Cv**: 3.5385
- **Std Cv**: 4.7358
- **Balance Contribution**: 0.3333
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 5
- **Size**: 96
- **Percentage**: 20.0000
- **Feature Coefficient Of Variation**: {'PC_1': 2.6329056406960674, 'PC_2': 3.503594467613003, 'PC_3': 7.31665542543082, 'PC_4': 0.8463757885589166, 'PC_5': 4.911996635640736, 'PC_6': 3.3619021263482587, 'PC_7': 2.272857494507034, 'PC_8': 4.027821730884104, 'PC_9': 29.78812600220165, 'PC_10': 3.6497257103732355, 'PC_11': 7.326049006320179, 'PC_12': 4.63702249402679, 'PC_13': 7.447163687416648, 'PC_14': 3.9959725482445814, 'PC_15': 59.45092371703085, 'PC_16': 8.778806855763701, 'PC_17': 2.2450100830685216, 'PC_18': 3.1634259437180177, 'PC_19': 2.712780198582582, 'PC_20': 4.202127578885515}
- **Mean Cv**: 8.3136
- **Std Cv**: 13.1278
- **Balance Contribution**: 1.6000
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 6
- **Size**: 13
- **Percentage**: 2.7083
- **Feature Coefficient Of Variation**: {'PC_1': 239.3280738347463, 'PC_2': 0.45783169879628793, 'PC_3': 1.957465041826724, 'PC_4': 0.5294633962385462, 'PC_5': 14.941723753405514, 'PC_6': 0.33307373846533916, 'PC_7': 1.3141772768078366, 'PC_8': 0.4322082113066489, 'PC_9': 0.6912998980996335, 'PC_10': 0.42815169310195844, 'PC_11': 0.35556715217808993, 'PC_12': 1.1482544570437911, 'PC_13': 1.1544131061859995, 'PC_14': 5.311435181474179, 'PC_15': 0.49196957488528414, 'PC_16': 4.8143860563744445, 'PC_17': 1.6842519550443424, 'PC_18': 9.47843101836858, 'PC_19': 2.811939912719866, 'PC_20': 58.39992012899691}
- **Mean Cv**: 17.3032
- **Std Cv**: 52.4813
- **Balance Contribution**: 0.2167
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 7
- **Size**: 75
- **Percentage**: 15.6250
- **Feature Coefficient Of Variation**: {'PC_1': 0.28913793077688504, 'PC_2': 1.0139589693192788, 'PC_3': 4.097755246562968, 'PC_4': 2.78255074679136, 'PC_5': 3.4966715955640426, 'PC_6': 10.374184290333107, 'PC_7': 2.728118709171577, 'PC_8': 2.2567647508996, 'PC_9': 1.11336993542859, 'PC_10': 2.142435158247993, 'PC_11': 19.68991350652321, 'PC_12': 1.831641342042545, 'PC_13': 3.431767232660678, 'PC_14': 10.758491202483686, 'PC_15': 87.37903942014324, 'PC_16': 2.616201070482448, 'PC_17': 79.28495310084728, 'PC_18': 25.282284889784226, 'PC_19': 17.376998416581607, 'PC_20': 2.0365996319097093}
- **Mean Cv**: 13.9991
- **Std Cv**: 24.1208
- **Balance Contribution**: 1.2500
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

### GMM Model Parameters
- **Number of Components**: 8
- **Covariance Type**: full
- **Random State**: 42

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

**Score: 0.836**

| Score Range | Interpretation |
|-------------|----------------|
| 0.70 - 1.00 | Excellent: Highly distinct regimes with strong temporal stability |
| 0.50 - 0.70 | Good: Clear regime separation with reasonable stability |
| 0.30 - 0.50 | Moderate: Some regime distinction, room for improvement |
| 0.00 - 0.30 | Poor: Weak regime separation, consider parameter tuning |

**Current Status**: Excellent

---

*Generated by GMM Regime Discovery at 2025-10-30T22:31:23.913016*

