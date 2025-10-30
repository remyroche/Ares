# GMM Regime Discovery Comprehensive Report

**Generated**: 2025-10-30T21:25:36.790272  
**Report ID**: `gmm_regime_discovery_ETHUSDT_1h_20251030_212536`

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Symbol** | ETHUSDT |
| **Exchange** | binance |
| **Timeframe** | 1h |
| **Processing Time** | 1.97 seconds |
| **Success Status** | ⚠️ PARTIAL SUCCESS |
| **Regimes Discovered** | 6 |
| **Quality Score** | 0.811 |
| **Noise Ratio** | 0.0% |

### Optimization Targets Achievement

**Targets Met** (2/3):
- ✅ Temporal Smoothness (0.937)
- ✅ Cluster Count (6)

**Targets Not Met**:
- ❌ Silhouette Score (0.083 < 0.10)

---

## 🔍 Regime Discovery Results

### Cluster Statistics
- **Total Regimes**: 6
- **Noise Points**: 0.0% of total samples (GMM: 0.0% - no noise)
- **Average Regime Size**: 80 samples per regime
- **Balance Score**: 0.591 (higher is better)

### Regime Distribution
| Regime ID | Sample Count | Percentage |
|-----------|--------------|------------|
| **Regime 0** | 175 | 36.5% |
| **Regime 1** | 101 | 21.0% |
| **Regime 2** | 10 | 2.1% |
| **Regime 3** | 78 | 16.2% |
| **Regime 4** | 19 | 4.0% |
| **Regime 5** | 97 | 20.2% |

---

## 📈 Comprehensive Quality Metrics (from cluster_quality_assessor.py)

### Overall Quality Score: 0.811

**Quality Score Breakdown:**

| Metric | Normalized Value | Weight | Contribution |
|--------|------------------|--------|--------------|
| **CV Ratio** | 0.8749 | 30.00% | 0.2625 |
| **Silhouette Score** | 0.5417 | 20.00% | 0.1083 |
| **Temporal Smoothness** | 0.9374 | 30.00% | 0.2812 |
| **Balance Score** | 0.5911 | 10.00% | 0.0591 |
| **Noise Ratio (inverted)** | 1.0000 | 10.00% | 0.1000 |

**Total Weight**: 100.00%  
**Weighted Score**: 0.8111

---

### Core Clustering Metrics
- **Silhouette Score**: 0.0835 (range: [-1, 1], higher is better)
  - *Interpretation*: Fair cluster separation
- **Calinski-Harabasz Score**: 21.66 (higher is better)
- **Davies-Bouldin Score**: 2.7173 (lower is better)

### Coefficient of Variation (CV) Metrics

**CV Ratio**: 1.3534

- **Within-Regime CV**: 11.658267 (lower = more cohesive regimes)
- **Between-Regime CV**: 15.778603 (higher = better separation)
- **CV Ratio Interpretation**: Good separation

### Temporal Metrics

- **Temporal Smoothness**: 0.9374 (range: [0, 1], higher = more stable over time)
  - *Interpretation*: Excellent temporal stability

- **Regime Persistence**: 15.97 periods (average duration)

### Balance Metrics

- **Balance Score**: 0.5911 (range: [0, 1], higher = more balanced)
- **Min Cluster Size**: 208.3% of total samples
- **Max Cluster Size**: 3645.8% of total samples
- **Cluster Size Std Dev**: 55.35

### Per-Regime Details

#### 🎯 Regime 0
- **Size**: 175
- **Percentage**: 36.4583
- **Feature Coefficient Of Variation**: {'PC_1': 1.7417251587466327, 'PC_2': 10.450437335026187, 'PC_3': 6.23145389323896, 'PC_4': 4.981808279929938, 'PC_5': 12.715378354409717, 'PC_6': 5.887654463792497, 'PC_7': 3.5183641559480217, 'PC_8': 5.351805111978788, 'PC_9': 37.90405852830385, 'PC_10': 2.4979638658030097, 'PC_11': 2.3545301104756486, 'PC_12': 6.020342753214573, 'PC_13': 3.5465927783167657, 'PC_14': 8.426662898359238, 'PC_15': 5.003482914402057, 'PC_16': 5.991807406103618, 'PC_17': 1.6230666168478458, 'PC_18': 222.96551500013194, 'PC_19': 4.515563492160103, 'PC_20': 42.60139253576352}
- **Mean Cv**: 19.7165
- **Std Cv**: 47.8640
- **Balance Contribution**: 2.1875
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 1
- **Size**: 101
- **Percentage**: 21.0417
- **Feature Coefficient Of Variation**: {'PC_1': 488.9599424573531, 'PC_2': 2.063728498318927, 'PC_3': 1.5552631467614533, 'PC_4': 0.9793433262085054, 'PC_5': 3.590247530876713, 'PC_6': 8.849487489499289, 'PC_7': 4.228793222221157, 'PC_8': 2.136761309772419, 'PC_9': 7.769338983469461, 'PC_10': 2.0760700702106476, 'PC_11': 13.161364079114044, 'PC_12': 7.6336589655377844, 'PC_13': 15.080778404358199, 'PC_14': 19.06222429483554, 'PC_15': 1.2716766981637073, 'PC_16': 5.839878744648526, 'PC_17': 2.1328991310073624, 'PC_18': 17.810774187218747, 'PC_19': 7.7777421421274315, 'PC_20': 31.15755927763166}
- **Mean Cv**: 32.1569
- **Std Cv**: 105.0708
- **Balance Contribution**: 1.2625
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 2
- **Size**: 10
- **Percentage**: 2.0833
- **Feature Coefficient Of Variation**: {'PC_1': 0.08897172376783377, 'PC_2': 0.6148952362278532, 'PC_3': 0.49170475437449096, 'PC_4': 0.3252509778172042, 'PC_5': 0.9180705369351179, 'PC_6': 0.584676686410808, 'PC_7': 0.27202867959009924, 'PC_8': 0.4565805919744056, 'PC_9': 1.875395242128285, 'PC_10': 1.9549549857299846, 'PC_11': 0.9597453972374236, 'PC_12': 0.7422148924627422, 'PC_13': 0.6245429501816961, 'PC_14': 0.36604180268648306, 'PC_15': 1.8747326280021626, 'PC_16': 1.300099474813548, 'PC_17': 0.9909654307165316, 'PC_18': 0.5015843623618154, 'PC_19': 47.640011211531544, 'PC_20': 2.341394706861604}
- **Mean Cv**: 3.2462
- **Std Cv**: 10.2036
- **Balance Contribution**: 0.1250
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 3
- **Size**: 78
- **Percentage**: 16.2500
- **Feature Coefficient Of Variation**: {'PC_1': 18.844074329247242, 'PC_2': 1.3751896503214844, 'PC_3': 2.233762888155982, 'PC_4': 12.809190443653947, 'PC_5': 1.675994806034161, 'PC_6': 49.57871283347647, 'PC_7': 2.7755669451027445, 'PC_8': 2.215925424550449, 'PC_9': 24.01112872237553, 'PC_10': 6.156263776615632, 'PC_11': 1.5508706808032469, 'PC_12': 9.501372022472061, 'PC_13': 8.40124466688649, 'PC_14': 5.1368165893624536, 'PC_15': 1.4970716427049258, 'PC_16': 2.9325927660527618, 'PC_17': 1.8613564817271544, 'PC_18': 5.771116658869051, 'PC_19': 2.0703072247731975, 'PC_20': 4.641777328170381}
- **Mean Cv**: 8.2520
- **Std Cv**: 11.2141
- **Balance Contribution**: 0.9750
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 4
- **Size**: 19
- **Percentage**: 3.9583
- **Feature Coefficient Of Variation**: {'PC_1': 0.44274181922913763, 'PC_2': 1.5611119158483056, 'PC_3': 0.8239389296803256, 'PC_4': 0.36380122397827697, 'PC_5': 6.990504633673792, 'PC_6': 0.8119719215465204, 'PC_7': 0.31458362080242686, 'PC_8': 0.15431489995590003, 'PC_9': 0.35768234859888814, 'PC_10': 0.21289223377574515, 'PC_11': 0.20769892434170642, 'PC_12': 0.5307768541431799, 'PC_13': 0.3343423549755308, 'PC_14': 0.20814342100706815, 'PC_15': 0.7991156935485596, 'PC_16': 1.2740847227541736, 'PC_17': 0.44750975525179665, 'PC_18': 1.3981412507697872, 'PC_19': 2.2362689679571046, 'PC_20': 2.125091434138806}
- **Mean Cv**: 1.0797
- **Std Cv**: 1.4894
- **Balance Contribution**: 0.2375
- **Regime Type**: unknown
- **Classification Scores**: {}
- **Regime Specific Metrics**: {}

#### 🎯 Regime 5
- **Size**: 97
- **Percentage**: 20.2083
- **Feature Coefficient Of Variation**: {'PC_1': 1.8505625896815292, 'PC_2': 5.531450982218547, 'PC_3': 2.678315391725964, 'PC_4': 3.656389994851428, 'PC_5': 1.6261353810820274, 'PC_6': 1.881135811317288, 'PC_7': 3.645462842280744, 'PC_8': 4.5013715702904475, 'PC_9': 1.5833705123799795, 'PC_10': 3.9940987749667154, 'PC_11': 1.70948347564617, 'PC_12': 1.187365068545809, 'PC_13': 1.9897531993760458, 'PC_14': 9.873209246302363, 'PC_15': 2.279274959575113, 'PC_16': 4.5541600307489105, 'PC_17': 1.9292871844291835, 'PC_18': 21.549968891037196, 'PC_19': 30.859644650735916, 'PC_20': 3.085582650468923}
- **Mean Cv**: 5.4983
- **Std Cv**: 7.3137
- **Balance Contribution**: 1.2125
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
- **Number of Components**: 6
- **Covariance Type**: full
- **Random State**: 42

---

## 🎯 Optimization Goals & Targets

This GMM regime discovery run was guided by the following optimization goals from `clustering_optimization_goals.py`:

### Cluster Configuration Targets
- **Target Cluster Count**: 4-6 clusters
- **Minimum Cluster Size**: 2.0% of total samples
- **Maximum Cluster Size**: 20.0% of total samples

### Quality Targets
- **Minimum Silhouette Score**: 0.10
- **Target Silhouette Score**: 0.30
- **Minimum Temporal Smoothness**: 0.60
- **Target Temporal Smoothness**: 0.90
- **Minimum CV Score**: 1.20
- **Target CV Score**: 2.00

### Economic Targets (for future integration)
- **Minimum Sharpe Ratio**: 0.50
- **Target Sharpe Ratio**: 1.50
- **Max Drawdown Threshold**: 30.0%

---

## 📊 Quality Score Interpretation

**Score: 0.811**

| Score Range | Interpretation |
|-------------|----------------|
| 0.70 - 1.00 | Excellent: Highly distinct regimes with strong temporal stability |
| 0.50 - 0.70 | Good: Clear regime separation with reasonable stability |
| 0.30 - 0.50 | Moderate: Some regime distinction, room for improvement |
| 0.00 - 0.30 | Poor: Weak regime separation, consider parameter tuning |

**Current Status**: Excellent

---

*Generated by GMM Regime Discovery at 2025-10-30T21:25:36.790272*

