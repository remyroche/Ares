# HDBSCAN Regime Discovery Comprehensive Report

**Generated**: 2025-10-25T19:41:23.754539  
**Report ID**: `hdbscan_regime_discovery_ETHUSDT_15m_20251025_194123`

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Symbol** | ETHUSDT |
| **Exchange** | binance |
| **Timeframe** | 15m |
| **Execution Mode** | light |
| **Processing Time** | 82.72 seconds |
| **Success Status** | ✅ SUCCESS |
| **Regimes Discovered** | 2 |
| **Noise Ratio** | 19.4% |
| **Economic Separation** | 0.0% |
| **Validation Status** | ❌ FAILED |

---

## 🔍 Regime Discovery Results

### Cluster Statistics
- **Total Regimes**: 2
- **Noise Points**: 19.4% of total samples
- **Average Regime Size**: 774 samples per regime
- **Economic Separation Score**: 0.000 (0.0 = identical, 1.0 = completely separated)

### Regime Distribution
| Regime ID | Sample Count | Percentage |
|-----------|--------------|------------|
| **Noise (-1)** | 373 | 19.4% |
| **Regime 0** | 331 | 17.2% |
| **Regime 1** | 1,216 | 63.3% |


### 📊 Detailed Per-Cluster Analysis

**Total Clusters**: 2 (excluding noise)

#### 🎯 Regime 0
- **Size**: 331 samples (17.2%)
- **Silhouette Score**: -0.0202
- **Calinski-Harabasz Score**: 0.1142
- **Davies-Bouldin Score**: 78.8751
- **Cluster Characteristics**:
  - Density: Medium
  - Stability: High
  - **Economic Profile**: Regime_0
  - **Avg Duration**: 33.1 periods

#### 🎯 Regime 1
- **Size**: 1,216 samples (63.3%)
- **Silhouette Score**: -0.0202
- **Calinski-Harabasz Score**: 0.1142
- **Davies-Bouldin Score**: 78.8751
- **Cluster Characteristics**:
  - Density: High
  - Stability: High
  - **Economic Profile**: Regime_1
  - **Avg Duration**: 121.6 periods


### 📈 Quality Metrics Summary

- **Silhouette Score**: -0.0202 (higher is better)
- **Calinski-Harabasz Score**: 0.1142 (higher is better)
- **Davies-Bouldin Score**: 78.8751 (lower is better)
- **Number of Regimes**: 2
- **Noise Ratio**: 19.4%



---

## 🏗️ Processing Pipeline Details

### 1. Feature Extraction
- **Feature Families Enabled**: Returns, Volatility, Volume/Flow, Entropy, Spectral
- **Total Features Generated**: 17 selected features
- **PID Features**: ✅ Enabled
- **Hybrid Features**: ✅ Enabled
- **Hardware Optimization**: ✅ Enabled

#### Feature Family Breakdown
- **Returns**: Features capturing returns patterns
- **Volatility**: Features capturing volatility patterns
- **Volume/Flow**: Features capturing volume/flow patterns
- **Entropy**: Features capturing entropy patterns
- **Spectral**: Features capturing spectral patterns


### 2. Preprocessing Pipeline
- **Transformer Type**: StandardScaler
- **Correlation Threshold**: 0.7
- **Mutual Information Threshold**: 0.9
- **HSIC Threshold**: 0.05
- **Per-Asset Transformers**: ✅ Enabled

### 3. Dimensionality Reduction
- **Method**: PCA_ONLY
- **PCA Variance Threshold**: 100.0%
- **UMAP Components**: 2
- **UMAP Neighbors**: 15
- **UMAP Min Distance**: 0.1

### 4. HDBSCAN Clustering
- **Min Cluster Size**: 5.0% (50 minimum)
- **Cluster Selection Method**: EOM
- **Selection Epsilon**: 0.3
- **Prediction Data**: ✅ Enabled

### 5. Post-Clustering Optimization
- **Change Budget**: 10.0% of samples
- **Max Optimization Rounds**: 5
- **Condensed Tree Usage**: ✅ Enabled
- **Reallocation Moves**: 0
- **Merges Performed**: 0

### 6. Temporal Stabilization
- **Smoothing Window**: 2 periods
- **Min Dwell Time**: 1 bars
- **Cooldown Period**: 1 bars
- **Stabilization Changes**: 0

---

## 💰 Economic Analysis

### Economic Validation Results
- **Minimum Separation Required**: 25.0%
- **Actual Separation Achieved**: 0.0%
- **Validation Status**: ❌ FAILED

### Interpretable Economic Axes
- **Trend Pc**: trend_pc
- **Vol Pc**: vol_pc
- **Breadth**: breadth
- **Skew**: skew
- **Liquidity Stress**: liquidity_stress
- **Momentum Strength**: momentum_strength

### Economic Profiles by Regime

#### Regime 0: Regime_0

**Key Economic Statistics:**
- **Avg Return**: 0.0094
- **Volatility**: 0.0400
- **Sharpe Ratio**: 0.2351

**Confidence Intervals:**
- **Return Ci**: [-0.0006, 0.0194]
- **Volatility Ci**: [0.0350, 0.0450]

**Temporal Characteristics:**
- **Average Duration**: 33.1 periods
- **Transitions From Others**: 0
- **Transitions To Others**: 0
- **Self-Transitions**: 330

**Trading Recommendations:**
- **Works Best For**: mean_reversion
- **Risk Caveats**: high_volatility

**Radar Plot Data:**
- **Return Score**: 0.735
- **Volatility Score**: 0.801
- **Sharpe Score**: 0.618

---

#### Regime 1: Regime_1

**Key Economic Statistics:**
- **Avg Return**: 0.0013
- **Volatility**: 0.0270
- **Sharpe Ratio**: 0.0492

**Confidence Intervals:**
- **Return Ci**: [-0.0087, 0.0113]
- **Volatility Ci**: [0.0220, 0.0320]

**Temporal Characteristics:**
- **Average Duration**: 121.6 periods
- **Transitions From Others**: 0
- **Transitions To Others**: 0
- **Self-Transitions**: 1215

**Trading Recommendations:**
- **Works Best For**: mean_reversion
- **Risk Caveats**: low_volatility

**Radar Plot Data:**
- **Return Score**: 0.533
- **Volatility Score**: 0.540
- **Sharpe Score**: 0.525

---


---

## 🔧 Technical Configuration

### Hardware Optimization
- **M1 GPU Acceleration**: ❌ Not Available
- **Matrix Operations**: ❌ Not Available
- **Memory Optimization**: ✅ Enabled

### Determinism Settings
- **Random State**: 42
- **BLAS Threading**: ✅ Pinned
- **Numba Threads**: 4

### Data Quality Metrics
- **Effective Sample Size**: N/A
- **Window Size**: N/A
- **Overlap Percentage**: N/A

---

## 📈 Performance Metrics

### Processing Times
- **Feature Extraction**: 0.00s (0.0%)
- **Preprocessing**: 0.00s (0.0%)
- **Dimensionality Reduction**: 0.00s (0.0%)
- **Clustering**: 0.00s (0.0%)
- **Optimization**: 0.00s (0.0%)
- **Economic Validation**: 0.00s (0.0%)
- **Temporal Stabilization**: 0.00s (0.0%)
- **Total Processing Time**: 0.00s


### Memory Usage
- **Peak Memory Usage**: N/A MB
- **Final Memory Usage**: N/A MB

---

## 📁 Generated Artifacts

### Data Files
- **Regime Labels**: `hdbscan_regime_labels_ETHUSDT_15m_20251025_194123.parquet`
- **Full Artifacts**: `hdbscan_regime_artifacts_ETHUSDT_15m_20251025_194123.pkl`
- **Economic Profiles**: `hdbscan_economic_profiles_ETHUSDT_15m_20251025_194123.json`

### Report Files
- **This Report**: `hdbscan_regime_discovery_report_ETHUSDT_15m_20251025_194123.md`

### Data Directory Structure
```
historical_data/hdbscan_regime_discovery/ETHUSDT/
├── hdbscan_regime_labels_ETHUSDT_15m_20251025_194123.parquet
├── hdbscan_regime_artifacts_ETHUSDT_15m_20251025_194123.pkl
├── hdbscan_economic_profiles_ETHUSDT_15m_20251025_194123.json
└── hdbscan_regime_discovery_report_ETHUSDT_15m_20251025_194123.md
```

---

## 🎯 Key Insights

### Regime Characteristics

- **Average Regime Duration**: 77.3 periods
- **Shortest Regime Duration**: 33.1 periods
- **Longest Regime Duration**: 121.6 periods
- **Regime Types Discovered**: Regime_1, Regime_0


### Trading Implications
- **Number of Actionable Regimes**: 2 (excluding noise)
- **Regime Stability**: Medium
- **Economic Separation**: Poor

### Model Performance
- **Validation Status**: ❌ FAILED - Requires Review
- **Economic Significance**: Low

---

## 🔄 Next Steps

### Immediate Actions
1. **Review Economic Profiles**: Examine each regime's characteristics and trading recommendations
2. **Validate Regime Stability**: Monitor regime transitions and duration patterns
3. **Test Trading Strategies**: Implement regime-aware trading strategies based on economic profiles

### Model Integration
1. **Feature Engineering**: Use regime labels as features in downstream models
2. **Regime-Aware Training**: Train models with regime-specific parameters
3. **Risk Management**: Implement regime-based position sizing and risk controls

### Monitoring
1. **Regime Drift Detection**: Monitor for changes in regime characteristics over time
2. **Performance Tracking**: Track strategy performance across different regimes
3. **Model Retraining**: Schedule periodic regime discovery updates

---

## 📊 Appendix

### Full Configuration
```yaml
regime_discovery_config:
            enabled_feature_families: ['technical', 'regime', 'entropy', 'spectral']
            total_max_features: 26
  transformer_type: N/A
  dim_reduction_mode: pca_only
  min_cluster_size_pct: 0.05
  change_budget_pct: 0.1
  min_economic_separation_pct: 0.25
  random_state: 42
```

### Processing Metadata
```json
{'processing_time': 81.84867215156555, 'memory_usage_mb': 0.0, 'optimization_stats': {'total_processing_time': 0.0, 'feature_generation_time': 81.48601293563843, 'hyperparameter_optimization_time': 1.0967254638671875e-05, 'clustering_time': 0.16708922386169434, 'post_processing_time': 0.0, 'memory_optimizations': 0, 'vectorized_operations': 0, 'caching_hits': 0, 'optimization_improvements': 0, 'memory_optimizer_stats': {'current_memory_mb': 1206.8125, 'peak_memory_mb': 0.0, 'memory_optimizations': 0, 'data_validations': 0, 'safe_operations': 0, 'memory_savings_mb': 0.0, 'processing_time': 0.0, 'memory_history_count': 0}, 'vectorized_processor_stats': {'vectorized_operations': 2, 'rolling_operations': 0, 'distance_calculations': 1, 'clustering_operations': 1, 'vectorbt_usage_rate': 0.0, 'gpu_usage_rate': 0.0, 'memory_optimizations': 0, 'processing_time': 0.2105400562286377, 'vectorization_stats': {'vectorization_time': 0.0, 'vectorization_operations': 0, 'vectorization_efficiency': 1.0}, 'rolling_optimizer_stats': {'vectorbt_operations': 732, 'pandas_fallbacks': 0, 'numpy_fallbacks': 0, 'gpu_operations': 0, 'memory_optimizations': 33, 'hardware_optimizations': 0, 'chunk_operations': 366, 'parallel_operations': 0, 'total_operations': 366, 'total_time': 5.465266227722168, 'errors': 0, 'fast_failures': 0, 'validation_errors': 0, 'avg_time_per_operation': 0.014932421387219038, 'vectorbt_usage_rate': 2.0, 'gpu_usage_rate': 0.0}}, 'features_common_stats': {'total_processing_time': 81.4851667881012, 'vectorbt_operations': 80, 'normalization_operations': 0, 'volatility_labeling_operations': 1, 'caching_hits': 0, 'optimization_improvements': 1, 'memory_optimizations': 1, 'vectorization_stats': {'vectorization_time': 0.0, 'vectorization_operations': 0, 'vectorization_efficiency': 1.0}, 'rolling_optimizer_stats': {'method': 'VectorBTRollingOptimizer', 'status': 'performance_stats_not_available'}}}, 'feature_importance': {'open': 0.010119428921767874, 'high': 0.00985668676926363, 'low': 0.010488969569722346, 'close': 0.010637974623859271, 'volume': 0.6717309979673602, 'quote_volume': 0.03324539757976521, 'trades': 0.022874689157966588, 'open_time': 0.005984884313181415, 'close_time': 0.005983061942906937, 'day': 0.004832403813062003, 'close_return': 0.017670398510815484, 'close_log_return': 0.017688982746136662, 'volume_return': 0.010906350687926943, 'volume_log_return': 0.011705564109325158, 'price_range': 0.01932275363508472, 'price_range_pct': 0.01983884758273433, 'body_size': 0.02435624556002692, 'body_size_pct': 0.02382430029315603, 'hour': 0.006496842343242979, 'day_of_week': 0.006575013984939072, 'is_weekend': 0.05379385460205186, 'volatility_label_t_103bps': 0.001188176238561362, 'unnamed_feature_0': 0.0008781750471428563}, 'cluster_persistence': array([0.28453494, 0.32605467]), 'condensed_tree': <hdbscan.plots.CondensedTree object at 0x33769c810>, 'mst': None, 'glosh_scores': None, 'cluster_centers': None, 'cluster_sizes': None}
```

---

*Report generated by HDBSCAN Regime Discovery Step v1.0.0*  
*Generated at: 2025-10-25T19:41:23.754539*  
*Processing completed in: 0.00 seconds*
