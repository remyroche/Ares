# HDBSCAN Regime Discovery Comprehensive Report

**Generated**: 2025-10-25T19:16:55.092612  
**Report ID**: `hdbscan_regime_discovery_ETHUSDT_15m_20251025_191655`

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Symbol** | ETHUSDT |
| **Exchange** | binance |
| **Timeframe** | 15m |
| **Execution Mode** | light |
| **Processing Time** | 100.73 seconds |
| **Success Status** | ✅ SUCCESS |
| **Regimes Discovered** | 2 |
| **Noise Ratio** | 23.8% |
| **Economic Separation** | 0.0% |
| **Validation Status** | ❌ FAILED |

---

## 🔍 Regime Discovery Results

### Cluster Statistics
- **Total Regimes**: 2
- **Noise Points**: 23.8% of total samples
- **Average Regime Size**: 732 samples per regime
- **Economic Separation Score**: 0.000 (0.0 = identical, 1.0 = completely separated)

### Regime Distribution
| Regime ID | Sample Count | Percentage |
|-----------|--------------|------------|
| **Noise (-1)** | 457 | 23.8% |
| **Regime 0** | 293 | 15.3% |
| **Regime 1** | 1,170 | 60.9% |


### 📊 Detailed Per-Cluster Analysis

**Total Clusters**: 2 (excluding noise)

#### 🎯 Regime 0
- **Size**: 293 samples (15.3%)
- **Silhouette Score**: -0.0248
- **Calinski-Harabasz Score**: 0.1631
- **Davies-Bouldin Score**: 62.3714
- **Cluster Characteristics**:
  - Density: Medium
  - Stability: High
  - **Economic Profile**: Regime_0
  - **Avg Duration**: 0.0 periods

#### 🎯 Regime 1
- **Size**: 1,170 samples (60.9%)
- **Silhouette Score**: -0.0248
- **Calinski-Harabasz Score**: 0.1631
- **Davies-Bouldin Score**: 62.3714
- **Cluster Characteristics**:
  - Density: High
  - Stability: High
  - **Economic Profile**: Regime_1
  - **Avg Duration**: 0.0 periods


### 📈 Quality Metrics Summary

- **Silhouette Score**: -0.0248 (higher is better)
- **Calinski-Harabasz Score**: 0.1631 (higher is better)
- **Davies-Bouldin Score**: 62.3714 (lower is better)
- **Number of Regimes**: 2
- **Noise Ratio**: 23.8%



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
- **Correlation Threshold**: 0.9
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
- **Min Cluster Size**: 0.5% (10 minimum)
- **Cluster Selection Method**: EOM
- **Selection Epsilon**: 0.05
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
- **Minimum Separation Required**: 15.0%
- **Actual Separation Achieved**: 0.0%
- **Validation Status**: ❌ FAILED

### Interpretable Economic Axes
- **Trend Pc**: trend_pc
- **Vol Pc**: vol_pc
- **Breadth**: breadth
- **Skew**: skew

### Economic Profiles by Regime

#### Regime 0: Regime_0

**Key Economic Statistics:**
- **Avg Return**: 0.0000
- **Volatility**: 0.0000
- **Sharpe Ratio**: 0.0000

**Confidence Intervals:**
- No confidence intervals available

**Temporal Characteristics:**
- **Average Duration**: 0.0 periods
- **Transitions From Others**: 0
- **Transitions To Others**: 0
- **Self-Transitions**: 0

**Trading Recommendations:**
- **Works Best For**: N/A
- **Risk Caveats**: N/A

**Radar Plot Data:**
- No radar plot data available

---

#### Regime 1: Regime_1

**Key Economic Statistics:**
- **Avg Return**: 0.0000
- **Volatility**: 0.0000
- **Sharpe Ratio**: 0.0000

**Confidence Intervals:**
- No confidence intervals available

**Temporal Characteristics:**
- **Average Duration**: 0.0 periods
- **Transitions From Others**: 0
- **Transitions To Others**: 0
- **Self-Transitions**: 0

**Trading Recommendations:**
- **Works Best For**: N/A
- **Risk Caveats**: N/A

**Radar Plot Data:**
- No radar plot data available

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
- **Regime Labels**: `hdbscan_regime_labels_ETHUSDT_15m_20251025_191655.parquet`
- **Full Artifacts**: `hdbscan_regime_artifacts_ETHUSDT_15m_20251025_191655.pkl`
- **Economic Profiles**: `hdbscan_economic_profiles_ETHUSDT_15m_20251025_191655.json`

### Report Files
- **This Report**: `hdbscan_regime_discovery_report_ETHUSDT_15m_20251025_191655.md`

### Data Directory Structure
```
historical_data/hdbscan_regime_discovery/ETHUSDT/
├── hdbscan_regime_labels_ETHUSDT_15m_20251025_191655.parquet
├── hdbscan_regime_artifacts_ETHUSDT_15m_20251025_191655.pkl
├── hdbscan_economic_profiles_ETHUSDT_15m_20251025_191655.json
└── hdbscan_regime_discovery_report_ETHUSDT_15m_20251025_191655.md
```

---

## 🎯 Key Insights

### Regime Characteristics

- **Average Regime Duration**: 0.0 periods
- **Shortest Regime Duration**: 0.0 periods
- **Longest Regime Duration**: 0.0 periods
- **Regime Types Discovered**: Regime_0, Regime_1


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
  min_cluster_size_pct: 0.005
  change_budget_pct: 0.1
  min_economic_separation_pct: 0.15
  random_state: 42
```

### Processing Metadata
```json
{'processing_time': 99.89146375656128, 'memory_usage_mb': 0.0, 'optimization_stats': {'total_processing_time': 0.0, 'feature_generation_time': 99.57379913330078, 'hyperparameter_optimization_time': 2.9802322387695312e-05, 'clustering_time': 0.14738106727600098, 'post_processing_time': 0.0, 'memory_optimizations': 0, 'vectorized_operations': 0, 'caching_hits': 0, 'optimization_improvements': 0, 'memory_optimizer_stats': {'current_memory_mb': 1231.5625, 'peak_memory_mb': 0.0, 'memory_optimizations': 0, 'data_validations': 0, 'safe_operations': 0, 'memory_savings_mb': 0.0, 'processing_time': 0.0, 'memory_history_count': 0}, 'vectorized_processor_stats': {'vectorized_operations': 2, 'rolling_operations': 0, 'distance_calculations': 1, 'clustering_operations': 1, 'vectorbt_usage_rate': 0.0, 'gpu_usage_rate': 0.0, 'memory_optimizations': 0, 'processing_time': 0.17572784423828125, 'vectorization_stats': {'vectorization_time': 0.0, 'vectorization_operations': 0, 'vectorization_efficiency': 1.0}, 'rolling_optimizer_stats': {'vectorbt_operations': 732, 'pandas_fallbacks': 0, 'numpy_fallbacks': 0, 'gpu_operations': 0, 'memory_optimizations': 33, 'hardware_optimizations': 0, 'chunk_operations': 366, 'parallel_operations': 0, 'total_operations': 366, 'total_time': 5.845604181289673, 'errors': 0, 'fast_failures': 0, 'validation_errors': 0, 'avg_time_per_operation': 0.015971596123742274, 'vectorbt_usage_rate': 2.0, 'gpu_usage_rate': 0.0}}, 'features_common_stats': {'total_processing_time': 99.57339429855347, 'vectorbt_operations': 80, 'normalization_operations': 0, 'volatility_labeling_operations': 1, 'caching_hits': 0, 'optimization_improvements': 1, 'memory_optimizations': 1, 'vectorization_stats': {'vectorization_time': 0.0, 'vectorization_operations': 0, 'vectorization_efficiency': 1.0}, 'rolling_optimizer_stats': {'method': 'VectorBTRollingOptimizer', 'status': 'performance_stats_not_available'}}}, 'feature_importance': {'open': 0.010119428921767874, 'high': 0.00985668676926363, 'low': 0.010488969569722346, 'close': 0.010637974623859271, 'volume': 0.6717309979673602, 'quote_volume': 0.03324539757976521, 'trades': 0.022874689157966588, 'open_time': 0.005984884313181415, 'close_time': 0.005983061942906937, 'day': 0.004832403813062003, 'close_return': 0.017670398510815484, 'close_log_return': 0.017688982746136662, 'volume_return': 0.010906350687926943, 'volume_log_return': 0.011705564109325158, 'price_range': 0.01932275363508472, 'price_range_pct': 0.01983884758273433, 'body_size': 0.02435624556002692, 'body_size_pct': 0.02382430029315603, 'hour': 0.006496842343242979, 'day_of_week': 0.006575013984939072, 'is_weekend': 0.05379385460205186, 'volatility_label_t_103bps': 0.001188176238561362, 'unnamed_feature_0': 0.0008781750471428563}, 'cluster_persistence': array([0.26886982, 0.32790646]), 'condensed_tree': <hdbscan.plots.CondensedTree object at 0x338e19890>, 'mst': None, 'glosh_scores': None, 'cluster_centers': None, 'cluster_sizes': None}
```

---

*Report generated by HDBSCAN Regime Discovery Step v1.0.0*  
*Generated at: 2025-10-25T19:16:55.092612*  
*Processing completed in: 0.00 seconds*
