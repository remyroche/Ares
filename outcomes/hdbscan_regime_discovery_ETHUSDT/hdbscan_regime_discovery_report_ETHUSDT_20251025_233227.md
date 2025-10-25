# HDBSCAN Regime Discovery Comprehensive Report

**Generated**: 2025-10-25T23:32:27.009875  
**Report ID**: `hdbscan_regime_discovery_ETHUSDT_15m_20251025_233227`

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Symbol** | ETHUSDT |
| **Exchange** | binance |
| **Timeframe** | 15m |
| **Execution Mode** | light |
| **Processing Time** | 15.48 seconds |
| **Success Status** | ✅ SUCCESS |
| **Regimes Discovered** | 3 |
| **Noise Ratio** | 19.5% |
| **Economic Separation** | 10.0% |
| **Validation Status** | ❌ FAILED |

---

## 🔍 Regime Discovery Results

### Cluster Statistics
- **Total Regimes**: 3
- **Noise Points**: 19.5% of total samples
- **Average Regime Size**: 515 samples per regime
- **Economic Separation Score**: 0.100 (0.0 = identical, 1.0 = completely separated)

### Regime Distribution
| Regime ID | Sample Count | Percentage |
|-----------|--------------|------------|
| **Noise (-1)** | 374 | 19.5% |
| **Regime 0** | 1,224 | 63.7% |
| **Regime 1** | 259 | 13.5% |
| **Regime 2** | 63 | 3.3% |


### 📊 Detailed Per-Cluster Analysis

**Total Clusters**: 3 (excluding noise)

#### 🎯 Regime 0
- **Size**: 1,224 samples (63.7%)
- **Silhouette Score**: -0.2611
- **Calinski-Harabasz Score**: 16.1854
- **Davies-Bouldin Score**: 10.7740
- **Cluster Characteristics**:
  - Density: High
  - Stability: High
  - **Economic Profile**: Regime_0
  - **Avg Duration**: 122.4 periods

#### 🎯 Regime 1
- **Size**: 259 samples (13.5%)
- **Silhouette Score**: -0.2611
- **Calinski-Harabasz Score**: 16.1854
- **Davies-Bouldin Score**: 10.7740
- **Cluster Characteristics**:
  - Density: Medium
  - Stability: High
  - **Economic Profile**: Regime_1
  - **Avg Duration**: 25.9 periods

#### 🎯 Regime 2
- **Size**: 63 samples (3.3%)
- **Silhouette Score**: -0.2611
- **Calinski-Harabasz Score**: 16.1854
- **Davies-Bouldin Score**: 10.7740
- **Cluster Characteristics**:
  - Density: Low
  - Stability: Medium
  - **Economic Profile**: Regime_2
  - **Avg Duration**: 6.3 periods


### 📈 Quality Metrics Summary

- **Silhouette Score**: -0.2611 (higher is better)
- **Calinski-Harabasz Score**: 16.1854 (higher is better)
- **Davies-Bouldin Score**: 10.7740 (lower is better)
- **Number of Regimes**: 3
- **Noise Ratio**: 19.5%



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
- **Actual Separation Achieved**: 10.0%
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
- **Avg Return**: 4.8437588702654466e-05
- **Volatility**: 0.0023167445324361324
- **Sharpe Ratio**: 0.020907608792185783

**Confidence Intervals:**
- **Return Ci**: [-0.0001, 0.0002]
- **Volatility Ci**: [0.0000, 0.0073]

**Temporal Characteristics:**
- **Average Duration**: 122.4 periods
- **Transitions From Others**: 0
- **Transitions To Others**: 0
- **Self-Transitions**: 1223

**Trading Recommendations:**
- **Works Best For**: mean_reversion
- **Risk Caveats**: low_volatility

**Radar Plot Data:**
- **Return Score**: 0.501
- **Volatility Score**: 0.046
- **Sharpe Score**: 0.510

---

#### Regime 1: Regime_1

**Key Economic Statistics:**
- **Avg Return**: 6.555567233590409e-05
- **Volatility**: 0.0013375245034694672
- **Sharpe Ratio**: 0.0490126870572567

**Confidence Intervals:**
- **Return Ci**: [-0.0001, 0.0002]
- **Volatility Ci**: [0.0000, 0.0063]

**Temporal Characteristics:**
- **Average Duration**: 25.9 periods
- **Transitions From Others**: 0
- **Transitions To Others**: 0
- **Self-Transitions**: 258

**Trading Recommendations:**
- **Works Best For**: mean_reversion
- **Risk Caveats**: low_volatility

**Radar Plot Data:**
- **Return Score**: 0.502
- **Volatility Score**: 0.027
- **Sharpe Score**: 0.525

---

#### Regime 2: Regime_2

**Key Economic Statistics:**
- **Avg Return**: -0.00018227857071906328
- **Volatility**: 0.0015525316121056676
- **Sharpe Ratio**: -0.11740731447935104

**Confidence Intervals:**
- **Return Ci**: [-0.0006, 0.0002]
- **Volatility Ci**: [0.0000, 0.0066]

**Temporal Characteristics:**
- **Average Duration**: 6.3 periods
- **Transitions From Others**: 0
- **Transitions To Others**: 0
- **Self-Transitions**: 62

**Trading Recommendations:**
- **Works Best For**: mean_reversion
- **Risk Caveats**: low_volatility

**Radar Plot Data:**
- **Return Score**: 0.495
- **Volatility Score**: 0.031
- **Sharpe Score**: 0.441

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
- **Regime Labels**: `hdbscan_regime_labels_ETHUSDT_15m_20251025_233227.parquet`
- **Full Artifacts**: `hdbscan_regime_artifacts_ETHUSDT_15m_20251025_233227.pkl`
- **Economic Profiles**: `hdbscan_economic_profiles_ETHUSDT_15m_20251025_233227.json`

### Report Files
- **This Report**: `hdbscan_regime_discovery_report_ETHUSDT_15m_20251025_233227.md`

### Data Directory Structure
```
historical_data/hdbscan_regime_discovery/ETHUSDT/
├── hdbscan_regime_labels_ETHUSDT_15m_20251025_233227.parquet
├── hdbscan_regime_artifacts_ETHUSDT_15m_20251025_233227.pkl
├── hdbscan_economic_profiles_ETHUSDT_15m_20251025_233227.json
└── hdbscan_regime_discovery_report_ETHUSDT_15m_20251025_233227.md
```

---

## 🎯 Key Insights

### Regime Characteristics

- **Average Regime Duration**: 51.5 periods
- **Shortest Regime Duration**: 6.3 periods
- **Longest Regime Duration**: 122.4 periods
- **Regime Types Discovered**: Regime_1, Regime_2, Regime_0


### Trading Implications
- **Number of Actionable Regimes**: 3 (excluding noise)
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
{'processing_time': 14.706064224243164, 'memory_usage_mb': 0.0, 'optimization_stats': {'total_processing_time': 0.0, 'feature_generation_time': 10.933759212493896, 'hyperparameter_optimization_time': 9.298324584960938e-06, 'clustering_time': 0.4343249797821045, 'post_processing_time': 0.0, 'memory_optimizations': 0, 'vectorized_operations': 0, 'caching_hits': 0, 'optimization_improvements': 0, 'memory_optimizer_stats': {'current_memory_mb': 1116.203125, 'peak_memory_mb': 0.0, 'memory_optimizations': 0, 'data_validations': 0, 'safe_operations': 0, 'memory_savings_mb': 0.0, 'processing_time': 0.0, 'memory_history_count': 0}, 'vectorized_processor_stats': {'vectorized_operations': 2, 'rolling_operations': 0, 'distance_calculations': 1, 'clustering_operations': 1, 'vectorbt_usage_rate': 0.0, 'gpu_usage_rate': 0.0, 'memory_optimizations': 0, 'processing_time': 0.6793570518493652, 'vectorization_stats': {'vectorization_time': 0.0, 'vectorization_operations': 0, 'vectorization_efficiency': 1.0}, 'rolling_optimizer_stats': {'vectorbt_operations': 0, 'pandas_fallbacks': 0, 'numpy_fallbacks': 0, 'gpu_operations': 0, 'memory_optimizations': 0, 'hardware_optimizations': 0, 'chunk_operations': 0, 'parallel_operations': 0, 'total_operations': 0, 'total_time': 0.0, 'errors': 0, 'fast_failures': 0, 'validation_errors': 0}}, 'features_common_stats': {'total_processing_time': 10.933353185653687, 'vectorbt_operations': 0, 'normalization_operations': 0, 'volatility_labeling_operations': 0, 'caching_hits': 0, 'optimization_improvements': 1, 'memory_optimizations': 1, 'vectorization_stats': {'vectorization_time': 0.0, 'vectorization_operations': 0, 'vectorization_efficiency': 1.0}, 'rolling_optimizer_stats': {'method': 'VectorBTRollingOptimizer', 'status': 'performance_stats_not_available'}}}, 'feature_importance': {'open': 0.00047094859129034346, 'high': 0.00045872082156627397, 'low': 0.0004881466613518184, 'close': 0.000470659678326683, 'volume': 0.0015297246582915269, 'quote_volume': 0.001547209163493245, 'trades': 0.0010645662634158647, 'open_time': 0.00027853082008071013, 'close_time': 0.000278446008702498, 'day': 0.00022489547442861504, 'close_return': 0.0008223635296559503, 'close_log_return': 0.0008232284222810559, 'volume_return': 0.000507571181368636, 'volume_log_return': 0.0005447658133837072, 'price_range': 0.0008992625645819279, 'price_range_pct': 0.0009232810857354512, 'body_size': 0.0011335164883605354, 'body_size_pct': 0.0011087602618962615, 'hour': 0.00030235685956586634, 'day_of_week': 0.000305994893373924, 'is_weekend': 0.002503514797205373, 'enhanced_volatility_100': 0.006469167999747657, 'enhanced_volatility_50': 0.006469167999747657, 'enhanced_volatility_10': 0.006469167999747657, 'enhanced_volatility_20': 0.006469167999747657, 'enhanced_volatility_14': 0.006469167999747657, 'vectorbt_atr_10': 0.006469167999747657, 'enhanced_volatility_30': 0.006469167999747657, 'vectorbt_volatility_comprehensive_10': 0.006469167999747657, 'vectorbt_volatility_comprehensive_14': 0.006469167999747657, 'vectorbt_atr_14': 0.006469167999747657, 'vectorbt_volatility_comprehensive_20': 0.006469167999747657, 'vectorbt_volatility_comprehensive_30': 0.006469167999747657, 'vectorbt_volatility_comprehensive_50': 0.006469167999747657, 'vectorbt_atr_20': 0.006469167999747657, 'vectorbt_atr_30': 0.006469167999747657, 'vectorbt_atr_50': 0.006469167999747657, 'vectorbt_bbands_10_1.5': 0.006469167999747657, 'vectorbt_bbands_10_2.5': 0.006469167999747657, 'vectorbt_bbands_14_1.5': 0.006469167999747657, 'vectorbt_bbands_10_2.0': 0.006469167999747657, 'vectorbt_bbands_14_2.5': 0.006469167999747657, 'vectorbt_bbands_14_2.0': 0.006469167999747657, 'vectorbt_bbands_20_2.5': 0.006469167999747657, 'vectorbt_bbands_20_2.0': 0.006469167999747657, 'vectorbt_bbands_20_1.5': 0.006469167999747657, 'vectorbt_rogers_satchell_volatility_10': 0.006469167999747657, 'vectorbt_parkinson_volatility_10': 0.006469167999747657, 'vectorbt_parkinson_volatility_14': 0.006469167999747657, 'vectorbt_garman_klass_volatility_10': 0.006469167999747657, 'vectorbt_yang_zhang_volatility_10': 0.006469167999747657, 'vectorbt_rogers_satchell_volatility_14': 0.006469167999747657, 'vectorbt_yang_zhang_volatility_14': 0.006469167999747657, 'vectorbt_garman_klass_volatility_14': 0.006469167999747657, 'vectorbt_parkinson_volatility_20': 0.006469167999747657, 'vectorbt_rogers_satchell_volatility_20': 0.006469167999747657, 'vectorbt_yang_zhang_volatility_20': 0.006469167999747657, 'vectorbt_yang_zhang_volatility_30': 0.006469167999747657, 'vectorbt_parkinson_volatility_30': 0.006469167999747657, 'vectorbt_garman_klass_volatility_20': 0.006469167999747657, 'vectorbt_rogers_satchell_volatility_30': 0.006469167999747657, 'vectorbt_garman_klass_volatility_30': 0.006469167999747657, 'vectorbt_parkinson_volatility_50': 0.006469167999747657, 'vectorbt_rogers_satchell_volatility_50': 0.006469167999747657, 'vectorbt_yang_zhang_volatility_50': 0.006469167999747657, 'vectorbt_garman_klass_volatility_50': 0.006469167999747657, 'macd_12_26_9_returns_vwap': 0.006469167999747657, 'rsi_14_returns_vwap': 0.006469167999747657, 'rsi_21_returns_vwap': 0.006469167999747657, 'rsi_30_returns_vwap': 0.006469167999747657, 'momentum_endpoints_sma_20': 0.006469167999747657, 'macd_delta_12_26_9': 0.006469167999747657, 'rsi_zscore_14_20': 0.006469167999747657, 'stochastic_14_3_price_returns': 0.006469167999747657, 'williams_r_14_price_returns': 0.006469167999747657, 'roc_14_price_returns': 0.006469167999747657, 'stochastic_21_3_price_returns': 0.006469167999747657, 'williams_r_21_price_returns': 0.006469167999747657, 'roc_21_price_returns': 0.006469167999747657, 'stochastic_30_3_price_returns': 0.006469167999747657, 'williams_r_30_price_returns': 0.006469167999747657, 'roc_30_price_returns': 0.006469167999747657, 'stochastic_kd_14_3': 0.006469167999747657, 'donchian_channel_20': 0.006469167999747657, 'vectorbt_momentum_comprehensive_9': 0.006469167999747657, 'vectorbt_momentum_comprehensive_14': 0.006469167999747657, 'momentum_features': 0.006469167999747657, 'vectorbt_momentum_comprehensive_30': 0.006469167999747657, 'vectorbt_momentum_comprehensive_21': 0.006469167999747657, 'momentum_14_price_returns': 0.006469167999747657, 'momentum_21_price_returns': 0.006469167999747657, 'momentum_30_price_returns': 0.006469167999747657, 'advanced_momentum_10_30': 0.006469167999747657, 'advanced_momentum_5_20': 0.006469167999747657, 'analyst_momentum_15m': 0.006469167999747657, 'analyst_momentum_1h': 0.006469167999747657, 'analyst_momentum_alignment': 0.006469167999747657, 'analyst_momentum_5m': 0.006469167999747657, 'sma_10_returns_vwap': 0.006469167999747657, 'sma_5_returns_vwap': 0.006469167999747657, 'sma_20_returns_vwap': 0.006469167999747657, 'sma_50_returns_vwap': 0.006469167999747657, 'sma_100_returns_vwap': 0.006469167999747657, 'ema_50_returns_vwap': 0.006469167999747657, 'ema_12_returns_vwap': 0.006469167999747657, 'ema_26_returns_vwap': 0.006469167999747657, 'dema_21_price_returns': 0.006469167999747657, 'tema_21_price_returns': 0.006469167999747657, 'wma_20_price_returns': 0.006469167999747657, 'keltner_channels_20_14_price_returns': 0.006469167999747657, 'mama_21_0.05_price_returns': 0.006469167999747657, 'vwma_20_price_returns': 0.006469167999747657, 'directional_signal': 0.006469167999747657, 'trend_score_14': 0.006469167999747657, 'volume_ema_5': 0.006469167999747657, 'volume_sma_5': 0.006469167999747657, 'volume_sma_20': 0.006469167999747657, 'volume_sma_10': 0.006469167999747657, 'volume_ema_10': 0.006469167999747657, 'volume_ema_20': 0.006469167999747657, 'volume_sma_50': 0.006469167999747657, 'volume_ema_50': 0.006469167999747657, 'volume_ratio_20': 0.006469167999747657, 'volume_ratio_10': 0.006469167999747657, 'volume_ratio_50': 0.006469167999747657, 'volume_roc_1': 0.006469167999747657, 'volume_roc_5': 0.006469167999747657, 'volume_roc_20': 0.006469167999747657, 'volume_roc_10': 0.006469167999747657, 'volume_std_10': 0.006469167999747657, 'volume_std_20': 0.006469167999747657, 'volume_std_50': 0.006469167999747657, 'volume_percentile_50': 0.006469167999747657, 'volume_percentile_20': 0.006469167999747657, 'volume_percentile_100': 0.006469167999747657, 'volume_oscillator_10_20': 0.006469167999747657, 'volume_oscillator_5_15': 0.006469167999747657, 'volume_vwap_20': 0.006469167999747657, 'volume_vwap_50': 0.006469167999747657, 'volume_vwap_10': 0.006469167999747657, 'volume_price_trend': 0.006469167999747657, 'volume_accumulation_distribution': 0.006469167999747657, 'volume_price_correlation_20': 0.006469167999747657, 'volume_price_correlation_10': 0.006469167999747657, 'volume_price_divergence_10': 0.006469167999747657, 'volume_price_divergence_20': 0.006469167999747657, 'price_volume_oscillator_10_20': 0.006469167999747657, 'price_volume_oscillator_5_15': 0.006469167999747657, 'analyst_volume_pressure': 0.006469167999747657, 'volume_zscore_60_252': 0.006469167999747657, 'volume_ma_ratios_20_10': 0.006469167999747657, 'analyst_volume_trend': 0.006469167999747657, 'cmf_20': 0.006469167999747657, 'vwap_deviations_20': 0.006469167999747657, 'order_flow_imbalance_20': 0.006469167999747657, 'volume_volatility_elasticity_20': 0.006469167999747657, 'volume_momentum_20': 0.006469167999747657, 'vectorbt_enhanced_obv_10': 0.006469167999747657, 'volume_momentum_5': 0.006469167999747657, 'volume_momentum_10': 0.006469167999747657, 'vectorbt_enhanced_obv_20': 0.006469167999747657, 'vectorbt_enhanced_ad_line_10': 0.006469167999747657, 'vectorbt_enhanced_obv_50': 0.006469167999747657, 'vectorbt_enhanced_ad_line_20': 0.006469167999747657, 'vectorbt_enhanced_ad_line_50': 0.006469167999747657, 'vectorbt_volume_weighted_ad_line_10': 0.006469167999747657, 'vectorbt_volume_weighted_ad_line_20': 0.006469167999747657, 'vectorbt_volume_weighted_ad_line_50': 0.006469167999747657, 'vectorbt_smoothed_obv_20': 0.006469167999747657, 'vectorbt_smoothed_obv_10': 0.006469167999747657, 'vectorbt_smoothed_obv_50': 0.006469167999747657, 'volume_trend_strength_20_50': 0.006469167999747657, 'volume_trend_strength_10_30': 0.006469167999747657}, 'cluster_persistence': None, 'condensed_tree': None, 'mst': None, 'glosh_scores': None, 'cluster_centers': None, 'cluster_sizes': None}
```

---

*Report generated by HDBSCAN Regime Discovery Step v1.0.0*  
*Generated at: 2025-10-25T23:32:27.009875*  
*Processing completed in: 0.00 seconds*
