# HDBSCAN Regime Discovery Comprehensive Report

**Generated**: 2025-10-26T13:52:08.017050  
**Report ID**: `hdbscan_regime_discovery_ETHUSDT_1h_20251026_135208`

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Symbol** | ETHUSDT |
| **Exchange** | binance |
| **Timeframe** | 1h |
| **Execution Mode** | light |
| **Processing Time** | 11.31 seconds |
| **Success Status** | ✅ SUCCESS |
| **Regimes Discovered** | 2 |
| **Noise Ratio** | 38.3% |
| **Economic Separation** | 11.0% |
| **Validation Status** | ❌ FAILED |

---

## 🔍 Regime Discovery Results

### Cluster Statistics
- **Total Regimes**: 2
- **Noise Points**: 38.3% of total samples
- **Average Regime Size**: 148 samples per regime
- **Economic Separation Score**: 0.110 (0.0 = identical, 1.0 = completely separated)

### Regime Distribution
| Regime ID | Sample Count | Percentage |
|-----------|--------------|------------|
| **Noise (-1)** | 184 | 38.3% |
| **Regime 0** | 64 | 13.3% |
| **Regime 1** | 232 | 48.3% |


### 📊 Detailed Per-Cluster Analysis

**Total Clusters**: 2 (excluding noise)

#### 🎯 Regime 0
- **Size**: 64 samples (13.3%)
- **Silhouette Score**: -0.0266
- **Calinski-Harabasz Score**: 0.5963
- **Davies-Bouldin Score**: 14.4442
- **Cluster Characteristics**:
  - Density: Medium
  - Stability: Medium
  - **Economic Profile**: Regime_0
  - **Avg Duration**: 6.4 periods

#### 🎯 Regime 1
- **Size**: 232 samples (48.3%)
- **Silhouette Score**: -0.0266
- **Calinski-Harabasz Score**: 0.5963
- **Davies-Bouldin Score**: 14.4442
- **Cluster Characteristics**:
  - Density: High
  - Stability: High
  - **Economic Profile**: Regime_1
  - **Avg Duration**: 23.2 periods


### 📈 Quality Metrics Summary

- **Silhouette Score**: -0.0266 (higher is better)
- **Calinski-Harabasz Score**: 0.5963 (higher is better)
- **Davies-Bouldin Score**: 14.4442 (lower is better)
- **Number of Regimes**: 2
- **Noise Ratio**: 38.3%



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
- **Correlation Threshold**: 0.85
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
- **Min Cluster Size**: 1.0% (10 minimum)
- **Cluster Selection Method**: EOM
- **Selection Epsilon**: 0.001
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
- **Actual Separation Achieved**: 11.0%
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
- **Avg Return**: -0.00018029473721981049
- **Volatility**: 0.0024956408888101578
- **Sharpe Ratio**: -0.07224386185407639

**Confidence Intervals:**
- **Return Ci**: [-0.0008, 0.0004]
- **Volatility Ci**: [0.0000, 0.0075]

**Temporal Characteristics:**
- **Average Duration**: 6.4 periods
- **Transitions From Others**: 0
- **Transitions To Others**: 0
- **Self-Transitions**: 63

**Trading Recommendations:**
- **Works Best For**: mean_reversion
- **Risk Caveats**: low_volatility

**Radar Plot Data:**
- **Return Score**: 0.495
- **Volatility Score**: 0.050
- **Sharpe Score**: 0.464

---

#### Regime 1: Regime_1

**Key Economic Statistics:**
- **Avg Return**: 0.00016426012734882534
- **Volatility**: 0.0037543068174272776
- **Sharpe Ratio**: 0.04375245049595833

**Confidence Intervals:**
- **Return Ci**: [-0.0003, 0.0006]
- **Volatility Ci**: [0.0000, 0.0088]

**Temporal Characteristics:**
- **Average Duration**: 23.2 periods
- **Transitions From Others**: 0
- **Transitions To Others**: 0
- **Self-Transitions**: 231

**Trading Recommendations:**
- **Works Best For**: mean_reversion
- **Risk Caveats**: low_volatility

**Radar Plot Data:**
- **Return Score**: 0.504
- **Volatility Score**: 0.075
- **Sharpe Score**: 0.522

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
- **Regime Labels**: `hdbscan_regime_labels_ETHUSDT_1h_20251026_135208.parquet`
- **Full Artifacts**: `hdbscan_regime_artifacts_ETHUSDT_1h_20251026_135208.pkl`
- **Economic Profiles**: `hdbscan_economic_profiles_ETHUSDT_1h_20251026_135208.json`

### Report Files
- **This Report**: `hdbscan_regime_discovery_report_ETHUSDT_1h_20251026_135208.md`

### Data Directory Structure
```
historical_data/hdbscan_regime_discovery/ETHUSDT/
├── hdbscan_regime_labels_ETHUSDT_1h_20251026_135208.parquet
├── hdbscan_regime_artifacts_ETHUSDT_1h_20251026_135208.pkl
├── hdbscan_economic_profiles_ETHUSDT_1h_20251026_135208.json
└── hdbscan_regime_discovery_report_ETHUSDT_1h_20251026_135208.md
```

---

## 🎯 Key Insights

### Regime Characteristics

- **Average Regime Duration**: 14.8 periods
- **Shortest Regime Duration**: 6.4 periods
- **Longest Regime Duration**: 23.2 periods
- **Regime Types Discovered**: Regime_0, Regime_1


### Trading Implications
- **Number of Actionable Regimes**: 2 (excluding noise)
- **Regime Stability**: Low
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
  min_cluster_size_pct: 0.01
  change_budget_pct: 0.1
  min_economic_separation_pct: 0.25
  random_state: 42
```

### Processing Metadata
```json
{'processing_time': 9.568185806274414, 'memory_usage_mb': 0.0, 'optimization_stats': {'total_processing_time': 0.0, 'feature_generation_time': 5.517344951629639, 'hyperparameter_optimization_time': 1.2159347534179688e-05, 'clustering_time': 0.023009061813354492, 'post_processing_time': 0.0, 'memory_optimizations': 0, 'vectorized_operations': 0, 'caching_hits': 0, 'optimization_improvements': 0, 'memory_optimizer_stats': {'current_memory_mb': 957.03125, 'peak_memory_mb': 0.0, 'memory_optimizations': 0, 'data_validations': 0, 'safe_operations': 0, 'memory_savings_mb': 0.0, 'processing_time': 0.0, 'memory_history_count': 0}, 'vectorized_processor_stats': {'vectorized_operations': 2, 'rolling_operations': 0, 'distance_calculations': 1, 'clustering_operations': 1, 'vectorbt_usage_rate': 0.0, 'gpu_usage_rate': 0.0, 'memory_optimizations': 0, 'processing_time': 0.03385734558105469, 'vectorization_stats': {'vectorization_time': 0.0, 'vectorization_operations': 0, 'vectorization_efficiency': 1.0}, 'rolling_optimizer_stats': {'vectorbt_operations': 0, 'pandas_fallbacks': 0, 'numpy_fallbacks': 0, 'gpu_operations': 0, 'memory_optimizations': 0, 'hardware_optimizations': 0, 'chunk_operations': 0, 'parallel_operations': 0, 'total_operations': 0, 'total_time': 0.0, 'errors': 0, 'fast_failures': 0, 'validation_errors': 0}}, 'features_common_stats': {'total_processing_time': 5.51701807975769, 'vectorbt_operations': 0, 'normalization_operations': 0, 'volatility_labeling_operations': 0, 'caching_hits': 0, 'optimization_improvements': 1, 'memory_optimizations': 1, 'vectorization_stats': {'vectorization_time': 0.0, 'vectorization_operations': 0, 'vectorization_efficiency': 1.0}, 'rolling_optimizer_stats': {'method': 'VectorBTRollingOptimizer', 'status': 'performance_stats_not_available'}}}, 'feature_importance': {'open': 0.00010278153526067396, 'high': 9.3815375712604e-05, 'low': 0.00010300224696402744, 'close': 0.0001026088791273236, 'volume': 0.00021795208106613228, 'quote_volume': 0.00022743789106794514, 'trades': 0.00016605890166854984, 'open_time': 5.702410827463605e-05, 'close_time': 5.701100266946887e-05, 'day': 4.58902821571892e-05, 'close_return': 0.000191289978407224, 'close_log_return': 0.0001918710249738498, 'volume_return': 0.00010131525447060744, 'volume_log_return': 8.914734408694645e-05, 'price_range': 0.00021834111875238302, 'price_range_pct': 0.00022091180454959712, 'body_size': 0.0002532263172886964, 'body_size_pct': 0.0002562684488875591, 'hour': 6.172043806992948e-05, 'day_of_week': 6.241901421358125e-05, 'is_weekend': 0.0005110452272190163, 'enhanced_volatility_20': 0.006557031985033634, 'enhanced_volatility_10': 0.006557031985033634, 'enhanced_volatility_50': 0.006557031985033634, 'enhanced_volatility_100': 0.006557031985033634, 'enhanced_volatility_14': 0.006557031985033634, 'vectorbt_volatility_comprehensive_10': 0.006557031985033634, 'vectorbt_atr_10': 0.006557031985033634, 'enhanced_volatility_30': 0.006557031985033634, 'vectorbt_atr_14': 0.006557031985033634, 'vectorbt_volatility_comprehensive_14': 0.006557031985033634, 'vectorbt_atr_20': 0.006557031985033634, 'vectorbt_volatility_comprehensive_20': 0.006557031985033634, 'vectorbt_volatility_comprehensive_50': 0.006557031985033634, 'vectorbt_atr_30': 0.006557031985033634, 'vectorbt_volatility_comprehensive_30': 0.006557031985033634, 'vectorbt_bbands_10_2.0': 0.006557031985033634, 'vectorbt_bbands_10_1.5': 0.006557031985033634, 'vectorbt_atr_50': 0.006557031985033634, 'vectorbt_bbands_10_2.5': 0.006557031985033634, 'vectorbt_bbands_14_1.5': 0.006557031985033634, 'vectorbt_bbands_14_2.5': 0.006557031985033634, 'vectorbt_bbands_20_1.5': 0.006557031985033634, 'vectorbt_bbands_14_2.0': 0.006557031985033634, 'vectorbt_bbands_20_2.0': 0.006557031985033634, 'vectorbt_bbands_20_2.5': 0.006557031985033634, 'vectorbt_parkinson_volatility_10': 0.006557031985033634, 'vectorbt_rogers_satchell_volatility_10': 0.006557031985033634, 'vectorbt_yang_zhang_volatility_10': 0.006557031985033634, 'vectorbt_garman_klass_volatility_10': 0.006557031985033634, 'vectorbt_parkinson_volatility_14': 0.006557031985033634, 'vectorbt_rogers_satchell_volatility_14': 0.006557031985033634, 'vectorbt_garman_klass_volatility_14': 0.006557031985033634, 'vectorbt_yang_zhang_volatility_14': 0.006557031985033634, 'vectorbt_parkinson_volatility_20': 0.006557031985033634, 'vectorbt_rogers_satchell_volatility_20': 0.006557031985033634, 'vectorbt_yang_zhang_volatility_20': 0.006557031985033634, 'vectorbt_parkinson_volatility_30': 0.006557031985033634, 'vectorbt_garman_klass_volatility_20': 0.006557031985033634, 'vectorbt_rogers_satchell_volatility_30': 0.006557031985033634, 'vectorbt_yang_zhang_volatility_30': 0.006557031985033634, 'vectorbt_parkinson_volatility_50': 0.006557031985033634, 'vectorbt_garman_klass_volatility_30': 0.006557031985033634, 'vectorbt_rogers_satchell_volatility_50': 0.006557031985033634, 'vectorbt_yang_zhang_volatility_50': 0.006557031985033634, 'vectorbt_garman_klass_volatility_50': 0.006557031985033634, 'macd_12_26_9_returns_vwap': 0.006557031985033634, 'rsi_14_returns_vwap': 0.006557031985033634, 'rsi_21_returns_vwap': 0.006557031985033634, 'rsi_30_returns_vwap': 0.006557031985033634, 'momentum_endpoints_sma_20': 0.006557031985033634, 'macd_delta_12_26_9': 0.006557031985033634, 'rsi_zscore_14_20': 0.006557031985033634, 'stochastic_14_3_price_returns': 0.006557031985033634, 'williams_r_14_price_returns': 0.006557031985033634, 'roc_14_price_returns': 0.006557031985033634, 'stochastic_21_3_price_returns': 0.006557031985033634, 'williams_r_21_price_returns': 0.006557031985033634, 'stochastic_30_3_price_returns': 0.006557031985033634, 'roc_21_price_returns': 0.006557031985033634, 'williams_r_30_price_returns': 0.006557031985033634, 'roc_30_price_returns': 0.006557031985033634, 'stochastic_kd_14_3': 0.006557031985033634, 'donchian_channel_20': 0.006557031985033634, 'momentum_features': 0.006557031985033634, 'vectorbt_momentum_comprehensive_9': 0.006557031985033634, 'vectorbt_momentum_comprehensive_14': 0.006557031985033634, 'vectorbt_momentum_comprehensive_30': 0.006557031985033634, 'vectorbt_momentum_comprehensive_21': 0.006557031985033634, 'momentum_30_price_returns': 0.006557031985033634, 'momentum_21_price_returns': 0.006557031985033634, 'momentum_14_price_returns': 0.006557031985033634, 'advanced_momentum_5_20': 0.006557031985033634, 'advanced_momentum_10_30': 0.006557031985033634, 'analyst_momentum_15m': 0.006557031985033634, 'analyst_momentum_5m': 0.006557031985033634, 'analyst_momentum_1h': 0.006557031985033634, 'analyst_momentum_alignment': 0.006557031985033634, 'sma_10_returns_vwap': 0.006557031985033634, 'sma_20_returns_vwap': 0.006557031985033634, 'sma_5_returns_vwap': 0.006557031985033634, 'sma_100_returns_vwap': 0.006557031985033634, 'sma_50_returns_vwap': 0.006557031985033634, 'ema_12_returns_vwap': 0.006557031985033634, 'ema_26_returns_vwap': 0.006557031985033634, 'dema_21_price_returns': 0.006557031985033634, 'tema_21_price_returns': 0.006557031985033634, 'ema_50_returns_vwap': 0.006557031985033634, 'keltner_channels_20_14_price_returns': 0.006557031985033634, 'wma_20_price_returns': 0.006557031985033634, 'mama_21_0.05_price_returns': 0.006557031985033634, 'directional_signal': 0.006557031985033634, 'vwma_20_price_returns': 0.006557031985033634, 'trend_score_14': 0.006557031985033634, 'volume_sma_5': 0.006557031985033634, 'volume_sma_10': 0.006557031985033634, 'volume_ema_10': 0.006557031985033634, 'volume_ema_5': 0.006557031985033634, 'volume_sma_20': 0.006557031985033634, 'volume_ema_50': 0.006557031985033634, 'volume_ema_20': 0.006557031985033634, 'volume_sma_50': 0.006557031985033634, 'volume_ratio_50': 0.006557031985033634, 'volume_roc_1': 0.006557031985033634, 'volume_ratio_10': 0.006557031985033634, 'volume_ratio_20': 0.006557031985033634, 'volume_roc_5': 0.006557031985033634, 'volume_roc_10': 0.006557031985033634, 'volume_roc_20': 0.006557031985033634, 'volume_std_10': 0.006557031985033634, 'volume_std_20': 0.006557031985033634, 'volume_std_50': 0.006557031985033634, 'volume_percentile_100': 0.006557031985033634, 'volume_percentile_50': 0.006557031985033634, 'volume_oscillator_10_20': 0.006557031985033634, 'volume_percentile_20': 0.006557031985033634, 'volume_oscillator_5_15': 0.006557031985033634, 'volume_vwap_20': 0.006557031985033634, 'volume_vwap_50': 0.006557031985033634, 'volume_vwap_10': 0.006557031985033634, 'volume_accumulation_distribution': 0.006557031985033634, 'volume_price_trend': 0.006557031985033634, 'volume_price_correlation_20': 0.006557031985033634, 'volume_price_correlation_10': 0.006557031985033634, 'volume_price_divergence_10': 0.006557031985033634, 'volume_price_divergence_20': 0.006557031985033634, 'price_volume_oscillator_10_20': 0.006557031985033634, 'analyst_volume_trend': 0.006557031985033634, 'analyst_volume_pressure': 0.006557031985033634, 'price_volume_oscillator_5_15': 0.006557031985033634, 'volume_zscore_60_252': 0.006557031985033634, 'volume_ma_ratios_20_10': 0.006557031985033634, 'cmf_20': 0.006557031985033634, 'vwap_deviations_20': 0.006557031985033634, 'order_flow_imbalance_20': 0.006557031985033634, 'volume_momentum_20': 0.006557031985033634, 'volume_momentum_10': 0.006557031985033634, 'volume_volatility_elasticity_20': 0.006557031985033634, 'volume_momentum_5': 0.006557031985033634, 'vectorbt_enhanced_obv_10': 0.006557031985033634, 'vectorbt_enhanced_obv_20': 0.006557031985033634, 'vectorbt_enhanced_ad_line_20': 0.006557031985033634, 'vectorbt_enhanced_ad_line_10': 0.006557031985033634, 'vectorbt_enhanced_ad_line_50': 0.006557031985033634, 'vectorbt_enhanced_obv_50': 0.006557031985033634, 'vectorbt_smoothed_obv_10': 0.006557031985033634, 'vectorbt_volume_weighted_ad_line_10': 0.006557031985033634, 'vectorbt_volume_weighted_ad_line_20': 0.006557031985033634, 'vectorbt_smoothed_obv_20': 0.006557031985033634, 'vectorbt_volume_weighted_ad_line_50': 0.006557031985033634, 'vectorbt_smoothed_obv_50': 0.006557031985033634, 'volume_trend_strength_10_30': 0.006557031985033634, 'volume_trend_strength_20_50': 0.006557031985033634}, 'cluster_persistence': None, 'condensed_tree': None, 'mst': None, 'glosh_scores': None, 'cluster_centers': None, 'cluster_sizes': None}
```

---

*Report generated by HDBSCAN Regime Discovery Step v1.0.0*  
*Generated at: 2025-10-26T13:52:08.017050*  
*Processing completed in: 0.00 seconds*
