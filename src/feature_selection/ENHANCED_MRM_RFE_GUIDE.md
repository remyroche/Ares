# Enhanced mRMR and Multi-Stage RFE Guide

## 🚀 Overview

This guide covers the enhanced mRMR with rank-based scoring and the sophisticated multi-stage RFE approach with SHAP integration, z-score normalization, and stability selection.

## 🧠 Enhanced mRMR with Rank-based Approach

### **Key Features**
- **70% Mutual Information + 30% Spearman correlation**
- **Rank-based scoring with z-score normalization**
- **Quantile binning for MI calculation**
- **Greedy selection until 50% of original features**
- **Redundancy calculation with mRMR criterion**

### **Algorithm Details**

#### **Relevance Calculation**
1. **MI Relevance**: `MI_rel_i = mutual_info_(classif|regression)(Xi, y)`
2. **Spearman Relevance**: `Spearman_rel_i = |ρ_s(Xi, y)|`
3. **Rank and Z-score**: Rank each vector (descending), z-score the ranks
4. **Blend**: `Rel_i = 0.70 * zrank(MI_rel) + 0.30 * zrank(Spear_rel)`

#### **Redundancy Calculation**
1. **MI Redundancy**: `MI_red = mean_j MI(Xi, Xj)` for features in selected set S
2. **Spearman Redundancy**: `Spear_red = mean_j |ρ_s(Xi, Xj)|`
3. **Rank and Z-score**: Rank-desc both, z-score, then blend
4. **Blend**: `Red_i = 0.70 * zrank(MI_red) + 0.30 * zrank(Spear_red)`

#### **Selection Criterion**
- **mRMR**: `Score_i = Rel_i - Red_i`
- **High Collinearity**: `Score_i = Rel_i / (Red_i + ε)` when features are highly collinear

## 🔧 Enhanced Multi-Stage RFE

### **Stage 1: mRMR Pre-filtering**
- **Goal**: Cut to top 50% features
- **Method**: Improved mRMR with rank-based scoring
- **Output**: Reduced feature set for subsequent stages

### **Stage 2: Ensemble Filtering**
- **Goal**: Filter to top 25% with CV until `len(F) == final_k + 60`
- **Models**: LGBM SHAP + LASSO + RandomForest
- **Process**:
  1. Train models on CV folds
  2. Calculate SHAP values (LGBM), coefficients (LASSO), importance (RF)
  3. Rank-desc importances, z-score ranks
  4. Average z-scores across models → `zFold`
  5. Average `zFold` across folds → `zEnsemble`
  6. Keep features with highest `zEnsemble`
  7. Repeat until target reached

### **Stage 3: Batch RFE**
- **Goal**: Remove 10% each step with CV until `len(F) == final_k + 20`
- **Process**:
  1. Fit same three models on train folds
  2. Compute importances → aggregate to `zEnsemble`
  3. Drop bottom 10% by `zEnsemble`
  4. Track CV metric and feature stability
  5. Lock features with high stability (freq ≥ 0.9)
  6. Stop on plateau or target reached

### **Stage 4: Fine RFE**
- **Goal**: Remove one-by-one until final target
- **Process**:
  1. Try removing lowest-ranked feature
  2. Check CV metric degradation tolerance
  3. Accept removal if within tolerance
  4. Enforce stability threshold (freq ≥ 0.75)
  5. Stop at final target or plateau

## 📊 Usage Examples

### **Enhanced mRMR Selection**

```python
from src.feature_selection import ImprovedMRMR, create_improved_mrmr

# Create improved mRMR selector
mrmr_selector = create_improved_mrmr({
    'mi_weight': 0.7,
    'spearman_weight': 0.3,
    'target_ratio': 0.5,  # Select top 50%
    'quantile_bins': 10,
    'use_rank_based': True
})

# Select features
result = mrmr_selector.select_features(X, y, feature_names)

print(f"Selected {len(result['selected_features'])} features")
print(f"Selection ratio: {result['selection_ratio']:.2%}")
print(f"Relevance scores: {result['relevance_scores']}")
```

### **Enhanced Multi-Stage RFE**

```python
from src.feature_selection import EnhancedMultiStageRFE, create_enhanced_multi_stage_rfe

# Create enhanced multi-stage RFE selector
rfe_selector = create_enhanced_multi_stage_rfe({
    'target_features': 50,
    'stage2_buffer': 60,
    'stage3_buffer': 20,
    'stability_threshold': 0.6,
    'high_stability_threshold': 0.9,
    'plateau_threshold': 0.002,
    'cv_folds': 5,
    'cv_strategy': 'stratified'
})

# Select features
result = rfe_selector.select_features(X, y, target_features=50, feature_names=feature_names)

print(f"Selected {len(result['selected_features'])} features")
print(f"Stage results: {result['stage_results']}")
print(f"Execution time: {result['execution_time']:.3f}s")
```

### **Combined Approach**

```python
from src.feature_selection import (
    ImprovedMRMR, 
    EnhancedMultiStageRFE,
    create_improved_mrmr,
    create_enhanced_multi_stage_rfe
)

# Stage 1: mRMR pre-filtering
mrmr_selector = create_improved_mrmr({'target_ratio': 0.5})
mrmr_result = mrmr_selector.select_features(X, y, feature_names)

# Stage 2-4: Multi-stage RFE
rfe_selector = create_enhanced_multi_stage_rfe({
    'target_features': 50,
    'enable_stage1': False  # Skip mRMR since we already did it
})

# Use pre-filtered data
X_filtered = mrmr_result['X_selected']
feature_names_filtered = mrmr_result['selected_features']

rfe_result = rfe_selector.select_features(
    X_filtered, y, target_features=50, 
    feature_names=feature_names_filtered
)

print(f"Final selection: {len(rfe_result['selected_features'])} features")
```

## ⚙️ Configuration Options

### **Improved mRMR Configuration**

```python
mrmr_config = {
    'mi_weight': 0.7,  # Weight for mutual information
    'spearman_weight': 0.3,  # Weight for Spearman correlation
    'target_ratio': 0.5,  # Select top 50% of features
    'quantile_bins': 10,  # Number of quantile bins for MI
    'epsilon': 1e-8,  # Small constant for division
    'use_rank_based': True,  # Use rank-based scoring
    'enable_cv_relevance': True,  # Use CV for relevance calculation
    'cv_folds': 5,
    'enable_hardware_optimization': True,
    'n_jobs': -1,
    'random_state': 42
}
```

### **Enhanced Multi-Stage RFE Configuration**

```python
rfe_config = {
    'target_features': 50,
    'enable_stage1': True,  # Enable mRMR pre-filtering
    'enable_stage2': True,  # Enable ensemble filtering
    'enable_stage3': True,  # Enable batch RFE
    'enable_stage4': True,  # Enable fine RFE
    'stage2_buffer': 60,  # Keep 60 more than target
    'stage3_buffer': 20,  # Keep 20 more than target
    'stage2_ratio': 0.25,  # Keep top 25% in each iteration
    'stage3_batch_ratio': 0.1,  # Remove 10% in each batch
    'stability_threshold': 0.6,  # Minimum stability frequency
    'high_stability_threshold': 0.9,  # Lock high stability features
    'plateau_threshold': 0.002,  # AUC improvement threshold
    'plateau_patience': 2,  # Patience for plateau detection
    'cv_folds': 5,
    'cv_strategy': 'stratified',  # 'stratified', 'kfold', 'grouped', 'timeseries'
    'enable_bootstrap': True,
    'bootstrap_samples': 3,
    'lgb_params': {
        'max_depth': 8,
        'num_leaves': 256,
        'learning_rate': 0.1,
        'n_estimators': 100
    }
}
```

## 🔍 Advanced Features

### **Quantile Binning for MI**

```python
# Quantile binning reduces noise in MI calculation
def quantile_bin(data, n_bins=10):
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(data, quantiles)
    bin_edges = np.unique(bin_edges)
    return np.digitize(data, bin_edges[1:-1])
```

### **Z-score Normalization**

```python
# Z-score normalization of ranks
def zscore_normalize_ranks(scores):
    ranks = rankdata(-scores, method='dense')  # Descending order
    return zscore(ranks)
```

### **Stability Selection**

```python
# Track feature selection frequency across folds
def calculate_stability_counts(feature_scores, top_ratio=0.25):
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    top_count = max(1, int(len(feature_scores) * top_ratio))
    return {f[0]: 1 for f in sorted_features[:top_count]}
```

### **Plateau Detection**

```python
# Detect performance plateaus for early stopping
class PlateauDetector:
    def __init__(self, threshold=0.002, patience=2):
        self.threshold = threshold
        self.patience = patience
        self.best_score = -np.inf
        self.no_improvement = 0
    
    def check_plateau(self, current_score):
        if current_score > self.best_score + self.threshold:
            self.best_score = current_score
            self.no_improvement = 0
            return False
        else:
            self.no_improvement += 1
            return self.no_improvement >= self.patience
```

## 📈 Performance Monitoring

### **Get Selection Statistics**

```python
# mRMR statistics
mrmr_stats = mrmr_selector.get_performance_stats()
print(f"mRMR selections: {mrmr_stats['total_selections']}")
print(f"Avg features removed: {mrmr_stats['avg_features_removed']}")
print(f"MI usage ratio: {mrmr_stats['mi_usage_ratio']:.2%}")

# RFE statistics
rfe_stats = rfe_selector.get_performance_stats()
print(f"RFE runs: {rfe_stats['total_runs']}")
print(f"Stage success rates: {rfe_stats['stage1_success_rate']:.2%}")
print(f"Plateau detections: {rfe_stats['plateau_detections']}")
```

### **Get Selection Insights**

```python
# mRMR insights
mrmr_insights = mrmr_selector.get_selection_insights(result)
print(f"Relevance distribution: {mrmr_insights['relevance_distribution']}")
print(f"Selected features: {mrmr_insights['selected_feature_names']}")

# RFE insights
rfe_insights = rfe_selector.get_performance_stats()
print(f"Stability locks: {rfe_insights['stability_lock_rate']:.2%}")
```

## 🎯 Use Cases

### **High-Dimensional Data**
```python
# For high-dimensional data, use aggressive pre-filtering
mrmr_config = {
    'target_ratio': 0.3,  # Select only 30% initially
    'quantile_bins': 15,  # More bins for better MI estimation
    'use_rank_based': True
}

rfe_config = {
    'stage2_buffer': 100,  # Larger buffer
    'stage3_buffer': 50,
    'stability_threshold': 0.7  # Higher stability threshold
}
```

### **Small Sample Sizes**
```python
# For small samples, use fewer CV folds and simpler models
rfe_config = {
    'cv_folds': 3,  # Fewer folds
    'lgb_params': {
        'max_depth': 4,  # Simpler model
        'num_leaves': 16,
        'n_estimators': 50
    },
    'bootstrap_samples': 1  # Fewer bootstrap samples
}
```

### **Time Series Data**
```python
# For time series, use appropriate CV strategy
rfe_config = {
    'cv_strategy': 'timeseries',  # Use time series CV
    'enable_bootstrap': False,  # Disable bootstrap for time series
    'plateau_patience': 3  # More patience for time series
}
```

### **Critical Applications**
```python
# For critical applications, use high stability thresholds
rfe_config = {
    'stability_threshold': 0.8,  # High stability requirement
    'high_stability_threshold': 0.95,  # Very high for locking
    'plateau_threshold': 0.001,  # Stricter plateau detection
    'cv_folds': 10,  # More CV folds for robustness
    'bootstrap_samples': 5  # More bootstrap samples
}
```

## 🐛 Troubleshooting

### **Common Issues**

1. **Low Selection Quality**
   ```python
   # Check relevance scores
   relevance_scores = result['relevance_scores']
   print("Relevance distribution:", np.mean(list(relevance_scores.values())))
   
   # Check stability counts
   if 'stability_counts' in result['stage_results']['stage3']:
       stability = result['stage_results']['stage3']['stability_counts']
       print("Stability distribution:", np.mean(list(stability.values())))
   ```

2. **Slow Performance**
   ```python
   # Reduce complexity
   config = {
       'cv_folds': 3,  # Fewer CV folds
       'bootstrap_samples': 1,  # Fewer bootstrap samples
       'lgb_params': {'n_estimators': 50},  # Simpler LGBM
       'n_jobs': 1  # Reduce parallelism
   }
   ```

3. **Memory Issues**
   ```python
   # Use quantile binning and reduce features
   config = {
       'quantile_bins': 5,  # Fewer bins
       'target_ratio': 0.3,  # Select fewer features initially
       'enable_bootstrap': False  # Disable bootstrap
   }
   ```

### **Debug Mode**

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
from src.utils.tprint import tprint_debug
tprint_debug("Debug information will be shown")
```

## 📚 Complete Example

```python
import numpy as np
from src.feature_selection import (
    ImprovedMRMR,
    EnhancedMultiStageRFE,
    create_improved_mrmr,
    create_enhanced_multi_stage_rfe
)

# Generate sample data
X = np.random.rand(1000, 200)
y = np.random.rand(1000)
feature_names = [f"feature_{i}" for i in range(200)]

# Stage 1: mRMR pre-filtering
mrmr_selector = create_improved_mrmr({
    'mi_weight': 0.7,
    'spearman_weight': 0.3,
    'target_ratio': 0.5,
    'quantile_bins': 10,
    'use_rank_based': True
})

mrmr_result = mrmr_selector.select_features(X, y, feature_names)
print(f"mRMR selected: {len(mrmr_result['selected_features'])} features")

# Stage 2-4: Multi-stage RFE
rfe_selector = create_enhanced_multi_stage_rfe({
    'target_features': 50,
    'enable_stage1': False,  # Skip mRMR
    'stage2_buffer': 60,
    'stage3_buffer': 20,
    'stability_threshold': 0.6,
    'cv_folds': 5,
    'cv_strategy': 'stratified'
})

rfe_result = rfe_selector.select_features(
    mrmr_result['X_selected'], y, target_features=50,
    feature_names=mrmr_result['selected_features']
)

print(f"Final selection: {len(rfe_result['selected_features'])} features")
print(f"Total execution time: {rfe_result['execution_time']:.3f}s")

# Analyze results
print("Stage results:")
for stage, result in rfe_result['stage_results'].items():
    print(f"  {stage}: {result.get('stage', 'unknown')}")

# Get performance insights
mrmr_stats = mrmr_selector.get_performance_stats()
rfe_stats = rfe_selector.get_performance_stats()

print(f"mRMR performance: {mrmr_stats}")
print(f"RFE performance: {rfe_stats}")
```

## 🎉 Conclusion

The enhanced mRMR and multi-stage RFE approach provides:

- **Robust pre-filtering** with rank-based mRMR
- **Sophisticated ensemble methods** with SHAP integration
- **Stability selection** for reliable feature ranking
- **Plateau detection** for efficient early stopping
- **Z-score normalization** for fair model comparison
- **Comprehensive validation** with cross-validation

This implementation addresses the requirements for high-quality feature selection in complex, high-dimensional datasets while maintaining computational efficiency and providing extensive configuration options.