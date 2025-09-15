# Feature Selection Process Steps

## Overview

This document outlines the complete feature selection process steps in the unified framework, including the integration with feature generation, PID module independence, and matrix operations.

## 🔄 **Feature Selection Process Steps**

### **Step 1: Feature Generation Integration** (Optional)
**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 414-429

```python
# Step 1: Generate features from feature bank if enabled
if self.config.build_on_feature_generation and input_data is not None:
    self.logger.info("🔧 Step 1: Generating features from feature bank")
    generated_X, generated_names = self.generate_features_from_bank(input_data)
    
    if generated_X.size > 0:
        # Combine with existing features
        X_combined = np.column_stack([X, generated_X])
        feature_names_combined = feature_names + generated_names
        X = X_combined
        feature_names = feature_names_combined
```

**What happens**:
- Generates features from feature bank by category
- Combines generated features with existing features
- Categories: returns, momentum, volume, volatility, trend, oscillator, support/resistance, candlestick patterns, HMM regime, cross-timeframe, microstructure, entropy, autoencoder, order flow, time

### **Step 2: Data Preparation**
**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 507-508

```python
# Step 2: Prepare data
X_processed, y_processed, feature_names_processed = self._prepare_data(X, y, feature_names)
```

**What happens**:
- Handle missing values (NaN)
- Handle infinite values (Inf)
- Remove constant features
- Validate data quality
- Clean feature names

### **Step 3: PID Module Independence** (Note)
**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 510-511

```python
# Note: PID-based features are now independent and should be called separately
# when needed for market_analysis/cross_timeframe_analysis pipeline
```

**What this means**:
- PID module is **independent** from feature selection pipeline
- PID features are **not automatically created** during selection
- PID must be called **explicitly** when needed
- PID is designed for **market_analysis/cross_timeframe_analysis** pipeline

### **Step 4: Feature Selection for Multiple Sizes**
**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 513-520

```python
# Set default target sizes if not provided
if target_sizes is None:
    target_sizes = [120, 100, 80, 60]

# Perform feature selection for each target size
for target_size in target_sizes:
    result = self._perform_feature_selection(X_processed, y_processed, feature_names_processed, config)
    results[f'top_{target_size}'] = result
    self.feature_sets[f'top_{target_size}'] = result['selected_features']
```

**What happens**:
- Selects features for each target size: 120, 100, 80, 60
- Uses hybrid method with iteration limits
- Stores results for each size

### **Step 5: Hybrid Selection Method**
**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 705-740

#### **5a. Filter-based Pre-selection**
```python
# Step 1: Filter-based pre-selection
filter_result = self._filter_selection(X, y, feature_names, config)
filter_features = filter_result['selected_features']
```

**Methods used**:
- Mutual information with target
- F-statistic
- Correlation with target
- Variance threshold filtering

#### **5b. mRMR-based Refinement** (with iteration limits)
```python
# Step 2: mRMR-based refinement
if len(filter_features) > config.target_features:
    mrmr_result = self._mrmr_selection_with_limits(X_filtered, y, filter_features, config.target_features)
    if mrmr_result['selected_features']:
        final_features = mrmr_result['selected_features']
    else:
        # Fallback to wrapper method
        wrapper_result = self._wrapper_selection(X_filtered, y, filter_features, config)
        final_features = wrapper_result['selected_features']
```

**Iteration limits**:
- **Full mode**: 50 iterations max
- **Blank mode**: 5 iterations max  
- **Light mode**: 2 iterations max

#### **5c. Embedded Method for Final Optimization**
```python
# Step 3: Embedded method for final optimization
if len(final_features) > config.target_features:
    embedded_result = self._embedded_selection(X_final, y, final_features, config)
    final_features = embedded_result['selected_features']
```

**Methods used**:
- **LASSO** (with iteration limits) for regression
- **Random Forest** for classification

### **Step 6: HMM Regime-Specific Selection** (If applicable)
**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 500-508

```python
# Special handling for HMM regime prediction
if self.config.prediction_target == "hmm_regime":
    self.logger.info("🎯 Performing HMM regime-specific feature selection")
    hmm_result = self._perform_hmm_regime_selection(
        X_processed, y_processed, feature_names_processed
    )
    results['hmm_regime_top_100'] = hmm_result
    self.feature_sets['hmm_regime_top_100'] = hmm_result['selected_features']
```

**What happens**:
- Uses classification-based methods
- Analyzes regime separation scores
- Creates top 100 HMM regime-specific features
- Target: HMM regimes (not price prediction)

### **Step 7: Results Compilation**
**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 540-550

```python
# Compile final results
execution_time = time.time() - start_time
results['execution_time'] = execution_time
results['total_features_processed'] = X_processed.shape[1]
results['feature_sets_generated'] = list(self.feature_sets.keys())

# Store results
self.results = results
```

**What happens**:
- Compiles all results into final dictionary
- Calculates execution time
- Stores feature sets and scores
- Returns comprehensive results

## 🎯 **PID Module Independence**

### **PID Module Location**: `src/utils/ml_common/partial_information_decomposition.py`

### **PID Features Created**:
1. **Polynomial Features**: Up to 50 features based on synergistic information
2. **Interaction Features**: Up to 100 most relevant interaction features  
3. **Cross-timeframe Features**: Up to 50 cross-timeframe features

### **PID Usage** (Independent):
```python
from src.utils.ml_common.partial_information_decomposition import create_pid_module

# Create PID module
pid_module = create_pid_module()

# Compute PID and create features
pid_results = pid_module.compute_pid(X, y, feature_names)
polynomial_features = pid_module.create_polynomial_features(X, feature_names)  # Up to 50
interaction_features = pid_module.create_interaction_features(X, feature_names)  # Up to 100
cross_timeframe_features = pid_module.create_cross_timeframe_features(X, feature_names, timeframe_data)  # Up to 50
```

### **Integration with market_analysis/cross_timeframe_analysis**:
- PID determines feature complementarity
- Cross-timeframe analysis generates features based on this complementarity
- PID is called explicitly when needed, not automatically

## 🔧 **Matrix Operations Integration**

### **Matrix Operations Location**: `src/utils/ml_common/matrix_feature_operations.py`

### **Uses Existing System**:
```python
# Import from existing matrix_operations system
from ...utils.matrix_operations.unified_operations import UnifiedMatrixOperations
from ...utils.matrix_operations.vectorized_core import VectorizedCore
from ...utils.matrix_operations.enhanced_operations import EnhancedOperations
```

### **Operations Provided**:
- Optimized correlation matrix calculation
- Efficient PCA computation
- Memory-aware matrix operations
- GPU acceleration when available
- Parallel processing support

## 📊 **Feature Selection Output**

### **Generated Feature Sets**:
1. **`top_120`**: Comprehensive feature set (120 features)
2. **`top_100`**: Refined feature set (100 features)  
3. **`top_80`**: Further refined (80 features)
4. **`top_60`**: Most selective set (60 features)
5. **`hmm_regime_top_100`**: HMM regime-specific features (100 features)

### **Each Feature Set Contains**:
- `selected_features`: List of selected feature names
- `feature_scores`: Dictionary of feature importance scores
- `method`: Selection method used
- `n_selected`: Number of features selected
- `selection_ratio`: Ratio of selected to total features
- `execution_time`: Time taken for selection

## 🚀 **Usage Examples**

### **Basic Feature Selection**:
```python
from src.utils.ml_common.unified_feature_selection import UnifiedFeatureSelector

# Configure
config = UnifiedFeatureSelectionConfig(
    build_on_feature_generation=True,
    execution_mode="full"  # or "blank", "light"
)

# Initialize
selector = UnifiedFeatureSelector(config)

# Select features
results = selector.select_features(
    X, y, feature_names, 
    target_sizes=[120, 100, 80, 60],
    input_data=raw_data  # For feature generation
)

# Get specific feature sets
top_120 = selector.get_feature_set('top_120')
hmm_features = selector.get_hmm_regime_features()
```

### **Independent PID Usage**:
```python
from src.utils.ml_common.partial_information_decomposition import create_pid_module

# Create PID module
pid_module = create_pid_module()

# Use in market_analysis/cross_timeframe_analysis pipeline
pid_results = pid_module.compute_pid(X, y, feature_names)
polynomial_features = pid_module.create_polynomial_features(X, feature_names)
interaction_features = pid_module.create_interaction_features(X, feature_names)
cross_timeframe_features = pid_module.create_cross_timeframe_features(X, feature_names, timeframe_data)
```

## 📋 **Summary of Changes**

1. ✅ **PID Module Independence**: PID is now independent from feature selection pipeline
2. ✅ **Feature Generation Integration**: Builds on features generated by feature bank
3. ✅ **Iteration Limits**: LASSO & mRMR respect limits (50/5/2 based on mode)
4. ✅ **Matrix Operations**: Uses existing matrix_operations/ system
5. ✅ **HMM Regime Selection**: Specialized selection for regime prediction
6. ✅ **Feature Limits**: Up to 50 polynomial + 100 interaction + 50 cross-timeframe features

The feature selection process is now modular, efficient, and integrates seamlessly with your existing systems while providing the requested enhancements.