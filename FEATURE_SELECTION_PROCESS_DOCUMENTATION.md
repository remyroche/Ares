# Feature Selection Process Documentation

## Overview

This document explains how the unified feature selection process works, including the integration with feature generation, PID module, iteration limits, and HMM regime-specific selection.

## Feature Selection Process Flow

### 1. **Feature Generation Integration** ✅

The feature selection process **builds on the features generated** by the feature generation system:

```python
# Step 1: Generate features from feature bank
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

**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 414-429

**Feature Categories Generated**:
- Returns features
- Momentum features  
- Volume features
- Volatility features
- Trend features
- Oscillator features
- Support/Resistance features
- Candlestick pattern features
- HMM regime features
- Cross-timeframe features
- Microstructure features
- Entropy features
- Autoencoder features
- Order flow features
- Time features

### 2. **Partial Information Decomposition (PID) Module** ✅

The PID module creates polynomial and cross-timeframe features:

```python
# Step 3: Create PID-based features if enabled
if self.config.enable_pid:
    self.logger.info("🔍 Step 2: Creating PID-based features")
    X_processed, feature_names_processed = self.create_pid_features(
        X_processed, y_processed, feature_names_processed, timeframe_data
    )
```

**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 435-439

**PID Features Created**:
- **Polynomial Features**: Based on synergistic information between feature pairs
- **Cross-timeframe Features**: Based on redundant information across timeframes

**PID Configuration**:
```python
pid_config = PIDConfig(
    method="bivariate",  # or "trivariate", "multivariate"
    max_polynomial_degree=3,
    max_interaction_terms=5,
    polynomial_threshold=0.1,
    timeframes=["1m", "5m", "15m", "1h", "4h", "1d"],
    cross_timeframe_threshold=0.15
)
```

### 3. **Iteration Limits for LASSO & mRMR** ✅

Both LASSO and mRMR respect iteration limits based on execution mode:

#### LASSO Iteration Limits
```python
# LASSO with iteration limit
estimator = LassoCV(
    cv=config.cv_folds, 
    random_state=config.random_state,
    max_iter=self.config.lasso_max_iterations  # 50, 5, or 2
)
```

**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 767-771

#### mRMR Iteration Limits
```python
# mRMR with iteration limit
mrmr_config = {
    'max_iterations': self.config.mrmr_max_iterations,  # 50, 5, or 2
    'relevance_method': 'mutual_info',
    'redundancy_method': 'correlation'
}
mrmr_selector = MRMRSelector(mrmr_config)
```

**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 262-267

#### Execution Mode Limits
```python
def _adjust_iteration_limits(self):
    """Adjust iteration limits based on execution mode."""
    if self.config.execution_mode == "blank":
        self.config.lasso_max_iterations = 5
        self.config.mrmr_max_iterations = 5
    elif self.config.execution_mode == "light":
        self.config.lasso_max_iterations = 2
        self.config.mrmr_max_iterations = 2
    # "full" mode uses the default values (50)
```

**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 187-195

### 4. **HMM Regime-Specific Feature Selection** ✅

HMM regime-specific feature selection is implemented in the `_perform_hmm_regime_selection` method:

```python
def _perform_hmm_regime_selection(
    self,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str]
) -> Dict[str, Any]:
    """Perform HMM regime-specific feature selection."""
    self.logger.info("🎯 Performing HMM regime-specific selection")
    
    # For HMM regime prediction, we want features that are good at distinguishing regimes
    # Use classification-based methods with regime-specific considerations
    
    # Create a classification config
    hmm_config = UnifiedFeatureSelectionConfig(
        target_features=100,
        task_type="classification",
        prediction_target="hmm_regime",
        primary_method="hybrid"
    )
    
    # Perform selection
    result = self._perform_feature_selection(X, y, feature_names, hmm_config)
    
    # Add regime-specific analysis
    result['regime_analysis'] = self._analyze_regime_features(X, y, result['selected_features'], feature_names)
    
    return result
```

**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 850-875

#### HMM Regime Analysis
```python
def _analyze_regime_features(
    self,
    X: np.ndarray,
    y: np.ndarray,
    selected_features: List[str],
    feature_names: List[str]
) -> Dict[str, Any]:
    """Analyze how well features distinguish between regimes."""
    # Get feature indices
    feature_indices = [feature_names.index(feat) for feat in selected_features]
    X_selected = X[:, feature_indices]
    
    # Calculate regime separation metrics
    unique_regimes = np.unique(y)
    regime_separation = {}
    
    for i, feature_idx in enumerate(feature_indices):
        feature_name = selected_features[i]
        feature_values = X_selected[:, i]
        
        # Calculate separation between regimes
        regime_means = {}
        regime_stds = {}
        
        for regime in unique_regimes:
            regime_mask = y == regime
            regime_values = feature_values[regime_mask]
            regime_means[regime] = np.mean(regime_values)
            regime_stds[regime] = np.std(regime_values)
        
        # Calculate separation score
        separation_score = 0
        for regime1 in unique_regimes:
            for regime2 in unique_regimes:
                if regime1 != regime2:
                    mean_diff = abs(regime_means[regime1] - regime_means[regime2])
                    std_combined = np.sqrt(regime_stds[regime1]**2 + regime_stds[regime2]**2)
                    if std_combined > 0:
                        separation_score += mean_diff / std_combined
        
        regime_separation[feature_name] = separation_score
    
    return {
        'regime_separation_scores': regime_separation,
        'unique_regimes': unique_regimes.tolist(),
        'n_regimes': len(unique_regimes)
    }
```

**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 877-920

#### HMM Regime Selection Usage
```python
# Special handling for HMM regime prediction
if self.config.prediction_target == "hmm_regime":
    self.logger.info("🎯 Performing HMM regime-specific feature selection")
    hmm_result = self._perform_hmm_regime_selection(
        X_processed, y_processed, feature_names_processed
    )
    results['hmm_regime_top_100'] = hmm_result
    self.feature_sets['hmm_regime_top_100'] = hmm_result['selected_features']
    self.feature_scores['hmm_regime_top_100'] = hmm_result['feature_scores']
```

**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 500-508

## Complete Feature Selection Process

### Step-by-Step Process

1. **Feature Generation** (if enabled)
   - Generate features from feature bank by category
   - Combine with existing features
   - Result: Enhanced feature matrix with generated features

2. **Data Preparation**
   - Handle missing values
   - Handle infinite values
   - Remove constant features
   - Result: Clean feature matrix

3. **PID Feature Creation** (if enabled)
   - Compute Partial Information Decomposition
   - Create polynomial features based on synergistic information
   - Create cross-timeframe features based on redundant information
   - Result: Enhanced feature matrix with PID-based features

4. **Feature Selection for Multiple Sizes**
   - Select top 120 features
   - Select top 100 features
   - Select top 80 features
   - Select top 60 features
   - Each selection uses hybrid method with iteration limits

5. **HMM Regime-Specific Selection** (if prediction_target="hmm_regime")
   - Use classification-based methods
   - Analyze regime separation
   - Select top 100 features for HMM regime prediction

### Hybrid Selection Method

The hybrid method combines multiple approaches:

1. **Filter-based pre-selection**
   - Mutual information
   - F-statistic
   - Correlation with target

2. **mRMR-based refinement** (with iteration limits)
   - Minimum Redundancy Maximum Relevance
   - Respects max_iterations limit

3. **Embedded method for final optimization**
   - LASSO (with iteration limits) for regression
   - Random Forest for classification

### Iteration Limits Summary

| Execution Mode | LASSO Max Iterations | mRMR Max Iterations |
|----------------|---------------------|-------------------|
| **Full**       | 50                  | 50                |
| **Blank**      | 5                   | 5                 |
| **Light**      | 2                   | 2                 |

### HMM Regime-Specific Features

The HMM regime-specific selection creates a **top 100 feature set** specifically optimized for HMM regime prediction:

- **Task Type**: Classification (not regression)
- **Target**: HMM regime labels (not price)
- **Method**: Hybrid with regime separation analysis
- **Output**: `hmm_regime_top_100` feature set
- **Analysis**: Regime separation scores for each feature

### Usage Examples

#### Basic Usage with Feature Generation
```python
from src.utils.ml_common.unified_feature_selection import UnifiedFeatureSelector

# Configure for feature generation integration
config = UnifiedFeatureSelectionConfig(
    build_on_feature_generation=True,
    enable_pid=True,
    execution_mode="full",  # or "blank", "light"
    prediction_target="price"
)

selector = UnifiedFeatureSelector(config)

# Select features with raw data for feature generation
results = selector.select_features(
    X, y, feature_names, 
    target_sizes=[120, 100, 80, 60],
    input_data=raw_data  # For feature generation
)
```

#### HMM Regime-Specific Selection
```python
# Configure for HMM regime prediction
config = UnifiedFeatureSelectionConfig(
    task_type="classification",
    prediction_target="hmm_regime",
    target_features=100,
    execution_mode="full"
)

selector = UnifiedFeatureSelector(config)
results = selector.select_features(X, y_regime, feature_names)

# Get HMM regime features
hmm_features = selector.get_hmm_regime_features()
```

#### With PID and Cross-Timeframe Features
```python
# Configure with PID
config = UnifiedFeatureSelectionConfig(
    enable_pid=True,
    pid_config={
        'max_polynomial_degree': 3,
        'max_interaction_terms': 5,
        'timeframes': ["1m", "5m", "15m", "1h", "4h", "1d"]
    }
)

selector = UnifiedFeatureSelector(config)
results = selector.select_features(
    X, y, feature_names,
    timeframe_data=timeframe_data  # For cross-timeframe features
)
```

## Key Benefits

1. **Feature Generation Integration**: Builds on comprehensive feature generation system
2. **PID Module**: Creates polynomial and cross-timeframe features automatically
3. **Iteration Limits**: Prevents excessive computation in LASSO and mRMR
4. **HMM Regime Support**: Specialized selection for regime prediction
5. **Multiple Feature Sets**: Generates 120, 100, 80, 60 feature sets
6. **Backwards Compatibility**: Existing code continues to work
7. **Matrix Operations**: Leverages optimized matrix computations

## File Locations

- **Main Framework**: `src/utils/ml_common/unified_feature_selection.py`
- **PID Module**: `src/utils/ml_common/partial_information_decomposition.py`
- **Matrix Operations**: `src/utils/ml_common/matrix_feature_operations.py`
- **Backwards Compatibility**: `src/utils/ml_common/backwards_compatibility.py`
- **Feature Generation**: `src/feature_generation/core/feature_bank.py`
- **mRMR Implementation**: `src/training/utils/feature_selection/selection_methods.py`

The unified framework successfully consolidates all feature selection capabilities while providing the requested enhancements for feature generation integration, PID module, iteration limits, and HMM regime-specific selection.