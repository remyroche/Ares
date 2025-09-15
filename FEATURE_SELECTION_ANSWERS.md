# Feature Selection Framework - Answers to Your Questions

## 1. ✅ Feature Selection Builds on Features Generated

**Answer**: Yes, the feature selection process now **builds on the features generated** by the feature generation system.

### Implementation Details:

**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 414-429

```python
# Step 1: Generate features from feature bank if enabled
if self.config.build_on_feature_generation and input_data is not None:
    self.logger.info("🔧 Step 1: Generating features from feature bank")
    generated_X, generated_names = self.generate_features_from_bank(input_data)
    
    if generated_X.size > 0:
        # Combine with existing features
        if isinstance(X, pd.DataFrame):
            X_combined = pd.concat([X, pd.DataFrame(generated_X, columns=generated_names)], axis=1)
            feature_names_combined = X_combined.columns.tolist()
        else:
            X_combined = np.column_stack([X, generated_X])
            feature_names_combined = (feature_names or [f'feature_{i}' for i in range(X.shape[1])]) + generated_names
        
        X = X_combined
        feature_names = feature_names_combined
        self.logger.info(f"✅ Combined features: {X.shape[1]} total features")
```

### Feature Categories Generated:
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

## 2. ✅ Partial Information Decomposition (PID) Module Added

**Answer**: Yes, a comprehensive PID module has been added for creating polynomial & cross-timeframe features.

### Implementation Details:

**Location**: `src/utils/ml_common/partial_information_decomposition.py`

### Key Features:
- **Bivariate, Trivariate, and Multivariate PID**
- **Polynomial Feature Creation**: Based on synergistic information
- **Cross-timeframe Feature Creation**: Based on redundant information
- **Configurable Parameters**: Discretization, thresholds, timeframes

### Usage:
```python
from src.utils.ml_common.partial_information_decomposition import create_pid_module

# Create PID module
pid_module = create_pid_module()

# Compute PID and create features
pid_results = pid_module.compute_pid(X, y, feature_names)
polynomial_features = pid_module.create_polynomial_features(X, feature_names)
cross_timeframe_features = pid_module.create_cross_timeframe_features(X, feature_names, timeframe_data)
```

### Integration in Unified Framework:
**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 435-439

```python
# Step 3: Create PID-based features if enabled
if self.config.enable_pid:
    self.logger.info("🔍 Step 2: Creating PID-based features")
    X_processed, feature_names_processed = self.create_pid_features(
        X_processed, y_processed, feature_names_processed, timeframe_data
    )
```

## 3. ✅ Feature Selection Process with Iteration Limits

**Answer**: The feature selection process now includes strict iteration limits for LASSO & mRMR.

### Iteration Limits by Mode:

| Execution Mode | LASSO Max Iterations | mRMR Max Iterations |
|----------------|---------------------|-------------------|
| **Full**       | 50                  | 50                |
| **Blank**      | 5                   | 5                 |
| **Light**      | 2                   | 2                 |

### Implementation Details:

#### LASSO Iteration Limits:
**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 767-771

```python
# Choose estimator
if self.config.task_type == "regression":
    estimator = LassoCV(
        cv=config.cv_folds, 
        random_state=config.random_state,
        max_iter=self.config.lasso_max_iterations  # 50, 5, or 2
    )
```

#### mRMR Iteration Limits:
**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 262-267

```python
# Initialize mRMR selector
mrmr_config = {
    'max_iterations': self.config.mrmr_max_iterations,  # 50, 5, or 2
    'relevance_method': 'mutual_info',
    'redundancy_method': 'correlation'
}
mrmr_selector = MRMRSelector(mrmr_config)
```

#### Mode Adjustment:
**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 187-195

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

### Feature Selection Process Flow:

1. **Feature Generation** (if enabled)
   - Generate features from feature bank
   - Combine with existing features

2. **Data Preparation**
   - Handle missing/infinite values
   - Remove constant features

3. **PID Feature Creation** (if enabled)
   - Create polynomial features
   - Create cross-timeframe features

4. **Feature Selection for Multiple Sizes**
   - **Hybrid Method**:
     - Filter-based pre-selection
     - mRMR refinement (with iteration limits)
     - LASSO final optimization (with iteration limits)
   - Generate: 120, 100, 80, 60 feature sets

5. **HMM Regime-Specific Selection** (if applicable)
   - Classification-based methods
   - Regime separation analysis

## 4. ✅ HMM Regime-Specific Feature Selection Location

**Answer**: HMM regime-specific feature selection for top 100 HMM ML features is implemented in **multiple specific locations**:

### Primary Implementation:

**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 850-875

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

### HMM Regime Analysis:

**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 877-920

```python
def _analyze_regime_features(
    self,
    X: np.ndarray,
    y: np.ndarray,
    selected_features: List[str],
    feature_names: List[str]
) -> Dict[str, Any]:
    """Analyze how well features distinguish between regimes."""
    # Calculate regime separation metrics
    # Returns regime separation scores for each feature
```

### HMM Regime Selection Usage:

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
    self.feature_scores['hmm_regime_top_100'] = hmm_result['feature_scores']
```

### HMM Regime Features Retrieval:

**Location**: `src/utils/ml_common/unified_feature_selection.py` lines 1050-1056

```python
def get_hmm_regime_features(self) -> List[str]:
    """Get HMM regime-specific features."""
    if 'hmm_regime_top_100' in self.feature_sets:
        return self.feature_sets['hmm_regime_top_100']
    else:
        self.logger.warning("⚠️ HMM regime features not found")
        return []
```

### Usage Example:

```python
# Configure for HMM regime prediction
config = UnifiedFeatureSelectionConfig(
    task_type="classification",
    prediction_target="hmm_regime",
    target_features=100
)

selector = UnifiedFeatureSelector(config)
results = selector.select_features(X, y_regime, feature_names)

# Get HMM regime features (top 100)
hmm_features = selector.get_hmm_regime_features()
```

### Key Features of HMM Regime Selection:

1. **Task Type**: Classification (not regression)
2. **Target**: HMM regime labels (not price)
3. **Method**: Hybrid with regime separation analysis
4. **Output**: `hmm_regime_top_100` feature set
5. **Analysis**: Regime separation scores for each feature
6. **Regime Analysis**: Calculates how well each feature distinguishes between different regimes

## Summary

All four requirements have been successfully implemented:

1. ✅ **Feature selection builds on features generated** - Integrated with feature generation system
2. ✅ **PID module added** - Comprehensive module for polynomial & cross-timeframe features
3. ✅ **Iteration limits implemented** - LASSO & mRMR respect limits (50/5/2 based on mode)
4. ✅ **HMM regime-specific selection** - Located in `_perform_hmm_regime_selection` method, creates top 100 HMM ML features

The unified framework now provides a complete, integrated solution that builds on feature generation, includes PID capabilities, respects iteration limits, and provides specialized HMM regime selection.