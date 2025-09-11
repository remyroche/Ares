# LASSO Feature Selection Implementation

## 🎯 Overview

This document describes the comprehensive LASSO feature selection enhancement implemented in `src/utils/ml_common/feature_selection.py`. The implementation addresses the key challenges you identified:

1. **LASSO Instability**: Solved with stability selection using bootstrap sampling
2. **Method Complementarity**: Integrated with existing filter and wrapper methods
3. **Robust Selection**: Multi-stage approach with ensemble voting

## 🚀 Implementation Architecture

### **Phase 1: Core LASSO Methods**

#### **1. Standard LASSO Feature Selection**
```python
def lasso_feature_selection(self, X, y, feature_names, alpha=None, cv_folds=5, selection_criterion='cv'):
    """
    Standard LASSO with optional cross-validation for alpha selection.
    
    Key Features:
    - Automatic alpha selection via cross-validation
    - Support for both classification and regression
    - Feature standardization (critical for LASSO)
    - Coefficient interpretation
    """
```

#### **2. LASSO Stability Selection**
```python
def lasso_stability_selection(self, X, y, feature_names, n_bootstrap=100, 
                             bootstrap_fraction=0.8, alpha_range=(0.001, 1.0), 
                             stability_threshold=0.6, cv_folds=5):
    """
    LASSO with stability selection to overcome instability issues.
    
    Key Features:
    - Bootstrap sampling for stability assessment
    - Cross-validation for optimal alpha selection
    - Stability thresholding for robust feature selection
    - Handles correlated features gracefully
    """
```

### **Phase 2: Comprehensive Integration**

#### **3. Comprehensive Feature Selection**
```python
def comprehensive_feature_selection(self, X, y, feature_names, methods=None, 
                                  weights=None, n_features=None):
    """
    Multi-stage feature selection combining all methods.
    
    Stages:
    1. Filter methods (correlation, mRMR) for initial reduction
    2. Embedded methods (LASSO stability) for robust selection  
    3. Wrapper methods (RFE) for final validation
    4. Ensemble voting for consensus features
    """
```

## 🔧 Key Implementation Details

### **1. Stability Selection Algorithm**

```python
# Bootstrap sampling and LASSO selection
for bootstrap_idx in range(n_bootstrap):
    # Bootstrap sampling
    bootstrap_indices = np.random.choice(len(X_scaled), size=bootstrap_size, replace=True)
    X_bootstrap = X_scaled[bootstrap_indices]
    y_bootstrap = y[bootstrap_indices]
    
    # Find optimal alpha using cross-validation
    lasso_cv = LassoCV(alphas=alpha_range, cv=cv_folds)
    lasso_cv.fit(X_bootstrap, y_bootstrap)
    
    # Get selected features (non-zero coefficients)
    selected_mask = np.abs(lasso_cv.coef_) > 1e-6
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    # Update selection counts
    for feature in selected_features:
        feature_selection_counts[feature] += 1

# Calculate stability scores
for feature in feature_names:
    stability_score = feature_selection_counts[feature] / n_bootstrap
    if stability_score >= stability_threshold:
        stable_features.append(feature)
```

### **2. Multi-Stage Approach**

```python
# Stage 1: Filter methods (correlation, mRMR)
filter_methods = ['correlation', 'mrmr']
for method in filter_methods:
    result = getattr(self, f"{method}_selection")(current_X, y, current_features)
    current_features = result['selected_features']

# Stage 2: Embedded methods (LASSO stability)
embedded_methods = ['lasso_stability', 'lasso']
for method in embedded_methods:
    result = getattr(self, method)(current_X, y, current_features)
    method_results[method] = result

# Stage 3: Wrapper methods (RFE)
wrapper_methods = ['rfe']
for method in wrapper_methods:
    result = self.recursive_feature_elimination(base_model, current_X, y, current_features)
    method_results[method] = result

# Stage 4: Ensemble voting
feature_votes = {feature: 0.0 for feature in feature_names}
for method, result in method_results.items():
    weight = weights.get(method, 1.0)
    for feature in result['selected_features']:
        feature_votes[feature] += weight
```

### **3. Configuration Management**

```python
'lasso': {
    'alpha_range': (0.001, 1.0),
    'cv_folds': 5,
    'max_iter': 1000,
    'tol': 1e-4,
    'random_state': 42
},
'lasso_stability': {
    'n_bootstraps': 100,
    'bootstrap_fraction': 0.8,
    'stability_threshold': 0.6,
    'alpha_range': (0.001, 1.0),
    'cv_folds': 5
}
```

## 📊 Method Comparison

| Method | Type | Strengths | Use Cases |
|--------|------|-----------|-----------|
| **Correlation Filter** | Filter | Fast, removes redundancy | Preprocessing, high-correlation datasets |
| **mRMR** | Filter | Information theory, relevance-redundancy balance | Feature ranking, interpretable selection |
| **LASSO** | Embedded | Automatic selection, linear interpretability | Linear models, sparse solutions |
| **LASSO Stability** | Embedded | Robust to correlated features, stability assessment | High-dimensional data, correlated features |
| **RFE** | Wrapper | Model-aware selection, complex interactions | Tree-based models, non-linear relationships |
| **Comprehensive** | Ensemble | Best of all methods, consensus selection | Production systems, robust pipelines |

## 🎯 Addressing Your Key Points

### **1. LASSO vs. Correlation Filtering**
- **Correlation Filtering**: Fast preprocessing, removes obvious redundancy
- **LASSO**: More sophisticated, considers collective feature contribution
- **Integration**: Correlation filtering → LASSO stability → Ensemble voting

### **2. Stability Selection Benefits**
- **Problem**: LASSO unstable with correlated features
- **Solution**: Bootstrap sampling + stability thresholding
- **Result**: Features consistently selected across subsamples are more reliable

### **3. Method Complementarity**
- **Filter Methods**: Initial reduction, fast preprocessing
- **Embedded Methods**: Model-aware selection, automatic sparsity
- **Wrapper Methods**: Complex interactions, model-specific optimization
- **Ensemble**: Combines strengths, reduces individual method weaknesses

## 🚀 Usage Examples

### **Basic LASSO Selection**
```python
framework = FeatureSelectionFramework()
result = framework.lasso_feature_selection(X, y, feature_names, alpha=0.01)
selected_features = result['selected_features']
```

### **LASSO Stability Selection**
```python
result = framework.lasso_stability_selection(
    X, y, feature_names,
    n_bootstrap=100,
    stability_threshold=0.6
)
stable_features = result['selected_features']
stability_scores = result['feature_stability_scores']
```

### **Comprehensive Selection**
```python
result = framework.comprehensive_feature_selection(
    X, y, feature_names,
    methods=['correlation', 'mrmr', 'lasso_stability', 'rfe'],
    weights={'correlation': 0.2, 'mrmr': 0.3, 'lasso_stability': 0.5, 'rfe': 0.0},
    n_features=20
)
consensus_features = result['consensus_features']
```

## 📈 Expected Benefits

1. **Robustness**: Stability selection overcomes LASSO instability
2. **Completeness**: Covers filter, embedded, and wrapper paradigms
3. **Flexibility**: Configurable methods and weights
4. **Interpretability**: Provides coefficients, stability scores, and voting results
5. **Production Ready**: Comprehensive error handling and logging

## 🔧 Configuration Options

### **LASSO Parameters**
- `alpha_range`: Range of regularization strengths to test
- `cv_folds`: Cross-validation folds for alpha selection
- `max_iter`: Maximum iterations for LASSO convergence
- `tol`: Convergence tolerance

### **Stability Selection Parameters**
- `n_bootstraps`: Number of bootstrap samples
- `bootstrap_fraction`: Fraction of data in each bootstrap
- `stability_threshold`: Minimum stability score for selection

### **Comprehensive Selection Parameters**
- `methods`: List of methods to use
- `weights`: Weights for ensemble voting
- `n_features`: Target number of features

## ✅ Implementation Status

- ✅ **LASSO Feature Selection**: Standard LASSO with CV
- ✅ **LASSO Stability Selection**: Bootstrap-based stability assessment
- ✅ **Comprehensive Selection**: Multi-stage ensemble approach
- ✅ **Configuration Management**: Flexible parameter control
- ✅ **Error Handling**: Robust exception management
- ✅ **Logging**: Comprehensive progress tracking
- ✅ **Documentation**: Complete method documentation

The implementation is **production-ready** and addresses all the key points you raised about LASSO instability, method complementarity, and robust feature selection.