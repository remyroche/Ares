# Feature Selection Module

A comprehensive feature selection framework for the Ares trading system. This module consolidates all feature selection logic into a single, well-organized location.

## 📋 Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Components](#components)
- [Migration Guide](#migration-guide)
- [Examples](#examples)

## 🎯 Overview

The feature selection module provides a unified interface for various feature selection algorithms, organized into logical categories:

- **Core Framework**: Main selection engine and high-level APIs
- **Selection Methods**: Filter, wrapper, embedded, and stability-based methods
- **Specialized Selectors**: Domain-specific selectors for special use cases
- **Dimensionality Reduction**: PCA, VIF, and correlation handling
- **Analysis Tools**: Feature importance, stability, temporal, and causal analysis

## 📁 Directory Structure

```
src/feature_selection/
├── __init__.py                          # Public API exports
├── README.md                            # This file
│
├── core/                                # Core framework
│   ├── __init__.py
│   └── framework.py                     # Main feature selection framework
│
├── methods/                             # Selection algorithms
│   ├── __init__.py
│   ├── mrmr.py                         # Minimum Redundancy Maximum Relevance
│   ├── stability_selection.py          # Stability-based selection
│   ├── wrapper_methods.py              # RFE and wrapper methods
│   ├── importance.py                   # Feature importance ranking
│   └── regularization.py               # Regularization-based selection
│
├── specialized/                         # Domain-specific selectors
│   ├── __init__.py
│   ├── entropy_balancer.py             # Entropy-based filtering
│   ├── adaptive_selector.py            # Adaptive selection for small samples
│   └── directional_selector.py         # Directional feature selection (long/short)
│
├── dimensionality/                      # Dimensionality reduction
│   ├── __init__.py
│   ├── pca_module.py                   # Principal Component Analysis
│   └── vif_module.py                   # Variance Inflation Factor
│
├── analysis/                            # Analysis tools
│   ├── __init__.py
│   └── feature_importance_analyzer.py  # Feature importance analysis
│
└── utils/                               # Utility functions
    └── __init__.py
```

## 🚀 Quick Start

### Basic Feature Selection

```python
from src.feature_selection import select_features

# Select features using comprehensive method
result = select_features(
    X=feature_matrix,
    y=target_vector,
    method='comprehensive',
    max_features=80
)

selected_features = result['selected_features']
feature_scores = result['feature_scores']
```

### Adaptive Selection (Small Samples)

```python
from src.feature_selection.specialized import AdaptiveFeatureSelector

# Create adaptive selector for small sample sizes
selector = AdaptiveFeatureSelector()
result = selector.select_features(X, y)

print(f"Selected {result.n_features_selected} features")
print(f"Overfitting risk: {result.overfitting_risk}")
```

### Regularization-Based Selection

```python
from src.feature_selection.methods import (
    FeatureRegularizationSelector,
    FeatureRegularizationConfig
)

# Configure regularization selector
config = FeatureRegularizationConfig(
    max_features=60,
    stability_threshold=0.6,
    n_bootstrap=75
)

# Fit and transform
selector = FeatureRegularizationSelector(config)
selector.fit(X, y, feature_names=feature_names)
X_selected = selector.transform(X)

selected_features = selector.get_selected_features()
```

### Directional Feature Selection

```python
from src.feature_selection.specialized import DirectionalFeatureSelectionConfig

# Configure directional selection (for long/short trading)
config = DirectionalFeatureSelectionConfig(
    target_total_features=80,
    maintain_directional_balance=True,
    min_features_per_direction=20
)

# Selection logic for directional features
# (see specialized/directional_selector.py for full implementation)
```

## 🧩 Components

### Core Framework

**`get_feature_selection_framework()`**
- Returns global framework instance
- Provides access to all selection methods

**`select_features(X, y, method='comprehensive', ...)`**
- Unified API for feature selection
- Supports: comprehensive, filter, wrapper, embedded, hybrid methods

**`run_comprehensive_feature_selection(...)`**
- Full pipeline with stability, temporal, and causal analysis
- Configurable stages and thresholds

### Selection Methods

#### Filter Methods
- Mutual information
- Correlation-based filtering
- Variance threshold
- Statistical tests (ANOVA, chi-squared)

#### Wrapper Methods
- **RFE (Recursive Feature Elimination)**: Iteratively removes features
- Forward/backward selection
- Sequential feature selection

#### Embedded Methods
- **LASSO**: L1 regularization for sparse feature selection
- **ElasticNet**: Combined L1/L2 regularization
- Tree-based importance

#### Stability Selection
- Bootstrap-based stability assessment
- Block bootstrap for time series
- Feature fraction sampling
- Cluster-correlated feature handling

### Specialized Selectors

#### Entropy Balancer
- Entropy-based feature filtering
- Stability across regime changes
- Information content assessment

#### Adaptive Selector
- Adapts to small sample sizes
- Conservative learning parameters
- Progressive feature selection
- Cross-validation with small sample handling

#### Directional Selector
- Long/short trading feature management
- Maintains directional balance
- Respects 60-100 feature limits
- Performance-based prioritization

### Dimensionality Reduction

#### PCA Module
- Principal Component Analysis
- Variance explained analysis
- Optimal component selection

#### VIF Module
- Variance Inflation Factor calculation
- Multicollinearity detection
- Correlation-based feature removal

## 🔄 Migration Guide

### Old Import → New Import

```python
# OLD (deprecated)
from src.utils.feature_selection_regularization import FeatureRegularizationSelector

# NEW
from src.feature_selection.methods import FeatureRegularizationSelector
```

```python
# OLD (deprecated)
from src.utils.feature_selection import select_features

# NEW
from src.feature_selection import select_features
```

```python
# OLD (deprecated)
from src.utils.sr_clustering.adaptive_feature_selection import AdaptiveFeatureSelector

# NEW
from src.feature_selection.specialized import AdaptiveFeatureSelector
```

```python
# OLD (deprecated)
from src.training.steps.pre_training.feature_lookback_optimization.directional_feature_selection_adapter import DirectionalFeatureSelectionConfig

# NEW
from src.feature_selection.specialized import DirectionalFeatureSelectionConfig
```

### Backward Compatibility

All old import locations have compatibility shims that will:
1. Issue deprecation warnings
2. Import from the new location
3. Work without code changes (for now)

**⚠️ Note**: Compatibility shims will be removed in a future version. Please update your imports.

## 📚 Examples

### Example 1: Comprehensive Pipeline

```python
from src.feature_selection import run_comprehensive_feature_selection

result = run_comprehensive_feature_selection(
    X=features_df,
    y=target,
    feature_names=feature_names,
    target_features=80,
    model_type='lightgbm',
    enable_stability_analysis=True,
    enable_temporal_analysis=True,
)

print(f"Selected {len(result['selected_features'])} features")
print(f"Selection quality: {result['selection_quality_score']:.3f}")
```

### Example 2: Multi-Stage Selection

```python
from src.feature_selection.methods import FeatureRegularizationSelector
from src.feature_selection.specialized import AdaptiveFeatureSelector

# Stage 1: Regularization-based selection (120 → 80)
reg_selector = FeatureRegularizationSelector()
reg_selector.fit(X, y, feature_names=features)
X_stage1 = reg_selector.transform(X)
features_stage1 = reg_selector.get_selected_features()

# Stage 2: Adaptive selection (80 → 60)
adaptive_selector = AdaptiveFeatureSelector()
result = adaptive_selector.select_features(X_stage1, y, features_stage1)
final_features = result.selected_features

print(f"Stage 1: {len(features)} → {len(features_stage1)} features")
print(f"Stage 2: {len(features_stage1)} → {len(final_features)} features")
```

### Example 3: Custom Configuration

```python
from src.feature_selection import get_feature_selection_framework

# Get framework with custom configuration
framework = get_feature_selection_framework(config={
    'max_features': 100,
    'stability_threshold': 0.7,
    'use_temporal_analysis': True,
    'use_causal_analysis': False,
})

# Run selection
result = framework.select_features(
    X=features_df,
    y=target,
    method='comprehensive',
    max_features=80
)
```

## 🔍 Key Features

### Stability Selection
- **Block Bootstrap**: Respects time series structure
- **Feature Fraction Sampling**: Random subsampling (60-80%)
- **Stability Scores**: Features selected across multiple bootstraps
- **Cluster Awareness**: ≤1 feature per correlation cluster

### Time Series Aware
- **Temporal Analysis**: Considers time-based patterns
- **No Lookahead Bias**: Respects temporal ordering
- **Block Cross-Validation**: Time series CV methods
- **Progressive Selection**: Start simple, add complexity

### Model-Specific Optimization
- **Model Profiles**: Different configs for different models
- **Priority Categories**: Momentum, volatility, microstructure, etc.
- **Feature Limits**: Respects 60-100 feature range for ML models
- **Performance Weighting**: Prioritizes predictive features

## 📊 Performance Considerations

- **Memory Efficient**: Optimized for large feature sets
- **Hardware Accelerated**: M1/GPU acceleration where available
- **Parallel Processing**: Multi-threaded selection methods
- **Caching**: Results caching for expensive operations

## 🧪 Testing

Feature selection tests are located in `tests/feature_selection/`:

```bash
# Run all feature selection tests
pytest tests/feature_selection/

# Run specific test module
pytest tests/feature_selection/test_adaptive_selector.py
```

## 📝 Contributing

When adding new feature selection methods:

1. Choose the appropriate subdirectory:
   - `methods/` for general algorithms
   - `specialized/` for domain-specific selectors
   - `analysis/` for analysis tools

2. Follow the existing patterns:
   - Clear docstrings
   - Type hints
   - Error handling
   - Unit tests

3. Update `__init__.py` files to export new classes/functions

4. Add examples to this README

## 🔗 Related Modules

- **Training Framework**: `src/training/utils/feature_selection/` (training-specific logic)
- **Feature Engineering**: `src/feature_generation/` (feature creation)
- **Model Training**: `src/training/` (model training pipeline)

## 📖 Additional Resources

- [Feature Selection Theory](https://en.wikipedia.org/wiki/Feature_selection)
- [mRMR Algorithm](https://en.wikipedia.org/wiki/Feature_selection#Minimum-redundancy-maximum-relevance_(mRMR)_feature_selection)
- [Stability Selection Paper](https://arxiv.org/abs/0809.2932)
- [Recursive Feature Elimination](https://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination)

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Maintainers**: Ares Team
