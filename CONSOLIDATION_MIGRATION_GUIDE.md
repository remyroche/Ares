# Consolidation Migration Guide

This guide helps you migrate from the old redundant implementations to the new consolidated ones.

## Overview

We have consolidated multiple redundant implementations into two main systems:

1. **Consolidated Cross-Validation** (`src/utils/ml_common/validation/consolidated_cv.py`)
2. **Consolidated Hyperparameter Optimization** (`src/utils/ml_common/optimization/consolidated_hpo.py`)

## Cross-Validation Migration

### Old Imports → New Imports

| Old Import | New Import |
|------------|------------|
| `from enhanced_purged_cv import EnhancedPurgedTemporalKFold` | `from src.utils.ml_common.validation.consolidated_cv import ConsolidatedCrossValidator` |
| `from src.validation.walkforward_validation import WalkForwardValidator` | `from src.utils.ml_common.validation.consolidated_cv import ConsolidatedCrossValidator` |
| `from src.utils.ml_common.validation.unified_cv import UnifiedCrossValidator` | `from src.utils.ml_common.validation.consolidated_cv import ConsolidatedCrossValidator` |
| `from src.utils.purged_kfold import PurgedKFoldTime` | `from src.utils.ml_common.validation.consolidated_cv import ConsolidatedCrossValidator` |

### Usage Examples

#### Old Way (Purged CV)
```python
from enhanced_purged_cv import EnhancedPurgedTemporalKFold, PurgedCVConfig

config = PurgedCVConfig(n_splits=5, purge_length=1, embargo_length=1)
cv = EnhancedPurgedTemporalKFold(config)
```

#### New Way (Consolidated CV)
```python
from src.utils.ml_common.validation.consolidated_cv import (
    ConsolidatedCrossValidator, ConsolidatedCVConfig, ValidationType
)

config = ConsolidatedCVConfig(n_splits=5, purge_length=1, embargo_length=1)
cv = ConsolidatedCrossValidator(config, ValidationType.PURGED)
```

#### Old Way (Walk Forward)
```python
from src.validation.walkforward_validation import WalkForwardValidator, ValidationConfig

config = ValidationConfig(n_outer_folds=6, n_inner_folds=3)
validator = WalkForwardValidator(config)
```

#### New Way (Consolidated CV)
```python
from src.utils.ml_common.validation.consolidated_cv import (
    ConsolidatedCrossValidator, ConsolidatedCVConfig, ValidationType
)

config = ConsolidatedCVConfig(n_splits=6, enable_walk_forward=True)
cv = ConsolidatedCrossValidator(config, ValidationType.WALK_FORWARD)
```

### Convenience Functions

The consolidated system provides convenience functions for common use cases:

```python
from src.utils.ml_common.validation.consolidated_cv import (
    create_purged_cv, create_walk_forward_cv, create_temporal_cv, create_standard_cv
)

# Quick setup for common CV types
purged_cv = create_purged_cv(n_splits=5, purge_length=1, embargo_length=1)
walk_forward_cv = create_walk_forward_cv(n_splits=6, initial_train_size=0.6)
temporal_cv = create_temporal_cv(n_splits=5)
standard_cv = create_standard_cv(n_splits=5)
```

## Hyperparameter Optimization Migration

### Old Imports → New Imports

| Old Import | New Import |
|------------|------------|
| `from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization` | `from src.utils.ml_common.optimization.consolidated_hpo import ConsolidatedHPO` |
| `from src.utils.ml_common.optimization.hierarchical_hpo import HierarchicalHPO` | `from src.utils.ml_common.optimization.consolidated_hpo import ConsolidatedHPO` |
| `from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer` | `from src.utils.ml_common.optimization.consolidated_hpo import ConsolidatedHPO` |
| `from src.utils.ml_common.optimization.bohb_optimizer import BOHBOptimizer` | `from src.utils.ml_common.optimization.consolidated_hpo import ConsolidatedHPO` |

### Usage Examples

#### Old Way (HPO Utils)
```python
from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization

hpo = HyperparameterOptimization()
result = hpo.bayesian_optimization(model_factory, X, y, search_space)
```

#### New Way (Consolidated HPO)
```python
from src.utils.ml_common.optimization.consolidated_hpo import (
    ConsolidatedHPO, HPOConfig
)

config = HPOConfig(strategy='bayesian', n_trials=100)
hpo = ConsolidatedHPO(config)
result = hpo.optimize(model_factory, X, y, search_space)
```

#### Old Way (Hierarchical HPO)
```python
from src.utils.ml_common.optimization.hierarchical_hpo import HierarchicalHPO, HierarchicalHPOConfig

config = HierarchicalHPOConfig(...)
hpo = HierarchicalHPO(config)
result = hpo.optimize_ensemble(X_train, y_train)
```

#### New Way (Consolidated HPO)
```python
from src.utils.ml_common.optimization.consolidated_hpo import (
    ConsolidatedHPO, HPOConfig
)

config = HPOConfig(strategy='hierarchical', enable_hierarchical=True)
hpo = ConsolidatedHPO(config)
result = hpo.optimize(model_factory, X, y, search_space)
```

### Convenience Functions

The consolidated system provides convenience functions for common HPO strategies:

```python
from src.utils.ml_common.optimization.consolidated_hpo import (
    create_bayesian_hpo, create_bohb_hpo, create_grid_hpo, create_random_hpo
)

# Quick setup for common HPO strategies
bayesian_hpo = create_bayesian_hpo(n_trials=100, n_startup_trials=10)
bohb_hpo = create_bohb_hpo(n_trials=100, min_budget=1.0, max_budget=3.0)
grid_hpo = create_grid_hpo(n_trials=100, coarse_grid_points=5)
random_hpo = create_random_hpo(n_trials=100)
```

## Backward Compatibility

The consolidated implementations provide backward compatibility through legacy aliases:

### Cross-Validation Legacy Aliases
- `PurgedKFoldTime` → `ConsolidatedCrossValidator`
- `UniversalTemporalValidator` → `ConsolidatedCrossValidator`
- `WalkForwardValidator` → `ConsolidatedCrossValidator`
- `UnifiedCrossValidator` → `ConsolidatedCrossValidator`

### HPO Legacy Aliases
- `HyperparameterOptimization` → `ConsolidatedHPO`
- `HierarchicalHPO` → `ConsolidatedHPO`
- `BayesianTPEOptimizer` → `ConsolidatedHPO`
- `BOHBOptimizer` → `ConsolidatedHPO`

## Key Benefits

1. **Unified Interface**: Single API for all CV and HPO strategies
2. **Reduced Redundancy**: Eliminated 20+ redundant implementations
3. **Better Maintainability**: Centralized code with consistent behavior
4. **Enhanced Features**: Combined best features from all implementations
5. **Backward Compatibility**: Legacy code continues to work
6. **Comprehensive Reporting**: Detailed validation and optimization reports

## Migration Checklist

- [ ] Update imports to use consolidated implementations
- [ ] Replace old class instantiations with new consolidated classes
- [ ] Update configuration objects to use new config classes
- [ ] Test that functionality works as expected
- [ ] Update any custom extensions or wrappers
- [ ] Remove any direct dependencies on deleted files

## Support

If you encounter issues during migration, please:

1. Check this migration guide first
2. Review the consolidated implementation documentation
3. Test with the convenience functions for common use cases
4. Use the legacy aliases if you need immediate compatibility

The consolidated implementations maintain all the functionality of the original implementations while providing a cleaner, more maintainable codebase.