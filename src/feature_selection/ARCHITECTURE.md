# Feature Selection Module - Architecture Documentation

## 📐 Architecture Overview

The `src/feature_selection/` module follows a **hybrid architecture pattern** that combines:
1. **Facade Pattern**: Re-exports from training framework for core algorithms
2. **Full Implementation Pattern**: Complete implementations for specialized selectors

This document clarifies which components use which pattern and why.

---

## 🏗️ Architecture Patterns

### Pattern A: Facade/Re-export Pattern

**Purpose**: Provide a unified API by re-exporting components from the training framework without duplication.

**Files Using This Pattern**:
- `methods/mrmr.py` - Re-exports `MRMRSelector`
- `methods/stability_selection.py` - Re-exports `ElasticNetStabilitySelector`, `StabilityAnalyzer`
- `methods/importance.py` - Re-exports `FeatureImportanceRanker`
- `methods/wrapper_methods.py` - Re-exports `RecursiveFeatureEliminator`
- `core/framework.py` - Delegates to `src.training.utils.feature_selection.main_framework`

**Why**: These are core ML algorithms that are:
- Already well-implemented in the training framework
- Used heavily during model training
- Better maintained in a single location
- Part of the training pipeline infrastructure

**Example**:
```python
# methods/mrmr.py
from src.training.utils.feature_selection.selection_methods import MRMRSelector
__all__ = ['MRMRSelector']
```

**Dependency Flow**:
```
src.feature_selection.methods.mrmr
  └─> src.training.utils.feature_selection.selection_methods.MRMRSelector
        └─> Training framework implementation
```

---

### Pattern B: Full Implementation Pattern

**Purpose**: Provide complete, standalone implementations for specialized or domain-specific functionality.

**Files Using This Pattern**:
- `methods/regularization.py` (328 lines) - Feature regularization for tree models
- `specialized/adaptive_selector.py` (522 lines) - Small sample feature selection
- `specialized/directional_selector.py` (646 lines) - Long/short directional features
- `specialized/entropy_balancer.py` - Entropy-based filtering
- `dimensionality/pca_module.py` (362 lines) - PCA dimensionality reduction
- `dimensionality/vif_module.py` (437 lines) - VIF multicollinearity detection

**Why**: These are:
- Domain-specific selectors (e.g., long/short trading)
- Specialized for particular use cases (small samples, entropy)
- Independent of training framework
- Self-contained with their own logic

**Example**:
```python
# specialized/adaptive_selector.py (522 lines of implementation)
class AdaptiveFeatureSelector:
    """Adaptive feature selector that works with small samples."""
    def __init__(self, config: Optional[AdaptiveFeatureSelectionConfig] = None):
        # ... full implementation ...
```

**Dependency Flow**:
```
src.feature_selection.specialized.adaptive_selector
  └─> sklearn, numpy, pandas (direct dependencies)
  └─> src.utils.tprint (logging)
  └─> No dependency on training framework
```

---

## 📊 Dependency Hierarchy

### High-Level View

```
┌─────────────────────────────────────────────────────────────┐
│  User Code                                                   │
│  from src.feature_selection import select_features          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  src.feature_selection/                                      │
│  ┌─────────────────┐        ┌──────────────────┐           │
│  │ Core (Facade)   │───────▶│ Training         │           │
│  │ - framework.py  │        │ Framework        │           │
│  └─────────────────┘        └──────────────────┘           │
│  ┌─────────────────┐        ┌──────────────────┐           │
│  │ Methods         │───────▶│ Training         │           │
│  │ - mrmr.py       │ facades│ Framework        │           │
│  │ - importance.py │        │ Implementation   │           │
│  └─────────────────┘        └──────────────────┘           │
│  ┌─────────────────┐                                        │
│  │ Specialized     │  standalone implementations           │
│  │ - adaptive_*.py │───────▶ sklearn, numpy, pandas        │
│  │ - directional_* │                                        │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Dependency Graph

```
src.feature_selection.__init__.py
├─> core/framework.py (FACADE)
│   └─> src.training.utils.feature_selection.main_framework
│
├─> methods/
│   ├─> mrmr.py (FACADE)
│   │   └─> src.training.utils.feature_selection.selection_methods
│   ├─> stability_selection.py (FACADE)
│   │   └─> src.training.utils.feature_selection.{selection_methods, stability_analysis}
│   ├─> importance.py (FACADE)
│   │   └─> src.training.utils.feature_selection.selection_methods
│   ├─> wrapper_methods.py (FACADE)
│   │   └─> src.training.utils.feature_selection.selection_methods
│   └─> regularization.py (IMPLEMENTATION)
│       └─> sklearn, numpy, pandas
│
├─> specialized/
│   ├─> adaptive_selector.py (IMPLEMENTATION)
│   │   └─> sklearn, numpy, pandas
│   ├─> directional_selector.py (IMPLEMENTATION)
│   │   └─> src.training.steps.pre_training.feature_lookback_optimization.directional_lookback_optimizer
│   │   └─> src.training.utils.feature_selection (optional)
│   └─> entropy_balancer.py (IMPLEMENTATION)
│       └─> numpy, pandas
│
└─> dimensionality/
    ├─> pca_module.py (IMPLEMENTATION)
    │   └─> sklearn.decomposition.PCA
    └─> vif_module.py (IMPLEMENTATION)
        └─> sklearn, statsmodels
```

---

## 🔍 Design Rationale

### Why Use Facades?

**Pros**:
- ✅ Single source of truth for core algorithms
- ✅ No code duplication
- ✅ Consistent behavior with training framework
- ✅ Easier maintenance (fix once, works everywhere)
- ✅ Cleaner user-facing API

**Cons**:
- ⚠️ Dependency on training framework
- ⚠️ Less flexible for standalone use
- ⚠️ Potential circular dependency risk

### Why Use Full Implementations?

**Pros**:
- ✅ Complete control over logic
- ✅ No external dependencies (within reason)
- ✅ Easy to test in isolation
- ✅ Domain-specific optimizations
- ✅ Can evolve independently

**Cons**:
- ⚠️ More code to maintain
- ⚠️ Need to ensure consistency with similar components
- ⚠️ Potential duplication if not careful

---

## 🎯 When to Use Each Pattern

### Use Facade Pattern When:
- ✓ Component exists and is well-maintained in training framework
- ✓ Behavior should be identical across all use cases
- ✓ Component is core ML algorithm (mRMR, RFE, stability selection)
- ✓ Used primarily during model training
- ✓ Would duplicate significant existing code

### Use Full Implementation When:
- ✓ Domain-specific logic (e.g., long/short trading features)
- ✓ Specialized for particular scenarios (small samples, entropy filtering)
- ✓ Independent from training pipeline
- ✓ Requires custom optimizations
- ✓ Benefits from standalone testing

---

## 📝 Guidelines for Contributors

### Adding New Feature Selection Methods

**Step 1: Determine Pattern**

Ask these questions:
1. Does this already exist in `src.training.utils.feature_selection/`?
   - **Yes** → Use Facade Pattern
   - **No** → Continue to Q2

2. Is this a standard ML algorithm (mRMR, LASSO, RFE, etc.)?
   - **Yes** → Consider implementing in training framework first, then facade
   - **No** → Continue to Q3

3. Is this domain-specific (trading, directional, regime-specific)?
   - **Yes** → Use Full Implementation Pattern
   - **No** → Continue to Q4

4. Will this be used outside of training pipeline?
   - **Yes** → Use Full Implementation Pattern
   - **No** → Consider implementing in training framework, then facade

**Step 2: Choose Directory**

- **Core algorithms** → `methods/`
- **Domain-specific** → `specialized/`
- **Dimensionality reduction** → `dimensionality/`
- **Analysis tools** → `analysis/`

**Step 3: Implementation**

**For Facade Pattern**:
```python
"""
[Method Name] Feature Selection

This module provides [method name] imported from the training framework.
"""

from src.training.utils.feature_selection.[module] import [ClassName]

__all__ = ['[ClassName]']
```

**For Full Implementation**:
```python
"""
[Method Name] Feature Selection

This module provides [description of what it does].

Key Features:
- [Feature 1]
- [Feature 2]
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
# ... other imports

from src.utils.tprint import tprint
from src.utils.math_validation import validate_array

logger = logging.getLogger(__name__)

@dataclass
class [YourClass]Config:
    """Configuration for [your class]."""
    # ... config fields

class [YourClass]:
    """[Description]."""
    
    def __init__(self, config: Optional[[YourClass]Config] = None):
        """Initialize [your class]."""
        self.config = config or [YourClass]Config()
        tprint(f"🚀 Initialized {self.__class__.__name__}")
        # ... implementation

    def select_features(self, X, y, **kwargs):
        """Select features using [your method]."""
        tprint(f"🔍 Starting feature selection with {self.__class__.__name__}")
        # ... implementation
        tprint(f"✅ Selected {n} features")
        return result

def create_[your_class](**kwargs) -> [YourClass]:
    """Factory function for [your class]."""
    config = [YourClass]Config(**kwargs)
    return [YourClass](config)

__all__ = ['[YourClass]', '[YourClass]Config', 'create_[your_class]']
```

**Step 4: Update Exports**

1. Add to module's `__init__.py`
2. Add to main `src/feature_selection/__init__.py` if public API
3. Update README.md with usage example

---

## 🔗 Related Documentation

- [README.md](./README.md) - User guide and examples
- [MIGRATION_SUMMARY.md](./MIGRATION_SUMMARY.md) - Migration from old locations
- [Training Framework](../training/utils/feature_selection/) - Core implementations

---

## 🏛️ Architectural Principles

### 1. Single Responsibility
Each module should have one clear purpose:
- `methods/` - Standard selection algorithms
- `specialized/` - Domain-specific selectors
- `dimensionality/` - Dimensionality reduction
- `analysis/` - Feature analysis tools

### 2. Don't Repeat Yourself (DRY)
- Use facades to avoid duplicating training framework code
- Share common utilities through imports
- Extract repeated logic into helper functions

### 3. Separation of Concerns
- Keep training-specific logic in training framework
- Keep domain-specific logic in specialized selectors
- Keep general utilities in utility modules

### 4. Dependency Inversion
- Depend on abstractions (interfaces) not implementations
- Use optional dependencies with try-except imports
- Provide fallback behavior when dependencies unavailable

### 5. Open/Closed Principle
- Open for extension (easy to add new selectors)
- Closed for modification (existing selectors stable)
- Use configuration objects for extensibility

---

## 🚀 Future Enhancements

### Planned Improvements
1. **Plugin System**: Allow external selectors to register
2. **Caching Layer**: Cache expensive feature selection results
3. **Async Support**: Async feature selection for large datasets
4. **Visualization**: Built-in visualization of selection results
5. **Benchmarking**: Performance comparison between methods

### Migration Path
- Gradually move stable implementations from training framework
- Maintain backward compatibility through facades
- Deprecate old import paths over time
- Eventually consolidate all feature selection here

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Maintainers**: Ares Team
