# Feature Optimization Integration

## Issue

**Current State:** `feature_engineering_roadmap/` has **locked, fixed features** (32 parents).  
**Desired State:** Use **optimized feature selection** from `feature_lookback_optimization/` before creating interactions.

## Solution: Hybrid Approach

### Architecture

```
[Market Data]
      │
      ├──→ feature_generation/ 
      │         ↓
      │    [Generate 100+ candidate features]
      │         ↓
      │    feature_lookback_optimization/
      │         ↓
      │    [Optimize lookback & select best features]
      │         ↓
      │    [Optimized feature set]
      │         ↓
      ├──→ feature_engineering_roadmap/
      │    (USE AS TRANSFORM/INTERACTION ENGINE ONLY)
      │         ↓
      │    [Apply transforms to optimized features]
      │         ↓
      │    [Generate interactions from optimized features]
      │         ↓
      └──→ feature_selection/
               ↓
          [Final reduction & optimization]
               ↓
          [Model Training]
```

### Key Changes

#### 1. feature_engineering_roadmap/ → Transform/Interaction Engine (Not Feature Generator)

**OLD role:** Generate 32 locked features + 15 interactions  
**NEW role:** Apply transforms & create interactions from OPTIMIZED features

**Changes:**
- ✅ `feature_registry.py` - Keep as reference/fallback
- ✅ `transforms.py` - PRIMARY USE (apply to any features)
- ✅ `interactions.py` - ADAPT to work with optimized features (not just locked 32)
- ✅ `lookback_selection.py` - INTEGRATE with feature_lookback_optimization/

#### 2. Use Feature Optimization Pipeline

**Primary logic:** `src/training/steps/pre_training/feature_lookback_optimization/`

**Process:**
```python
# Step 1: Generate candidates (feature_generation)
from src.feature_generation import FeatureBank

bank = FeatureBank()
candidate_features = bank.generate_features(
    data=data,
    categories=['momentum', 'volatility', 'volume', 'returns'],
    lookback_ranges={'momentum': [5, 10, 20], 'volatility': [10, 20, 50]}
)

# Step 2: Optimize lookback & select best (feature_lookback_optimization)
from src.training.steps.pre_training.feature_lookback_optimization import (
    FeatureLookbackOptimizer
)

optimizer = FeatureLookbackOptimizer(
    use_bayesian=True,
    n_trials=50,
    optimization_metric='ic'  # Information Coefficient
)

optimized_features = optimizer.optimize_and_select(
    features=candidate_features,
    targets=targets,
    n_features_to_select=32,  # Or dynamic
    validation_split=0.2
)

# Step 3: Apply transforms (feature_engineering_roadmap)
from src.feature_engineering_roadmap.transforms import TransformRouter

transformer = TransformRouter(transform_config)
transformed = transformer.fit_transform(
    train_data=optimized_features['train'],
    val_data=optimized_features['val']
)

# Step 4: Create interactions (feature_engineering_roadmap)
from src.feature_engineering_roadmap.interactions import InteractionEngine

engine = InteractionEngine(interaction_config)
interactions = engine.build_interactions(
    transformed_data=transformed,  # Uses OPTIMIZED features
    patch_features=model_predictions  # Optional model outputs
)

# Step 5: Final selection (feature_selection)
from src.feature_selection.main_framework import FeatureSelectionFramework

selector = FeatureSelectionFramework(n_features=50)
final_features = selector.select_features(
    features=pd.concat([transformed, interactions], axis=1),
    targets=targets
)
```

---

## Implementation

### 1. Create Dynamic Feature Selection Wrapper

**File:** `src/feature_engineering_roadmap/dynamic_feature_selector.py`

```python
"""
Dynamic Feature Selector for End-to-End Roadmap

Wraps feature_lookback_optimization to provide optimized feature selection
before applying transforms and interactions.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from src.training.steps.pre_training.feature_lookback_optimization import (
    FeatureLookbackOptimizer
)
from src.feature_generation import FeatureBank
from .transforms import TransformRouter
from .interactions import InteractionEngine

class DynamicRoadmapPipeline:
    """
    Pipeline that uses OPTIMIZED feature selection instead of locked features.
    
    Process:
    1. Generate candidate features from feature_generation
    2. Optimize lookback & select best features  
    3. Apply transforms (from roadmap)
    4. Generate interactions (from roadmap)
    """
    
    def __init__(self, 
                 n_candidate_features: int = 50,
                 n_final_features: int = 32,
                 use_bayesian: bool = True):
        self.n_candidate_features = n_candidate_features
        self.n_final_features = n_final_features
        self.use_bayesian = use_bayesian
        
        # Initialize components
        self.feature_bank = FeatureBank()
        self.lookback_optimizer = FeatureLookbackOptimizer(
            use_bayesian=use_bayesian
        )
    
    def generate_optimized_features(self,
                                     data: pd.DataFrame,
                                     targets: pd.Series,
                                     categories: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate optimized features using feature selection logic.
        
        Returns:
            Dict with 'train' and 'val' DataFrames of optimized features
        """
        if categories is None:
            categories = ['returns', 'momentum', 'volatility', 'volume']
        
        # 1. Generate candidates with feature_generation
        candidates = self.feature_bank.generate_features(
            data=data,
            categories=categories,
            lookback_optimization=False  # Will optimize next
        )
        
        # 2. Optimize lookback & select best
        optimized = self.lookback_optimizer.optimize_and_select(
            features=candidates,
            targets=targets,
            n_features=self.n_final_features
        )
        
        return optimized
    
    def apply_transforms_and_interactions(self,
                                           optimized_features: Dict[str, pd.DataFrame],
                                           transform_config: Any,
                                           interaction_config: Any) -> pd.DataFrame:
        """
        Apply roadmap transforms and interactions to optimized features.
        """
        # Apply transforms
        transformer = TransformRouter(transform_config)
        transformed = transformer.fit_transform(
            train_data=optimized_features['train'],
            val_data=optimized_features['val']
        )
        
        # Create interactions
        engine = InteractionEngine(interaction_config)
        interactions = engine.build_interactions(transformed)
        
        # Combine
        final = pd.concat([transformed, interactions], axis=1)
        
        return final
```

---

### 2. Update feature_engineering_roadmap Documentation

The roadmap system should be documented as:

**PRIMARY PURPOSE:** Transform & Interaction Engine (not feature generator)

**Use roadmap for:**
- ✅ Applying statistical transforms (EW-Z, TOD Rank, etc.) to ANY features
- ✅ Creating interactions from ANY features (not just the 32 locked)
- ✅ Regime-dependent interactions

**DON'T use roadmap for:**
- ❌ Feature generation (use feature_generation + lookback_optimization)
- ❌ Locked 32 features (keep as reference/fallback only)

---

### 3. Integration Example

**File:** `src/training/steps/pre_training/optimized_roadmap_integration.py`

```python
"""
Optimized Roadmap Integration

Demonstrates how to use feature_lookback_optimization with
feature_engineering_roadmap transforms/interactions.
"""

import pandas as pd
from typing import Dict, List

# Feature generation & optimization
from src.feature_generation import FeatureBank
from src.training.steps.pre_training.feature_lookback_optimization import (
    FeatureLookbackOptimizer
)

# Roadmap transforms & interactions (NOT feature generation)
from src.feature_engineering_roadmap.transforms import (
    TransformRouter, create_default_transform_config
)
from src.feature_engineering_roadmap.interactions import (
    InteractionEngine, create_default_interaction_config
)

def optimized_roadmap_pipeline(data: pd.DataFrame,
                                 targets: pd.Series,
                                 categories: List[str] = None) -> pd.DataFrame:
    """
    Complete optimized pipeline using best of both systems.
    
    Args:
        data: Market data
        targets: Target labels
        categories: Feature categories to generate
        
    Returns:
        Final feature set with optimized features + transforms + interactions
    """
    if categories is None:
        categories = ['returns', 'momentum', 'volatility', 'volume']
    
    # STEP 1: Generate candidate features (flexible)
    print("🔧 Step 1: Generating candidate features...")
    bank = FeatureBank()
    candidates = bank.generate_features(
        data=data,
        categories=categories,
        lookback_ranges={
            'momentum': [5, 10, 14, 20],
            'volatility': [10, 20, 50],
            'volume': [10, 20],
            'returns': [1, 3, 5, 10]
        }
    )
    print(f"✅ Generated {len(candidates.columns)} candidate features")
    
    # STEP 2: Optimize lookback & select best
    print("🎯 Step 2: Optimizing lookback periods...")
    optimizer = FeatureLookbackOptimizer(
        use_bayesian=True,
        n_trials=50
    )
    
    optimized = optimizer.optimize_and_select(
        features=candidates,
        targets=targets,
        n_features=32  # Or use dynamic selection
    )
    print(f"✅ Selected {len(optimized['train'].columns)} optimized features")
    
    # STEP 3: Apply roadmap transforms
    print("🔄 Step 3: Applying transforms...")
    transform_config = create_default_transform_config(
        optimized['train'].columns.tolist()
    )
    
    transformer = TransformRouter(transform_config)
    transformed_train = transformer.fit_transform(
        train_data=optimized['train'],
        val_data=optimized['val']
    )
    print(f"✅ Applied transforms, created {len(transformed_train.columns)} features")
    
    # STEP 4: Create interactions from optimized features
    print("🔗 Step 4: Generating interactions...")
    interaction_config = create_default_interaction_config()
    
    engine = InteractionEngine(interaction_config)
    interactions = engine.build_interactions(transformed_train)
    print(f"✅ Created {len(interactions.columns)} interactions")
    
    # STEP 5: Combine all
    final_features = pd.concat([
        optimized['train'],      # Original optimized features
        transformed_train,       # Transformed features
        interactions             # Interactions
    ], axis=1)
    
    print(f"✅ Final feature set: {len(final_features.columns)} features")
    
    return final_features
```

---

## Fixed Issues

### ✅ 1. Circular Import Issues
**Fixed:** Removed non-existent class exports from `__init__.py` and `interaction.py`

**Before:**
```python
from .interaction import RegimeDependentFeatureGenerator  # Doesn't exist!
```

**After:**
```python
# Removed from imports - not implemented
# Added comment explaining removal
```

### ✅ 2. Removed dollarvol_z18
**Fixed:** Removed from all files

**Changes:**
- ❌ `LiquidityMicroFeatures.dollarvol_z18()` - Removed
- ❌ `p/dollarvol_z18` metadata - Removed
- ❌ `i/micro/dollarvol_x_widespread` interaction - Removed
- ✅ Now 31 parent features (was 32)
- ✅ Now 14 interactions (was 15)

### ✅ 3. Use Optimized Feature Selection
**Implemented:** Integration guide and dynamic pipeline

**Key Points:**
- feature_engineering_roadmap is now a **TRANSFORM/INTERACTION ENGINE**
- NOT a feature generator
- Works with ANY features from feature_lookback_optimization

---

## Recommended Usage

### DON'T: Use Locked 32 Features

```python
# ❌ OLD APPROACH (locked, not recommended)
from src.feature_engineering_roadmap.feature_registry import FeatureRegistry

registry = FeatureRegistry()
# This generates locked features - not optimal!
features = {name: registry.compute_feature(name, data) 
            for name in registry.get_all_features()}
```

### DO: Use Optimized Selection + Roadmap Transforms

```python
# ✅ NEW APPROACH (optimized)
from src.feature_generation import FeatureBank
from src.training.steps.pre_training.feature_lookback_optimization import (
    FeatureLookbackOptimizer
)
from src.feature_engineering_roadmap.transforms import TransformRouter
from src.feature_engineering_roadmap.interactions import InteractionEngine

# 1. Generate flexible candidates
bank = FeatureBank()
candidates = bank.generate_features(data, categories=['all'])

# 2. OPTIMIZE (this is the key!)
optimizer = FeatureLookbackOptimizer(use_bayesian=True)
optimized = optimizer.optimize_and_select(candidates, targets, n_features=32)

# 3. Apply roadmap transforms to OPTIMIZED features
transformer = TransformRouter(config)
transformed = transformer.fit_transform(optimized['train'], optimized['val'])

# 4. Generate interactions from OPTIMIZED features
engine = InteractionEngine(config)
interactions = engine.build_interactions(transformed)
```

---

## Benefits

### Using Optimized Selection:
- ✅ Data-driven feature choice
- ✅ Bayesian optimization of lookback periods
- ✅ IC/AUC based selection
- ✅ Adapts to different markets/timeframes
- ✅ Better performance than locked features

### Keeping Roadmap Transforms/Interactions:
- ✅ Theory-driven transformations (EW-Z, TOD Rank)
- ✅ Regime-aware interactions
- ✅ Proven interaction patterns
- ✅ Statistical rigor

---

## Migration Path

### Phase 1: Keep Both (Current)
- feature_engineering_roadmap available as fallback
- Can use locked 31 features if optimization fails
- Backwards compatible

### Phase 2: Default to Optimized (Recommended)
- Use `DynamicRoadmapPipeline` by default
- Fall back to locked features only if needed
- Documented in training pipelines

### Phase 3: Fully Optimized (Future)
- Remove feature_registry.py locked features (optional)
- Keep only transforms.py and interactions.py
- Always use optimized selection

---

##Summary

**Changes Made:**
1. ✅ Fixed circular imports (removed non-existent classes)
2. ✅ Removed dollarvol_z18 (31 features now)
3. ✅ Created integration guide for optimized approach
4. ✅ Documented hybrid usage pattern

**Recommended Approach:**
- Use `feature_lookback_optimization/` for feature selection
- Use `feature_engineering_roadmap/` for transforms & interactions
- Apply to OPTIMIZED features, not locked list

**Status:** Ready for implementation in training pipelines
