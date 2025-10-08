# Quick Feature Systems Reference Guide

## TL;DR - Which System Should I Use?

```
┌─────────────────────────────────────────────────────┐
│ NEED TO GENERATE FEATURES?                          │
└─────────────────────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌───────────────────┐      ┌──────────────────────┐
│ End-to-End        │      │ Anything Else        │
│ Roadmap Training? │      │ (Backtesting,        │
│                   │      │  Exploration, etc.)  │
└───────────────────┘      └──────────────────────┘
        │                             │
        ▼                             ▼
┌───────────────────┐      ┌──────────────────────┐
│ USE:              │      │ USE:                 │
│ feature_          │      │ feature_generation/  │
│ engineering/      │      │                      │
└───────────────────┘      └──────────────────────┘
```

---

## Quick Comparison

| Aspect | feature_generation/ | feature_engineering/ |
|--------|---------------------|----------------------|
| **Purpose** | General-purpose feature generation | End-to-end roadmap only |
| **Features** | 100+ flexible generators | 32 locked parent features |
| **Interactions** | 9+ general types | 15 theory-driven interactions |
| **Registry** | Dynamic, category-based | Fixed, family-based |
| **Usage** | Exploration, backtesting, all models | End-to-end roadmap training |
| **Flexibility** | High - configurable | Low - immutable formulas |
| **Base Classes** | `FeatureGenerator`, `FeatureConfig` | `ParentFeature`, `FeatureMetadata` |

---

## Import Examples

### feature_generation/

```python
# Generating momentum features
from src.feature_generation.categories.momentum import (
    MomentumGenerator,
    RSIGenerator
)
from src.feature_generation.core.feature_registry import FeatureRegistry

# Create registry
registry = FeatureRegistry()

# Create and register generators
mom_gen = MomentumGenerator(period=14)
registry.register(mom_gen)

# Generate features
features = mom_gen.generate(data)
```

### feature_engineering/

```python
# End-to-end roadmap features
from src.feature_engineering.feature_registry import (
    FeatureRegistry,
    PriceReturnsFeatures,
    VolatilityFeatures
)
from src.feature_engineering.interactions import (
    InteractionEngine,
    create_default_interaction_config
)
from src.feature_engineering.transforms import (
    TransformRouter,
    create_default_transform_config
)

# Create feature registry
registry = FeatureRegistry()

# Compute parent features
r1 = registry.compute_feature('p/r1', data)

# Apply transforms
transform_config = create_default_transform_config(['p/r1'])
transformer = TransformRouter(transform_config)
transformed = transformer.fit_transform(train_data, val_data)

# Build interactions
interaction_config = create_default_interaction_config()
engine = InteractionEngine(interaction_config)
interactions = engine.build_interactions(transformed)
```

---

## Key Differences in Code

### Feature Definition

**feature_generation/**
```python
# Flexible, class-based
class CustomMomentumGenerator(FeatureGenerator):
    def __init__(self, period: int = 14):
        config = FeatureConfig(
            name=f"custom_momentum_{period}",
            category=FeatureCategory.MOMENTUM,
            description="Custom momentum calculation",
            required_columns=["close"],
            default_lookback=period
        )
        super().__init__(config)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Custom logic here
        return data['close'].pct_change(self.period)
```

**feature_engineering/**
```python
# Fixed, function-based
class PriceReturnsFeatures:
    @staticmethod
    def r1(data: pd.DataFrame) -> pd.Series:
        """1-bar return: log(Ct/Ct-1)"""
        return np.log(data['close'] / data['close'].shift(1))
    
    @staticmethod
    def mom5(data: pd.DataFrame) -> pd.Series:
        """5-bar momentum: (Ct/Ct-5) - 1"""
        return (data['close'] / data['close'].shift(5)) - 1
```

### Interaction Definition

**feature_generation/**
```python
# General-purpose interaction
class MomentumVolumeGenerator(FeatureGenerator):
    def __init__(self, period: int = 5):
        # ... config setup
        
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        price_momentum = data['close'].pct_change(self.period)
        volume_momentum = data['volume'].pct_change(self.period)
        return price_momentum * volume_momentum
```

**feature_engineering/**
```python
# Theory-driven, locked interaction
class InteractionEngine:
    def _tension_mom5_x_negmom20(self, data: pd.DataFrame) -> pd.Series:
        """t/mom5/* × (-t/mom20/*)
        
        Theory: Captures momentum tension between short and long periods.
        """
        mom5 = self._get_transformed_feature(data, 't/p/mom5')
        mom20 = self._get_transformed_feature(data, 't/p/mom20')
        return mom5 * (-mom20)
```

---

## Common Patterns

### Pattern 1: Feature Generation (Exploration)

**Use `feature_generation/`**

```python
from src.feature_generation.core.factory import FeatureFactory
from src.feature_generation.categories.momentum import MomentumGenerator
from src.feature_generation.categories.volatility import VolatilityGenerator

# Create factory
factory = FeatureFactory()

# Generate multiple features
generators = [
    MomentumGenerator(period=14),
    VolatilityGenerator(window=20),
]

# Generate all features
all_features = factory.generate_all(data, generators)
```

### Pattern 2: End-to-End Roadmap Training

**Use `feature_engineering/`**

```python
from src.feature_engineering.feature_registry import FeatureRegistry
from src.feature_engineering.transforms import TransformRouter
from src.feature_engineering.interactions import InteractionEngine
from src.feature_engineering.lookback_selection import LookbackSelector

# 1. Generate parent features
registry = FeatureRegistry()
parent_features = pd.DataFrame({
    name: registry.compute_feature(name, data)
    for name in registry.get_all_features()
})

# 2. Apply transforms
transformer = TransformRouter(transform_config)
transformed = transformer.fit_transform(train_data, val_data)

# 3. Build interactions
engine = InteractionEngine(interaction_config)
interactions = engine.build_interactions(transformed)

# 4. Select lookbacks
selector = LookbackSelector()
lookback_choices = selector.select_lookbacks(
    features=transformed,
    targets=targets,
    feature_families=feature_families
)
```

### Pattern 3: Backtesting

**Use `feature_generation/`**

```python
from src.feature_generation.core.feature_bank import FeatureBank
from src.feature_generation.core.feature_cache import FeatureCache

# Create feature bank with caching
bank = FeatureBank(cache=FeatureCache())

# Register features
bank.register_generator(MomentumGenerator(period=14))
bank.register_generator(RSIGenerator(period=14))

# Generate with caching for performance
features = bank.generate_all(data, use_cache=True)
```

---

## Overlap Areas & Recommendations

### ⚠️ If You Need Common Functionality

| Functionality | Recommended Approach |
|--------------|----------------------|
| **Z-score normalization** | Use `feature_generation/categories/normalization.py` for general use, `feature_engineering/transforms.py` (EW-Z) for roadmap |
| **Momentum calculation** | Use `feature_generation/categories/momentum.py` for flexible momentum, `feature_engineering/feature_registry.py` for exact formulas |
| **Feature interactions** | Use `feature_generation/categories/interaction.py` for exploration, `feature_engineering/interactions.py` for roadmap |
| **Lookback optimization** | Use `feature_generation/utils/optimization/lookback_optimizer.py` for general, `feature_engineering/lookback_selection.py` for roadmap |

---

## Migration Checklist

### Switching from feature_generation to feature_engineering (for roadmap)

- [ ] Verify you're working on end-to-end roadmap training
- [ ] Map your features to the 32 parent features
- [ ] Understand transform requirements (EW-Z, TOD Rank, etc.)
- [ ] Review the 15 locked interactions
- [ ] Test with end-to-end pipeline

### Switching from feature_engineering to feature_generation (for general use)

- [ ] Verify you need flexible feature generation
- [ ] Understand FeatureGenerator base class
- [ ] Create FeatureConfig for your features
- [ ] Register with FeatureRegistry
- [ ] Test with your specific use case

---

## FAQ

### Q: Can I use features from both systems?
**A:** Not recommended. Choose one system per model/pipeline to avoid confusion.

### Q: I found a useful feature in feature_engineering, can I use it elsewhere?
**A:** Yes, but consider:
1. Port it to feature_generation as a new generator
2. Extract common logic to a shared utility
3. If it's truly general-purpose, consider moving to features_common/

### Q: Which system is faster?
**A:** `feature_generation/` has more optimization utilities (matrix ops, GPU acceleration). `feature_engineering/` is optimized for its specific use case.

### Q: Can I add new features to feature_engineering?
**A:** Generally no - it's designed for locked, immutable features. For new features, use feature_generation/ unless you're modifying the end-to-end roadmap specification.

### Q: How do I know if my code is using the right system?
**A:** Check your imports:
- `from src.feature_generation...` → General purpose
- `from src.feature_engineering...` → End-to-end roadmap

---

## Getting Help

**For feature_generation:**
- Read: `src/feature_generation/README.md`
- Examples: `src/feature_generation/examples/`
- Support: General features team

**For feature_engineering:**
- Read: `src/feature_engineering/README.md` (to be created)
- Examples: `src/feature_engineering/` docstrings
- Support: End-to-end roadmap team

---

## Best Practices

### ✅ DO:
- Use feature_generation for exploratory work
- Use feature_engineering for end-to-end roadmap training
- Document which system you're using in your code
- Keep feature definitions in the appropriate system

### ❌ DON'T:
- Mix imports from both systems in the same pipeline
- Duplicate feature logic across both systems
- Add general-purpose features to feature_engineering
- Modify locked features in feature_engineering without team approval

---

## Quick Command Reference

### Generate features with feature_generation

```bash
# Example: Generate momentum features
python -c "
from src.feature_generation.categories.momentum import MomentumGenerator
import pandas as pd

data = pd.read_csv('your_data.csv')
gen = MomentumGenerator(period=14)
result = gen.generate(data)
print(result.data)
"
```

### Generate features with feature_engineering

```bash
# Example: Generate parent features for roadmap
python -c "
from src.feature_engineering.feature_registry import FeatureRegistry
import pandas as pd

data = pd.read_csv('your_data.csv')
registry = FeatureRegistry()
r1 = registry.compute_feature('p/r1', data)
print(r1)
"
```

---

Last updated: 2025-10-08
Maintained by: Platform Team
