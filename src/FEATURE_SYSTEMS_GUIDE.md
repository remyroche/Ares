# Feature Systems Guide

## Overview

This codebase has **two distinct feature systems** that serve different purposes. This guide helps you choose the right system for your needs.

## Quick Decision Tree

```
┌─────────────────────────────────────────────────────┐
│ Need to generate or work with features?             │
└─────────────────────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌───────────────────┐      ┌──────────────────────┐
│ End-to-End        │      │ General Purpose      │
│ Roadmap Training? │      │ (Backtesting, etc.)  │
└───────────────────┘      └──────────────────────┘
        │                             │
        ▼                             ▼
┌───────────────────┐      ┌──────────────────────┐
│ USE:              │      │ USE:                 │
│ feature_          │      │ feature_generation/  │
│ engineering_      │      │                      │
│ roadmap/          │      │                      │
└───────────────────┘      └──────────────────────┘
```

---

## System 1: feature_generation/ - General Purpose

### Purpose
Flexible, general-purpose feature generation framework for exploration, backtesting, and most models.

### Key Characteristics
- **100+ feature generators** organized by category
- **Dynamic registration** - create and register new features at runtime
- **Flexible parameters** - customize lookback periods, calculations, etc.
- **Optimized** - matrix operations, GPU support, caching
- **Category-based** - momentum, volatility, volume, oscillators, etc.

### Directory Structure
```
feature_generation/
├── core/
│   ├── feature_generator.py     # Base classes
│   ├── feature_registry.py      # Dynamic registry
│   ├── factory.py               # Feature factory
│   └── feature_bank.py          # Feature storage
├── categories/
│   ├── momentum.py              # Momentum features
│   ├── volatility.py            # Volatility features
│   ├── volume.py                # Volume features
│   ├── interaction.py           # Feature interactions
│   └── [30+ other categories]
└── utils/
    └── optimization/            # Performance optimizations
```

### When to Use
- ✅ Exploratory feature engineering
- ✅ Backtesting with custom features
- ✅ Analyst model features
- ✅ Tactician model features
- ✅ General trading strategies
- ✅ Feature discovery and research

### Example Usage
```python
from src.feature_generation.categories.momentum import MomentumGenerator
from src.feature_generation.core.feature_registry import FeatureRegistry

# Create generators with flexible parameters
registry = FeatureRegistry()
mom_gen = MomentumGenerator(period=14)  # Customizable
registry.register(mom_gen)

# Generate features
features = mom_gen.generate(data)
print(features.data)
```

### Base Classes
- `FeatureGenerator` - Base for all generators
- `FeatureConfig` - Configuration for generators
- `FeatureRegistry` - Registry for dynamic features
- `VectorizedFeatureGenerator` - Optimized generators

---

## System 2: feature_engineering_roadmap/ - End-to-End Roadmap

### Purpose
Locked, theory-driven features specifically for end-to-end roadmap training. Features are immutable with exact formulas.

### Key Characteristics
- **32 parent features** with exact, locked formulas
- **15 theory-driven interactions** (e.g., momentum tension)
- **Transform pipeline** (EW-Z, TOD Rank, Signed Log, MAD Scaler)
- **Lookback selection** with hysteresis and simplicity prior
- **Immutable** - features don't change once defined
- **Family-based** - organized by feature families

### Directory Structure
```
feature_engineering_roadmap/
├── feature_registry.py        # 32 parent features (locked)
├── interactions.py            # 15 theory-driven interactions
├── transforms.py              # Transform system
├── lookback_selection.py      # Lookback optimization
├── assembly_dag.py            # Feature assembly
└── disagreement_meta_features.py  # Meta-features
```

### When to Use
- ✅ End-to-end roadmap model training **ONLY**
- ✅ When you need the exact 32 parent features
- ✅ When you need the 15 locked interactions
- ✅ When using the roadmap transform pipeline

### Example Usage
```python
from src.feature_engineering_roadmap.feature_registry import FeatureRegistry
from src.feature_engineering_roadmap.transforms import TransformRouter
from src.feature_engineering_roadmap.interactions import InteractionEngine

# 1. Generate parent features (locked formulas)
registry = FeatureRegistry()
r1 = registry.compute_feature('p/r1', data)  # Always log(Ct/Ct-1)
mom5 = registry.compute_feature('p/mom5', data)  # Always (Ct/Ct-5) - 1

# 2. Apply transforms
transformer = TransformRouter(transform_config)
transformed = transformer.fit_transform(train_data, val_data)

# 3. Build interactions
engine = InteractionEngine(interaction_config)
interactions = engine.build_interactions(transformed)
```

### Base Classes
- `ParentFeature` - Base for parent features
- `FeatureMetadata` - Metadata for features
- `InteractionEngine` - Interaction builder
- `TransformRouter` - Transform pipeline

---

## Shared Utilities: features_common/

### Purpose
Common base classes and utilities shared between both systems to reduce duplication.

### Structure
```
features_common/
├── transforms/
│   └── base_scaler.py       # BaseScaler interface
├── optimization/
│   └── cv_base.py           # BaseCVSplitter
└── registry/
    └── base_registry.py     # BaseFeatureRegistry
```

### Usage
Both systems inherit from these common base classes:

```python
from src.features_common.transforms.base_scaler import BaseScaler

# feature_generation uses it
class ZScoreNormalizer(BaseScaler):
    ...

# feature_engineering_roadmap uses it  
class OnlineEWZ(BaseScaler):
    ...
```

---

## Key Differences

| Aspect | feature_generation/ | feature_engineering_roadmap/ |
|--------|---------------------|------------------------------|
| **Purpose** | General purpose | End-to-end roadmap only |
| **Features** | 100+ generators | 32 locked parent features |
| **Flexibility** | High - configurable | Low - immutable formulas |
| **Registration** | Dynamic at runtime | Fixed at definition |
| **Interactions** | 9+ flexible types | 15 locked interactions |
| **Use Cases** | Exploration, backtesting, most models | Roadmap training only |
| **Parameters** | Customizable (period=14) | Fixed (r1 = log(Ct/Ct-1)) |

---

## Best Practices

### ✅ DO:
- Use `feature_generation/` for general-purpose work
- Use `feature_engineering_roadmap/` only for end-to-end roadmap training
- Clearly document which system you're using
- Leverage `features_common/` base classes for new features
- Keep feature definitions in the appropriate system

### ❌ DON'T:
- Mix imports from both systems in the same pipeline
- Add general-purpose features to `feature_engineering_roadmap/`
- Modify locked features in `feature_engineering_roadmap/` without approval
- Duplicate feature logic across both systems
- Use `feature_engineering_roadmap/` for non-roadmap models

---

## Common Scenarios

### Scenario 1: Backtesting a New Strategy
**Use:** `feature_generation/`

```python
from src.feature_generation.categories.momentum import RSIGenerator
from src.feature_generation.categories.volatility import ATRGenerator

# Flexible feature generation for backtesting
rsi = RSIGenerator(period=14)
atr = ATRGenerator(period=20)

features = pd.DataFrame({
    'rsi': rsi.generate(data).data,
    'atr': atr.generate(data).data
})
```

### Scenario 2: Training End-to-End Roadmap
**Use:** `feature_engineering_roadmap/`

```python
from src.feature_engineering_roadmap.feature_registry import FeatureRegistry
from src.feature_engineering_roadmap.transforms import TransformRouter
from src.feature_engineering_roadmap.interactions import InteractionEngine

# Locked features for roadmap
registry = FeatureRegistry()
parent_features = {
    name: registry.compute_feature(name, data)
    for name in registry.get_all_features()
}

# Apply transforms
transformer = TransformRouter(config)
transformed = transformer.fit_transform(train_data, val_data)

# Build interactions
engine = InteractionEngine(config)
interactions = engine.build_interactions(transformed)
```

### Scenario 3: Creating a New Feature
**General feature:** Add to `feature_generation/`
**Roadmap-specific feature:** Discuss with team before adding to `feature_engineering_roadmap/`

---

## Migration Guide

### From feature_engineering_roadmap to feature_generation
If you need a roadmap feature for general use:

1. Create a new generator in `feature_generation/categories/`
2. Inherit from `BaseScaler` or `FeatureGenerator`
3. Make it flexible (allow parameter customization)
4. Register and test

### From feature_generation to feature_engineering_roadmap
Generally **not recommended** - roadmap features are locked by design. Discuss with team.

---

## FAQ

### Q: Can I use features from both systems?
**A:** Not recommended. Choose one system per model/pipeline to avoid confusion.

### Q: Which system is faster?
**A:** `feature_generation/` has more optimization utilities. `feature_engineering_roadmap/` is optimized for its specific use case.

### Q: Can I add new features to feature_engineering_roadmap?
**A:** Generally no - it's designed for locked features. For new features, use `feature_generation/` unless modifying the roadmap specification.

### Q: How do I know which system my code uses?
**A:** Check your imports:
- `from src.feature_generation...` → General purpose
- `from src.feature_engineering_roadmap...` → Roadmap only

### Q: What if I need common functionality?
**A:** Use or extend `features_common/` base classes.

---

## Getting Help

**For feature_generation:**
- Read: `src/feature_generation/README.md`
- Examples: `src/feature_generation/examples/`

**For feature_engineering_roadmap:**
- Read: `src/feature_engineering_roadmap/README.md`
- Examples: Docstrings in each module

**For features_common:**
- Read: Base class docstrings in `src/features_common/`

---

Last updated: 2025-10-08  
Strategy C Implementation
