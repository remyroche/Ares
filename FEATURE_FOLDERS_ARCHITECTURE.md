# Feature Folders Architecture - Complete Guide

## Overview Map

Your codebase now has **4 distinct feature-related directories** after Strategy C implementation:

```
src/
├── features_common/              # 🆕 Shared utilities (NEW)
├── feature_generation/           # 🔵 General purpose (100+ generators)
├── feature_engineering_roadmap/  # 🟢 End-to-end roadmap (32 features)
└── feature_selection/            # 🟣 Feature selection & optimization

PLUS utilities that support these:
├── utils/ml_common/              # ML utilities (CV, HPO, validation)
├── utils/matrix_operations/      # Hardware-optimized operations
├── utils/hardware/               # M1-specific optimizations
└── utils/[data, tprint, etc.]    # Common utilities
```

---

## 1. features_common/ 🆕 **SHARED UTILITIES**

### Role
**Shared base classes and utilities** to reduce duplication between feature_generation and feature_engineering_roadmap.

### Location
```
src/features_common/
├── __init__.py
├── transforms/
│   ├── base_scaler.py        # BaseScaler interface + utility methods
│   └── __init__.py
├── optimization/
│   ├── cv_base.py             # BaseCVSplitter, PurgedCVSplitter
│   └── __init__.py
└── registry/
    ├── base_registry.py       # BaseFeatureRegistry interface
    └── __init__.py
```

### Key Classes
1. **BaseScaler** - Abstract base for all scaling/transformation
   ```python
   from src.features_common.transforms.base_scaler import BaseScaler
   
   class MyScaler(BaseScaler):
       def fit_transform(self, data: pd.Series) -> pd.Series:
           # Your implementation
       def transform(self, data: pd.Series) -> pd.Series:
           # Transform logic
   ```
   
   **Features:**
   - ✅ Integrates `tprint` for logging
   - ✅ Uses `math_validation` for safety
   - ✅ Provides `_safe_divide`, `_log_info`, `_log_success`, `_log_warning`
   - ✅ State persistence (get_state/set_state)

2. **BaseCVSplitter** - Time series CV with embargo
   ```python
   from src.features_common.optimization.cv_base import BaseCVSplitter
   
   splitter = BaseCVSplitter(n_folds=5, embargo_pct=0.1)
   for train_idx, val_idx in splitter.split_with_embargo(X):
       # Train/validate
   ```

3. **BaseFeatureRegistry** - Registry interface

### Who Uses It
- ✅ `feature_engineering_roadmap/transforms.py` (OnlineEWZ, MADScaler, etc.)
- ✅ `feature_generation/categories/normalization.py` (ZScoreNormalizer, RobustScaler, etc.)
- ✅ Any future scalers/transforms should inherit from BaseScaler

### When to Use
- Creating new scaling/normalization methods
- Need consistent interface across systems
- Want to leverage shared utilities (tprint, math_validation)

---

## 2. feature_generation/ 🔵 **GENERAL PURPOSE**

### Role
**Flexible feature generation framework** for exploration, backtesting, and most trading models.

### Location
```
src/feature_generation/
├── core/                         # Framework core
│   ├── feature_generator.py     # FeatureGenerator, VectorizedFeatureGenerator
│   ├── feature_registry.py      # Dynamic registry (100+ generators)
│   ├── factory.py               # FeatureFactory
│   ├── feature_bank.py          # Storage + caching
│   └── feature_cache.py         # Caching mechanisms
│
├── categories/                   # 35 feature categories
│   ├── momentum.py              # RSI, MACD, momentum indicators
│   ├── volatility.py            # ATR, Bollinger, volatility measures
│   ├── volume.py                # OBV, VWAP, volume features
│   ├── oscillator.py            # Stochastic, Williams %R
│   ├── trend.py                 # SMA, EMA, ADX, trend indicators
│   ├── returns.py               # Return calculations
│   ├── normalization.py         # Z-score, robust, min-max (now uses BaseScaler!)
│   ├── interaction.py           # Feature interactions
│   ├── support_resistance.py   # S/R levels
│   ├── microstructure.py       # Market microstructure
│   ├── entropy.py               # Entropy-based features
│   ├── regime.py                # Regime detection
│   └── [25+ more categories]
│
├── base_calculations/            # Base calculation types
│   ├── base_calculator.py       # BaseCalculator interface
│   └── [calculation implementations]
│
├── utils/                        # Utilities (50+ files!)
│   ├── optimization/
│   │   ├── lookback_optimizer.py      # Lookback optimization
│   │   └── unified_optimizer.py       # Unified optimization
│   ├── vectorization_optimizer.py     # Vectorization utilities
│   ├── enhanced_matrix_accelerator.py # Matrix acceleration
│   └── [40+ other utilities]
│
└── examples/                     # Usage examples
```

### Key Components

**Base Classes:**
- `FeatureGenerator` - Base for all generators
- `VectorizedFeatureGenerator` - Optimized with matrix ops
- `FeatureConfig` - Configuration dataclass
- `FeatureCategory` - Category enum

**Features:**
- 🔢 100+ feature generators
- 📊 35+ categories
- ⚙️ Dynamic registration
- 🚀 Performance optimized (GPU, vectorization, caching)
- 🔧 Flexible parameters

### When to Use
- ✅ Exploratory feature engineering
- ✅ Backtesting strategies
- ✅ Analyst model features
- ✅ Tactician model features
- ✅ General trading indicators
- ✅ Feature research & discovery
- ✅ Custom feature development

### Example
```python
from src.feature_generation.categories.momentum import RSIGenerator
from src.feature_generation.categories.volatility import ATRGenerator
from src.feature_generation.core.feature_registry import FeatureRegistry

# Create flexible generators
rsi = RSIGenerator(period=14)  # Can customize period
atr = ATRGenerator(period=20)   # Can customize period

# Generate features
rsi_features = rsi.generate(data)
atr_features = atr.generate(data)

# Use registry for management
registry = FeatureRegistry()
registry.register(rsi)
registry.register(atr)
```

### Integration with Utilities
- ✅ **matrix_operations/** - Vectorized processing
- ✅ **hardware/m1_*.py** - GPU/CPU/Memory optimization
- ✅ **ml_common/optimization/** - Lookback optimization
- ✅ **features_common/** - BaseScaler for normalization

---

## 3. feature_engineering_roadmap/ 🟢 **END-TO-END ROADMAP**

### Role
**Locked, theory-driven features** specifically for end-to-end roadmap training. Features are immutable with exact formulas.

### Location
```
src/feature_engineering_roadmap/      # (Renamed from feature_engineering)
├── feature_registry.py      # 32 parent features (locked formulas)
├── interactions.py          # 15 theory-driven interactions
├── transforms.py            # Transform pipeline (now uses BaseScaler!)
├── lookback_selection.py    # Lookback optimization with hysteresis
├── assembly_dag.py          # Feature assembly DAG
├── data_contracts.py        # Data contracts
├── disagreement_meta_features.py    # Ensemble meta-features
├── ensemble_meta_features.py        # Additional meta-features
└── step06_labeling_components/      # Labeling components
```

### Key Components

**32 Parent Features** (immutable):
- `p/r1`, `p/r3`, `p/r5`, `p/r10` - Returns
- `p/mom5`, `p/mom10`, `p/mom20` - Momentum
- `p/sigma_ew`, `p/gk_w`, `p/rv_bipower_12` - Volatility
- `p/rsi7`, `p/rsi14`, `p/stochk14` - Mean reversion
- `p/volume_z18`, `p/dollarvol_z18` - Liquidity
- `p/vwap_session_dist`, `p/vwap_roll12_dist` - Anchors
- [See feature_registry.py for all 32]

**15 Interactions** (theory-driven):
- **Tension:** `i/tension/mom5_x_negmom20`, `i/tension/rsi14_x_highvol`
- **Micro:** `i/micro/ofi_x_spread`, `i/micro/tradecount_x_spread`
- **Vol:** `i/vol/r1_x_rvshort`, `i/vol/sigmaew_x_posmom5_guard`
- **Model:** `i/model/yhat1_x_rvshort`, `i/model/yhatconf_x_widespread`

**Transforms** (now inherits from BaseScaler):
- `OnlineEWZ` - Exponential weighted z-score
- `TODRank` - Time-of-day ranking
- `SignedLog` - Signed log for heavy tails
- `MADScaler` - Median absolute deviation
- `Winsorization` - Quantile clipping

### When to Use
- ✅ End-to-end roadmap model training **ONLY**
- ✅ Need exact 32 parent features
- ✅ Need 15 locked interactions
- ✅ Using roadmap transform pipeline

### Example
```python
from src.feature_engineering_roadmap.feature_registry import FeatureRegistry
from src.feature_engineering_roadmap.transforms import TransformRouter
from src.feature_engineering_roadmap.interactions import InteractionEngine

# 1. Generate parent features (locked formulas)
registry = FeatureRegistry()
r1 = registry.compute_feature('p/r1', data)    # Always log(Ct/Ct-1)
mom5 = registry.compute_feature('p/mom5', data) # Always (Ct/Ct-5) - 1

# 2. Apply transforms
transformer = TransformRouter(transform_config)
transformed = transformer.fit_transform(train_data, val_data)

# 3. Build interactions
engine = InteractionEngine(interaction_config)
interactions = engine.build_interactions(transformed)
```

### Integration with Utilities
- ✅ **features_common/BaseScaler** - All transforms inherit
- ✅ **ml_common/optimization/** - Lookback selection
- ⚠️ **tprint/math_validation** - Available via BaseScaler methods

---

## 4. feature_selection/ 🟣 **FEATURE SELECTION**

### Role
**Feature selection and optimization** - reduces feature sets, removes redundancy, selects best features.

### Location
```
src/feature_selection/
├── base_framework.py          # Base selection framework
├── selection_methods.py       # Selection algorithms
├── stability_analysis.py      # Stability metrics
├── performance_monitoring.py  # Performance tracking
├── quality_metrics.py         # Quality assessment
├── temporal_analysis.py       # Temporal stability
├── causal_analysis.py         # Causal feature analysis
└── main_framework.py          # Main selection pipeline
```

### Key Capabilities
- 📉 Feature reduction (dimensionality)
- 🔍 Redundancy detection
- 📊 Information coefficient (IC) analysis
- ⏰ Temporal stability checks
- 🎯 Causal feature identification
- 🔬 Performance monitoring

### When to Use
- After generating features (from either system)
- Need to reduce feature dimensionality
- Want to remove redundant/correlated features
- Optimize for model performance
- Analyze feature importance

### Example
```python
from src.feature_selection.main_framework import FeatureSelectionFramework

# After generating features
all_features = ... # From feature_generation or feature_engineering_roadmap

# Select best features
selector = FeatureSelectionFramework(
    selection_methods=['mutual_info', 'correlation', 'stability'],
    n_features_to_select=50
)

selected_features = selector.select_features(
    features=all_features,
    targets=targets,
    validation_data=val_data
)
```

### Integration with Utilities
- ✅ **ml_common/validation/** - Validation framework
- ✅ **ml_common/optimization/** - Optimization framework
- ✅ **Matrix operations** - Fast computation

---

## Complete Feature Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE PIPELINE FLOW                        │
└─────────────────────────────────────────────────────────────────┘

STEP 1: FEATURE GENERATION
    ↓
┌─────────────────────────────────────────────┐
│ Choose System:                               │
│                                              │
│ A) feature_generation/                      │
│    • For: Exploration, backtesting          │
│    • Creates: 100+ flexible features        │
│    • Example: RSI(14), ATR(20), custom      │
│                                              │
│ B) feature_engineering_roadmap/             │
│    • For: End-to-end roadmap training       │
│    • Creates: 32 parent features            │
│    • Example: p/r1, p/mom5, p/sigma_ew      │
└─────────────────────────────────────────────┘
    ↓
STEP 2: TRANSFORMATION (Optional)
    ↓
┌─────────────────────────────────────────────┐
│ Apply Scaling/Normalization:                │
│                                              │
│ • features_common/BaseScaler               │
│   - ZScoreNormalizer                        │
│   - RobustScaler                            │
│   - MinMaxScaler                            │
│                                              │
│ • feature_engineering_roadmap/transforms   │
│   - OnlineEWZ                               │
│   - TODRank                                 │
│   - SignedLog                               │
│   - MADScaler                               │
│   - Winsorization                           │
└─────────────────────────────────────────────┘
    ↓
STEP 3: INTERACTIONS (Optional)
    ↓
┌─────────────────────────────────────────────┐
│ Create Feature Interactions:                │
│                                              │
│ A) feature_generation/categories/           │
│    interaction.py                           │
│    • MomentumVolumeGenerator                │
│    • VolatilityVolumeGenerator              │
│    • CrossTimeframeInteractionGenerator     │
│                                              │
│ B) feature_engineering_roadmap/             │
│    interactions.py                          │
│    • i/tension/mom5_x_negmom20              │
│    • i/micro/ofi_x_spread                   │
│    • i/vol/r1_x_rvshort                     │
└─────────────────────────────────────────────┘
    ↓
STEP 4: FEATURE SELECTION
    ↓
┌─────────────────────────────────────────────┐
│ feature_selection/                          │
│    • Remove redundant features              │
│    • Select top N by IC/AUC                 │
│    • Stability analysis                     │
│    • Causal feature identification          │
└─────────────────────────────────────────────┘
    ↓
STEP 5: MODEL TRAINING
    ↓
    [Final feature set ready for ML models]
```

---

## Decision Matrix

### Which System Should I Use?

| Scenario | System | Reason |
|----------|--------|--------|
| **Backtesting new strategy** | `feature_generation/` | Need flexible, customizable features |
| **Exploring momentum indicators** | `feature_generation/` | Has 100+ generators, easy to experiment |
| **Training Analyst models** | `feature_generation/` | General-purpose features |
| **Training Tactician models** | `feature_generation/` | General-purpose features |
| **End-to-end roadmap training** | `feature_engineering_roadmap/` | Need exact 32 parent features |
| **Creating new scaler** | `features_common/` | Inherit from BaseScaler |
| **Reducing features** | `feature_selection/` | After generation, optimize set |

---

## Utility Integration Map

### utils/ml_common/ **ML UTILITIES**
**Used by:** All feature systems

**Key Modules:**
```
ml_common/
├── optimization/
│   ├── bayesian_tpe_optimizer.py    # Used by: lookback optimization
│   ├── grid_search.py               # Used by: hyperparameter tuning
│   └── multi_objective.py           # Used by: pareto optimization
│
├── validation/
│   ├── enhanced_validation.py       # Used by: data validation
│   ├── lookahead_protection.py      # Used by: CV splitting
│   └── data_leakage.py              # Used by: train/val separation
│
├── cross_validation/
│   ├── purged_kfold.py              # Used by: time series CV
│   └── oof_predictions.py           # Used by: out-of-fold validation
│
└── utils/
    ├── memory_optimization.py       # Used by: large dataset handling
    └── memory_integration.py        # Used by: memory-aware operations
```

**Integration Examples:**
```python
# In feature_generation lookback optimization
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer

# In feature_engineering_roadmap lookback selection  
from src.utils.ml_common.validation.lookahead_protection import check_lookahead

# In features_common CV splitting
from sklearn.model_selection import TimeSeriesSplit  # Standard, but ml_common adds validation
```

---

### utils/matrix_operations/ **MATRIX OPERATIONS**
**Used by:** feature_generation (heavily)

**Key Modules:**
```
matrix_operations/
├── unified_operations.py         # Main interface
├── hardware_integration.py       # Hardware optimization
├── vectorized_core.py           # Vectorized operations
└── enhanced_operations.py       # Custom operations
```

**Integration:**
```python
# feature_generation/core/feature_generator.py
from ...utils.matrix_operations import get_unified_matrix_operations

class VectorizedFeatureGenerator(FeatureGenerator):
    def __init__(self):
        self.matrix_ops = get_unified_matrix_operations()
        
    def _vectorized_operation(self, operation, data, **kwargs):
        return self.matrix_ops.batch_process(data, operation, **kwargs)
```

---

### utils/hardware/ **M1 OPTIMIZATION**
**Used by:** All systems (via matrix_operations)

**Key Modules:**
```
hardware/
├── m1_gpu_utils.py              # GPU acceleration
├── m1_memory_optimizer.py       # Memory management
└── m1_cpu_optimizer.py          # CPU optimization
```

**Integration:**
```python
# Automatically detected and used
# See logs:
# ✅ M1 GPU acceleration available
# ✅ M1 Memory optimizer initialized
# ✅ M1 CPU optimizer initialized
```

---

### utils/ **COMMON UTILITIES**
**Used by:** All systems

**Key Files:**
```
utils/
├── tprint.py                 # ✅ NOW used via BaseScaler._log_*
├── math_validation.py        # ✅ NOW used via BaseScaler._safe_divide
├── common_operations.py      # ⚠️ Evaluate for additional usage
├── common_utilities.py       # ⚠️ Evaluate for additional usage
└── data/                     # Data quality, cleaning, streaming
    ├── data_quality.py
    ├── data_cleaner.py
    └── data_streaming.py
```

**Integration:**
```python
# features_common/transforms/base_scaler.py
from src.utils.tprint import tprint
from src.utils.math_validation import safe_divide, check_for_inf_nan

class BaseScaler(ABC):
    def _log_success(self, message):
        tprint(message, color="green")  # ✅ Enhanced UX
    
    def _safe_divide(self, numerator, denominator, default=0.0):
        return safe_divide(numerator, denominator, default)  # ✅ Safe math
```

---

## Usage Patterns

### Pattern 1: General Feature Generation

```python
# Use feature_generation for exploration
from src.feature_generation.categories.momentum import MomentumGenerator
from src.feature_generation.categories.volatility import VolatilityGenerator

# Generate flexible features
momentum = MomentumGenerator(period=14).generate(data)
volatility = VolatilityGenerator(window=20).generate(data)

# Optional: Normalize with features_common
from src.features_common.transforms.base_scaler import ZScoreNormalizer

normalizer = ZScoreNormalizer()
normalized = normalizer.fit_transform(momentum.data)
```

### Pattern 2: End-to-End Roadmap Training

```python
# Use feature_engineering_roadmap for roadmap
from src.feature_engineering_roadmap.feature_registry import FeatureRegistry
from src.feature_engineering_roadmap.transforms import TransformRouter
from src.feature_engineering_roadmap.interactions import InteractionEngine

# 1. Generate 32 parent features (locked)
registry = FeatureRegistry()
parents = {name: registry.compute_feature(name, data) 
           for name in registry.get_all_features()}

# 2. Transform (uses BaseScaler internally)
transformer = TransformRouter(config)
transformed = transformer.fit_transform(train_data, val_data)

# 3. Build 15 interactions
engine = InteractionEngine(config)
interactions = engine.build_interactions(transformed)

# 4. Combine
final_features = pd.concat([transformed, interactions], axis=1)
```

### Pattern 3: Feature Selection

```python
# Use feature_selection after generation
from src.feature_selection.main_framework import FeatureSelectionFramework

# Generate features (from either system)
features = ...  # From feature_generation or feature_engineering_roadmap

# Select best features
selector = FeatureSelectionFramework(
    selection_methods=['mutual_info', 'stability', 'ic'],
    n_features_to_select=50
)

selected = selector.select_features(features, targets)
```

---

## System Comparison

| Aspect | features_common | feature_generation | feature_engineering_roadmap | feature_selection |
|--------|----------------|-------------------|----------------------------|------------------|
| **Role** | Shared utilities | General features | Roadmap features | Feature optimization |
| **Size** | 7 files | 80+ files | 10 files | 10 files |
| **Features** | 3 base classes | 100+ generators | 32 locked + 15 interactions | Selection algorithms |
| **Flexibility** | N/A | HIGH | LOW | MEDIUM |
| **Use Case** | Base classes | General purpose | Roadmap only | Post-generation |
| **Utilities** | tprint, math_val | matrix_ops, hardware | BaseScaler, lookback | IC, stability, causal |

---

## Integration Points

### How Systems Work Together

```
[Market Data]
      │
      ├──→ feature_generation/ ──→ [100+ features]
      │         ↓
      │    Uses: matrix_operations, hardware/m1_*, ml_common
      │         ↓
      │    Normalizes via: features_common/BaseScaler
      │         ↓
      └──→ feature_engineering_roadmap/ ──→ [32 parents + 15 interactions]
            ↓
       Uses: features_common/BaseScaler, ml_common/lookback
            ↓
       Both feed into:
            ↓
┌───────────────────────────────┐
│   feature_selection/          │
│   • Reduces to top N          │
│   • Removes redundancy        │
│   • Stability analysis        │
└───────────────────────────────┘
            ↓
      [Optimized Feature Set]
            ↓
      [Model Training]
```

---

## Directory Summary

### 📁 features_common/
- **What:** Shared base classes
- **For:** Both feature_generation and feature_engineering_roadmap
- **Size:** Small (7 files)
- **Purpose:** Reduce duplication
- **Utilities:** tprint ✅, math_validation ✅

### 📁 feature_generation/
- **What:** General-purpose feature framework
- **For:** Exploration, backtesting, most models
- **Size:** Large (80+ files)
- **Purpose:** Flexible feature engineering
- **Utilities:** matrix_ops ✅, hardware ✅, ml_common ✅, BaseScaler ✅

### 📁 feature_engineering_roadmap/
- **What:** Locked features for roadmap
- **For:** End-to-end roadmap training only
- **Size:** Small (10 files)
- **Purpose:** Theory-driven, immutable features
- **Utilities:** BaseScaler ✅, ml_common ✅, tprint (via BaseScaler) ✅

### 📁 feature_selection/
- **What:** Feature optimization
- **For:** Post-generation optimization
- **Size:** Medium (10 files)
- **Purpose:** Select best features, remove redundancy
- **Utilities:** ml_common ✅, validation ✅

---

## Quick Reference Cheat Sheet

| I need to... | Use this folder | Entry point |
|-------------|-----------------|-------------|
| Generate RSI/MACD/ATR | `feature_generation/categories/` | `momentum.py`, `volatility.py` |
| Create custom indicator | `feature_generation/` | Inherit from `FeatureGenerator` |
| Train roadmap models | `feature_engineering_roadmap/` | `feature_registry.py` |
| Normalize features | `features_common/` or `feature_generation/` | `BaseScaler` or `normalization.py` |
| Create interactions | `feature_generation/categories/interaction.py` or `feature_engineering_roadmap/interactions.py` | Depends on use case |
| Optimize lookback | `feature_generation/utils/optimization/` or `feature_engineering_roadmap/lookback_selection.py` | Depends on system |
| Select features | `feature_selection/` | `main_framework.py` |
| Scale data robustly | `features_common/transforms/` | `BaseScaler` implementations |
| Cross-validate safely | `features_common/optimization/` | `BaseCVSplitter` |

---

## Visual Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    ARES FEATURE ARCHITECTURE                    │
└────────────────────────────────────────────────────────────────┘

                         [MARKET DATA]
                              │
                ┌─────────────┴─────────────┐
                │                           │
         ┌──────▼──────┐            ┌──────▼──────┐
         │  feature_   │            │  feature_   │
         │ generation/ │            │ engineering_│
         │             │            │  roadmap/   │
         │ 100+ gens   │            │ 32 parents  │
         └──────┬──────┘            └──────┬──────┘
                │                           │
                └──────────┬────────────────┘
                           │
                    ┌──────▼──────┐
                    │ features_   │
                    │  common/    │
                    │ BaseScaler  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  feature_   │
                    │ selection/  │
                    │ Optimize    │
                    └──────┬──────┘
                           │
                    [FINAL FEATURES]
                           │
                    [MODEL TRAINING]

                POWERED BY:
    ┌───────────────────────────────────────┐
    │ utils/matrix_operations/   (GPU, M1)  │
    │ utils/hardware/m1_*.py     (Hardware) │
    │ utils/ml_common/           (ML Utils) │
    │ utils/tprint, math_validation (Utils) │
    └───────────────────────────────────────┘
```

---

## Best Practices

### ✅ DO:
- Use `feature_generation/` for general purpose
- Use `feature_engineering_roadmap/` only for roadmap training
- Inherit from `BaseScaler` for new transforms
- Use `feature_selection/` to optimize feature sets
- Leverage `ml_common/` for CV, HPO, validation
- Use matrix_operations for performance

### ❌ DON'T:
- Mix features from both generation systems in same pipeline
- Add general features to `feature_engineering_roadmap/`
- Skip feature selection for high-dimensional data
- Ignore hardware optimizations (M1 utils)
- Forget to use BaseScaler utilities (tprint, math_validation)

---

## Summary

You now have a **well-organized, utility-enhanced** feature architecture:

1. **features_common/** - Shared foundation with tprint & math_validation ✅
2. **feature_generation/** - Flexible generation (100+ features) ✅
3. **feature_engineering_roadmap/** - Locked roadmap features (32+15) ✅
4. **feature_selection/** - Post-generation optimization ✅

All integrated with:
- ✅ M1 hardware optimization
- ✅ Matrix operations
- ✅ ML common utilities
- ✅ tprint for logging
- ✅ math_validation for safety

**Status:** Production ready with clear boundaries and excellent utility integration.

---

Last updated: October 8, 2025  
Post Strategy C Implementation + Utility Enhancement
