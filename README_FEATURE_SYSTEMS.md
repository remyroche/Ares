# Feature Systems - Quick Reference

> **After Strategy C Implementation + Utility Integration**

## The 4 Feature Directories Explained

### 1️⃣ features_common/ 🆕 **[SHARED FOUNDATION]**
```
Location: src/features_common/
Size: 7 files
Purpose: Shared base classes to eliminate duplication
```

**What it provides:**
- `BaseScaler` - Abstract base for all scalers (with tprint & math_validation!)
- `BaseCVSplitter` - Time series CV with embargo
- `BaseFeatureRegistry` - Registry interface

**Who uses it:**
- Both feature_generation AND feature_engineering_roadmap
- Any new scaler/transform you create

**Key benefit:** Inherit once, get utilities for free (tprint, math_validation)

---

### 2️⃣ feature_generation/ 🔵 **[GENERAL PURPOSE - YOUR MAIN TOOL]**
```
Location: src/feature_generation/
Size: 80+ files, 100+ generators
Purpose: Flexible feature generation for EVERYTHING except roadmap
```

**What it provides:**
- **100+ feature generators** (RSI, MACD, ATR, Bollinger, etc.)
- **35+ categories** (momentum, volatility, volume, oscillators, trends, interactions)
- **Flexible parameters** (RSI can be period 7, 14, 21, or any number)
- **Performance optimized** (GPU, M1, vectorization, caching)

**Use this for:**
- ✅ Backtesting strategies
- ✅ Exploring new indicators
- ✅ Analyst models
- ✅ Tactician models
- ✅ Research
- ✅ 95% of your feature needs

**Example:**
```python
from src.feature_generation.categories.momentum import RSIGenerator

# Flexible - you control parameters
rsi14 = RSIGenerator(period=14)
rsi21 = RSIGenerator(period=21)
```

---

### 3️⃣ feature_engineering_roadmap/ 🟢 **[ROADMAP ONLY - LOCKED]**
```
Location: src/feature_engineering_roadmap/
Size: 10 files, 32 features + 15 interactions
Purpose: Locked features for end-to-end roadmap training ONLY
```

**What it provides:**
- **32 parent features** with EXACT formulas (never change)
  - p/r1 = log(Ct/Ct-1) ← Always this, never changes
  - p/mom5 = (Ct/Ct-5) - 1 ← Always this formula
- **15 theory-driven interactions** (tension, micro, vol, model)
- **Transform pipeline** (OnlineEWZ, TODRank, SignedLog, etc.)

**Use this ONLY for:**
- ✅ End-to-end roadmap training
- ❌ NOT for backtesting
- ❌ NOT for exploration
- ❌ NOT for general models

**Why it exists:** Roadmap requires exact, reproducible formulas for research

**Example:**
```python
from src.feature_engineering_roadmap.feature_registry import FeatureRegistry

# Locked - formula never changes
registry = FeatureRegistry()
r1 = registry.compute_feature('p/r1', data)  # Always log(Ct/Ct-1)
```

---

### 4️⃣ feature_selection/ 🟣 **[OPTIMIZATION - USE AFTER GENERATION]**
```
Location: src/feature_selection/
Size: 10 files
Purpose: Reduce and optimize features AFTER generation
```

**What it provides:**
- Feature reduction (100 features → 20-50 best)
- Redundancy removal (drop correlated features)
- IC (Information Coefficient) analysis
- Stability analysis
- Causal feature identification

**Use this AFTER:**
- Generated features from feature_generation OR feature_engineering_roadmap
- Have too many features (100+)
- Need to optimize for model performance

**Example:**
```python
from src.feature_selection.main_framework import FeatureSelectionFramework

# After generating 100+ features
selector = FeatureSelectionFramework(n_features=50)
selected = selector.select_features(all_features, targets)
```

---

## Complete Pipeline

```
START: Market Data (OHLCV)
    ↓
STEP 1: Generate Features
    ├─ feature_generation/ (for general use)
    └─ feature_engineering_roadmap/ (for roadmap only)
    ↓
STEP 2: Apply Transforms (optional)
    └─ features_common/BaseScaler
       • ZScoreNormalizer, RobustScaler, etc.
       • OnlineEWZ, TODRank, MADScaler, etc.
    ↓
STEP 3: Optimize Features
    └─ feature_selection/
       • Reduce from 100+ to 20-50
       • Remove redundancy
    ↓
END: Final Feature Set → Model Training
```

---

## Utilities That Power Everything

### Already Integrated ✅
- **utils/matrix_operations/** → GPU acceleration, vectorization
- **utils/hardware/m1_*.py** → M1 optimization (auto-detected)
- **utils/ml_common/** → Bayesian TPE, CV, validation, HPO
- **utils/tprint.py** → Now in BaseScaler._log_* ✅
- **utils/math_validation.py** → Now in BaseScaler._safe_divide ✅

### Available for Use
- **utils/common_operations.py** → Case-by-case evaluation
- **utils/data/** → Data quality framework

---

## Quick Decision Chart

| I want to... | Use this folder |
|-------------|-----------------|
| Generate RSI with period 14 | `feature_generation/categories/momentum.py` |
| Generate ATR with period 20 | `feature_generation/categories/volatility.py` |
| Backtest a strategy | `feature_generation/` (all categories) |
| Explore custom indicators | `feature_generation/` |
| Train Analyst models | `feature_generation/` |
| Train Tactician models | `feature_generation/` |
| **Train roadmap models** | `feature_engineering_roadmap/` **ONLY** |
| Create a new normalizer | Inherit from `features_common/BaseScaler` |
| Reduce 100 features to 50 | `feature_selection/main_framework.py` |
| Remove correlated features | `feature_selection/` |

---

## Key Takeaways

### 🔵 feature_generation/ is your MAIN TOOL
- Use this for 95% of feature needs
- Flexible, powerful, optimized
- 100+ generators ready to use

### 🟢 feature_engineering_roadmap/ is SPECIALIZED
- ONLY for end-to-end roadmap training
- Locked formulas (research requirement)
- Don't use for general work

### 🆕 features_common/ is THE GLUE
- Inherit from BaseScaler when creating scalers
- Gets you tprint & math_validation automatically
- Consistent interface across both systems

### 🟣 feature_selection/ is POST-PROCESSING
- Use AFTER generating features
- Optimizes your feature set
- Removes redundancy

---

## Documentation Index

**Start here:**
1. This file (README_FEATURE_SYSTEMS.md)
2. FEATURE_FOLDERS_ARCHITECTURE.md
3. src/FEATURE_SYSTEMS_GUIDE.md

**For implementation:**
4. COMPLETE_IMPLEMENTATION_SUMMARY.md
5. STRATEGY_C_IMPLEMENTATION_COMPLETE.md

**For reference:**
6. QUICK_FEATURE_SYSTEMS_REFERENCE.md
7. UTILITY_USAGE_AUDIT.md

---

**Status:** ✅ Complete, tested, production ready  
**Last updated:** October 8, 2025
