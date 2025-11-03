# Ares Launcher Performance Analysis

## Timing Breakdown from Logs (Second Run)

### Total Startup Time: **~21 seconds**

| Phase | Start Time | End Time | Duration | % of Total |
|-------|-----------|----------|----------|-----------|
| **1. Core System Init** | 20:01:45.622 | 20:01:48.884 | ~3.3s | 15.7% |
| **2. Hardware Detection** | 20:01:48.884 | 20:01:50.313 | ~1.4s | 6.7% |
| **3. Feature Bank Setup** | 20:01:50.313 | 20:01:53.231 | ~2.9s | 13.8% |
| **4. Feature Registration** | 20:01:53.231 | 20:01:53.600 | ~0.4s | 1.9% |
| **5. ML Common Loading** | 20:01:53.600 | 20:01:53.613 | ~0.01s | 0.05% |
| **🔴 6. SLOWEST: Feature Generation Steps** | 20:01:53.613 | 20:02:05.670 | **~12.1s** | **57.6%** |
| **7. Market Analysis Steps** | 20:02:05.670 | 20:02:05.921 | ~0.25s | 1.2% |
| **8. Model Training Steps** | 20:02:05.921 | 20:02:06.849 | ~0.9s | 4.3% |
| **TOTAL** | 20:01:45.622 | 20:02:06.849 | **~21.2s** | **100%** |

---

## 🔴 CRITICAL BOTTLENECK: Feature Generation Steps Registration

**Time**: 12.1 seconds (57.6% of total startup)  
**Location**: Between lines registering `data_download` step and `feature_generation_gate_feature_step`

### What's Happening in This Gap?

Looking at the code path:
```python
# src/launcher/ares_launcher.py line 46
import src.training.steps.market_analysis  # Takes ~0.25s
```

But then there's a **11.8 second gap** before we see:
```
[2025-10-31 20:02:05.384] ✅ Feature generation gate feature step registered SUCCESS
```

### The Problem

The gap occurs between:
```
20:01:53,613 - ares.registry - Registered step: data_download
[12 SECOND GAP HERE]
20:02:05,383 - ares.registry - Registered step: feature_generation_gate_feature_step
```

This suggests that **importing feature generation steps** is extremely slow.

---

## Detailed Breakdown by Component

### Fast Components (< 1 second each)

✅ **Data Quality Framework**: 0.0s (singleton, already initialized)  
✅ **ML Common Utilities**: 0.01s (well optimized)  
✅ **Market Analysis Registration**: 0.25s (efficient)  
✅ **Model Training Registration**: 0.9s (reasonable)

### Medium Components (1-3 seconds)

🟡 **Core System Init**: 3.3s
- DataQualityFramework, DataCleaner, AdvancedQualityMetrics
- Features common utilities
- Could be optimized with lazy loading

🟡 **Feature Bank Setup**: 2.9s
- VectorBT components
- Unified Vectorization Manager
- 533 generators registered
- Acceptable for amount of work being done

🟡 **Hardware Detection**: 1.4s  
- M1 GPU/CPU/Memory initialization
- Multiple manager instantiations
- Could be cached or parallelized

### Slow Components (> 5 seconds)

🔴 **Feature Generation Steps Import**: **12.1 seconds** (57.6% of total!)

This is where `src.training.steps.feature_generation` modules are being imported.

---

## Root Cause Analysis

### Why Feature Generation Steps Are Slow

Looking at the import chain in `ares_launcher.py`:

```python
import src.training.steps.feature_generation  # Line ~44
```

This triggers imports of:
1. `feature_generation_gate_feature_step.py`
2. `feature_generation_final_feature_selection_step.py`
3. `feature_generation_interaction_generation_step.py`
4. `feature_generation_feature_selection_step.py`
5. etc.

Each of these likely imports heavy dependencies:
- Feature bank (already loaded, but being accessed)
- ML optimization utilities
- Feature importance analyzers
- Complementarity analysis (CMI)
- Cross-validation systems

### Specific Culprits

From the logs, the warning appears:
```
[2025-10-31 20:02:05.669] WARNING: ⚠️ HPO utilities not available: 
cannot import name 'HierarchicalHPOptimizer' from 
'src.utils.ml_common.optimization.hierarchical_hpo'
```

This suggests **import errors are being caught and retried**, adding latency.

---

## Recommendations

### 🎯 Priority 1: Fix Import Issues (Quick Win)

**Problem**: `HierarchicalHPOptimizer` import failing  
**Impact**: Likely causing retry delays in feature_generation_interaction_generation_step.py  
**Fix**: 
1. Check if class name is correct (might be `HierarchicalHPO` not `HierarchicalHPOptimizer`)
2. Add proper error handling to fail fast

```python
# Current (slow):
try:
    from src.utils.ml_common.optimization.hierarchical_hpo import HierarchicalHPOptimizer
except ImportError:
    HierarchicalHPOptimizer = None  # Silently fails, but adds delay

# Better:
try:
    from src.utils.ml_common.optimization.hierarchical_hpo import HierarchicalHPO as HierarchicalHPOptimizer
except ImportError:
    HierarchicalHPOptimizer = None
```

### 🎯 Priority 2: Lazy Load Feature Generation Steps (Medium Win)

**Problem**: All 10 feature generation steps imported even if not needed  
**Impact**: 12 seconds wasted for analyst_base training  
**Fix**: Use lazy imports in `src/training/steps/feature_generation/__init__.py`

```python
# Instead of:
from .feature_generation_gate_feature_step import FeatureGenerationGateFeatureStep

# Use:
def __getattr__(name):
    if name == 'FeatureGenerationGateFeatureStep':
        from .feature_generation_gate_feature_step import FeatureGenerationGateFeatureStep
        return FeatureGenerationGateFeatureStep
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Expected savings**: ~10 seconds

### 🎯 Priority 3: Lazy Load Market Analysis (Small Win)

**Problem**: Market analysis components loaded even though analyst training doesn't need them  
**Fix**: Only import when running market analysis stages

**Expected savings**: ~0.3 seconds (minor)

### 🎯 Priority 4: Cache Hardware Detection (Small Win)

**Problem**: Hardware detection runs multiple times  
**Fix**: Cache results in singleton

**Expected savings**: ~0.5 seconds

---

## Expected Time After Optimizations

| Scenario | Current | After P1 | After P1+P2 | After All |
|----------|---------|----------|-------------|-----------|
| **Startup Time** | 21s | 18s | 8s | 6.5s |
| **Training Time** | +3-5min | +3-5min | +3-5min | +3-5min |
| **Total** | ~3.5-5.5min | ~3.3-5.3min | ~3.1-5.1min | ~3.1-5.1min |

---

## Immediate Action

The slowest part is **Feature Generation Steps import (~12s, 57.6% of time)**.

### Quick Fix Option 1: Skip Unused Imports

Modify `src/launcher/ares_launcher.py` to conditionally import:

```python
# Current:
import src.training.steps.feature_generation  # Always imports all

# Better:
if args.stage in ['FEATURE_GENERATION', 'PRE_TRAINING']:
    import src.training.steps.feature_generation
# Otherwise skip
```

### Quick Fix Option 2: Fix the Import Error

The warning about `HierarchicalHPOptimizer` suggests there's an import retry happening. Fixing this could save several seconds.

---

## What You Can Do Right Now

1. **Just let it run** - The 21s is one-time overhead, HPO will start after
2. **Fix the import error** - Address the `HierarchicalHPOptimizer` warning
3. **Refactor launcher** - Make imports conditional/lazy

**My recommendation**: Let it run this time to test HPO, then we can optimize the launcher for future runs.

The actual HPO testing is the valuable part - the 21s startup is annoying but acceptable for now.

