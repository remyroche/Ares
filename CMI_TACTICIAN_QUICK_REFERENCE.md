# CMI in Tactician Mode - Quick Reference Guide

## TL;DR

**Problem**: How do we use CMI (Conditional Mutual Information) in Tactician mode to avoid redundancy with the Analyst?

**Solution**: In each of the three feature generation steps, use CMI to select features that maximize **I(X; Y | A)** where:
- **X** = Tactician feature
- **Y** = Target variable
- **A** = Analyst side information (outputs/predictions/features)

This ensures Tactician features are **complementary** to Analyst, not redundant.

---

## The Three Steps

### 1. `feature_generation_period_lookback_optimization_step`

**What it does**: Optimizes lookback periods for feature families

**How to use CMI**:
- Score each lookback period by **I(X_lookback; Y | A)** instead of just **I(X_lookback; Y)**
- Allocate more budget to feature families with high CMI complementarity

**Current Status**: ❌ **Not integrated** - needs implementation

**Code Pattern**:
```python
# Instead of:
score = compute_mutual_information(feature, target)

# Do this in Tactician mode:
if tactician_mode:
    score = compute_cmi(feature, target, analyst_side_info)
```

---

### 2. `feature_generation_interaction_generation_step`

**What it does**: Generates feature interactions through 3-phase LGBM+SHAP pipeline

**How to use CMI**:
- **Phase 1**: After generating variants, apply CMI prefiltering to keep top 40% complementary features
- **Phase 2**: Check each refined feature has minimum CMI diversity threshold
- **Phase 3**: Score interactions by CMI complementarity + synergy bonus

**Current Status**: ❌ **Not integrated** - needs implementation

**Code Pattern**:
```python
# Phase 1: CMI prefiltering
if tactician_mode:
    cmi_scores = score_features_by_cmi(variants, target, analyst_side_info)
    variants = variants[cmi_scores > threshold]

# Phase 2: CMI diversity
if tactician_mode:
    for feature in refined_features:
        if cmi_score(feature) < min_threshold:
            remove(feature)  # Not diverse enough from Analyst

# Phase 3: CMI interaction scoring
if tactician_mode:
    for interaction in interactions:
        score = cmi_score(interaction) + synergy_bonus
```

---

### 3. `feature_generation_final_feature_selection_step`

**What it does**: Selects final feature sets (60, 50, 40 features)

**How to use CMI**: Already has partial integration! Just needs to be enabled.

**Current Status**: ✅ **Partially integrated** - needs import fix

**Code Pattern**:
```python
# Already exists in lines 898-1104:
def _detect_tactician_mode(self, features_df, config):
    """Detect if we're in Tactician mode."""
    return (
        'tactician' in step_name.lower() or
        'tactician' in config.get('execution_context', '') or
        config.get('tactician_mode', False)
    )

def _perform_cmi_aware_selection(self, features_df, targets, config, sizes):
    """Perform CMI-aware feature selection for Tactician mode."""
    analyst_side_info = self._extract_analyst_side_info(config)
    
    if analyst_side_info:
        # Use CMI scorer to select features
        selected = self.cmi_scorer.select_features(
            features, targets, analyst_side_info
        )
```

**Problem**: CMI components are set to `None` (placeholder)

**Fix**: Change lines 74-86 to properly import:
```python
from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_complementarity import (
    CMIComplementarityScorer,
    CMIComplementarityConfig
)
from src.training.steps.pre_training.unified_data_driven_pipeline.utils.analyst_side_info import (
    AnalystSideInfoHandler
)
```

---

## How CMI Works

### Basic Concept

**Standard Feature Selection (Analyst mode)**:
- Score features by: **I(X; Y)** = Mutual Information between feature and target
- Problem: Might select features redundant with Analyst outputs

**CMI-Based Selection (Tactician mode)**:
- Score features by: **I(X; Y | A)** = Conditional MI given Analyst side information
- Benefit: Selects features that provide **new information** beyond what Analyst has

### Mathematical Formulation

```
I(X; Y | A) = I(X; Y) - I(X; A) + correction_terms

Where:
- I(X; Y) = How much X predicts Y (standard MI)
- I(X; A) = How much X overlaps with Analyst info (redundancy penalty)
- correction_terms = Adjustments for synergy and complementarity
```

### Practical Interpretation

- **High I(X; Y | A)**: Feature X is useful for predicting Y **even after** knowing A
  - ✅ Good for Tactician (complementary to Analyst)

- **Low I(X; Y | A)**: Feature X doesn't add much beyond what A already tells us
  - ❌ Bad for Tactician (redundant with Analyst)

---

## Implementation Checklist

### For Each Step, Add:

#### 1. **Import CMI Components**
```python
try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_complementarity import (
        CMIComplementarityScorer, CMIComplementarityConfig
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.analyst_side_info import (
        AnalystSideInfoHandler
    )
    CMI_AVAILABLE = True
except ImportError:
    CMI_AVAILABLE = False
```

#### 2. **Initialize in __init__**
```python
if CMI_AVAILABLE:
    self.cmi_scorer = CMIComplementarityScorer(CMIComplementarityConfig())
    self.analyst_handler = AnalystSideInfoHandler()
```

#### 3. **Mode Detection**
```python
def _is_tactician_mode(self, config):
    return (
        'tactician' in self.step_name.lower() or
        config.get('tactician_mode', False)
    )
```

#### 4. **Extract Analyst Info**
```python
def _extract_analyst_side_info(self, config):
    if not CMI_AVAILABLE:
        return None
    
    pipeline_state = config.get('pipeline_state', {})
    return self.analyst_handler.emit_analyst_side_info(pipeline_state)
```

#### 5. **Apply CMI Scoring**
```python
if self._is_tactician_mode(config) and CMI_AVAILABLE:
    analyst_info = self._extract_analyst_side_info(config)
    
    if analyst_info:
        cmi_scores = self.cmi_scorer.score_features(
            features, targets, analyst_info.analyst_outputs
        )
        # Use cmi_scores instead of regular MI scores
```

---

## Configuration Parameters

### CMI Complementarity Config

```python
CMIComplementarityConfig(
    per_family_budget=(5, 15),          # Min/max features per family
    upstream_multiplier=3,              # Total budget = 3x per-family
    max_total_features=60,              # Maximum features to select
    enable_regime_awareness=True,       # Use regime-aware CMI
    compute_timeout_seconds=300.0,      # 5 min timeout
    enable_synergy=True,                # Enable synergy computation
    beta_synergy=0.25,                  # Synergy bonus weight (25%)
    estimator_type='ksg',               # KSG, GCMI, or binned
    ksg_k=3,                            # KSG estimator parameter
    enable_caching=True,                # Cache CMI computations
    enable_parallel=True                # Parallel computation
)
```

### When to Use Each Estimator

| Estimator | Speed | Accuracy | Use When |
|-----------|-------|----------|----------|
| **KSG** | Slow | High | n_features < 600, n_samples > 2000 |
| **GCMI** | Medium | Medium | Balanced scenarios |
| **Binned** | Fast | Low | n_features > 800, n_samples < 1500 |

---

## Common Patterns

### Pattern 1: CMI Prefiltering
```python
# Filter features by CMI threshold
def prefilter_by_cmi(features, targets, analyst_info, threshold=0.01):
    cmi_scores = compute_cmi_scores(features, targets, analyst_info)
    return features[cmi_scores > threshold]
```

### Pattern 2: CMI Budget Allocation
```python
# Allocate budget proportional to CMI scores
def allocate_budget_by_cmi(families, analyst_info):
    family_cmi = {
        family: avg_cmi_score(features, analyst_info)
        for family, features in families.items()
    }
    return proportional_allocation(family_cmi, total_budget=60)
```

### Pattern 3: CMI Interaction Scoring
```python
# Score interactions by CMI + synergy
def score_interaction_by_cmi(feat1, feat2, targets, analyst_info):
    interaction = create_interaction(feat1, feat2)
    cmi_score = compute_cmi(interaction, targets, analyst_info)
    synergy = compute_synergy(feat1, feat2, analyst_info)
    return cmi_score + (0.25 * synergy)
```

---

## Testing Your Integration

### Quick Test

```python
# Run in Tactician mode
config = {
    'tactician_mode': True,
    'enable_cmi_complementarity': True,
    'symbol': 'ETHUSDT',
    'execution_mode': 'light'
}

result = asyncio.run(step.execute(config))

# Check CMI was used
assert result['success'] == True
assert result.get('diagnostics', {}).get('cmi_enabled') == True
```

### Verify Output

Check that results include:
- ✅ `cmi_enabled: true` in diagnostics
- ✅ `complementarity_scores` in metadata
- ✅ Lower correlation with Analyst features
- ✅ Higher complementarity metrics

---

## Debugging Tips

### CMI Not Activating?
- ✅ Check `tactician_mode=True` in config
- ✅ Check CMI imports successful (`CMI_AVAILABLE=True`)
- ✅ Check Analyst side info extracted (`analyst_info is not None`)
- ✅ Check step name contains 'tactician'

### Performance Issues?
- ✅ Use 'binned' estimator for large feature sets
- ✅ Enable caching (`enable_caching=True`)
- ✅ Reduce `compute_timeout_seconds`
- ✅ Use parallel computation (`enable_parallel=True`)

### Low CMI Scores?
- ✅ Check Analyst side info quality
- ✅ Try different estimator types
- ✅ Adjust `beta_synergy` parameter
- ✅ Check for data alignment issues

---

## Next Steps

### Immediate Actions

1. **Fix Final Feature Selection Step** (5 min)
   - File: `feature_generation_final_feature_selection_step.py`
   - Lines: 74-86
   - Action: Replace placeholder imports with actual imports

2. **Add CMI to Interaction Generation** (30 min)
   - File: `feature_generation_interaction_generation_step.py`
   - Add: Phase 1 prefiltering, Phase 2 diversity, Phase 3 scoring

3. **Add CMI to Lookback Optimization** (20 min)
   - File: `feature_generation_period_lookback_optimization_step.py`
   - Add: CMI-based lookback scoring and budget allocation

### Testing

4. **Unit Tests** (20 min)
   - Test each step in Tactician mode
   - Verify CMI activation
   - Check complementarity scores

5. **Integration Tests** (30 min)
   - Test full pipeline with CMI
   - Compare Analyst vs Tactician features
   - Benchmark performance

---

## Resources

- **Full Analysis**: `CMI_TACTICIAN_MODE_INTEGRATION_ANALYSIS.md`
- **CMI Guide**: `docs/CMI_COMPLEMENTARITY_GUIDE.md`
- **CMI Source**: `src/training/steps/pre_training/unified_data_driven_pipeline/utils/cmi_complementarity.py`
- **Analyst Handler**: `src/training/steps/pre_training/unified_data_driven_pipeline/utils/analyst_side_info.py`

---

## Summary

**Key Takeaway**: Use CMI to score features by **I(X; Y | A)** in Tactician mode to ensure features are **complementary** to the Analyst, not redundant.

**Current Status**:
- ✅ Final feature selection: Partially integrated (needs import fix)
- ❌ Interaction generation: Not integrated (needs implementation)
- ❌ Lookback optimization: Not integrated (needs implementation)

**Priority**: Fix imports first, then add CMI to other two steps.
