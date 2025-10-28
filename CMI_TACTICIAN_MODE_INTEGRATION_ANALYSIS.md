# CMI Integration in Tactician Mode - Complete Analysis

## Executive Summary

This document analyzes how **Conditional Mutual Information (CMI)** should be integrated into the three key feature generation steps to avoid redundancy with the Analyst mode when in Tactician mode:

1. **`feature_generation_period_lookback_optimization_step`** - Optimize lookback periods
2. **`feature_generation_interaction_generation_step`** - Generate feature interactions
3. **`feature_generation_final_feature_selection_step`** - Select final features

**Key Principle**: In Tactician mode, we want to maximize **I(X; Y | A)** where:
- **X** = Tactician features
- **Y** = Target variable
- **A** = Analyst side information (outputs, predictions, features)

This ensures Tactician features provide **complementary** information to the Analyst, not redundant information.

---

## Current State of CMI Integration

### 1. `feature_generation_final_feature_selection_step` ✅ **ALREADY INTEGRATED**

**Status**: ✅ **CMI is already integrated** (lines 74-173, 898-1104)

**Current Implementation**:
```python
# Lines 74-86: CMI components initialization
try:
    CMIComplementarityScorer = None
    CMIComplementarityConfig = None
    AnalystSideInfoHandler = None
    CMI_COMPLEMENTARITY_AVAILABLE = False
except ImportError:
    # Placeholder
    pass

# Lines 156-173: CMI scorer initialization
if CMI_COMPLEMENTARITY_AVAILABLE:
    self.cmi_config = CMIComplementarityConfig(
        per_family_budget=(5, 15),
        upstream_multiplier=3,
        max_total_features=60,
        enable_regime_awareness=True,
        compute_timeout_seconds=300.0,
        enable_synergy=True,
        beta_synergy=0.25
    )
    self.cmi_scorer = CMIComplementarityScorer(self.cmi_config)
    self.analyst_handler = AnalystSideInfoHandler()
```

**Detection Logic** (lines 898-957):
```python
def _detect_tactician_mode(self, features_df: pd.DataFrame, config: Dict[str, Any]) -> bool:
    """Detect if we're in Tactician mode."""
    # Primary: Check step name
    is_tactician_training_step = (
        'tactician_base_training' in current_step_name or
        'tactician_ensemble_training' in current_step_name or
        'tactician' in current_step_name.lower()
    )
    
    # Secondary: Check execution context
    is_tactician_context = 'tactician' in config.get('execution_context', '').lower()
    
    # Tertiary: Check for Tactician-specific features
    tactician_features = [col for col in features_df.columns if 'tactician' in col.lower()]
    
    # Determine mode
    is_tactician_mode = (
        is_tactician_training_step or
        is_tactician_context or
        len(tactician_features) > 0 or
        config.get('enable_cmi_complementarity', False)
    )
    
    return is_tactician_mode
```

**CMI-Aware Selection** (lines 1032-1104):
```python
def _perform_cmi_aware_selection(self, features_df, targets, config, feature_set_sizes):
    """Perform CMI-aware feature selection for Tactician mode."""
    # Extract Analyst side information
    analyst_side_info = self._extract_analyst_side_info_for_cmi(features_df, config)
    
    if not analyst_side_info.get('cmi_enabled', False):
        # Fallback to standard selection
        return self._perform_standard_selection(...)
    
    # Separate Tactician and Analyst features
    tactician_features = [col for col in features_df.columns 
                        if 'tactician' in col.lower() or 'cmi' in col.lower()]
    analyst_features = [col for col in features_df.columns 
                      if 'analyst' in col.lower()]
    
    # Use CMI scorer for feature selection
    for size in feature_set_sizes:
        selected_features = self.cmi_scorer.select_features(
            features=X,
            targets=y,
            analyst_side_info=analyst_side_info['side_info']
        )
        # Limit to requested size
        selected_features = selected_features[:size]
```

**Issues**:
1. ❌ CMI components are set to `None` (placeholder implementation)
2. ❌ `CMI_COMPLEMENTARITY_AVAILABLE` is always `False`
3. ❌ No actual import from the CMI modules
4. ⚠️ Need to properly import from `src/training/steps/pre_training/unified_data_driven_pipeline/utils/`

---

### 2. `feature_generation_interaction_generation_step` ⚠️ **NEEDS INTEGRATION**

**Current Status**: ⚠️ **No CMI integration** - uses standard three-phase LGBM+SHAP pipeline

**Where CMI Should Be Applied**:

#### **Phase 1: Variant Generation & Initial Selection**
After generating feature variants, apply CMI filtering to prioritize variants that are complementary to Analyst:

```python
# CURRENT: Line ~200-300 in interaction generation step
def _phase1_variant_generation_and_selection(self, ...):
    # Generate variants
    variants = self._generate_feature_variants(base_features)
    
    # MISSING: CMI-based prefiltering for Tactician mode
    # Should add here:
    if self._is_tactician_mode():
        variants = self._apply_cmi_prefiltering(
            variants, 
            targets, 
            analyst_side_info,
            budget=int(len(variants) * 0.4)  # Keep top 40%
        )
    
    # Continue with LGBM+SHAP selection
    selected_features = self._lgbm_shap_selection(variants, targets)
```

#### **Phase 2: Middle Refinement**
After narrowing down to top features, apply CMI to ensure diversity from Analyst:

```python
def _phase2_middle_refinement(self, selected_features, ...):
    # LGBM refinement
    refined_features = self._deeper_lgbm_refinement(selected_features)
    
    # MISSING: CMI-based diversity check
    if self._is_tactician_mode():
        refined_features = self._ensure_cmi_diversity(
            refined_features,
            analyst_side_info,
            min_diversity_threshold=0.3  # Minimum CMI(X; Y | A)
        )
```

#### **Phase 3: Interaction Discovery**
When generating interactions, prioritize pairs that maximize complementarity:

```python
def _phase3_interaction_discovery(self, top_features, ...):
    # Generate interaction candidates
    interaction_candidates = self._generate_interaction_pairs(top_features)
    
    # MISSING: CMI-based interaction scoring
    if self._is_tactician_mode():
        # Score interactions by CMI complementarity
        scored_interactions = self._score_interactions_by_cmi(
            interaction_candidates,
            targets,
            analyst_side_info,
            synergy_bonus=0.25  # Bonus for synergistic interactions
        )
        
        # Select top interactions
        top_interactions = self._select_top_cmi_interactions(
            scored_interactions,
            n_interactions=50
        )
```

---

### 3. `feature_generation_period_lookback_optimization_step` ⚠️ **NEEDS INTEGRATION**

**Current Status**: ⚠️ **No CMI integration** - optimizes lookback periods independently

**Where CMI Should Be Applied**:

#### **Lookback Period Search**
When optimizing lookback periods, consider complementarity with Analyst:

```python
def optimize_lookback_periods(self, features, targets, ...):
    """Optimize lookback periods for feature families."""
    
    # MISSING: Get Analyst side information for Tactician mode
    analyst_side_info = None
    if self._is_tactician_mode():
        analyst_side_info = self._extract_analyst_side_info()
    
    for family in feature_families:
        # For each lookback period candidate
        for lookback in lookback_candidates:
            # Compute features with this lookback
            features_with_lookback = self._compute_features(family, lookback)
            
            # CURRENT: Score by regular MI
            score = self._score_by_mutual_information(features_with_lookback, targets)
            
            # IMPROVED: Score by CMI in Tactician mode
            if analyst_side_info is not None:
                score = self._score_by_cmi_complementarity(
                    features_with_lookback,
                    targets,
                    analyst_side_info,
                    lookback_period=lookback
                )
        
        # Select best lookback period (highest CMI score in Tactician mode)
        best_lookback = self._select_best_lookback(scores)
```

#### **Feature Family Budgeting**
Allocate more budget to feature families that show high CMI complementarity:

```python
def allocate_family_budgets(self, optimized_features, ...):
    """Allocate feature budget per family."""
    
    if self._is_tactician_mode():
        # MISSING: CMI-based budget allocation
        family_cmi_scores = {}
        for family in feature_families:
            family_features = optimized_features[family]
            cmi_score = self._compute_family_cmi_score(
                family_features,
                targets,
                analyst_side_info
            )
            family_cmi_scores[family] = cmi_score
        
        # Allocate more budget to high-CMI families
        budgets = self._allocate_budgets_by_cmi(
            family_cmi_scores,
            total_budget=60,
            per_family_range=(5, 15)
        )
```

---

## Implementation Recommendations

### Step 1: Fix CMI Imports in Final Feature Selection Step

**File**: `src/training/steps/pre_training/feature_generation_final_feature_selection_step.py`

**Lines 74-86**: Replace placeholder with actual imports:

```python
# Import CMI complementarity components for Tactician mode
try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_complementarity import (
        CMIComplementarityScorer,
        CMIComplementarityConfig
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.analyst_side_info import (
        AnalystSideInfoHandler
    )
    CMI_COMPLEMENTARITY_AVAILABLE = True
    tprint_info("✅ CMI complementarity components loaded successfully")
except ImportError as e:
    CMI_COMPLEMENTARITY_AVAILABLE = False
    CMIComplementarityScorer = None
    CMIComplementarityConfig = None
    AnalystSideInfoHandler = None
    tprint_warning(f"⚠️ CMI complementarity components not available: {e}")
```

---

### Step 2: Add CMI Integration to Interaction Generation Step

**File**: `src/training/steps/pre_training/feature_generation_interaction_generation_step.py`

**Add at top of file** (after imports):

```python
# Import CMI complementarity for Tactician mode
try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_complementarity import (
        CMIComplementarityScorer,
        CMIComplementarityConfig
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.analyst_side_info import (
        AnalystSideInfoHandler
    )
    CMI_COMPLEMENTARITY_AVAILABLE = True
except ImportError as e:
    CMI_COMPLEMENTARITY_AVAILABLE = False
    tprint_warning(f"⚠️ CMI complementarity not available: {e}")
```

**Add in `__init__` method**:

```python
def __init__(self, step_name="feature_generation_interaction_generation_step"):
    super().__init__(step_name)
    
    # Initialize CMI components for Tactician mode
    if CMI_COMPLEMENTARITY_AVAILABLE:
        self.cmi_config = CMIComplementarityConfig(
            per_family_budget=(5, 15),
            upstream_multiplier=3,
            max_total_features=60,
            enable_regime_awareness=True,
            compute_timeout_seconds=300.0,
            enable_synergy=True,
            beta_synergy=0.25
        )
        self.cmi_scorer = CMIComplementarityScorer(self.cmi_config)
        self.analyst_handler = AnalystSideInfoHandler()
        tprint_info("✅ CMI components initialized for interaction generation")
    else:
        self.cmi_scorer = None
        self.analyst_handler = None
```

**Add CMI filtering in Phase 1**:

```python
def _phase1_variant_generation_and_selection(self, base_features, targets, config):
    """Phase 1: Generate variants and apply initial selection."""
    
    # Generate feature variants
    variants_df = self._generate_feature_variants(base_features)
    
    # Apply CMI prefiltering in Tactician mode
    if self._is_tactician_mode(config) and CMI_COMPLEMENTARITY_AVAILABLE:
        tprint_info("🎯 Applying CMI prefiltering in Tactician mode (Phase 1)")
        
        # Extract Analyst side information
        analyst_side_info = self._extract_analyst_side_info(config)
        
        if analyst_side_info is not None:
            # Score variants by CMI complementarity
            cmi_result = self.cmi_scorer.score_features(
                features=variants_df,
                targets=targets,
                analyst_outputs=analyst_side_info.analyst_outputs,
                regime_labels=analyst_side_info.regime_labels
            )
            
            # Filter to top 40% by CMI score
            top_40_percent = int(len(variants_df.columns) * 0.4)
            variants_df = variants_df[cmi_result.selected_features[:top_40_percent]]
            
            tprint_success(f"✅ CMI prefiltering: {len(variants_df.columns)} variants selected")
    
    # Continue with standard LGBM+SHAP selection
    return self._lgbm_shap_selection(variants_df, targets, config)
```

**Add CMI diversity check in Phase 2**:

```python
def _phase2_middle_refinement(self, selected_features, targets, config):
    """Phase 2: Deeper LGBM refinement with CMI diversity."""
    
    # Standard LGBM refinement
    refined_features = self._deeper_lgbm_refinement(selected_features, targets)
    
    # Apply CMI diversity check in Tactician mode
    if self._is_tactician_mode(config) and CMI_COMPLEMENTARITY_AVAILABLE:
        tprint_info("🎯 Applying CMI diversity check (Phase 2)")
        
        analyst_side_info = self._extract_analyst_side_info(config)
        
        if analyst_side_info is not None:
            # Ensure minimum CMI complementarity for each feature
            diverse_features = []
            min_cmi_threshold = 0.01  # Minimum I(X; Y | A)
            
            for feature in refined_features:
                feature_data = selected_features[[feature]]
                cmi_score = self._compute_cmi_score(
                    feature_data,
                    targets,
                    analyst_side_info
                )
                
                if cmi_score >= min_cmi_threshold:
                    diverse_features.append(feature)
                else:
                    tprint_warning(f"⚠️ Feature {feature} filtered (CMI={cmi_score:.4f} < {min_cmi_threshold})")
            
            refined_features = diverse_features
            tprint_success(f"✅ CMI diversity check: {len(refined_features)} features retained")
    
    return refined_features
```

**Add CMI interaction scoring in Phase 3**:

```python
def _phase3_interaction_discovery(self, top_features, targets, config):
    """Phase 3: Discover interactions with CMI complementarity scoring."""
    
    # Generate interaction candidates
    interaction_candidates = self._generate_interaction_pairs(top_features)
    
    # Apply CMI interaction scoring in Tactician mode
    if self._is_tactician_mode(config) and CMI_COMPLEMENTARITY_AVAILABLE:
        tprint_info("🎯 Applying CMI interaction scoring (Phase 3)")
        
        analyst_side_info = self._extract_analyst_side_info(config)
        
        if analyst_side_info is not None:
            # Score each interaction by CMI complementarity + synergy
            interaction_scores = {}
            
            for (feat1, feat2), interaction_df in interaction_candidates.items():
                # Compute CMI complementarity score
                cmi_score = self._compute_cmi_score(
                    interaction_df,
                    targets,
                    analyst_side_info
                )
                
                # Compute synergy bonus
                synergy_score = self._compute_synergy(
                    feat1, feat2, targets, analyst_side_info
                )
                
                # Combined score
                total_score = cmi_score + (self.cmi_config.beta_synergy * synergy_score)
                interaction_scores[(feat1, feat2)] = total_score
            
            # Select top interactions by CMI score
            sorted_interactions = sorted(
                interaction_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            top_interactions = [
                interaction_candidates[pair] 
                for pair, score in sorted_interactions[:50]
            ]
            
            tprint_success(f"✅ CMI interaction scoring: {len(top_interactions)} interactions selected")
            return top_interactions
    
    # Fallback to standard interaction selection
    return self._standard_interaction_selection(interaction_candidates, targets)
```

---

### Step 3: Add CMI Integration to Period Lookback Optimization Step

**File**: `src/training/steps/pre_training/feature_generation_period_lookback_optimization_step.py`

**Add at top of file** (after imports):

```python
# Import CMI complementarity for Tactician mode
try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_complementarity import (
        CMIComplementarityScorer,
        CMIComplementarityConfig
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.analyst_side_info import (
        AnalystSideInfoHandler
    )
    CMI_COMPLEMENTARITY_AVAILABLE = True
except ImportError as e:
    CMI_COMPLEMENTARITY_AVAILABLE = False
    tprint_warning(f"⚠️ CMI complementarity not available: {e}")
```

**Add CMI-based lookback scoring**:

```python
def _score_lookback_period(self, features, targets, lookback_period, config):
    """Score a lookback period configuration."""
    
    # Standard MI scoring
    mi_score = self._compute_mutual_information(features, targets)
    
    # Apply CMI scoring in Tactician mode
    if self._is_tactician_mode(config) and CMI_COMPLEMENTARITY_AVAILABLE:
        analyst_side_info = self._extract_analyst_side_info(config)
        
        if analyst_side_info is not None:
            # Compute CMI complementarity: I(X; Y | A)
            cmi_score = self.cmi_scorer.score_features(
                features=features,
                targets=targets,
                analyst_outputs=analyst_side_info.analyst_outputs,
                regime_labels=analyst_side_info.regime_labels
            )
            
            tprint_info(f"Lookback={lookback_period}: MI={mi_score:.4f}, CMI={cmi_score.feature_scores[features.columns[0]]:.4f}")
            
            # Use CMI score instead of MI in Tactician mode
            return cmi_score.feature_scores[features.columns[0]]
    
    # Fallback to MI score
    return mi_score
```

**Add CMI-based budget allocation**:

```python
def _allocate_feature_budgets(self, optimized_features, config):
    """Allocate feature budget per family with CMI awareness."""
    
    if self._is_tactician_mode(config) and CMI_COMPLEMENTARITY_AVAILABLE:
        tprint_info("🎯 CMI-based budget allocation (Tactician mode)")
        
        analyst_side_info = self._extract_analyst_side_info(config)
        
        if analyst_side_info is not None:
            # Compute CMI score for each family
            family_cmi_scores = {}
            
            for family_name, family_features in optimized_features.items():
                # Score family by average CMI complementarity
                cmi_result = self.cmi_scorer.score_features(
                    features=family_features,
                    targets=self.targets,
                    analyst_outputs=analyst_side_info.analyst_outputs
                )
                
                avg_cmi = np.mean(list(cmi_result.complementarity_scores.values()))
                family_cmi_scores[family_name] = avg_cmi
            
            # Allocate budget proportional to CMI scores
            total_budget = 60
            min_per_family = 5
            max_per_family = 15
            
            budgets = self._proportional_budget_allocation(
                family_cmi_scores,
                total_budget=total_budget,
                min_budget=min_per_family,
                max_budget=max_per_family
            )
            
            tprint_success(f"✅ CMI-based budgets: {budgets}")
            return budgets
    
    # Fallback to standard budget allocation
    return self._standard_budget_allocation(optimized_features, config)
```

---

## Helper Functions (Common Across All Steps)

Add these helper functions to each step file:

```python
def _is_tactician_mode(self, config: Dict[str, Any]) -> bool:
    """Check if we're in Tactician mode."""
    # Check step name
    is_tactician_step = 'tactician' in self.step_name.lower()
    
    # Check execution context
    is_tactician_context = 'tactician' in config.get('execution_context', '').lower()
    
    # Check explicit flag
    is_explicit_tactician = config.get('tactician_mode', False)
    
    return is_tactician_step or is_tactician_context or is_explicit_tactician

def _extract_analyst_side_info(self, config: Dict[str, Any]):
    """Extract Analyst side information from config/pipeline state."""
    if not CMI_COMPLEMENTARITY_AVAILABLE or self.analyst_handler is None:
        return None
    
    try:
        # Get pipeline state
        pipeline_state = config.get('pipeline_state', {})
        
        # Extract Analyst side information
        analyst_result = self.analyst_handler.emit_analyst_side_info(
            pipeline_state=pipeline_state,
            targets=None,  # Will be extracted from pipeline state
            data_index=None  # Will be extracted from pipeline state
        )
        
        if analyst_result.analyst_outputs is not None:
            tprint_info(f"✅ Analyst side information extracted: {analyst_result.analyst_outputs.shape}")
            return analyst_result
        else:
            tprint_warning("⚠️ No Analyst outputs available")
            return None
            
    except Exception as e:
        tprint_warning(f"⚠️ Failed to extract Analyst side information: {e}")
        return None

def _compute_cmi_score(self, features, targets, analyst_side_info):
    """Compute CMI complementarity score for features."""
    if not CMI_COMPLEMENTARITY_AVAILABLE or self.cmi_scorer is None:
        return 0.0
    
    try:
        result = self.cmi_scorer.score_features(
            features=features,
            targets=targets,
            analyst_outputs=analyst_side_info.analyst_outputs,
            regime_labels=analyst_side_info.regime_labels
        )
        
        # Return average complementarity score
        return np.mean(list(result.complementarity_scores.values()))
        
    except Exception as e:
        tprint_warning(f"⚠️ CMI score computation failed: {e}")
        return 0.0
```

---

## Testing Strategy

### 1. Unit Tests for Each Step

**File**: `tests/training/test_cmi_tactician_integration.py`

```python
import pytest
from src.training.steps.pre_training.feature_generation_final_feature_selection_step import (
    FeatureGenerationFinalFeatureSelectionStep
)
from src.training.steps.pre_training.feature_generation_interaction_generation_step import (
    FeatureGenerationInteractionGenerationStep
)
from src.training.steps.pre_training.feature_generation_period_lookback_optimization_step import (
    FeatureGenerationPeriodLookbackOptimizationStep
)

class TestCMITacticianIntegration:
    
    def test_final_selection_tactician_mode(self):
        """Test CMI integration in final feature selection (Tactician mode)."""
        step = FeatureGenerationFinalFeatureSelectionStep()
        
        config = {
            'tactician_mode': True,
            'enable_cmi_complementarity': True,
            'symbol': 'ETHUSDT',
            'execution_mode': 'light'
        }
        
        # Mock features with Analyst and Tactician features
        features_df = self._create_mock_features(
            n_analyst_features=20,
            n_tactician_features=30
        )
        
        # Execute step
        result = asyncio.run(step.execute(config))
        
        # Verify CMI was used
        assert result['success'] == True
        assert 'cmi_enabled' in result.get('diagnostics', {})
        assert result['diagnostics']['cmi_enabled'] == True
    
    def test_interaction_generation_tactician_mode(self):
        """Test CMI integration in interaction generation (Tactician mode)."""
        step = FeatureGenerationInteractionGenerationStep()
        
        config = {
            'tactician_mode': True,
            'execution_context': 'tactician_training',
            'symbol': 'ETHUSDT'
        }
        
        result = asyncio.run(step.execute(config))
        
        # Verify CMI filtering was applied in phases
        assert 'phase1_cmi_filtered' in result.get('metrics', {})
        assert 'phase2_cmi_diversity' in result.get('metrics', {})
        assert 'phase3_cmi_interactions' in result.get('metrics', {})
    
    def test_lookback_optimization_tactician_mode(self):
        """Test CMI integration in lookback optimization (Tactician mode)."""
        step = FeatureGenerationPeriodLookbackOptimizationStep()
        
        config = {
            'tactician_mode': True,
            'symbol': 'ETHUSDT'
        }
        
        result = asyncio.run(step.execute(config))
        
        # Verify CMI-based lookback scoring
        assert 'cmi_lookback_scores' in result.get('metadata', {})
        assert 'cmi_budget_allocation' in result.get('metadata', {})
```

### 2. Integration Tests

**File**: `tests/integration/test_cmi_full_pipeline.py`

```python
def test_full_tactician_pipeline_with_cmi():
    """Test full Tactician pipeline with CMI integration."""
    
    # Execute all three steps in sequence
    config = {
        'tactician_mode': True,
        'enable_cmi_complementarity': True,
        'symbol': 'ETHUSDT',
        'execution_mode': 'light'
    }
    
    # Step 1: Lookback optimization with CMI
    lookback_step = FeatureGenerationPeriodLookbackOptimizationStep()
    lookback_result = asyncio.run(lookback_step.execute(config))
    assert lookback_result['success']
    
    # Step 2: Interaction generation with CMI
    interaction_step = FeatureGenerationInteractionGenerationStep()
    interaction_result = asyncio.run(interaction_step.execute(config))
    assert interaction_result['success']
    
    # Step 3: Final selection with CMI
    selection_step = FeatureGenerationFinalFeatureSelectionStep()
    selection_result = asyncio.run(selection_step.execute(config))
    assert selection_result['success']
    
    # Verify CMI was used throughout
    assert 'cmi_enabled' in selection_result.get('diagnostics', {})
```

---

## Summary and Action Items

### Current Status

| Step | CMI Integration Status | Action Required |
|------|----------------------|-----------------|
| **Final Feature Selection** | ✅ Partially integrated (placeholder) | 🔧 Fix imports, enable CMI components |
| **Interaction Generation** | ❌ Not integrated | ➕ Add CMI filtering in all 3 phases |
| **Lookback Optimization** | ❌ Not integrated | ➕ Add CMI-based lookback scoring |

### Priority Action Items

1. **HIGH PRIORITY**: Fix CMI imports in `feature_generation_final_feature_selection_step.py`
   - Replace placeholder imports with actual module imports
   - Test that `CMI_COMPLEMENTARITY_AVAILABLE` becomes `True`

2. **MEDIUM PRIORITY**: Add CMI integration to `feature_generation_interaction_generation_step.py`
   - Phase 1: CMI prefiltering
   - Phase 2: CMI diversity check
   - Phase 3: CMI interaction scoring

3. **MEDIUM PRIORITY**: Add CMI integration to `feature_generation_period_lookback_optimization_step.py`
   - CMI-based lookback scoring
   - CMI-based budget allocation

4. **LOW PRIORITY**: Add comprehensive testing
   - Unit tests for each step
   - Integration tests for full pipeline
   - Performance benchmarks

### Key Principles to Remember

1. **Mode Detection**: Always check `tactician_mode` before applying CMI
2. **Graceful Degradation**: If CMI not available, fallback to standard methods
3. **Complementarity, Not Redundancy**: Maximize I(X; Y | A), not I(X; Y)
4. **Analyst Protection**: Never modify Analyst mode behavior
5. **Performance Monitoring**: Track CMI computation time and fallback events

---

## Conclusion

This analysis provides a complete roadmap for integrating CMI into all three feature generation steps. The key insight is that **CMI should be used to score features by their complementarity to the Analyst**, not just by their predictive power alone. This ensures that Tactician features provide **additional information** beyond what the Analyst already captures, leading to more effective ensemble models.

By following this implementation guide, the Tactician mode will properly leverage CMI to avoid redundancy with the Analyst while maximizing predictive performance through complementary feature selection.
