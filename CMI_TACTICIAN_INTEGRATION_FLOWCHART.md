# CMI Tactician Mode Integration - Visual Flowchart

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE GENERATION PIPELINE                  │
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │   STEP 1:     │  │   STEP 2:     │  │   STEP 3:     │      │
│  │   Lookback    │→ │ Interaction   │→ │    Final      │      │
│  │ Optimization  │  │  Generation   │  │  Selection    │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
│         ↓                   ↓                   ↓               │
│  ┌───────────────────────────────────────────────────────┐     │
│  │     CMI-AWARE PROCESSING (Tactician Mode Only)       │     │
│  └───────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step CMI Integration

### Step 1: Period Lookback Optimization

```
INPUT: Market Data + Feature Families
  │
  ├─ Mode Detection ──────────────────┐
  │                                    │
  ▼                                    ▼
┌─────────────┐              ┌──────────────────┐
│ ANALYST MODE│              │  TACTICIAN MODE  │
│   (Regular) │              │  (CMI-Aware)     │
└─────────────┘              └──────────────────┘
      │                               │
      │ Score by MI                   │ Score by CMI
      │ I(X; Y)                        │ I(X; Y | A)
      │                               │
      ▼                               ▼
┌─────────────────────┐    ┌──────────────────────┐
│ Standard Lookback   │    │ Complementary        │
│ Optimization        │    │ Lookback Optimization│
│                     │    │                      │
│ • Maximize I(X;Y)   │    │ • Maximize I(X;Y|A)  │
│ • Equal budgets     │    │ • CMI-based budgets  │
│ • No complementarity│    │ • Analyst-aware      │
└─────────────────────┘    └──────────────────────┘
      │                               │
      └───────────────┬───────────────┘
                      │
                      ▼
              OUTPUT: Optimized Features
```

**Key CMI Integration Points**:
1. **Lookback Scoring**: Use `I(X_lookback; Y | A)` instead of `I(X_lookback; Y)`
2. **Budget Allocation**: Assign more budget to families with high CMI scores
3. **Analyst Awareness**: Penalize features that correlate highly with Analyst outputs

---

### Step 2: Interaction Generation (3-Phase Pipeline)

```
┌────────────────────────────────────────────────────────────────┐
│                     PHASE 1: Variant Generation                │
│                                                                │
│  Base Features → Generate Variants → [ CMI Prefilter ]        │
│                                            ↓                   │
│                                    Keep Top 40% by CMI         │
│                                            ↓                   │
│                                    LGBM+SHAP Selection         │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│                  PHASE 2: Middle Refinement                    │
│                                                                │
│  Selected Features → Deeper LGBM → [ CMI Diversity Check ]    │
│                                            ↓                   │
│                                  Remove Low-CMI Features       │
│                                  (CMI < threshold)             │
│                                            ↓                   │
│                                    Top 40 Features             │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│                PHASE 3: Interaction Discovery                  │
│                                                                │
│  Top Features → Generate Pairs → [ CMI Interaction Scoring ]  │
│                                            ↓                   │
│                              Score = CMI + (β × Synergy)       │
│                                            ↓                   │
│                                  Top 50 Interactions           │
└────────────────────────────────────────────────────────────────┘
                            ↓
                   OUTPUT: CMI-Filtered Interactions
```

**Key CMI Integration Points**:
- **Phase 1**: Prefilter variants by CMI before expensive LGBM
- **Phase 2**: Ensure each feature has minimum CMI diversity
- **Phase 3**: Score interactions by `CMI + (β × synergy)`

---

### Step 3: Final Feature Selection

```
INPUT: All Generated Features (Lookback + Interactions)
  │
  ├─ Mode Detection ──────────────────┐
  │                                    │
  ▼                                    ▼
┌─────────────────┐          ┌─────────────────────┐
│  ANALYST MODE   │          │   TACTICIAN MODE    │
│   (Standard)    │          │   (CMI-Aware)       │
└─────────────────┘          └─────────────────────┘
      │                               │
      │ Mutual Information            │ Conditional MI
      │ I(X; Y)                        │ I(X; Y | A)
      │                               │
      ▼                               ▼
┌─────────────────────┐    ┌──────────────────────────┐
│ Standard Selection  │    │ CMI-Aware Selection      │
│                     │    │                          │
│ • Score by MI       │    │ • Extract Analyst Info   │
│ • Top N features    │    │ • Score by CMI           │
│ • No filtering      │    │ • Synergy bonus          │
│                     │    │ • Family budgets         │
└─────────────────────┘    └──────────────────────────┘
      │                               │
      └───────────────┬───────────────┘
                      │
                      ▼
          OUTPUT: Selected Feature Sets
         (60, 50, 40 features)
```

**Key CMI Integration Points**:
1. **Analyst Side Info Extraction**: Get Analyst outputs/predictions/features
2. **CMI Scoring**: Use `cmi_scorer.score_features(X, Y, A)`
3. **Synergy Bonus**: Add `β × synergy_score` to CMI
4. **Family Budgets**: Allocate (5-15) features per family based on CMI

---

## CMI Computation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CMI COMPUTATION PIPELINE                     │
│                                                                 │
│  1. Detect Tactician Mode                                       │
│     ├─ Check: 'tactician' in step_name?                         │
│     ├─ Check: config.get('tactician_mode', False)?              │
│     └─ Check: 'tactician' in execution_context?                 │
│                       ↓                                         │
│  2. Extract Analyst Side Information                            │
│     ├─ analyst_outputs (predictions, probabilities)             │
│     ├─ feature_importance (scores from Analyst)                 │
│     └─ regime_labels (optional)                                 │
│                       ↓                                         │
│  3. Select CMI Estimator                                        │
│     ├─ KSG   (high accuracy, slow)                              │
│     ├─ GCMI  (balanced, medium speed)                           │
│     └─ Binned (fast, lower accuracy)                            │
│                       ↓                                         │
│  4. Compute CMI Scores                                          │
│     For each feature X:                                         │
│       CMI(X; Y | A) = I(X; Y) - I(X; A) + corrections          │
│                       ↓                                         │
│  5. Apply Complementarity Filter                                │
│     ├─ Keep features with CMI > threshold                       │
│     ├─ Apply synergy bonus for complementary pairs              │
│     └─ Allocate budget per family                               │
│                       ↓                                         │
│  6. Return Selected Features                                    │
│     └─ Features with highest CMI complementarity                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Mode Detection Logic

```
┌─────────────────────────────────────────────────────────────────┐
│                      MODE DETECTION TREE                        │
└─────────────────────────────────────────────────────────────────┘

Is 'tactician' in step_name.lower()?
│
├─ YES ──→ TACTICIAN MODE
│
└─ NO
   │
   └─ Is 'tactician' in config.get('execution_context', '')?
      │
      ├─ YES ──→ TACTICIAN MODE
      │
      └─ NO
         │
         └─ Is config.get('tactician_mode', False) == True?
            │
            ├─ YES ──→ TACTICIAN MODE
            │
            └─ NO
               │
               └─ Are there 'tactician_*' features in dataframe?
                  │
                  ├─ YES + enable_cmi_complementarity=True ──→ TACTICIAN MODE
                  │
                  └─ NO ──→ ANALYST MODE (default)

```

---

## CMI Estimator Selection

```
┌──────────────────────────────────────────────────────────────────┐
│                   ESTIMATOR SELECTION LOGIC                      │
└──────────────────────────────────────────────────────────────────┘

Input: n_features, n_samples, stage (prefilter/shortlist/final)
│
├─ n_features > 800 OR n_samples < 1500?
│  │
│  ├─ YES
│  │  │
│  │  ├─ stage == 'prefilter'?  ──→ BINNED (fastest)
│  │  ├─ stage == 'shortlist'?  ──→ GCMI (balanced)
│  │  └─ stage == 'final'?       ──→ KSG (most accurate)
│  │
│  └─ NO
│     │
│     ├─ n_features <= 600 AND n_samples >= 2000?
│     │  │
│     │  ├─ stage == 'prefilter'?  ──→ GCMI (balanced)
│     │  └─ stage == 'final'?       ──→ KSG (most accurate)
│     │
│     └─ DEFAULT ──→ GCMI (balanced)
│
└─ Fallback mechanisms:
   ├─ Timeout exceeded?       ──→ Switch to faster estimator
   ├─ Memory exceeded?        ──→ Switch to binned
   └─ Accuracy insufficient?  ──→ Switch to KSG

```

---

## Analyst Side Information Extraction

```
┌──────────────────────────────────────────────────────────────────┐
│               ANALYST SIDE INFO EXTRACTION FLOW                  │
└──────────────────────────────────────────────────────────────────┘

Pipeline State
│
├─ Extract Analyst Outputs
│  ├─ Check: pipeline_state['analyst_outputs']
│  ├─ Check: pipeline_state['analyst_predictions']
│  ├─ Check: pipeline_state['analyst_model_outputs']
│  └─ Check: pipeline_state['analyst_features']
│     │
│     └─ Convert to DataFrame if needed
│
├─ Extract Feature Importance
│  ├─ Check: pipeline_state['feature_importance']
│  ├─ Check: pipeline_state['feature_scores']
│  └─ Check: pipeline_state['analyst_feature_importance']
│     │
│     └─ Convert to Dict[str, float] if needed
│
├─ Extract Regime Labels (optional)
│  ├─ Check: pipeline_state['regime_labels']
│  ├─ Check: pipeline_state['regime_clusters']
│  └─ Check: pipeline_state['cluster_labels']
│     │
│     └─ Convert to Series if needed
│
└─ Package as AnalystSideInfoResult
   ├─ analyst_outputs: DataFrame
   ├─ feature_importance: Dict[str, float]
   ├─ regime_labels: Optional[Series]
   └─ metadata: Dict[str, Any]

```

---

## Feature Scoring Comparison

### Analyst Mode (Standard MI)

```
Feature Score = I(X; Y)

Where:
  X = Feature values
  Y = Target values

Higher score = Better predictive power
```

### Tactician Mode (CMI Complementarity)

```
Feature Score = I(X; Y | A) + (β × Synergy(X, A))

Where:
  X = Tactician feature
  Y = Target values
  A = Analyst side information
  β = Synergy bonus weight (typically 0.25)

Breakdown:
  I(X; Y | A) = I(X; Y) - I(X; A) + correction_terms
  
  ├─ I(X; Y)    = Predictive power (standard MI)
  ├─ -I(X; A)   = Redundancy penalty (correlation with Analyst)
  └─ corrections = Cross-terms and synergy adjustments

Higher score = Better complementarity to Analyst
```

---

## Implementation Status Matrix

| Step | File | Status | CMI Integration | Priority |
|------|------|--------|----------------|----------|
| **Final Feature Selection** | `feature_generation_final_feature_selection_step.py` | ✅ Partially | Lines 74-86 (placeholder imports), 898-1104 (CMI logic) | 🔴 HIGH |
| **Interaction Generation** | `feature_generation_interaction_generation_step.py` | ❌ Missing | Needs Phase 1/2/3 CMI integration | 🟡 MEDIUM |
| **Lookback Optimization** | `feature_generation_period_lookback_optimization_step.py` | ❌ Missing | Needs lookback scoring & budget allocation | 🟡 MEDIUM |

### Required Actions

#### 1. Fix Final Feature Selection (5 minutes)
```python
# Lines 74-86: Replace placeholder with:
from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_complementarity import (
    CMIComplementarityScorer, CMIComplementarityConfig
)
from src.training.steps.pre_training.unified_data_driven_pipeline.utils.analyst_side_info import (
    AnalystSideInfoHandler
)
CMI_COMPLEMENTARITY_AVAILABLE = True
```

#### 2. Add CMI to Interaction Generation (30 minutes)
- Phase 1: Add CMI prefiltering after variant generation
- Phase 2: Add CMI diversity check during refinement
- Phase 3: Add CMI interaction scoring

#### 3. Add CMI to Lookback Optimization (20 minutes)
- Add CMI-based lookback period scoring
- Add CMI-based family budget allocation

---

## Testing Checklist

### Unit Tests
- [ ] Test mode detection logic
- [ ] Test Analyst side info extraction
- [ ] Test CMI scoring computation
- [ ] Test estimator selection
- [ ] Test fallback mechanisms

### Integration Tests
- [ ] Test full pipeline with CMI (all 3 steps)
- [ ] Test Analyst mode protection (no CMI when `tactician_mode=False`)
- [ ] Test performance benchmarks
- [ ] Compare Analyst vs Tactician features

### Validation Tests
- [ ] Verify CMI features have lower correlation with Analyst
- [ ] Verify ensemble performance improves with CMI
- [ ] Verify complementarity metrics
- [ ] Verify no data leakage

---

## Performance Expectations

### With CMI Integration (Tactician Mode)

| Metric | Expected Improvement | Notes |
|--------|---------------------|-------|
| **Feature Redundancy** | ↓ 40-60% | Lower correlation with Analyst |
| **Ensemble Performance** | ↑ 10-20% | Better complementarity |
| **Feature Diversity** | ↑ 30-50% | More diverse information |
| **Computation Time** | ↑ 15-30% | CMI overhead (mitigated by caching) |

### Fallback Performance

If CMI components unavailable:
- ✅ Graceful degradation to standard MI
- ✅ No pipeline breakage
- ✅ Warning logged
- ✅ Standard feature selection continues

---

## Key Takeaways

### 1. **When to Use CMI**
- ✅ Always in Tactician mode (`tactician_mode=True`)
- ❌ Never in Analyst mode (standard MI only)

### 2. **What CMI Measures**
- **I(X; Y | A)** = Information in X about Y, **given** we already know A
- High CMI = X provides **new** information beyond A
- Low CMI = X is **redundant** with A

### 3. **Where to Apply CMI**
- **Lookback Optimization**: Score lookback periods by CMI
- **Interaction Generation**: Filter variants, check diversity, score interactions
- **Final Selection**: Select features with highest CMI complementarity

### 4. **How to Integrate CMI**
```python
# Standard pattern:
if tactician_mode and CMI_AVAILABLE:
    analyst_info = extract_analyst_side_info()
    cmi_scores = cmi_scorer.score_features(X, Y, analyst_info)
    # Use cmi_scores instead of MI
else:
    # Fallback to standard MI
    mi_scores = compute_mutual_information(X, Y)
```

---

## References

- **Full Analysis**: `CMI_TACTICIAN_MODE_INTEGRATION_ANALYSIS.md`
- **Quick Reference**: `CMI_TACTICIAN_QUICK_REFERENCE.md`
- **CMI Guide**: `docs/CMI_COMPLEMENTARITY_GUIDE.md`
- **Source Code**: `src/training/steps/pre_training/unified_data_driven_pipeline/utils/`
  - `cmi_complementarity.py`
  - `analyst_side_info.py`
  - `cmi_estimators.py`

---

**End of Flowchart Document**
