# Pipeline Descriptions Review

## Summary
Most of your descriptions are **correct**, but there are some important corrections and clarifications needed, particularly around timeframes and the scope of feature engineering steps.

---

## PRE_TRAINING/ Pipeline Steps

### ✅ CORRECT: multi_horizon_profit_labeler
**Your Description:**
> Apply triple barrier method-inspired, per-regime, volatility and noise-aware

**Status:** ✅ **CORRECT**

**Evidence:** The component uses enhanced multi-horizon profit labeling with:
- Triple barrier method inspiration
- Per-regime/per-cluster optimization
- Volatility-aware labeling
- Noise gating and quality scoring
- Trading-aware label definitions for both Analyst and Tactician

**File:** `src/training/steps/pre_training/profit_labeling/enhanced_multi_horizon_labeler.py`

---

### ⚠️ NEEDS CLARIFICATION: feature_lookback_optimization
**Your Description:**
> Optimize feature lookback periods for all features present in feature_engineering/'s feature bank, except interaction feature, cross-timeframe features, wavelets, auto encoders and regime-specific features. **15m timeframe by default**

**Status:** ⚠️ **PARTIALLY CORRECT - Timeframe depends on pipeline**

**Corrections:**
1. **Timeframe is NOT always 15m by default:**
   - **Analyst pipeline:** Uses **15m** timeframe (you were correct here)
   - **Tactician pipeline:** Also uses **15m** timeframe (per code)
   - The PRE_TRAINING components accept a timeframe parameter with **15m as default**

2. **Scope is correct:** Optimizes lookback periods for base features, excluding:
   - Interaction features (generated later)
   - Cross-timeframe features (generated later)
   - Wavelets, autoencoders (if used)
   - Regime-specific features (added from market_analysis)

**File:** `src/training/steps/pre_training/feature_lookback_optimization/feature_lookback_optimization.py`

**Suggested Correction:**
> Optimize feature lookback periods for base features (excludes interaction, cross-timeframe, wavelets, autoencoders, and regime features). Timeframe: 15m (default, configurable per pipeline)

---

### ✅ CORRECT: interactive_feature_generation
**Your Description:**
> Cross timeframe & interaction features

**Status:** ✅ **CORRECT** (but could be more descriptive)

**Expanded Description:**
The component generates:
- **Interaction features:** Polynomial combinations, multiplicative interactions
- **Cross-timeframe features:** Multi-timeframe aggregations
- Integration with optimized interaction orchestrator
- Hardware-accelerated feature generation

**File:** `src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/interactive_feature_generation_component.py`

**Suggested Enhancement:**
> Generate interaction features (polynomial, multiplicative) and cross-timeframe features with hardware acceleration

---

### ✅ CORRECT: final_feature_selection
**Your Description:**
> xx→120→100→80→60 features

**Status:** ✅ **CORRECT**

**Evidence:** Multi-stage feature selection pipeline:
- Stage 1: Initial pool → 120 features
- Stage 2: 120 → 100 features
- Stage 3: 100 → 80 features
- Stage 4: 80 → 60 features (final)

**File:** `src/training/steps/pre_training/final_feature_selection_step.py`

---

## MODEL_TRAINING/ Pipeline Steps

### ❌ INCORRECT: analyst_pre_ml_orchestration (you had tactician_pre_ml_orchestration)
**Your Description:**
> Applies differentiated horizon labeling + Optimizes feature lookback periods + Generates PID features + Selects final features. It does all this with the **60m timeframe**, with per-regime/cluster optimisation.

**Status:** ❌ **INCORRECT - Wrong timeframe**

**Corrections:**
1. **Timeframe:** Analyst uses **15m**, NOT 60m
2. **PID features:** Not mentioned in Analyst orchestrator (only in Tactician)
3. **Scope:** Calls the PRE_TRAINING sub-pipeline with all 4 steps

**Actual Implementation:**
- **Timeframe:** 15m (strategic IF-to-trade decisions)
- **Training Data:** ALL market data (not filtered)
- **Steps:** 
  1. Multi-horizon profit labeling
  2. Feature lookback optimization (per-regime/cluster)
  3. Interactive feature generation (interaction, polynomial, cross-timeframe)
  4. Final feature selection (120→100→80→60)

**File:** `src/training/steps/models_training/analyst_pre_ml_orchestration.py` (line 10-14)

**Suggested Correction:**
> **analyst_pre_ml_orchestration** - Applies multi-horizon profit labeling + Optimizes feature lookback periods + Generates interaction/cross-timeframe features + Selects final features. All on **15m timeframe** with per-regime/cluster optimization. Uses the PRE_TRAINING pipeline.

---

### ✅ CORRECT: analyst_models_training
**Your Description:**
> Per-regime individual model training with HPO, saving, and metrics. Trained on all features selected by PRE_TRAINING/final_feature_selection (the number of features depends on the type of ML model we are training) + regime features (from the Ensemble ML model in market_analysis/).

**Status:** ✅ **CORRECT**

**Evidence:** Trains base models per-regime with:
- Features from final_feature_selection
- Regime features from market_analysis
- Per-regime training integration
- HPO and metrics tracking

**File:** `src/training/steps/model_training/sub_pipeline.py` (line 801-918)

---

### ✅ CORRECT: analyst_ensemble_training
**Your Description:**
> Per-regime ensemble training with HPO, saving, and metrics. Trained on the same features as above + the outputs from the base Analyst models

**Status:** ✅ **CORRECT**

**Evidence:** Trains ensemble models using:
- Base features from analyst_models_training
- Base model predictions as meta-features
- Per-regime ensemble training

**File:** `src/training/steps/model_training/sub_pipeline.py` (line 920-1024)

---

### ❌ INCORRECT: tactician_pre_ml_orchestration
**Your Description:**
> Applies differentiated horizon labeling (based on Analyst targets) + Optimizes feature lookback periods + Generates interaction/cross timeframe features + Selects final features. It does all this with the **15m timeframe**. Uses the pipeline present in src/training/steps/PRE_TRAINING/

**Status:** ❌ **INCORRECT - Missing critical information**

**Corrections:**
1. **Analyst Filtering:** You missed that Tactician filters data based on Analyst predictions (confidence >= 0.4%)
2. **Subsequent Minutes:** Extracts data from subsequent 45 minutes after Analyst signals
3. **Differentiated Horizons:** Separate long & short signal processing
4. **PID Features:** Tactician DOES generate PID-based features (Analyst doesn't)

**Actual Implementation:**
1. Separate long & short signals from Analyst with confidence >= 0.5
2. Extract data from subsequent 45 minutes
3. Optimize features lookback periods
4. **Generate PID-based features** ← YOU MISSED THIS
5. Apply multi-horizon profit labeling
6. Select final features
7. Train Tactician models twice (longs and shorts) with differentiated features

**File:** `src/training/steps/model_training/tactician_pre_ml_orchestrator.py` (line 1-17)

**Suggested Correction:**
> **tactician_pre_ml_orchestration** - Filters on Analyst signals (confidence >= 0.4%) + Extracts subsequent 45 minutes + Optimizes feature lookback periods + **Generates PID-based features** + Applies differentiated horizon labeling + Generates interaction/cross-timeframe features + Selects final features. All on **15m timeframe** (filtered). Uses PRE_TRAINING pipeline + PID generation.

---

### ✅ CORRECT: tactician_models_training
**Your Description:**
> Individual model training with HPO, saving, and metrics. Trained on all features selected by PRE_TRAINING/final_feature_selection (the number of features depends on the type of ML model we are training) + regime features (from the Ensemble ML model in market_analysis/) + the outputs from the Analyst Ensemble model.

**Status:** ✅ **CORRECT**

**Evidence:** Trains base models with:
- Features from final_feature_selection
- Regime features from market_analysis
- **Analyst Ensemble predictions** as additional features
- Per-regime training

**File:** `src/training/steps/model_training/sub_pipeline.py` (line 1131-1280)

---

### ✅ CORRECT: tactician_ensemble_training
**Your Description:**
> Ensemble training with HPO, saving, and metrics. Trained on the same features as above + the outputs from the base Tactician models

**Status:** ✅ **CORRECT**

**Evidence:** Trains ensemble models using:
- Base features from tactician_models_training
- Base Tactician model predictions as meta-features
- Analyst features
- Per-regime ensemble training

**File:** `src/training/steps/model_training/sub_pipeline.py` (line 1282-1407)

---

## Key Corrections Summary

### 🔴 Critical Errors Found:
1. **Analyst timeframe:** You said 60m, but it's actually **15m**
2. **PID Features:** You placed this in Analyst, but it's actually in **Tactician only**
3. **Tactician filtering:** You didn't mention the Analyst-based filtering and 45-minute extraction

### 🟡 Minor Clarifications Needed:
1. **feature_lookback_optimization timeframe:** Clarify it's configurable (default 15m)
2. **interactive_feature_generation:** Could be more descriptive about what it generates

### ✅ Correct Descriptions:
- multi_horizon_profit_labeler ✅
- final_feature_selection ✅
- analyst_models_training ✅
- analyst_ensemble_training ✅
- tactician_models_training ✅
- tactician_ensemble_training ✅

---

## Recommended Updated Descriptions

### PRE_TRAINING/
```
multi_horizon_profit_labeler - Apply triple barrier method-inspired, per-regime, volatility and noise-aware multi-horizon profit labeling

feature_lookback_optimization - Optimize feature lookback periods for base features (excludes interaction, cross-timeframe, wavelets, autoencoders, and regime features). Timeframe: 15m (default, configurable)

interactive_feature_generation - Generate interaction features (polynomial, multiplicative) and cross-timeframe features with hardware acceleration

final_feature_selection - Multi-stage feature selection: Initial→120→100→80→60 features
```

### MODEL_TRAINING/
```
analyst_pre_ml_orchestration - Applies multi-horizon profit labeling + Optimizes feature lookback periods + Generates interaction/cross-timeframe features + Selects final features. All on 15m timeframe with per-regime/cluster optimization. Training data: ALL market data (unfiltered). Uses the PRE_TRAINING pipeline.

analyst_models_training - Per-regime individual model training with HPO, saving, and metrics. Trained on all features selected by PRE_TRAINING/final_feature_selection + regime features (from Ensemble ML model in market_analysis/)

analyst_ensemble_training - Per-regime ensemble training with HPO, saving, and metrics. Trained on same features as above + outputs from base Analyst models

tactician_pre_ml_orchestration - Filters on Analyst signals (confidence >= 0.4%) + Extracts subsequent 45 minutes + Optimizes feature lookback periods + Generates PID-based features + Applies differentiated horizon labeling (separate for longs/shorts) + Generates interaction/cross-timeframe features + Selects final features. All on 15m timeframe (filtered). Uses PRE_TRAINING pipeline + PID generation.

tactician_models_training - Individual model training with HPO, saving, and metrics. Trained on all features selected by PRE_TRAINING/final_feature_selection + regime features (from Ensemble ML model in market_analysis/) + outputs from Analyst Ensemble model

tactician_ensemble_training - Ensemble training with HPO, saving, and metrics. Trained on same features as above + outputs from base Tactician models
```

---

## Pipeline Flow Visualization

```
ANALYST PIPELINE (15m timeframe - "IF we trade"):
1. analyst_pre_ml_orchestration (15m, unfiltered data)
   └─> PRE_TRAINING: labeling → lookback → interaction → selection
2. analyst_models_training (per-regime base models)
3. analyst_ensemble_training (ensemble models)
   └─> Outputs predictions for Tactician

TACTICIAN PIPELINE (15m timeframe - "WHEN we trade"):
4. tactician_pre_ml_orchestration (15m, filtered on Analyst >= 0.4%)
   └─> Filter Analyst signals → Extract 45min → PRE_TRAINING + PID
5. tactician_models_training (with Analyst predictions as features)
6. tactician_ensemble_training (final ensemble)
```

---

## Files Referenced in This Review

- `src/training/steps/pre_training/sub_pipeline.py`
- `src/training/steps/model_training/sub_pipeline.py`
- `src/training/steps/models_training/analyst_pre_ml_orchestration.py`
- `src/training/steps/model_training/tactician_pre_ml_orchestrator.py`
- `src/training/steps/pre_training/profit_labeling/enhanced_multi_horizon_labeler.py`
- `src/training/steps/pre_training/feature_lookback_optimization/feature_lookback_optimization.py`
- `src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/interactive_feature_generation_component.py`
- `src/training/steps/pre_training/final_feature_selection_step.py`