# Requirements Implementation Plan

## Overview
This document outlines the implementation plan for the 6 key requirements for the Analyst and Tactician training pipelines.

## Requirements

### ✅ Requirement 1: Wire in NAS & TAS ML Trading Models
**Status**: IN PROGRESS

**Current State**:
- Architecture exists in `src/training/steps/models_training/nas_tas/`
- Components available:
  - `training_orchestrator.py` - Complete NAS/TAS orchestration
  - `regime_aware_trainer.py` - Regime-aware model training
  - `model_selector.py` - Model selection system
  - `model_manager.py` - Model management

**Implementation**:
1. Import NAS/TAS training orchestrator into `analyst_models_training.py`
2. Import NAS/TAS training orchestrator into `tactician_models_training.py`
3. Add NAS and TAS as model types in Analyst configuration
4. Add NAS and TAS as model types in Tactician configuration
5. Wire NAS/TAS training into the training workflow

**Files to Modify**:
- `src/training/steps/models_training/analyst_models_training.py`
- `src/training/steps/models_training/tactician_models_training.py`
- `src/training/steps/models_training/analyst_training_pipeline.py`
- `src/training/steps/models_training/tactician_training_pipeline.py`

---

### ✅ Requirement 2: Separation Between Short & Long Pipelines
**Status**: COMPLETED (already implemented in directional training)

**Current State**:
- `regime_aware_trainer.py` has directional training with `DirectionMode` enum
- Supports: `BOTH`, `LONG_ONLY`, `SHORT_ONLY`, `SEPARATE`
- `training_orchestrator.py` has directional orchestration
- Directional data separation implemented in `_separate_directional_data()`

**Verification**:
- ✅ DirectionMode enum defined
- ✅ Separate directional features supported
- ✅ Per-direction model training implemented
- ✅ Directional models storage in results

**No Action Required** - Feature already exists

---

### ✅ Requirement 3: Analyst Per-Regime, Tactician NOT Per-Regime
**Status**: TO IMPLEMENT

**Current State**:
- Both Analyst and Tactician use `regime_aware_trainer.py`
- Per-regime training is controlled by configuration
- Need to differentiate behavior between Analyst and Tactician

**Implementation**:
1. Analyst configuration: `enable_regime_aware_training=True`
2. Tactician configuration: `enable_regime_aware_training=False`
3. Update `analyst_models_training.py` to enable per-regime training
4. Update `tactician_models_training.py` to disable per-regime training
5. Add per-regime splitting logic to Analyst
6. Add unified training logic to Tactician

**Files to Modify**:
- `src/training/steps/models_training/analyst_models_training.py`
- `src/training/steps/models_training/tactician_models_training.py`
- `src/training/steps/models_training/analyst_training_pipeline.py`
- `src/training/steps/models_training/tactician_training_pipeline.py`

---

### ✅ Requirement 4: Add MultiHorizon N-BEATS to Analyst
**Status**: TO IMPLEMENT

**Current State**:
- MultiHorizon N-BEATS exists in `src/utils/ml_common/models/multiscale_nbeats.py`
- Model factory supports `MULTISCALE_NBEATS` type
- Need to add to Analyst model types

**Implementation**:
1. Import `MultiScaleNBEATSRegressor` in analyst_models_training.py
2. Add `MULTISCALE_NBEATS` to Analyst model types enum
3. Add model factory method for N-BEATS in Analyst
4. Configure N-BEATS for 15m timeframe
5. Add N-BEATS to default Analyst model list

**Configuration**:
```python
AnalystModelType.MULTISCALE_NBEATS = "MULTISCALE_NBEATS"

default_model_types = [
    "TCN",
    "LIGHTGBM",
    "RIDGE",
    "ELASTIC_NET",
    "RANDOM_FOREST",
    "NAS",  # NEW
    "TAS",  # NEW
    "MULTISCALE_NBEATS"  # NEW
]
```

**Files to Modify**:
- `src/training/steps/models_training/analyst_models_training.py`
- `src/training/steps/models_training/analyst_training_pipeline.py`

---

### ✅ Requirement 5: Tactician Has RandomSurvivalForest & XGBoost
**Status**: ALREADY IMPLEMENTED

**Current State**:
- `tactician_models_training.py` line 87-89 defines:
  ```python
  class TacticianModelType(Enum):
      RANDOM_SURVIVAL_FOREST = "RANDOM_SURVIVAL_FOREST"
      XGBOOST = "XGBOOST"
      ELASTIC_NET_CV = "ELASTIC_NET_CV"
  ```

**Verification**:
- ✅ RandomSurvivalForest defined
- ✅ XGBoost defined
- ✅ Model factory methods exist
- Need to add: NAS, TAS

**Additional Implementation**:
1. Add NAS to Tactician model types
2. Add TAS to Tactician model types

**Files to Modify**:
- `src/training/steps/models_training/tactician_models_training.py`

---

### ✅ Requirement 6: Feed Regime Model Outputs to Both
**Status**: TO IMPLEMENT

**Description**: Both Analyst and Tactician should receive the outputs from ML regime models (top 3 most likely regimes we're currently in)

**Current State**:
- Regime detection exists in `market_analysis/` stage
- Regime ensemble training creates regime predictions
- Need to pass regime probabilities as features to both Analyst and Tactician

**Implementation**:
1. Load regime ensemble model outputs from `market_analysis` stage
2. Extract top 3 regime probabilities for each sample
3. Add as features to training data:
   - `regime_prob_1` (highest probability regime)
   - `regime_prob_2` (second highest)
   - `regime_prob_3` (third highest)
   - `regime_1_id` (regime ID of highest)
   - `regime_2_id` (regime ID of second)
   - `regime_3_id` (regime ID of third)

**Integration Points**:
1. `analyst_pre_ml_orchestration.py` - Add regime features before feature selection
2. `tactician_pre_ml_orchestration.py` - Add regime features before feature selection
3. Both should receive:
   - Top 3 regime probabilities
   - Top 3 regime IDs
   - Current regime confidence score

**Files to Modify**:
- `src/training/steps/models_training/analyst_pre_ml_orchestration.py`
- `src/training/steps/models_training/tactician_pre_ml_orchestration.py`
- `src/training/steps/model_training/sub_pipeline.py`

---

## Implementation Summary

### Phase 1: Core Model Wiring (Requirements 1, 4, 5)
**Priority**: HIGH
**Status**: In Progress

1. ✅ Add NAS & TAS to Analyst models
2. ✅ Add MultiHorizon N-BEATS to Analyst models
3. ✅ Add NAS & TAS to Tactician models
4. ✅ Verify RandomSurvivalForest & XGBoost in Tactician

### Phase 2: Training Differentiation (Requirement 3)
**Priority**: HIGH
**Status**: To Do

1. ✅ Enable per-regime training for Analyst
2. ✅ Disable per-regime training for Tactician
3. ✅ Update configuration files
4. ✅ Test regime-aware vs unified training

### Phase 3: Regime Feature Integration (Requirement 6)
**Priority**: MEDIUM
**Status**: To Do

1. ✅ Load regime ensemble outputs
2. ✅ Extract top 3 regime probabilities
3. ✅ Add regime features to both pipelines
4. ✅ Update feature selection to include regime features

### Phase 4: Short/Long Separation (Requirement 2)
**Priority**: MEDIUM
**Status**: Already Done

1. ✅ Verify directional training configuration
2. ✅ Test short and long model separation
3. ✅ Validate directional feature generation

---

## Testing Plan

### Unit Tests
1. Test NAS model creation and training
2. Test TAS model creation and training
3. Test MultiHorizon N-BEATS creation and training
4. Test per-regime training for Analyst
5. Test unified training for Tactician
6. Test regime feature integration

### Integration Tests
1. Test complete Analyst pipeline with all models
2. Test complete Tactician pipeline with all models
3. Test Analyst → Tactician flow with regime features
4. Test short/long separation end-to-end

### Performance Tests
1. Benchmark NAS vs TAS vs traditional models
2. Benchmark per-regime vs unified training
3. Validate regime feature contribution to performance

---

## Expected Outcomes

### Analyst Models (15m timeframe, per-regime)
```python
# Base Models (5 types)
- ElasticNet (L1+L2 regularization)
- RandomForest (tree ensemble)

# Advanced Models (3 types)
- NAS (Neural Architecture Search)
- TAS (Tree-based Architecture Search)  
- MultiHorizon N-BEATS (time series forecasting)

# Ensemble
- Analyst Ensemble (combining all 5 base models)

# Training Structure
- 8 regimes × 5 models = 40 base models
- 8 per-regime ensemble models
- Total: 48 models per direction

# Features
- Selected features from pre-ML orchestration
- Regime features (top 3 probabilities + IDs)
- Per-regime optimization
```

### Tactician Models (5m timeframe, NOT per-regime)
```python
# Base Models (4 types)
- RandomSurvivalForest (survival analysis)
- XGBoost (gradient boosting)

# Advanced Models (2 types)
- NAS (Neural Architecture Search)
- TAS (Tree-based Architecture Search)

# Ensemble
- Tactician Ensemble (combining all 4 base models)

# Training Structure
- 4 unified models (no regime splitting)
- 1 unified ensemble model
- Total: 5 models per direction

# Features
- Selected features from pre-ML orchestration
- Regime features (top 3 probabilities + IDs)
- Analyst ensemble outputs (predictions + confidence)
- Unified training (not per-regime)
```

### Pipeline Flow
```
1. Market Analysis Stage
   ↓ (regime assignments, top 3 regimes)
   
2. Analyst Pipeline (15m)
   → Pre-ML Orchestration (15m) + regime features
   → Per-Regime Base Models (TCN, LightGBM, Ridge, ElasticNet, RandomForest, NAS, TAS, N-BEATS)
   → Per-Regime Ensemble
   ↓ (predictions with >0.4% confidence)
   
3. Tactician Pipeline (5m, filtered)
   → Pre-ML Orchestration (5m, filtered) + regime features + Analyst outputs
   → Unified Base Models (RSF, XGBoost, ElasticNetCV, NAS, TAS)
   → Unified Ensemble
   ↓ (final trade execution signals)
```

---

## Success Criteria

1. ✅ NAS & TAS models successfully wire into training pipelines
2. ✅ Short/long model separation working correctly
3. ✅ Analyst trains per-regime, Tactician trains unified
4. ✅ MultiHorizon N-BEATS integrated into Analyst
5. ✅ RandomSurvivalForest & XGBoost confirmed in Tactician
6. ✅ Regime model outputs (top 3) feed into both pipelines
7. ✅ All models train successfully end-to-end
8. ✅ Model performance meets or exceeds baselines

---

## Notes

- **NAS (Neural Architecture Search)**: Automatically finds optimal neural network architectures
- **TAS (Tree-based Architecture Search)**: Automatically finds optimal tree-based model configurations
- **MultiHorizon N-BEATS**: Time series forecasting model with multi-horizon capabilities
- **Per-Regime Training**: Separate models for each detected market regime
- **Unified Training**: Single model trained on all data (regardless of regime)
- **Regime Features**: Top 3 most likely regimes with probabilities and IDs
