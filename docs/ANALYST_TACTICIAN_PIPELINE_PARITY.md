# Analyst & Tactician Training Pipeline Parity Analysis

## Overview
This document verifies the parity between Analyst and Tactician training pipelines, ensuring both follow the same structure with appropriate differentiation for their distinct roles.

## Pipeline Architecture

### Common Structure (Both Pipelines)
Both Analyst and Tactician follow a 3-step training process:
1. **Pre-ML Orchestration**: Feature engineering and preparation
2. **Base Models Training**: Individual model training with HPO
3. **Ensemble Training**: Ensemble model training with HPO

---

## ANALYST PIPELINE (15m Timeframe - "IF we trade")

### Purpose
Train models to decide **IF** we should trade (strategic decision-making)

### Configuration
- **Timeframe**: 15m (higher timeframe for strategic decisions)
- **Training Data**: ALL market data (not filtered)
- **Role**: Strategic assessment of trading opportunities

### Step 1: analyst_pre_ml_orchestration
**Location**: `src/training/steps/models_training/analyst_pre_ml_orchestration.py`

**Operations**:
1. Multi-horizon profit labeling with differentiated horizons
2. Feature lookback period optimization (per-regime/cluster)
3. PID-based feature generation (interaction, polynomial, cross-timeframe)
4. Final feature selection (multi-stage: 120→100→80→60)

**Key Features**:
- Per-regime optimization: ✅
- Per-cluster optimization: ✅
- Timeframe: 15m
- Input: Raw market data (15m)
- Output: Optimized features for Analyst training

### Step 2: analyst_models_training
**Location**: `src/training/steps/models_training/analyst_models_training.py`

**Models Trained**:
- TCN (Temporal Convolutional Network)
- LightGBM
- Ridge Regression
- ElasticNet
- RandomForest

**Features Used**:
- All selected features from pre-ML orchestration
- Regime features (from Ensemble ML model in market_analysis/)
- Per-regime individual models with HPO

**Output**:
- Trained NAS (Neural Architecture Search) models (15m timeframe)
- Trained TAS (Tree-based Architecture Search) models (15m timeframe)
- Per-regime model variants

### Step 3: analyst_ensemble_training
**Location**: `src/training/steps/models_training/analyst_ensemble_training.py`

**Features Used**:
- All selected features from pre-ML orchestration
- Regime features (from Ensemble ML model)
- **Outputs from base Analyst models**

**Output**:
- Ensemble model combining all base models
- Predictions with confidence scores (used for Tactician filtering)

---

## TACTICIAN PIPELINE (5m Timeframe - "WHEN we trade")

### Purpose
Train models to decide **WHEN** to execute trades (tactical execution timing)

### Configuration
- **Timeframe**: 5m (lower timeframe for tactical decisions)
- **Training Data**: FILTERED on Analyst "green" signals (>0.4% confidence)
- **Role**: Tactical timing of trade execution

### Step 1: tactician_pre_ml_orchestration
**Location**: `src/training/steps/models_training/tactician_pre_ml_orchestration.py`

**Operations**:
0. **Data filtering on Analyst signals** (>0.4% confidence threshold) ⭐
1. Multi-horizon profit labeling with differentiated horizons
2. Feature lookback period optimization (per-regime/cluster)
3. PID-based feature generation (interaction, polynomial, cross-timeframe)
4. Final feature selection (multi-stage: 120→100→80→60)

**Key Features**:
- Per-regime optimization: ✅
- Per-cluster optimization: ✅
- Timeframe: 5m
- **Analyst signal filtering**: ✅ (0.4% threshold)
- Input: Raw market data (5m) + Analyst predictions
- Output: Optimized features for Tactician training (filtered dataset)

### Step 2: tactician_models_training
**Location**: `src/training/steps/models_training/tactician_models_training.py`

**Models Trained**:
- RandomSurvivalForest
- XGBoost
- ElasticNetCV

**Features Used**:
- All selected features from pre-ML orchestration
- Regime features (from Ensemble ML model in market_analysis/)
- **Outputs from Analyst Ensemble model** ⭐
- Per-regime individual models with HPO

**Output**:
- Trained NAS models (5m timeframe)
- Trained TAS models (5m timeframe)
- Per-regime model variants

### Step 3: tactician_ensemble_training
**Location**: `src/training/steps/models_training/tactician_ensemble_training.py`

**Features Used**:
- All selected features from pre-ML orchestration
- Regime features (from Ensemble ML model)
- **Outputs from Analyst Ensemble model** ⭐
- **Outputs from base Tactician models**

**Output**:
- Ensemble model combining all base models
- Final predictions for trade execution timing

---

## Parity Verification Matrix

| Aspect | Analyst | Tactician | Parity Status |
|--------|---------|-----------|---------------|
| **Pipeline Steps** | 3 steps | 3 steps | ✅ MATCH |
| **Pre-ML Orchestration** | ✅ | ✅ | ✅ MATCH |
| **Feature Engineering** | 4 sub-steps | 4 sub-steps | ✅ MATCH |
| **Base Models Training** | ✅ | ✅ | ✅ MATCH |
| **Ensemble Training** | ✅ | ✅ | ✅ MATCH |
| **Per-Regime Optimization** | ✅ | ✅ | ✅ MATCH |
| **Per-Cluster Optimization** | ✅ | ✅ | ✅ MATCH |
| **HPO (Hyperparameter Optimization)** | ✅ | ✅ | ✅ MATCH |
| **NAS Models** | ✅ (15m) | ✅ (5m) | ✅ MATCH |
| **TAS Models** | ✅ (15m) | ✅ (5m) | ✅ MATCH |
| **Separate Short/Long Models** | ✅ | ✅ | ✅ MATCH |

## Key Differences (By Design)

| Aspect | Analyst | Tactician | Reason |
|--------|---------|-----------|--------|
| **Timeframe** | 15m | 5m | Strategic vs. Tactical |
| **Training Data** | All data | Filtered (>0.4% conf) | Quality focus for Tactician |
| **Purpose** | IF to trade | WHEN to trade | Role separation |
| **Analyst Features** | N/A | Uses Analyst outputs | Hierarchical dependency |
| **Data Filtering** | None | Analyst signal-based | Leverage Analyst intelligence |

## Feature Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     MARKET ANALYSIS                          │
│  (Regime Detection, SR Clustering, Feature Engineering)     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├──────────────────────┬─────────────────┐
                     ▼                      ▼                 ▼
         ┌───────────────────┐  ┌──────────────────┐  ┌─────────────┐
         │ Regime Features   │  │   15m Data       │  │  5m Data    │
         │ (Ensemble Model)  │  │   (Analyst)      │  │ (Tactician) │
         └─────────┬─────────┘  └────────┬─────────┘  └──────┬──────┘
                   │                     │                    │
                   ▼                     ▼                    │
         ┌────────────────────────────────────┐              │
         │  ANALYST PRE-ML ORCHESTRATION      │              │
         │  • Horizon Labeling (15m)          │              │
         │  • Feature Optimization            │              │
         │  • PID Generation                  │              │
         │  • Feature Selection               │              │
         └──────────────┬─────────────────────┘              │
                        │                                     │
                        ▼                                     │
         ┌────────────────────────────────┐                  │
         │  ANALYST MODELS TRAINING       │                  │
         │  • TCN, LightGBM, Ridge, etc.  │                  │
         │  • Per-regime training         │                  │
         │  • 15m NAS & TAS models        │                  │
         └──────────────┬─────────────────┘                  │
                        │                                     │
                        ▼                                     │
         ┌────────────────────────────────┐                  │
         │  ANALYST ENSEMBLE TRAINING     │                  │
         │  • Combine base models         │                  │
         │  • Generate predictions        │                  │
         │  • Output confidence scores    │                  │
         └──────────────┬─────────────────┘                  │
                        │                                     │
                        │ Analyst Predictions (>0.4%)        │
                        └────────────┬────────────────────────┘
                                     │
                                     ▼
                   ┌──────────────────────────────────────┐
                   │  TACTICIAN PRE-ML ORCHESTRATION      │
                   │  • Filter on Analyst signals         │
                   │  • Horizon Labeling (5m)             │
                   │  • Feature Optimization              │
                   │  • PID Generation                    │
                   │  • Feature Selection                 │
                   └──────────────┬───────────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────────────┐
                   │  TACTICIAN MODELS TRAINING           │
                   │  • RSF, XGBoost, ElasticNetCV        │
                   │  • Uses Analyst features             │
                   │  • 5m NAS & TAS models               │
                   └──────────────┬───────────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────────────┐
                   │  TACTICIAN ENSEMBLE TRAINING         │
                   │  • Combine base models               │
                   │  • Uses Analyst ensemble outputs     │
                   │  • Final trade timing decisions      │
                   └──────────────────────────────────────┘
```

## Execution Commands

### Execute Complete Analyst Pipeline
```bash
# Full pipeline (15m timeframe)
python src/launcher/ares_launcher.py --mode sub_pipeline \
  --sub_pipeline analyst_pre_ml_orchestration \
  --execution-mode full \
  --timeframe 15m \
  --symbol ETHUSDT
```

### Execute Complete Tactician Pipeline
```bash
# Full pipeline (5m timeframe)
python src/launcher/ares_launcher.py --mode sub_pipeline \
  --sub_pipeline tactician_pre_ml_orchestration \
  --execution-mode full \
  --timeframe 5m \
  --symbol ETHUSDT
```

### Execute Individual Steps

#### Analyst Steps
```bash
# Step 1: Pre-ML Orchestration
python src/launcher/ares_launcher.py --mode sub_pipeline \
  --sub_pipeline analyst_pre_ml_orchestration --execution-mode full

# Step 2: Models Training
python src/launcher/ares_launcher.py --mode sub_pipeline \
  --sub_pipeline analyst_models_training --execution-mode full

# Step 3: Ensemble Training
python src/launcher/ares_launcher.py --mode sub_pipeline \
  --sub_pipeline analyst_ensemble_training --execution-mode full
```

#### Tactician Steps
```bash
# Step 1: Pre-ML Orchestration (requires Analyst predictions)
python src/launcher/ares_launcher.py --mode sub_pipeline \
  --sub_pipeline tactician_pre_ml_orchestration --execution-mode full

# Step 2: Models Training
python src/launcher/ares_launcher.py --mode sub_pipeline \
  --sub_pipeline tactician_models_training --execution-mode full

# Step 3: Ensemble Training
python src/launcher/ares_launcher.py --mode sub_pipeline \
  --sub_pipeline tactician_ensemble_training --execution-mode full
```

## Validation Checklist

- [x] Both pipelines have 3 steps
- [x] Both use pre-ML orchestration with same 4 sub-steps
- [x] Both train base models with HPO
- [x] Both train ensemble models with HPO
- [x] Both support per-regime optimization
- [x] Both support per-cluster optimization
- [x] Both generate NAS and TAS models
- [x] Both train separate short/long models
- [x] Analyst uses 15m timeframe
- [x] Tactician uses 5m timeframe
- [x] Tactician filters on Analyst signals (>0.4%)
- [x] Tactician uses Analyst ensemble outputs as features
- [x] Proper hierarchical dependency (Analyst → Tactician)

## Conclusion

✅ **PARITY VERIFIED**: Both Analyst and Tactician pipelines follow the same structure with appropriate differentiation for their distinct roles.

The key differentiators are:
1. **Timeframe**: 15m (Analyst) vs 5m (Tactician)
2. **Training Data**: All data (Analyst) vs Filtered >0.4% confidence (Tactician)
3. **Feature Integration**: Tactician uses Analyst ensemble outputs
4. **Purpose**: IF to trade (Analyst) vs WHEN to trade (Tactician)

Both pipelines maintain complete parity in:
- Pipeline structure (3 steps)
- Feature engineering approach (4 sub-steps)
- Model training methodology
- Hyperparameter optimization
- Per-regime/cluster optimization
- Ensemble methodology
