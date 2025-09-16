# ML Models Architecture Summary

## Overview

This document provides a comprehensive overview of the ML models trained in the `hmm_models_training` and ensemble training systems, reflecting the updated 3-tier ensemble architecture.

## HMM Models Training (`/workspace/src/training/steps/market_analysis/hmm_models_training/`)

### **Stack Approach Implementation**
The HMM models training system now uses a **2-model stack approach**:

#### 1. **Logistic Regression**
- **Purpose**: Primary classification model for regime-based predictions
- **Configuration**: 
  - C=1.0, max_iter=1000, random_state=42
  - class_weight='balanced' for handling imbalanced data
- **Use Case**: Binary/multi-class classification for market regime detection
- **Role**: First layer in the stack

#### 2. **XGBoost (XGBClassifier)**
- **Purpose**: Secondary gradient boosting classifier for complex pattern recognition
- **Configuration**:
  - n_estimators=100, learning_rate=0.1
  - max_depth=6, random_state=42, verbosity=0
  - objective='multi:softprob', eval_metric='mlogloss'
- **Use Case**: High-performance classification with feature importance analysis
- **Role**: Second layer in the stack

### **Key Features:**
- **Circuit Breaker Pattern**: Prevents cascading failures
- **Real-time Progress Reporting**: Shows training progress with ETA
- **Enhanced Validation**: Multi-level validation (BASIC, STANDARD, STRICT)
- **Feature Selection**: MRMR and LASSO stability selection
- **Comprehensive Reporting**: Real metrics with actionable insights

## 3-Tier Ensemble Architecture

The system now implements a **3-tier ensemble architecture** with each tier operating on different timeframes:

### **Tier 1: HMM Ensemble Training** (1h timeframe)
**File**: `/workspace/src/training/steps/model_training/hmm_ensemble_training.py`

**Purpose**: Market regime detection and classification

**Models Trained**:
1. **Logistic Regression** - Primary classification model
2. **XGBoost** - Gradient boosting classifier
3. **Random Forest** - Ensemble classifier for robustness
4. **Voting Classifier** - Meta-learner combining all models

**Configuration**:
- **Timeframe**: 1h
- **HPO Trials**: 100
- **Min Samples per Regime**: 1000
- **Evaluation Metrics**: accuracy, f1_score, precision, recall, auc

### **Tier 2: Analyst Ensemble Training** (5m timeframe)
**File**: `/workspace/src/training/steps/model_training/analyst_ensemble_training.py`

**Purpose**: Trade decision analysis and enhancement

**Models Trained**:
1. **TCN (Temporal Convolutional Network)** - Primary: Fast sequential data processing
2. **CatBoost** - Primary: Financial data + regime handling
3. **LightGBM** - Primary: Speed + robustness
4. **Random Forest** - Meta: Speed + efficiency

**Configuration**:
- **Timeframe**: 5m
- **HPO Trials**: 100
- **Min Samples per Regime**: 1000
- **Evaluation Metrics**: mse, mae, r2, mape, smape

### **Tier 3: Tactician Ensemble Training** (1m timeframe)
**File**: `/workspace/src/training/steps/model_training/tactician_ensemble_training.py`

**Purpose**: Timing decisions and final execution

**Models Trained**:
1. **NODE (Neural Oblivious Decision Ensembles)** - Primary: Tabular data optimization
2. **CatBoost** - Primary: Regime handling
3. **LightGBM** - Primary: Speed + robustness
4. **Elastic Net** - Meta: Fast baseline

**Configuration**:
- **Timeframe**: 1m
- **HPO Trials**: 100
- **Min Samples per Regime**: 1000
- **Evaluation Metrics**: mse, mae, r2, mape, smape

## Model Integration Flow

The system creates a **hierarchical model architecture** with the following flow:

```
1. HMM Models (1h) → Market regime detection
   ↓
2. Analyst Models (5m) → Trade decision analysis
   ↓
3. Tactician Models (1m) → Timing decisions
   ↓
4. Ensemble Models → Meta-learning combining all previous models
```

### **Integration Details:**

#### **HMM Ensemble Integration:**
- Receives individual HMM model predictions
- Uses base model performance metrics for weighting
- Creates regime-specific ensemble combinations
- Provides enhanced market regime detection signals

#### **Analyst Ensemble Integration:**
- Receives individual analyst model predictions
- Uses base model performance metrics for weighting
- Creates regime-specific ensemble combinations
- Provides enhanced trade decision signals

#### **Tactician Ensemble Integration:**
- Receives individual tactician model predictions
- Integrates analyst model predictions
- Integrates analyst ensemble predictions
- Integrates HMM regime data and features
- Creates final meta-learner for optimal timing decisions
- Provides comprehensive market intelligence

## Training Configuration Summary

### **HMM Models Training:**
- **Timeframe**: 1h
- **Features**: 100 features (configurable)
- **Sequence Length**: 20
- **Regimes**: 3 (configurable)
- **HPO Trials**: 50-100
- **Model Types**: ["logistic_regression", "xgboost"]

### **HMM Ensemble Training:**
- **Timeframe**: 1h
- **HPO Trials**: 100
- **Min Samples per Regime**: 1000
- **Model Types**: ["logistic_regression", "xgboost", "random_forest", "voting_classifier"]
- **Evaluation Metrics**: accuracy, f1_score, precision, recall, auc

### **Analyst Ensemble Training:**
- **Timeframe**: 5m
- **HPO Trials**: 100
- **Min Samples per Regime**: 1000
- **Model Types**: ["tcn", "catboost", "lightgbm", "ensemble_rf"]
- **Evaluation Metrics**: mse, mae, r2, mape, smape

### **Tactician Ensemble Training:**
- **Timeframe**: 1m
- **HPO Trials**: 100
- **Min Samples per Regime**: 1000
- **Model Types**: ["node", "catboost", "lightgbm", "elastic_net"]
- **Evaluation Metrics**: mse, mae, r2, mape, smape

## Key Benefits of the 3-Tier Architecture

### **1. Temporal Hierarchy:**
- **1h timeframe**: Long-term market regime detection
- **5m timeframe**: Medium-term trade decision analysis
- **1m timeframe**: Short-term timing decisions

### **2. Model Specialization:**
- **HMM Models**: Specialized for regime detection
- **Analyst Models**: Specialized for trade decisions
- **Tactician Models**: Specialized for timing execution

### **3. Ensemble Benefits:**
- **Robustness**: Multiple models reduce overfitting
- **Performance**: Meta-learning improves accuracy
- **Adaptability**: Different models excel in different market conditions

### **4. Comprehensive Intelligence:**
- **Multi-timeframe analysis**: Captures both short and long-term patterns
- **Regime-aware modeling**: Adapts to different market conditions
- **Meta-learning**: Combines insights from all model tiers

## Usage Examples

### **HMM Ensemble Training:**
```python
from src.training.steps.model_training.hmm_ensemble_training import create_hmm_ensemble_training_step

config = EnsembleTrainingConfig(
    model_name="hmm_ensemble_models",
    timeframe="1h",
    model_types=["logistic_regression", "xgboost", "random_forest", "voting_classifier"]
)

training_step = create_hmm_ensemble_training_step(config)
results = training_step.execute(X, y, regime_labels, feature_names, hmm_states)
```

### **Analyst Ensemble Training:**
```python
from src.training.steps.model_training.analyst_ensemble_training import create_analyst_ensemble_training_step

config = EnsembleTrainingConfig(
    model_name="analyst_ensemble_models",
    timeframe="5m",
    model_types=["tcn", "catboost", "lightgbm", "ensemble_rf"]
)

training_step = create_analyst_ensemble_training_step(config)
results = training_step.execute(X, y, regime_labels, feature_names, hmm_states)
```

### **Tactician Ensemble Training:**
```python
from src.training.steps.model_training.tactician_ensemble_training import create_tactician_ensemble_training_step

config = EnsembleTrainingConfig(
    model_name="tactician_ensemble_models",
    timeframe="1m",
    model_types=["node", "catboost", "lightgbm", "elastic_net"]
)

training_step = create_tactician_ensemble_training_step(config)
results = training_step.execute(X, y, regime_labels, feature_names, hmm_states)
```

## Conclusion

The updated ML models architecture provides:

1. **Streamlined HMM Models**: 2-model stack approach with Logistic Regression and XGBoost
2. **3-Tier Ensemble System**: HMM (1h), Analyst (5m), and Tactician (1m) ensembles
3. **Specialized Models**: Each tier optimized for its specific timeframe and purpose
4. **Comprehensive Integration**: Meta-learning combines insights from all tiers
5. **Robust Performance**: Ensemble methods reduce overfitting and improve accuracy

This architecture creates a comprehensive ML pipeline that combines regime-aware modeling with multi-timeframe ensemble techniques for robust financial market prediction and trading decision support.