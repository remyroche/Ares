# train_tactician_ensemble: Complete Implementation Analysis

## Overview
`train_tactician_ensemble` is a model training step that trains ensemble tactician models. It's part of a 4-phase sequential training pipeline that builds increasingly complex models by chaining artifacts from previous phases.

---

## 1. Implementation Location

### Primary Implementation Files

**Main Entry Point:**
- `/home/user/Ares/src/training/steps/model_training/tactician_ensemble_training_step.py` (93 lines)
  - Thin wrapper class: `TacticianEnsembleTrainingStep`
  - Delegates to `UnifiedModelsTrainingStep`

**Actual Implementation:**
- `/home/user/Ares/src/training/steps/model_training/unified_models_training_step.py` (31,322 lines)
  - Core training logic
  - Data loading and validation
  - Temporal splitting and cross-validation
  - HPO (Hyperparameter Optimization) orchestration
  - Model training and artifact management

**Related Files:**
- `/home/user/Ares/src/training/steps/model_training/__init__.py`
  - Step registration in the step registry
- `/home/user/Ares/src/launcher/ares_launcher.py` (line 601-603)
  - CLI interface for executing training
  - Maps `--train-tactician-ensemble` flag to the step

**Configuration:**
- `/home/user/Ares/src/training/steps/model_training/hpo_config.py`
  - HPO orchestrator and model parameter groups
- `/home/user/Ares/src/training/steps/model_training/dynamic_config_calculator.py`
  - Dynamic configuration based on execution mode

---

## 2. Current Structure

### Wrapper Pattern: TacticianEnsembleTrainingStep

```python
class TacticianEnsembleTrainingStep(BaseStep):
    """Trains ensemble tactician models using outputs from base models."""
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Set training type and execution context
        config['training_type'] = 'tactician_ensemble'
        config['execution_context'] = 'tactician'
        
        # Delegate to UnifiedModelsTrainingStep
        unified_step = UnifiedModelsTrainingStep()
        result = await unified_step.execute(config)
        return result
```

**Key Properties:**
- Minimal wrapper (only 93 lines)
- Sets `training_type = 'tactician_ensemble'` and `execution_context = 'tactician'`
- Defers all actual logic to `UnifiedModelsTrainingStep`
- Returns standard result format with `success`, `artifacts`, `metrics`, `error`

### UnifiedModelsTrainingStep: The Actual Implementation

**Main Execute Method Flow:**
1. **Configuration Loading** → Load YAML training config
2. **Data Retrieval** → Fetch training data and targets from artifacts
3. **Temporal Splitting** → Enforce train/val/test boundaries (walk-forward)
4. **Feature Engineering** → Cleanup, validation, feature selection
5. **Model Training** → Using appropriate HPO configuration
6. **Artifact Management** → Save models and predictions
7. **Result Compilation** → Return metrics and artifacts

**Data Flow for Tactician Ensemble:**

```
Input (from config):
├── symbol: Trading symbol (e.g., 'ETHUSDT')
├── exchange: Exchange name (e.g., 'binance')
├── timeframe: Timeframe (e.g., '15m')
├── direction: Trading direction ('long', 'short', 'both')
└── training_type: 'tactician_ensemble'

Processing:
├── Load YAML config for tactician_ensemble training
├── Load training data from feature_generation artifacts
├── Load tactician targets from labeling artifacts
├── Create walk-forward temporal split (3 folds + test)
├── Train ensemble models using HPO
└── Save artifacts (models, predictions, metrics)

Output:
├── success: bool
├── artifacts:
│   ├── tactician_ensemble_model: Trained ensemble model
│   ├── tactician_predictions: Predictions on training data
│   └── ... (model files, metrics)
└── metrics: Performance metrics (Sharpe, accuracy, etc.)
```

---

## 3. Relationships to Other Training Steps

### Sequential Pipeline Architecture

The system implements a **4-Phase Sequential Learning Pipeline** with **Artifact Chaining**:

```
Phase 1: Analyst Base Models
  Input:  Raw training data + Analyst targets
  Output: Base models + Predictions
  ↓
Phase 2: Analyst Ensemble ← Uses Phase 1 outputs
  Input:  Raw training data + Analyst targets + Analyst base predictions
  Output: Ensemble model + Enhanced predictions
  ↓
Phase 3: Tactician Base Models ← Uses Phase 2 outputs
  Input:  Raw training data + Tactician targets + Analyst ensemble predictions
  Output: Base models + Predictions
  ↓
Phase 4: Tactician Ensemble ← Uses Phase 3 outputs
  Input:  Raw training data + Tactician targets + Tactician base predictions
  Output: Ensemble model + Final predictions
```

### File Structure by Phase

**Phase 1 - Analyst Base:**
- `/home/user/Ares/src/training/steps/model_training/analyst_base_training_step.py`
- Delegates to `UnifiedModelsTrainingStep` with `training_type='analyst_base'`

**Phase 2 - Analyst Ensemble:**
- `/home/user/Ares/src/training/steps/model_training/analyst_ensemble_training_step.py` (93 lines)
- Delegates to `UnifiedModelsTrainingStep` with `training_type='analyst_ensemble'`

**Phase 3 - Tactician Base:**
- `/home/user/Ares/src/training/steps/model_training/tactician_base_training_step.py` (483 lines)
- More complex: Includes centralized configuration management
- Delegates to `UnifiedModelsTrainingStep` with `training_type='tactician_base'`

**Phase 4 - Tactician Ensemble:** (Current Subject)
- `/home/user/Ares/src/training/steps/model_training/tactician_ensemble_training_step.py` (93 lines)
- Same pattern as Analyst Ensemble
- Delegates to `UnifiedModelsTrainingStep` with `training_type='tactician_ensemble'`

### Launch Sequence (ares_launcher.py)

```bash
# CLI interface
python ares_launcher.py --stage MODEL_TRAINING --symbol ETHUSDT

# Or individual steps
python ares_launcher.py --train-analyst-base --symbol ETHUSDT
python ares_launcher.py --train-analyst-ensemble --symbol ETHUSDT
python ares_launcher.py --train-tactician-base --symbol ETHUSDT
python ares_launcher.py --train-tactician-ensemble --symbol ETHUSDT
```

**Launcher Registration (line 198-203):**
```python
'MODEL_TRAINING': [
    'analyst_base_training',
    'analyst_ensemble_training',
    'tactician_base_training',
    'tactician_ensemble_training'
]
```

### Relationship to Regime Ensemble Training

**Separate Pipeline (Market Analysis, NOT Model Training):**
- `/home/user/Ares/src/training/steps/market_analysis/regime_ensemble_training_step.py`
- Different purpose: Classifies market regimes (not trading decisions)
- Uses regime probabilities as input (not predictions from models)
- Runs in PRE_TRAINING phase, before MODEL_TRAINING
- Pipeline flow:
  ```
  MARKET_ANALYSIS phase:
  ├── regime_models_training (ML models for regime classification)
  └── regime_ensemble_training (meta-learning ensemble for regimes)
  
  MODEL_TRAINING phase: (Uses regime outputs as features)
  ├── analyst_base_training
  ├── analyst_ensemble_training
  ├── tactician_base_training
  └── tactician_ensemble_training
  ```

---

## 4. Data Loading Mechanisms

### Source: Feature Generation Artifacts

**Primary Data Source:**
The training data comes from the **Feature Generation Pipeline** final step:
- **Artifact Step:** `feature_generation_final_feature_selection_step`
- **Artifact Names (fallback priority):**
  1. `selected_feature_dataframe_{size}` (e.g., `selected_feature_dataframe_50`)
  2. `selected_features_{size}`
  3. `final_dataset_{size}` (from validation step)
  4. `final_analyst_dataset_{size}` (analyst-specific)
  5. Fallback to sizes: 60 → 50 → 40 features

**Search Strategy (unified_models_training_step.py, line 1582-1594):**
```python
feature_artifact_names = [
    f'selected_feature_dataframe_{feature_set_size}',  # Specific size
    f'selected_features_{feature_set_size}',           # Alternative name
    f'final_dataset_{feature_set_size}',               # Validation step generic
    f'final_analyst_dataset_{feature_set_size}',       # Analyst-specific alias
    'selected_feature_dataframe_60',                   # Fallback to 60 (try largest first)
    'selected_feature_dataframe_50',                   # Fallback to 50
    'selected_feature_dataframe_40',                   # Fallback to 40
    'final_dataset_60',                                # Validation step 60
    'final_dataset_50',                                # Validation step 50
    'final_dataset_40',                                # Validation step 40
]
```

### Target Data Loading

**Analyst Targets** (line 1764-1780):
- Direction-specific first:
  1. `analyst_targets_{direction}` (e.g., `analyst_targets_long`)
  2. `{direction}_analyst_targets` (e.g., `long_analyst_targets`)
  3. `{direction}_targets` (generic direction-specific)
  4. `analyst_targets` (generic)
  5. `targets` (fallback)

**Tactician Targets** (line 1782-1799):
- Same pattern as analyst targets:
  1. `tactician_targets_{direction}`
  2. `{direction}_tactician_targets`
  3. `{direction}_targets`
  4. `tactician_targets`
  5. `targets`

**Fallback: Extract from labeled_data** (line 1801-1850):
If separate target artifacts not found:
- Try to load `labeled_data` or `labeled_features`
- Extract target columns (direction-aware)
- Support new simplified structure: `target_long` and `target_short`
- Fallback to legacy target detection with direction-specific columns

### Data Cleaning Pipeline (line 1640-1757)

**Comprehensive Validation Steps:**

1. **Duplicate Column Removal**
   - Identify and remove duplicate columns
   
2. **Insufficient Data Filtering**
   - Drop columns with <1% valid data
   - Column-by-column threshold: `valid_ratio < 0.01`

3. **Non-Numeric Column Removal**
   - Remove categorical/string columns (break model training)
   - Exception: boolean columns converted to float

4. **Zero-Variance Column Detection**
   - Identify constant columns
   - Marked for removal during training

5. **Target Column Filtering**
   - Remove columns named: `target`, `label`, `*_target`, `*_label`
   - Prevents data leakage

6. **Metadata Column Removal**
   - Patterns: `labeling_method`, `labeling_timestamp`, `base_threshold`
   - `lookahead_periods`, `optimization_iteration`, `quality_acceptance_rate`

7. **Verification & Logging**
   - Memory usage calculation
   - Data type distribution
   - Missing value detection
   - Shape before/after cleaning

### Temporal Data Splitting

**Walk-Forward Cross-Validation** (line 131-194):

```python
walkforward_config = create_walkforward_split_config_for_pipeline(
    n_folds=3,              # 3 train/val pairs
    val_pct_per_fold=0.10,  # 10% validation per fold
    final_test_pct=0.15,    # 15% for final test
    min_train_pct=0.55,     # Start with 55% training
    embargo_days=1          # 1-day embargo (no look-ahead bias)
)
```

**Fold Strategy:**
```
Fold 1: Train [55%] | Validate [10%] | Embargo [1d]
Fold 2: Train [65%] | Validate [10%] | Embargo [1d]  (expanding)
Fold 3: Train [75%] | Validate [10%] | Embargo [1d]  (expanding)
Test:   [15%] (held out from training)
```

**Purpose:**
- Prevents data leakage
- Ensures proper temporal ordering
- Expanding window = more training data in later folds
- Test set never used during training/validation

### YAML Configuration Loading

**YAML Config Path** (resolved dynamically):
- Searches for: `tactician_ensemble_training.yaml`
- Includes training parameters, HPO settings, model configurations
- Falls back to inline defaults if file not found

### Artifact Management

**Get Artifact Method** (`_get_artifact`):
- Uses BaseStep's artifact retrieval system
- Supports multiple storage backends (HDF5, JSON, pickle)
- Automatic cache on first access
- Handles missing artifacts gracefully with logging

---

## Configuration Example

**Typical Execution Config:**
```python
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'direction': 'long',
    'training_type': 'tactician_ensemble',      # Set by wrapper
    'execution_context': 'tactician',           # Set by wrapper
    'execution_mode': 'light',                  # Options: full, light, blank
    'feature_set_size': 50,                     # Default feature count
}
```

---

## Execution Flow Summary

```
ares_launcher.py
  └─> TacticianEnsembleTrainingStep.execute()
       └─> UnifiedModelsTrainingStep.execute()
            ├─> Load YAML: tactician_ensemble_training.yaml
            ├─> _retrieve_training_data()
            │   ├─> Load features from feature_generation artifacts
            │   ├─> Load tactician targets from labeling artifacts
            │   └─> Clean and validate data
            ├─> Create walk-forward temporal split
            ├─> HPO training with appropriate config
            │   ├─> For 'tactician_ensemble': LightGBM, CatBoost, Neural Network ensemble
            │   └─> Use full HPO with early stopping
            ├─> Generate predictions
            ├─> Save artifacts:
            │   ├─> tactician_ensemble_model
            │   ├─> tactician_predictions
            │   └─> metrics.json
            └─> Return result with success/artifacts/metrics
```

---

## Key Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| `tactician_ensemble_training_step.py` | 93 | Thin wrapper, delegates to unified |
| `unified_models_training_step.py` | 31,322 | Core implementation, all training logic |
| `hpo_config.py` | - | HPO orchestrator and model configurations |
| `dynamic_config_calculator.py` | - | Dynamic config based on execution mode |
| `ares_launcher.py` | 685 | CLI interface and step orchestration |
| ARTIFACT_CHAINING_GUIDE.md | 361 | Documentation of sequential learning |

---

## Current Status

- ✅ **Implemented:** Full working implementation with artifact chaining
- ✅ **Registered:** In step registry as `tactician_ensemble_training`
- ✅ **CLI Support:** `--train-tactician-ensemble` flag
- ✅ **Data Loading:** Multi-source with comprehensive fallbacks
- ✅ **Temporal Splitting:** Walk-forward validation implemented
- ✅ **HPO:** Full hyperparameter optimization
- ✅ **Documentation:** ARTIFACT_CHAINING_GUIDE.md provides detailed explanations

