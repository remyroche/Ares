# Analyst Base Training - Implementation Documentation

## Overview

The **Analyst Base Training** component is responsible for training base analyst models that decide IF we should trade by analyzing market conditions. This document describes the complete implementation including data loading, model training, metrics tracking, and persistence.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Analyst Base Training                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ├── Data Loading (HDF5)
                            │   ├── Features (60)
                            │   ├── Labels/Targets
                            │   └── Regime Probabilities
                            │
                            ├── Model Training
                            │   ├── LightGBM
                            │   ├── LightGBM + PatchTST
                            │   ├── CatBoost
                            │   └── Stacker LGBM Calibrated
                            │
                            ├── Metrics Generation
                            │   ├── Markdown Report
                            │   └── JSON Metrics
                            │
                            └── Model Persistence (Pickle)
```

## Data Loading from HDF5 Versioned Artifacts

### 1. Features (60-Feature Set)

**Source:** `feature_generation_final_feature_selection_step`

**Storage Format:** HDF5 (via `src/utils/versioned_artifacts/`)

**Artifact Names** (in priority order):
- `selected_feature_dataframe_60` (primary)
- `selected_features_60` (alternative)
- `final_dataset_60` (validation step alias)
- `final_analyst_dataset_60` (analyst-specific alias)

**Implementation:**
```python
# File: src/training/steps/model_training/unified_models_training_step.py
# Lines: 1605-1650

# Default feature set size is 60
feature_set_size = config.get('feature_set_size', 60)

# Load from HDF5 via versioned artifacts
training_data = self._get_artifact('selected_feature_dataframe_60', 'data')
```

**Features Include:**
- Technical indicators
- Price action features
- Volume features
- Market microstructure features
- Regime-based features (added separately)

### 2. Labels/Targets

**Source:** `feature_generation_labeling_integration_step`

**Storage Format:** HDF5 (via `src/utils/versioned_artifacts/`)

**Artifact Names** (direction-aware):
- `analyst_targets_{direction}` (e.g., `analyst_targets_long`)
- `{direction}_analyst_targets`
- `labeled_data` (fallback with extraction)

**Implementation:**
```python
# File: src/training/steps/model_training/unified_models_training_step.py
# Lines: 1814-1900

direction = config.get('direction', 'long')
analyst_targets = self._get_artifact(f'analyst_targets_{direction}', 'data')
```

**Target Structure:**
- Direction-specific binary targets
- Volume-based confidence adjustments
- Simplified target structure (`target_long`, `target_short`)

### 3. Regime Probabilities

**Source:** `regime_ensemble_training`

**Storage Format:** HDF5 (via `src/utils/versioned_artifacts/`)

**Artifact Name:** `regime_ensemble_predictions`

**Implementation:**
```python
# File: src/training/steps/model_training/unified_models_training_step.py
# Lines: 2106-2150

regime_features = self._get_artifact('regime_ensemble_predictions', 'data')
```

**Regime Features Include:**
- Probability for each market regime
- Ensemble predictions from multiple regime models
- Resampled to match 15m timeframe

## Model Training

### Models Trained

1. **LightGBM Base Model**
   - Fast gradient boosting
   - High performance on tabular data
   - Configurable via HPO

2. **LightGBM + PatchTST Features**
   - Enhanced with time-series features
   - PatchTST embeddings for temporal patterns
   - Better capture of market dynamics

3. **CatBoost Model**
   - Handles categorical features well
   - Robust to overfitting
   - Alternative gradient boosting approach

4. **Stacker LGBM Calibrated (Meta-Learner)**
   - Combines predictions from base models
   - Calibrated probabilities
   - Ensemble learning for improved accuracy

### Training Configuration

**Timeframe:** 15m (analyst operates on 15-minute bars)

**Validation Strategy:** Walk-forward expanding window
- 3 train/validation folds
- 10% validation per fold
- 15% final test set
- 1-day embargo between periods

**HPO (Hyperparameter Optimization):**
- Optuna-based optimization
- Model-specific parameter groups
- Dynamic configuration based on data size

### Entry Points

#### Primary Entry Point
```python
# File: src/training/steps/model_training/analyst_base_training_step.py

from src.training.steps.model_training.analyst_base_training_step import AnalystBaseTrainingStep

step = AnalystBaseTrainingStep()
result = await step.execute({
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'direction': 'long',
    'feature_set_size': 60  # Optional, defaults to 60
})
```

#### Unified Training Pipeline
```python
# File: src/training/steps/model_training/unified_models_training_step.py

from src.training.steps.model_training.unified_models_training_step import UnifiedModelsTrainingStep

step = UnifiedModelsTrainingStep()
result = await step.execute({
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'direction': 'long',
    'training_type': 'analyst_base',
    'execution_context': 'analyst'
})
```

## Metrics and Reporting

### Markdown Report

**Format:** Comprehensive markdown (.md) file

**Location:** `outcomes/analyst_base_{symbol}_{timeframe}_{direction}_report_{timestamp}.md`

**Generator:** `ModelTrainingReportGenerator`

**File:** `src/training/steps/model_training/model_training_report_generator.py`

**Sections:**
1. **Executive Summary**
   - Training metadata
   - Symbol, timeframe, direction
   - Execution time

2. **Models Trained**
   - Model types and algorithms
   - Training samples
   - Hyperparameters

3. **Performance Metrics**
   - Overall accuracy, precision, recall, F1 score
   - R² score, MSE, MAE
   - Per-model metrics

4. **HPO Results**
   - Best parameters found
   - Optimization metrics
   - Trial history

5. **Regime-Based Performance**
   - Performance by market regime
   - Regime-specific metrics

6. **Feature Information**
   - Feature count (60)
   - Feature names
   - Feature sources

7. **Data Sources**
   - HDF5 artifact references
   - Pipeline step sources

8. **Model Persistence**
   - Pickle format details
   - Storage locations
   - Model file names

### JSON Metrics Report

**Format:** Structured JSON file

**Location:** `outcomes/analyst_base_{symbol}_{timeframe}_{direction}_metrics_{timestamp}.json`

**Structure:**
```json
{
  "metadata": {
    "generated_at": "2025-11-08T...",
    "training_type": "analyst_base",
    "symbol": "ETHUSDT",
    "timeframe": "15m",
    "direction": "long",
    "execution_time_seconds": 123.45
  },
  "models_trained": {
    "count": 4,
    "model_names": ["lightgbm", "lightgbm_patchtst", "catboost", "stacker_lgbm_calibrated"],
    "details": { ... }
  },
  "metrics": {
    "overall_accuracy": 0.8523,
    "overall_precision": 0.8214,
    "overall_recall": 0.8876,
    ...
  },
  "hpo_results": { ... },
  "regime_performance": { ... },
  "feature_info": {
    "feature_count": 60,
    "feature_source": "feature_generation_final_feature_selection_step",
    ...
  },
  "data_sources": {
    "features": "feature_generation_final_feature_selection_step",
    "labels": "feature_generation_labeling_integration_step",
    "regime_probabilities": "regime_ensemble_training"
  },
  "persistence": {
    "format": "pickle",
    "location": "/artifacts",
    "files": ["analyst_base_lightgbm.pkl", ...]
  }
}
```

## Model Persistence

### Format

**Serialization:** Pickle (.pkl)

**Location:** `/artifacts` directory

**File Naming:** `{training_type}_{model_name}.pkl`

**Examples:**
- `analyst_base_lightgbm.pkl`
- `analyst_base_lightgbm_patchtst.pkl`
- `analyst_base_catboost.pkl`
- `analyst_base_stacker_lgbm_calibrated.pkl`

### Implementation

**Saving:**
```python
# File: src/utils/artifact_router.py
# Lines: 395-414

def _save_pickle(self, data: Any, artifact_name: str, metadata: Optional[Dict] = None) -> str:
    filepath = self.base_dir / f"{artifact_name}.pkl"

    if metadata:
        save_data = {
            'data': data,
            'metadata': metadata,
            'saved_at': datetime.now().isoformat()
        }
    else:
        save_data = data

    success = save_pickle(save_data, str(filepath))
    return str(filepath)
```

**Loading:**
```python
# Load a trained model
import pickle

with open('/artifacts/analyst_base_lightgbm.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['data']  # If saved with metadata
# or
model = model_data  # If saved without metadata wrapper
```

### Metadata Included

Models are saved with metadata containing:
- `training_type`: 'analyst_base'
- `symbol`: Trading symbol
- `timeframe`: Timeframe
- `direction`: Trading direction
- `created_at`: Timestamp

## Code Files

### Main Implementation Files

1. **analyst_base_training_step.py**
   - Entry point for analyst base training
   - Delegates to unified training step
   - Lines: 93

2. **unified_models_training_step.py**
   - Unified training orchestrator
   - Handles all data loading from HDF5
   - Coordinates model training
   - Generates reports
   - Lines: 2317

3. **model_training_report_generator.py** ✨ NEW
   - Generates markdown and JSON reports
   - Comprehensive metrics tracking
   - Lines: 450+

4. **analyst_models_training_modular.py**
   - Modular component implementation
   - Individual model training logic
   - Lines: 650

### Utility Files

1. **artifact_router.py**
   - Routes artifacts to appropriate storage
   - Handles pickle serialization
   - Location: `src/utils/artifact_router.py`

2. **versioned_artifacts/store.py**
   - HDF5 data storage and retrieval
   - Versioned artifact management
   - Location: `src/utils/versioned_artifacts/store.py`

## Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│  1. Feature Generation & Labeling Integration               │
│     - feature_generation_final_feature_selection_step        │
│     - feature_generation_labeling_integration_step           │
│     → Outputs: HDF5 artifacts (60 features + labels)         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  2. Regime Ensemble Training                                 │
│     - regime_ensemble_training                               │
│     → Outputs: HDF5 artifact (regime probabilities)          │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  3. Analyst Base Training                                    │
│     - analyst_base_training_step                             │
│     → Loads: All HDF5 artifacts                              │
│     → Trains: 4 base models                                  │
│     → Outputs:                                               │
│       - Pickle models (.pkl)                                 │
│       - Markdown report (.md)                                │
│       - JSON metrics (.json)                                 │
└──────────────────────────────────────────────────────────────┘
```

## Verification Checklist

### ✅ Data Loading (HDF5 Versioned Artifacts)

- [x] Features loaded from `feature_generation_final_feature_selection_step`
- [x] 60-feature set is default and explicitly configured
- [x] Labels/targets loaded from `feature_generation_labeling_integration_step`
- [x] Direction-aware target selection
- [x] Regime probabilities loaded from `regime_ensemble_training`
- [x] HDF5 format via versioned artifacts confirmed
- [x] Explicit logging for all data sources

### ✅ Model Training

- [x] LightGBM model training
- [x] LightGBM + PatchTST features model training
- [x] CatBoost model training
- [x] Stacker LGBM Calibrated (meta-learner) training
- [x] HPO integration
- [x] Walk-forward validation

### ✅ Metrics Reporting

- [x] Markdown report generation
- [x] Per-model metrics included
- [x] HPO results included
- [x] Accuracy, Precision, Recall, F1 Score
- [x] R² Score, MSE, MAE
- [x] Regime-based performance
- [x] JSON metrics file generation
- [x] Structured metrics data

### ✅ Model Persistence

- [x] Pickle format (.pkl) confirmed
- [x] Models saved to `/artifacts` directory
- [x] Metadata included with models
- [x] Proper file naming convention
- [x] Loadable and reusable

## Usage Example

```python
import asyncio
from src.training.steps.model_training.analyst_base_training_step import AnalystBaseTrainingStep

async def train_analyst_base():
    """Train analyst base models."""

    # Initialize step
    step = AnalystBaseTrainingStep()

    # Configure training
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'long',
        'feature_set_size': 60,  # Explicit 60-feature set
        'use_hpo': True,
        'hpo_trials': 50
    }

    # Execute training
    result = await step.execute(config)

    # Check results
    if result['success']:
        print(f"✅ Training successful!")
        print(f"Models: {result['artifacts'].keys()}")
        print(f"Metrics: {result['metrics']}")
        print(f"Markdown report: {result['artifacts']['training_report_markdown']}")
        print(f"JSON metrics: {result['artifacts']['training_report_json']}")
    else:
        print(f"❌ Training failed: {result.get('error')}")

# Run training
asyncio.run(train_analyst_base())
```

## Expected Outputs

After successful training, you will find:

1. **Pickle Models** (in `/artifacts`):
   - `analyst_base_lightgbm.pkl`
   - `analyst_base_lightgbm_patchtst.pkl`
   - `analyst_base_catboost.pkl`
   - `analyst_base_stacker_lgbm_calibrated.pkl`

2. **Markdown Report** (in `outcomes/`):
   - `analyst_base_ETHUSDT_15m_long_report_20251108_120000.md`

3. **JSON Metrics** (in `outcomes/`):
   - `analyst_base_ETHUSDT_15m_long_metrics_20251108_120000.json`

## Troubleshooting

### Missing HDF5 Artifacts

**Error:** `CRITICAL: No training data found in artifacts!`

**Solution:**
1. Ensure `feature_generation_final_feature_selection_step` has run
2. Check for `selected_feature_dataframe_60` artifact
3. Verify HDF5 files in `src/utils/versioned_artifacts/`

### Missing Regime Probabilities

**Error:** `CRITICAL: regime_ensemble_predictions artifact not found!`

**Solution:**
1. Run `regime_ensemble_training` step first
2. Verify `regime_ensemble_predictions` artifact exists
3. Check HDF5 versioned artifacts store

### Model Persistence Issues

**Error:** Failed to save model to pickle

**Solution:**
1. Check `/artifacts` directory exists and is writable
2. Verify `src/utils/artifact_router.py` is functioning
3. Check disk space

## References

- **Main Implementation:** `src/training/steps/model_training/unified_models_training_step.py`
- **Report Generator:** `src/training/steps/model_training/model_training_report_generator.py`
- **Artifact Router:** `src/utils/artifact_router.py`
- **Versioned Artifacts:** `src/utils/versioned_artifacts/store.py`

---

**Last Updated:** 2025-11-08
**Version:** 1.0
**Author:** Claude Code Implementation
