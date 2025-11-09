# Ensemble Training Setup - Complete

## ✅ Changes Made

### 1. Per-Model Predictions Storage

**Modified Files:**
- `src/training/steps/models_training/core/pipeline_orchestrator.py`
- `src/training/steps/model_training/unified_models_training_step.py`

**Changes:**
1. **Pipeline Orchestrator** (`_execute_analyst_base_training`):
   - Extract all trained models from metadata (not just best model)
   - Generate predictions for each model individually
   - Compute confidence scores (absolute values of predictions)
   - Return predictions and confidence in result dict

2. **Unified Models Training Step** (`_save_training_artifacts`):
   - Save `analyst_base_predictions` (per-model predictions)
   - Save `analyst_base_confidence` (per-model confidence scores)
   - Both saved to versioned HDF5 storage with `data_category='predictions'`

### 2. Data Flow for Ensemble Training

```
analyst_base_training:
├── Train 3 models (LightGBM, DEPTHWISE_CNN, CatBoost)
├── Generate predictions for each model
├── Compute confidence scores
└── Save to HDF5:
    ├── analyst_base_predictions (DataFrame with columns: lightgbm, depthwise_cnn, catboost)
    └── analyst_base_confidence (DataFrame with absolute values)

analyst_ensemble_training:
├── Load from HDF5:
│   ├── labeled_data (from feature_generation_labeling_integration_step)
│   ├── regime_probabilities (from regime_ensemble_training)
│   ├── analyst_base_predictions (base model outputs)
│   └── analyst_base_confidence (base model confidence)
├── Generate disagreement features
└── Train ensemble model
```

### 3. Artifact Names

**Saved by analyst_base:**
- `analyst_base_predictions` - Per-model predictions
- `analyst_base_confidence` - Per-model confidence scores

**Expected by analyst_ensemble:**
- `analyst_base_predictions` - Required
- `analyst_base_confidence` - Optional
- `labeled_data` - From labeling step
- `regime_probabilities` - From regime training

### 4. Prediction Format

**analyst_base_predictions DataFrame:**
```
Index: DatetimeIndex (same as training data)
Columns:
  - lightgbm: float64 (predictions from LightGBM)
  - depthwise_cnn: float64 (predictions from DEPTHWISE_CNN)
  - catboost: float64 (predictions from CatBoost)
```

**analyst_base_confidence DataFrame:**
```
Index: DatetimeIndex (same as training data)
Columns:
  - lightgbm: float64 (|predictions| from LightGBM)
  - depthwise_cnn: float64 (|predictions| from DEPTHWISE_CNN)
  - catboost: float64 (|predictions| from CatBoost)
```

---

## 🔧 Implementation Details

### Code Changes

#### 1. Pipeline Orchestrator - Generate Per-Model Predictions

```python
# Get all trained models for per-model predictions
models_to_predict = {}
if hasattr(result, 'metadata') and 'trained_models' in result.metadata:
    models_to_predict = result.metadata['trained_models']
elif hasattr(result, 'metadata') and 'model_instances' in result.metadata:
    models_to_predict = result.metadata['model_instances']
else:
    # Fallback to single best model
    models_to_predict = {'best_model': result.model}

predictions = await self._generate_predictions(models_to_predict, data_for_prediction)

# Store predictions for ensemble training
if predictions is not None:
    tprint_info(f"📊 Generated predictions from {len(models_to_predict)} models: {predictions.shape}")
    # Also compute confidence scores
    confidence = predictions.abs()
    tprint_info(f"📊 Computed confidence scores: {confidence.shape}")
```

#### 2. Unified Models Training Step - Save Predictions

```python
# Save predictions for ensemble training (analyst_base only)
if training_type == 'analyst_base' and 'predictions' in result and result['predictions'] is not None:
    try:
        predictions_path = self._save_artifact(
            data=result['predictions'],
            artifact_name='analyst_base_predictions',
            artifact_type='data',
            data_category='predictions'
        )
        artifacts['analyst_base_predictions'] = predictions_path
        tprint_success(f"✅ Saved analyst_base_predictions: {result['predictions'].shape}")
        
        # Save confidence scores
        if 'confidence' in result and result['confidence'] is not None:
            confidence_path = self._save_artifact(
                data=result['confidence'],
                artifact_name='analyst_base_confidence',
                artifact_type='data',
                data_category='predictions'
            )
            artifacts['analyst_base_confidence'] = confidence_path
            tprint_success(f"✅ Saved analyst_base_confidence: {result['confidence'].shape}")
    except Exception as e:
        tprint_warning(f"⚠️ Failed to save predictions/confidence: {e}")
```

---

## 📋 Usage

### Step 1: Train Base Models
```bash
python3 src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT --execution-mode light
```

**Output:**
- ✅ 3 models trained
- ✅ Per-model predictions saved to HDF5
- ✅ Confidence scores saved to HDF5

### Step 2: Train Ensemble Model
```bash
python3 src/launcher/ares_launcher.py --train-analyst-ensemble --symbol ETHUSDT --execution-mode light
```

**Expected:**
- ✅ Loads analyst_base_predictions
- ✅ Loads analyst_base_confidence
- ✅ Loads labeled_data
- ✅ Loads regime_probabilities
- ✅ Generates disagreement features
- ✅ Trains ensemble model

---

## 🎯 Verification

### Check Saved Predictions
```python
from src.utils.artifact_manager import ArtifactManager

# Initialize artifact manager
am = ArtifactManager()
am.set_context(
    step_name='analyst_ensemble_training',
    symbol='ETHUSDT',
    exchange='binance',
    timeframe='15m',
    direction='long',
    model='analyst'
)

# Load predictions
predictions = am.get_artifact('analyst_base_predictions', artifact_type='data')
confidence = am.get_artifact('analyst_base_confidence', artifact_type='data')

print(f"Predictions shape: {predictions.shape}")
print(f"Predictions columns: {list(predictions.columns)}")
print(f"Confidence shape: {confidence.shape}")
print(f"Confidence columns: {list(confidence.columns)}")
```

---

## ✅ Status

- ✅ **Per-model predictions** - Implemented and saving
- ✅ **Confidence scores** - Implemented and saving
- ✅ **Artifact names** - Match ensemble expectations
- ⏳ **Testing** - Running analyst_base training now
- ⏳ **Ensemble training** - Will test after base completes

---

## 📝 Notes

1. **Confidence Calculation**: Using absolute values of predictions as confidence scores. Higher absolute values = more confident predictions.

2. **Model Selection**: All trained models are included in predictions (not just the best model).

3. **Data Alignment**: Predictions use the same index as training data for proper alignment.

4. **Storage Format**: HDF5 versioned artifacts with `data_category='predictions'` for easy retrieval.

5. **Backward Compatibility**: Existing code still works - predictions are added to the result dict without breaking existing functionality.
