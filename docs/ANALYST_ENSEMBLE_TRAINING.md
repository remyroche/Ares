# Analyst Ensemble Training

## Overview

The Analyst Ensemble Training step is a comprehensive machine learning pipeline that trains ensemble models for the Analyst component of the Ares trading system. It combines multiple data sources and base model predictions to create a powerful meta-learning ensemble.

## Architecture

### Data Sources

The Analyst Ensemble Training loads and combines data from four primary HDF5 sources:

1. **Feature Generation & Labeling Integration**
   - **Source**: `feature_generation_labeling_integration_step`
   - **Location**: `src/utils/versioned_artifacts/` (HDF5)
   - **Content**: Engineered features and target labels
   - **Context**: `analyst` model, `15m` timeframe

2. **Regime Ensemble Probabilities**
   - **Source**: `regime_ensemble_training`
   - **Location**: `src/utils/versioned_artifacts/` (HDF5)
   - **Content**: Regime classification probabilities from ensemble models
   - **Context**: `regime` model, `1h` timeframe
   - **Note**: Optional - system continues without regime features if unavailable

3. **Analyst Base Model Outputs**
   - **Source**: `analyst_base_training`
   - **Location**: `src/utils/versioned_artifacts/` (HDF5)
   - **Content**: Predictions and confidence scores from base analyst models
   - **Context**: `analyst` model, `15m` timeframe
   - **Required**: Yes - training cannot proceed without base model outputs

4. **Disagreement Features**
   - **Generated**: Computed from base model predictions
   - **Content**: Variance, standard deviation, range, MAD, and CV metrics
   - **Purpose**: Captures model agreement/disagreement patterns

## Features

### Disagreement Metrics

The following disagreement features are computed from base model predictions:

- **Variance**: Variance across model predictions
- **Standard Deviation**: Standard deviation of predictions
- **Range**: Maximum - minimum prediction values
- **Mean Absolute Deviation (MAD)**: Average absolute deviation from mean
- **Coefficient of Variation (CV)**: Normalized measure of dispersion

### Feature Combination

All features are combined with intelligent alignment:
- Regime probabilities are aligned from 1h to 15m timeframe using forward-fill
- All DataFrames are aligned to a common index
- Missing values are handled with forward-fill, backward-fill, and zero-filling

## Model Serialization

### Pickle Format

Models are saved using the `StandardizedModelManager` which uses `joblib.dump()`:

- **Format**: `.joblib` files (pickle-compatible)
- **Location**: `data_cache/models/{step_name}/`
- **Naming**: `{step_name}_{version}_{timestamp}.joblib`
- **Metadata**: Stored in companion `{model_id}_metadata.json` file

### Model Registry

All models are registered in:
- **File**: `data_cache/models/model_registry.json`
- **Content**: Model ID, step name, type, metrics, features, file paths

## Metrics Output

### Markdown Format

Metrics are saved to `outcomes/analyst_ensemble_metrics_{symbol}_{timeframe}_{direction}_{timestamp}.md`:

```markdown
# Analyst Ensemble Training Metrics

**Symbol**: ETHUSDT
**Timeframe**: 15m
**Direction**: long
**Timestamp**: 2025-11-08T12:34:56.789

## Performance Metrics

- **accuracy**: 0.856000
- **precision**: 0.842000
- **recall**: 0.871000
...

## Detailed Metrics

```json
{
  "accuracy": 0.856,
  "precision": 0.842,
  ...
}
```
```

### JSON Format

Metrics are also saved to `outcomes/analyst_ensemble_metrics_{symbol}_{timeframe}_{direction}_{timestamp}.json`:

```json
{
  "symbol": "ETHUSDT",
  "timeframe": "15m",
  "direction": "long",
  "timestamp": "2025-11-08T12:34:56.789",
  "metrics": {
    "accuracy": 0.856,
    "precision": 0.842,
    "recall": 0.871,
    ...
  }
}
```

## Usage

### Command Line

```bash
# Train analyst ensemble
python3 src/launcher/ares_launcher.py --train-analyst-ensemble \
  --symbol ETHUSDT \
  --timeframe 15m \
  --direction long \
  --execution-mode light
```

### Prerequisites

Before running analyst ensemble training, ensure the following steps have been completed:

1. **Feature Generation**:
   ```bash
   python3 src/launcher/ares_launcher.py feature_generation_labeling_integration_step \
     --symbol ETHUSDT --timeframe 15m
   ```

2. **Regime Training** (optional but recommended):
   ```bash
   python3 src/launcher/ares_launcher.py regime_ensemble_training \
     --symbol ETHUSDT --timeframe 1h --execution-mode blank
   ```

3. **Analyst Base Training** (required):
   ```bash
   python3 src/launcher/ares_launcher.py --train-analyst-base \
     --symbol ETHUSDT --timeframe 15m --direction long
   ```

### Configuration

Key configuration parameters:

- **symbol**: Trading symbol (e.g., 'ETHUSDT')
- **exchange**: Exchange name (default: 'binance')
- **timeframe**: Analyst timeframe (default: '15m')
- **regime_timeframe**: Regime timeframe (default: '1h')
- **direction**: Trading direction ('long', 'short', 'both')
- **execution_mode**: Training mode ('full', 'light', 'blank')

## Implementation Details

### Data Loading Process

1. **Context Setting**: Sets versioned artifacts context for each data source
2. **Artifact Retrieval**: Uses `_get_artifact()` to load HDF5 data
3. **Validation**: Checks for required artifacts and provides helpful error messages
4. **Feature Generation**: Computes disagreement features from base predictions
5. **Combination**: Merges all features with proper alignment

### Training Pipeline

1. **Feature Preparation**: Combines all data sources
2. **Model Training**: Delegates to `UnifiedModelsTrainingStep`
3. **Metrics Computation**: Calculates performance metrics (accuracy, precision, recall, etc.)
4. **Serialization**: Saves model using `StandardizedModelManager`
5. **Metrics Export**: Saves metrics to both .md and .json formats

### Error Handling

The step provides comprehensive error messages:

- Missing feature generation data → Suggests running prerequisite step
- Missing base model outputs → Provides exact command to run
- Missing regime data → Continues with warning (optional feature)
- Training failures → Detailed error logging with stack traces

## File Structure

```
Ares/
├── src/
│   ├── training/
│   │   └── steps/
│   │       └── model_training/
│   │           └── analyst_ensemble_training_step.py
│   └── utils/
│       └── versioned_artifacts/
│           └── store.py                 # HDF5 storage
├── data_cache/
│   └── models/
│       ├── model_registry.json          # Model registry
│       └── analyst_ensemble_training/
│           ├── {model_id}.joblib        # Trained model (pickle)
│           └── {model_id}_metadata.json # Model metadata
└── outcomes/
    ├── analyst_ensemble_metrics_*.md    # Metrics (Markdown)
    └── analyst_ensemble_metrics_*.json  # Metrics (JSON)
```

## Code Example

```python
from src.training.steps.model_training.analyst_ensemble_training_step import (
    AnalystEnsembleTrainingStep
)

# Create step
step = AnalystEnsembleTrainingStep()

# Configure
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'direction': 'long',
    'execution_mode': 'light'
}

# Execute
result = await step.execute(config)

# Check results
if result['success']:
    print(f"Model saved at: {result['model_path']}")
    print(f"Metrics saved to: {result['metrics_files']}")
    print(f"Accuracy: {result['metrics']['accuracy']}")
else:
    print(f"Training failed: {result['error']}")
```

## Verification

### Model Verification

To verify a model was saved correctly:

```python
from src.utils.standardized_model_manager import standardized_model_manager

# Load model
model, metadata = standardized_model_manager.load_model(model_id)

# Check metadata
print(f"Model type: {metadata.model_type}")
print(f"File size: {metadata.file_size} bytes")
print(f"Features: {len(metadata.features)}")
```

### Metrics Verification

Check that metrics files exist:

```bash
# List recent metrics files
ls -lth outcomes/analyst_ensemble_metrics_*.md | head -5
ls -lth outcomes/analyst_ensemble_metrics_*.json | head -5
```

## Troubleshooting

### Common Issues

**Issue**: "No feature generation data found"
- **Solution**: Run `feature_generation_labeling_integration_step` first

**Issue**: "No analyst base model outputs found"
- **Solution**: Run `analyst_base_training` with same symbol/timeframe/direction

**Issue**: "Model path not found in result"
- **Solution**: Check `UnifiedModelsTrainingStep` is returning `model_path` in result

**Issue**: "Metrics not saved"
- **Solution**: Ensure `outcomes/` directory is writable and has sufficient disk space

## Performance Considerations

- **HDF5 Loading**: Efficient column-wise loading from versioned artifacts
- **Feature Alignment**: Minimal memory overhead with pandas reindex
- **Model Size**: Typical ensemble models are 10-100 MB (joblib compressed)
- **Training Time**: Depends on execution_mode (blank < light < full)

## Future Enhancements

- [ ] Add hyperparameter optimization (HPO) for ensemble meta-learner
- [ ] Support for multiple timeframe ensembles
- [ ] Implement online learning for continuous model updates
- [ ] Add model explainability outputs (SHAP, LIME)
- [ ] Create ensemble visualization dashboard

## References

- [Versioned Artifacts Documentation](../src/utils/versioned_artifacts/README.md)
- [Standardized Model Manager](../src/utils/standardized_model_manager.py)
- [Base Step Interface](../src/training/steps/base_step.py)
- [Regime Ensemble Training](../src/training/steps/market_analysis/regime_ensemble_training_step.py)
