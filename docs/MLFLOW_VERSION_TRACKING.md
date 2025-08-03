# MLFlow Version Tracking

This document describes the MLFlow version tracking system implemented in the Ares trading bot.

## Overview

The bot now automatically includes the bot version in all MLFlow runs, providing better traceability and reproducibility of model training experiments.

## Features

### 1. Automatic Bot Version Logging

Every MLFlow run now includes the following tags:
- `bot_version`: Current bot version (e.g., "2.0.0")
- `training_date`: Date when training was performed
- `model_type`: Type of model being trained
- `symbol`: Trading symbol being trained on
- `timeframe`: Timeframe used for training

### 2. Utility Functions

The system provides utility functions for working with versioned runs:

```python
from src.utils.mlflow_utils import (
    log_bot_version_to_mlflow,
    log_training_metadata_to_mlflow,
    get_run_with_bot_version,
)

# Log bot version to current run
log_bot_version_to_mlflow()

# Log comprehensive training metadata
log_training_metadata_to_mlflow(
    symbol="ETHUSDT",
    timeframe="15m",
    model_type="enhanced_training"
)

# Get run information including bot version
run_info = get_run_with_bot_version("run_id_here")
```

### 3. Integration Points

The version tracking is automatically integrated into:
- Enhanced training manager (`src/training/enhanced_training_manager.py`)
- Training pipeline steps
- Model training processes

## Configuration

### Bot Version

The bot version is defined in `src/config.py`:

```python
ARES_VERSION = "2.0.0"  # Used for model versioning
```

### MLFlow Configuration

MLFlow is configured in `src/config.py`:

```python
"MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"),
"MLFLOW_EXPERIMENT_NAME": "Ares_Trading_Models",
```

## Usage Examples

### 1. Manual Version Logging

```python
import mlflow
from src.utils.mlflow_utils import log_bot_version_to_mlflow

with mlflow.start_run():
    # Your training code here
    mlflow.log_metric("accuracy", 0.85)
    
    # Log bot version
    log_bot_version_to_mlflow()
```

### 2. Comprehensive Metadata Logging

```python
from src.utils.mlflow_utils import log_training_metadata_to_mlflow

log_training_metadata_to_mlflow(
    symbol="ETHUSDT",
    timeframe="15m",
    model_type="ensemble_model"
)
```

### 3. Querying Runs by Version

```python
from src.utils.mlflow_utils import get_run_with_bot_version

# Get run information including bot version
run_info = get_run_with_bot_version("abc123")
if run_info:
    print(f"Bot version: {run_info['bot_version']}")
    print(f"Training date: {run_info['training_date']}")
    print(f"Symbol: {run_info['symbol']}")
```

## Benefits

### 1. Reproducibility

- Track which bot version was used for each model
- Reproduce results with specific bot versions
- Understand model performance across different bot versions

### 2. Audit Trail

- Complete history of bot changes
- Link model performance to bot improvements
- Track the evolution of the trading system

### 3. Model Comparison

- Compare models trained with different bot versions
- Identify which bot changes improved model performance
- Maintain backward compatibility

### 4. Debugging

- Quickly identify which bot version trained a problematic model
- Trace issues back to specific bot changes
- Validate fixes across different bot versions

## Changelog Integration

The MLFlow version tracking works in conjunction with the manual changelog system:

1. **Version Updates**: When updating the bot version in `src/config.py`
2. **Changelog Entry**: Add an entry to `docs/BOT_CHANGELOG.md`
3. **Automatic Tracking**: All subsequent MLFlow runs will include the new version
4. **Documentation**: Changes are documented in the changelog for reference

## Demo Script

Run the demo script to see the version tracking in action:

```bash
python scripts/demo_mlflow_version_tracking.py
```

This script demonstrates:
- Creating an MLFlow run with bot version tracking
- Logging training metadata
- Retrieving run information with version details
- Displaying changelog information

## Best Practices

### 1. Version Management

- Update bot version when making significant changes
- Document changes in the changelog
- Use semantic versioning (major.minor.patch)

### 2. MLFlow Usage

- Always use the utility functions for consistent logging
- Include comprehensive metadata for each training run
- Query runs by version for analysis

### 3. Documentation

- Keep the changelog up to date
- Document major changes and their impact
- Include version information in model reports

## Troubleshooting

### Common Issues

1. **MLFlow not running**: Ensure MLFlow tracking URI is configured
2. **Version not logged**: Check that utility functions are being called
3. **Run not found**: Verify run ID and MLFlow experiment name

### Debug Commands

```python
# Check current bot version
from src.config import ARES_VERSION
print(f"Current bot version: {ARES_VERSION}")

# List all runs with bot version
from src.utils.mlflow_utils import list_runs_by_bot_version
runs = list_runs_by_bot_version("2.0.0")
for run in runs:
    print(f"Run {run['run_id']}: {run['bot_version']}")
```

## Future Enhancements

1. **Version Comparison Tools**: Compare models across different bot versions
2. **Automated Changelog**: Generate changelog entries from git commits
3. **Version Analytics**: Analyze performance trends across bot versions
4. **Rollback Capability**: Easy rollback to previous bot versions 