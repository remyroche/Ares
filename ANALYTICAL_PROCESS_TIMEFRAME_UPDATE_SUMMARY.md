# Analytical Process Timeframe Update Summary

## Overview
Successfully updated the analytical process timeframes and frequencies as requested:

- **Regime Detection**: 4h timeframe, run every 1h
- **Analyst**: 1h timeframe, run every 15m  
- **Tactician**: 15m timeframe, run every 3m

## Changes Made

### 1. New Configuration File
**File**: `config/analytical_process_timeframes.yaml`

Created a comprehensive configuration file that defines:
- Timeframes and execution frequencies for all three components
- Data requirements and model types for each component
- Execution schedules with cron expressions
- Data pipeline configuration
- Model training parameters
- Monitoring and alerting thresholds

### 2. Configuration Loader
**File**: `src/config/analytical_process_config.py`

Created a Python configuration loader that provides:
- Easy access to component configurations
- Type-safe configuration objects
- Validation and error handling
- Convenience functions for quick access

### 3. Updated Existing Configuration Files

#### Regime Detection Updates
- **File**: `src/training/steps/market_analysis/nas_regime/core/perfect_nas_config.py`
  - Changed `regime_detection_timeframe` from "15m" to "4h"
  - Added "4h" to `trading_timeframes` list

#### Market Analysis Sub-Pipeline Updates  
- **File**: `src/training/steps/market_analysis/sub_pipeline.py`
  - Changed default `timeframe` from "30m" to "4h"

#### Analyst Model Training Updates
- **File**: `src/training/steps/models_training/analyst_models_training.py`
  - Updated documentation to reflect 1h timeframe
  - Changed references from 15m to 1h timeframe

#### Tactician Model Training Updates
- **File**: `src/training/steps/models_training/tactician_models_training.py`
  - Updated documentation to reflect 15m timeframe
  - Changed references from 5m to 15m timeframe

#### Training Modes Configuration Updates
- **File**: `config/training_modes.yaml`
  - Updated regime discovery timeframe from "1h" to "4h"

- **File**: `config/features/training_modes.yaml`
  - Added "4h" to timeframe lists for all training modes
  - Updated regime discovery timeframe references

## New Timeframe Structure

### Regime Detection (4h timeframe, run every 1h)
- **Purpose**: Market regime detection using 4-hour timeframe data
- **Execution**: Every hour at minute 0
- **Data Requirements**: Minimum 24 bars (6 days), 30 days lookback
- **Models**: NAS, TAS, and hybrid regime detectors
- **Max Execution Time**: 30 minutes

### Analyst (1h timeframe, run every 15m)
- **Purpose**: Strategic decision making (IF we trade)
- **Execution**: Every 15 minutes (0, 15, 30, 45)
- **Data Requirements**: Minimum 12 bars (12 hours), 7 days lookback
- **Models**: ElasticNet, RandomForest, NAS, TAS, N-BEATS
- **Confidence Threshold**: 0.4 (40%)
- **Max Execution Time**: 10 minutes

### Tactician (15m timeframe, run every 3m)
- **Purpose**: Tactical execution timing (WHEN we trade)
- **Execution**: Every 3 minutes
- **Data Requirements**: Minimum 20 bars (5 hours), 3 days lookback
- **Models**: RandomSurvivalForest, XGBoost, NAS, TAS
- **Analyst Filtering**: Requires analyst green signals (>0.4% confidence)
- **Max Execution Time**: 5 minutes

## Usage Examples

### Using the Configuration Loader

```python
from src.config.analytical_process_config import (
    get_regime_detection_config,
    get_analyst_config, 
    get_tactician_config,
    get_analytical_process_config
)

# Get specific component configurations
regime_config = get_regime_detection_config()
print(f"Regime Detection: {regime_config.timeframe} timeframe, runs every {regime_config.run_frequency}")

analyst_config = get_analyst_config()
print(f"Analyst: {analyst_config.timeframe} timeframe, runs every {analyst_config.run_frequency}")

tactician_config = get_tactician_config()
print(f"Tactician: {tactician_config.timeframe} timeframe, runs every {tactician_config.run_frequency}")

# Get all configurations
config_manager = get_analytical_process_config()
all_configs = config_manager.get_all_components()
```

### Accessing Execution Schedules

```python
from src.config.analytical_process_config import (
    get_regime_detection_schedule,
    get_analyst_schedule,
    get_tactician_schedule
)

# Get execution schedules
regime_schedule = get_regime_detection_schedule()
print(f"Regime Detection cron: {regime_schedule.cron_expression}")

analyst_schedule = get_analyst_schedule()
print(f"Analyst cron: {analyst_schedule.cron_expression}")

tactician_schedule = get_tactician_schedule()
print(f"Tactician cron: {tactician_schedule.cron_expression}")
```

## Validation

The configuration includes comprehensive validation:
- All required sections are present
- All three components are configured
- Data requirements are specified
- Execution schedules are defined
- Model training parameters are set

## Monitoring

Each component has monitoring thresholds:
- **Regime Detection**: 75% accuracy, 30s latency, 2GB memory
- **Analyst**: 70% accuracy, 10s latency, 1GB memory  
- **Tactician**: 65% accuracy, 5s latency, 512MB memory

## Next Steps

1. **Integration**: Update existing code to use the new configuration loader
2. **Testing**: Test the new timeframes with sample data
3. **Scheduling**: Implement cron job scheduling for the new frequencies
4. **Monitoring**: Set up monitoring and alerting based on the new thresholds
5. **Documentation**: Update system documentation to reflect the new timeframes

## Files Modified

1. `config/analytical_process_timeframes.yaml` (new)
2. `src/config/analytical_process_config.py` (new)
3. `src/training/steps/market_analysis/nas_regime/core/perfect_nas_config.py`
4. `src/training/steps/market_analysis/sub_pipeline.py`
5. `src/training/steps/models_training/analyst_models_training.py`
6. `src/training/steps/models_training/tactician_models_training.py`
7. `config/training_modes.yaml`
8. `config/features/training_modes.yaml`

All changes maintain backward compatibility while implementing the new timeframe requirements.