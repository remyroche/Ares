# Training Modes Guide

## Overview

The Ares trading system now features a centralized training mode configuration system that provides three distinct training modes: **light**, **blank**, and **full**. Each mode is optimized for different use cases and computational requirements, with configurable parameters and lookback periods.

## Training Modes

### 1. Light Mode (30 days) - 2% Intensity
- **Purpose**: Quick testing, development, and debugging
- **Lookback Period**: 30 days
- **Computational Intensity**: Low (2% of full intensity)
- **Estimated Duration**: 5 minutes
- **Best For**:
  - Rapid prototyping
  - Code testing
  - Development iterations
  - Quick validation of changes

**Configuration**:
- Max Trials: 4 (2% of 200, minimum 3)
- N Trials: 3 (2% of 100, minimum 3)
- Advanced Model Training: Disabled
- Ensemble Training: Disabled
- Multi-timeframe Training: Disabled
- Adaptive Training: Disabled

**Minimum Requirements**: All modes enforce a minimum of 3 trials for both max_trials and n_trials to ensure meaningful training results.

### 2. Blank Mode (180 days) - 10% Intensity
- **Purpose**: Moderate testing and validation
- **Lookback Period**: 180 days (6 months)
- **Computational Intensity**: Medium (10% of full intensity)
- **Estimated Duration**: 15 minutes
- **Best For**:
  - Feature validation
  - Model testing
  - Performance evaluation
  - Experimentation

**Configuration**:
- Max Trials: 20 (10% of 200, minimum 3)
- N Trials: 10 (10% of 100, minimum 3)
- Advanced Model Training: Enabled
- Ensemble Training: Enabled
- Multi-timeframe Training: Disabled
- Adaptive Training: Disabled

### 3. Full Mode (730 days) - 100% Intensity
- **Purpose**: Production-ready model training
- **Lookback Period**: 730 days (2 years)
- **Computational Intensity**: High (100% intensity)
- **Estimated Duration**: 120 minutes
- **Best For**:
  - Production training
  - Final validation
  - Comprehensive model development
  - Maximum accuracy requirements

**Configuration**:
- Max Trials: 200 (100% intensity, minimum 3)
- N Trials: 100 (100% intensity, minimum 3)
- Advanced Model Training: Enabled
- Ensemble Training: Enabled
- Multi-timeframe Training: Enabled
- Adaptive Training: Enabled

## Usage

### Basic Commands

```bash
# Light training for quick testing
python ares_launcher.py light --symbol ETHUSDT --exchange BINANCE

# Blank training for moderate testing
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE

# Full training for production
python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE
```

### Custom Lookback Periods

You can override the default lookback period for any mode using the `--lookback-days` parameter:

```bash
# Light training with 15 days instead of 30
python ares_launcher.py light --symbol ETHUSDT --exchange BINANCE --lookback-days 15

# Blank training with 90 days instead of 180
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --lookback-days 90

# Full training with 365 days instead of 730
python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --lookback-days 365
```

### View Available Modes

To see all available modes and their configurations:

```bash
python ares_launcher.py modes
```

This will display:
- Mode descriptions
- Default parameters
- Computational requirements
- Usage recommendations
- Example commands

## Configuration System

### Centralized Configuration

All training mode configurations are centralized in `src/config/training_modes.py`. This provides:

1. **Single Source of Truth**: All mode parameters are defined in one place
2. **Easy Customization**: Parameters can be easily modified without touching multiple files
3. **Consistency**: Ensures all components use the same configuration
4. **Validation**: Built-in parameter validation to prevent invalid configurations
5. **Step-Specific Parameters**: Different pipeline steps get appropriate parameter scaling

### Parameter Application Across Pipeline Steps

The training mode parameters are applied across all pipeline steps with step-specific scaling. **All steps now use the centralized configuration instead of hardcoded values.**

#### **Step 12: Final Parameters Optimization**
- **Light Mode**: 3 trials for each optimization section (confidence, volatility, position sizing, etc.)
- **Blank Mode**: 4-6 trials for each optimization section
- **Full Mode**: 30-60 trials for each optimization section
- **Parameters**: `confidence_threshold_trials`, `volatility_trials`, `position_sizing_trials`, `risk_management_trials`, `ensemble_trials`, `regime_specific_trials`, `timing_trials`

#### **Step 6: Analyst Enhancement**
- **Light Mode**: 3 trials for each model type (LightGBM, XGBoost, SVM, etc.)
- **Blank Mode**: 3-5 trials for each model type
- **Full Mode**: 25-50 trials for each model type
- **Parameters**: `lightgbm_trials`, `xgboost_trials`, `svm_trials`, `random_forest_trials`, `neural_network_trials`, `catboost_trials`, `logistic_trials`

#### **Step 5.5: Unified Regime Intelligence**
- **Light Mode**: 3 HPO trials, 300s timeout
- **Blank Mode**: 3 HPO trials, 300s timeout
- **Full Mode**: 20 HPO trials, 900s timeout
- **Parameters**: `hpo_trials`, `hpo_timeout`

#### **SR Outcome Model Trainer**
- **Light Mode**: 3 trials for each model type
- **Blank Mode**: 3 trials for each model type
- **Full Mode**: 30 trials for each model type
- **Parameters**: `sr_lightgbm_trials`, `sr_xgboost_trials`

#### **Validation Steps (13, 14, 15)**
- **Light Mode**: 2-3 validation folds/runs, 3 trials
- **Blank Mode**: 2-10 validation folds/runs, 3 trials
- **Full Mode**: 5-100 validation folds/runs, 20-50 trials

### Configuration Structure

Each training mode is defined using a `TrainingModeConfig` dataclass:

```python
@dataclass
class TrainingModeConfig:
    name: str
    description: str
    lookback_days: int
    max_trials: int
    n_trials: int
    exclude_recent_days: int
    enable_advanced_model_training: bool
    enable_ensemble_training: bool
    enable_multi_timeframe_training: bool
    enable_adaptive_training: bool
    enhanced_training_interval: int
    max_enhanced_training_history: int
    min_data_points: int
    computational_intensity: str
    estimated_duration_minutes: int
```

### Backward Compatibility

The system maintains backward compatibility with existing code:

- Constants like `FULL_TRAINING_LOOKBACK_DAYS`, `BLANK_TRAINING_LOOKBACK_DAYS`, etc. are still available
- Environment variables like `BLANK_TRAINING_MODE` and `FULL_TRAINING_MODE` are still supported
- Existing training scripts continue to work without modification

## API Reference

### Core Functions

#### `get_training_mode_config(mode: str) -> TrainingModeConfig`
Get the configuration for a specific training mode.

```python
from src.config.training_modes import get_training_mode_config

config = get_training_mode_config("light")
print(f"Light mode uses {config.lookback_days} days")
```

#### `get_training_config_dict(mode: str) -> Dict[str, Any]`
Get the training configuration dictionary for a specific mode.

```python
from src.config.training_modes import get_training_config_dict

config_dict = get_training_config_dict("blank")
max_trials = config_dict["enhanced_training_manager"]["max_trials"]
```

#### `get_training_input_dict(mode: str, symbol: str, exchange: str, **kwargs) -> Dict[str, Any]`
Get the training input dictionary for a specific mode.

```python
from src.config.training_modes import get_training_input_dict

training_input = get_training_input_dict(
    mode="full",
    symbol="ETHUSDT",
    exchange="BINANCE",
    lookback_days=365  # Override default
)
```

#### `list_available_modes() -> Dict[str, str]`
Get a list of available training modes with their descriptions.

```python
from src.config.training_modes import list_available_modes

modes = list_available_modes()
for mode, description in modes.items():
    print(f"{mode}: {description}")
```

#### `validate_mode_parameters(mode: str, **kwargs) -> bool`
Validate that the provided parameters are appropriate for the specified mode.

```python
from src.config.training_modes import validate_mode_parameters

# Valid parameters
is_valid = validate_mode_parameters("light", lookback_days=30, max_trials=2)

# Invalid parameters
is_invalid = validate_mode_parameters("light", lookback_days=100, max_trials=10)
```

#### `get_mode_recommendations() -> Dict[str, str]`
Get recommendations for when to use each mode.

```python
from src.config.training_modes import get_mode_recommendations

recommendations = get_mode_recommendations()
for mode, recommendation in recommendations.items():
    print(f"{mode}: {recommendation}")
```

#### `get_step_specific_parameters(mode: str, step_name: str) -> Dict[str, Any]`
Get step-specific parameters for a particular pipeline step.

```python
from src.config.training_modes import get_step_specific_parameters

# Get parameters for step 12 with light mode
step_params = get_step_specific_parameters("light", "step12_final_parameters_optimization")
print(f"Confidence trials: {step_params['confidence_threshold_trials']}")
```

#### `get_optimization_parameters(mode: str, optimization_type: str) -> Dict[str, Any]`
Get optimization parameters for different optimization types.

```python
from src.config.training_modes import get_optimization_parameters

# Get hyperparameter optimization parameters for blank mode
opt_params = get_optimization_parameters("blank", "hyperparameter")
print(f"Trials: {opt_params['n_trials']}, Timeout: {opt_params['timeout']}s")
```

#### `apply_mode_parameters_to_config(config: Dict[str, Any], mode: str, step_name: str) -> Dict[str, Any]`
Apply training mode parameters to an existing configuration.

```python
from src.config.training_modes import apply_mode_parameters_to_config

base_config = {"some_param": "value"}
updated_config = apply_mode_parameters_to_config(base_config, "light", "step6_analyst_enhancement")
```

## Implementation Details

### Step Modifications

All pipeline steps have been updated to use the centralized training mode configuration instead of hardcoded values:

#### **Modified Steps**
1. **Step 12: Final Parameters Optimization** - All optimization sections now use configurable trials
2. **Step 6: Analyst Enhancement** - All model types now use configurable trials
3. **Step 5.5: Unified Regime Intelligence** - HPO uses configurable trials and timeout
4. **SR Outcome Model Trainer** - Both LightGBM and XGBoost use configurable trials

#### **Parameter Access Pattern**
Each step now follows this pattern to access training input parameters:

```python
# Get trials from training input or use default
confidence_trials = self.training_input.get("confidence_threshold_trials", 40)
study.optimize(objective, n_trials=confidence_trials)
```

#### **Fallback Mechanism**
If a step-specific parameter is not provided in the training input, the step falls back to a reasonable default value. This ensures backward compatibility and graceful degradation.

### Benefits of Step Modifications

1. **True Scalability**: Light and blank modes now use only 2% and 10% of full intensity across ALL steps
2. **Consistent Behavior**: All steps respect the training mode configuration
3. **Maintainable**: Easy to adjust parameters for specific steps
4. **Backward Compatible**: Steps work even without step-specific parameters

## Best Practices

### Choosing the Right Mode

1. **Development Phase**: Use **light mode** for rapid iteration and testing
2. **Validation Phase**: Use **blank mode** for feature validation and performance testing
3. **Production Phase**: Use **full mode** for final training and deployment

### Custom Lookback Periods

When using custom lookback periods:

1. **Light Mode**: Keep between 7-60 days for quick testing
2. **Blank Mode**: Keep between 30-365 days for moderate testing
3. **Full Mode**: Keep between 365-1095 days for production training

### Performance Considerations

- **Light Mode**: Minimal computational requirements, suitable for any system
- **Blank Mode**: Moderate requirements, suitable for most development systems
- **Full Mode**: High requirements, recommended for dedicated training systems

## Migration Guide

### From Old System

If you're migrating from the old system:

1. **Replace `short-blank` with `light`**:
   ```bash
   # Old
   python ares_launcher.py short-blank --symbol ETHUSDT --exchange BINANCE

   # New
   python ares_launcher.py light --symbol ETHUSDT --exchange BINANCE
   ```

2. **Use centralized configuration**:
   ```python
   # Old: Direct constant usage
   from src.config.constants import BLANK_TRAINING_LOOKBACK_DAYS

   # New: Use centralized configuration
   from src.config.training_modes import get_training_mode_config
   config = get_training_mode_config("blank")
   lookback_days = config.lookback_days
   ```

3. **Environment variables still work**:
   ```bash
   # These still work for backward compatibility
   export BLANK_TRAINING_MODE=1
   export FULL_TRAINING_MODE=1
   ```

## Troubleshooting

### Common Issues

1. **Invalid Mode Error**:
   ```
   ValueError: Unsupported training mode: invalid_mode
   ```
   **Solution**: Use one of the supported modes: `light`, `blank`, or `full`

2. **Parameter Validation Error**:
   ```
   ValueError: Invalid parameters for mode
   ```
   **Solution**: Check that your custom parameters are within the valid ranges for the mode

3. **Performance Issues**:
   - For slow systems, use **light mode**
   - For moderate systems, use **blank mode**
   - For fast systems, use **full mode**

### Getting Help

1. **View available modes**: `python ares_launcher.py modes`
2. **Check mode configurations**: Use the API functions to inspect configurations
3. **Validate parameters**: Use `validate_mode_parameters()` before training

## Future Enhancements

The training mode system is designed to be extensible:

1. **New Modes**: Easy to add new training modes with different configurations
2. **Dynamic Parameters**: Support for runtime parameter adjustment
3. **Mode Presets**: User-defined mode configurations
4. **Performance Profiling**: Automatic mode recommendation based on system capabilities

## Conclusion

The centralized training mode system provides a flexible, maintainable, and user-friendly way to configure training parameters for different use cases. By centralizing the configuration and providing clear mode definitions, the system ensures consistency and makes it easy to choose the right training configuration for your needs.