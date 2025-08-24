# Training Modes Guide

## Overview

The Ares trading system now features a centralized training mode configuration system that provides three distinct training modes: **light**, **blank**, and **full**. Each mode is optimized for different use cases and computational requirements, with configurable parameters and lookback periods.

## Training Modes

### 1. Light Mode (30 days)
- **Purpose**: Quick testing, development, and debugging
- **Lookback Period**: 30 days
- **Computational Intensity**: Low
- **Estimated Duration**: 5 minutes
- **Best For**: 
  - Rapid prototyping
  - Code testing
  - Development iterations
  - Quick validation of changes

**Configuration**:
- Max Trials: 2
- N Trials: 3
- Advanced Model Training: Disabled
- Ensemble Training: Disabled
- Multi-timeframe Training: Disabled
- Adaptive Training: Disabled

### 2. Blank Mode (180 days)
- **Purpose**: Moderate testing and validation
- **Lookback Period**: 180 days (6 months)
- **Computational Intensity**: Medium
- **Estimated Duration**: 15 minutes
- **Best For**:
  - Feature validation
  - Model testing
  - Performance evaluation
  - Experimentation

**Configuration**:
- Max Trials: 3
- N Trials: 5
- Advanced Model Training: Enabled
- Ensemble Training: Enabled
- Multi-timeframe Training: Disabled
- Adaptive Training: Disabled

### 3. Full Mode (730 days)
- **Purpose**: Production-ready model training
- **Lookback Period**: 730 days (2 years)
- **Computational Intensity**: High
- **Estimated Duration**: 120 minutes
- **Best For**:
  - Production training
  - Final validation
  - Comprehensive model development
  - Maximum accuracy requirements

**Configuration**:
- Max Trials: 200
- N Trials: 100
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