# New Configuration Structure

## Overview

The Ares trading system has been reorganized to separate parameters into two distinct categories:

1. **Static Configuration** (`src/config/config.py`) - Non-optimizable parameters
2. **Optimizable Configuration** - Parameters that can be optimized in step12

## Configuration Categories

### Static Configuration (Non-Optimizable)

Located in `src/config/config.py`, these parameters are fixed and should not be optimized:

- **DatabaseConfig**: Database connection settings
- **ExchangeConfig**: Exchange API settings
- **SystemConfig**: System-level settings (logging, performance, etc.)
- **EnvironmentConfig**: Environment-specific settings
- **TradingConfig**: Basic trading parameters (fees, timeouts, etc.)
- **TrainingConfig**: Training-specific settings

### Optimizable Configuration

These parameters are organized into categories and can be optimized in step12:

#### 1. Confidence Thresholds (`src/config/config_confidence.py`)
- Entry thresholds
- Analyst vs Tactician thresholds
- Position management thresholds
- Ensemble thresholds
- Model performance thresholds
- Regime-specific thresholds
- S/R confidence thresholds

#### 2. Position Sizing (`src/config/config_position_sizing.py`)
- Base position sizing
- Confidence-based scaling
- Volatility adjustment
- Regime-based adjustment
- Liquidation risk adjustment
- Successive position rules
- Risk limits
- Kelly criterion parameters

#### 3. Leverage (`src/config/config_leverage.py`)
- Base leverage settings
- Leverage risk levels
- Dynamic leverage adjustment
- Volatility-based leverage
- Regime-based leverage
- Confidence-based leverage
- Liquidation risk management
- Leverage decay

#### 4. Take Profit/Stop Loss (`src/config/config_tpsl.py`)
- Base TP/SL settings
- Dynamic TP/SL based on volatility
- Regime-based TP/SL
- Confidence-based TP/SL
- Trailing stop loss
- Time-based exits
- Risk-reward ratios

#### 5. Ensemble (`src/config/config_ensemble.py`)
- Ensemble method selection
- Threshold-based ensemble
- Weighted ensemble
- Meta-learner parameters
- Regime-specific ensemble
- Ensemble validation
- Confidence-weighted ensemble
- Ensemble diversity and stability

#### 6. Support/Resistance (`src/config/config_sr.py`)
- Strength score weights
- Level detection parameters
- Breakout thresholds
- Zone multipliers
- Confidence thresholds
- Optimization configuration
- Performance thresholds

## Usage

### Basic Usage

```python
from src.config.config_manager import (
    get_static_config,
    get_optimizable_config,
    get_parameter_value,
    update_optimizable_config,
)

# Get static configuration
static_config = get_static_config()

# Get optimizable configuration for a category
confidence_config = get_optimizable_config("confidence")

# Get a specific parameter value
entry_threshold = get_parameter_value("confidence.base_entry_threshold")

# Update optimizable parameters
update_optimizable_config("confidence", {"base_entry_threshold": 0.75})
```

### Step12 Integration

The new step12 optimization (`src/training/steps/step12_final_parameters_optimization_new.py`) uses the categorized structure:

```python
from src.training.steps.step12_final_parameters_optimization_new import FinalParametersOptimizationStepNew

# Initialize step12
step12 = FinalParametersOptimizationStepNew(config)

# Execute optimization
results = await step12.execute(training_input, pipeline_state)
```

## Parameter Organization

### Static Parameters (Non-Optimizable)

These parameters remain constant and are not optimized:

- Database connection settings
- Exchange API credentials
- System configuration (logging, performance)
- Environment settings
- Basic trading parameters (fees, timeouts)
- Training configuration (splits, model types)

### Optimizable Parameters

These parameters are optimized in step12, one category at a time:

1. **Confidence Thresholds**: All confidence-related thresholds
2. **Position Sizing**: Position size calculations and risk management
3. **Leverage**: Leverage settings and risk levels
4. **TP/SL**: Take profit and stop loss parameters
5. **Ensemble**: Ensemble model combination parameters
6. **S/R**: Support/Resistance analysis parameters

## Search Spaces

Each optimizable category has defined search spaces for optimization:

```python
from src.config.config_manager import get_search_space

# Get search space for confidence parameters
confidence_search_space = get_search_space("confidence")

# Example search space structure
{
    "base_entry_threshold": {"min": 0.5, "max": 0.9, "type": "float"},
    "analyst_confidence_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
    # ... more parameters
}
```

## Migration from Old Structure

The old `src/config_optuna.py` structure has been reorganized into the new categorized structure:

- **Old**: Single large configuration with all parameters mixed
- **New**: Separated into static and optimizable categories

### Key Changes

1. **Separation of Concerns**: Static vs optimizable parameters are clearly separated
2. **Categorized Optimization**: Parameters are optimized by category in step12
3. **Better Organization**: Related parameters are grouped together
4. **Improved Maintainability**: Easier to add new parameters and categories
5. **Type Safety**: Better type hints and validation

## Testing

Run the test script to verify the new configuration structure:

```bash
python test_new_config_structure.py
```

This will test:
- Configuration loading
- Parameter access
- Parameter updates
- Search spaces
- Step12 integration
- Configuration validation

## Benefits

1. **Clear Separation**: Static vs optimizable parameters are clearly defined
2. **Organized Optimization**: Parameters are optimized by category
3. **Better Performance**: Only relevant parameters are optimized together
4. **Easier Maintenance**: Related parameters are grouped together
5. **Type Safety**: Better type hints and validation
6. **Extensibility**: Easy to add new parameter categories

## Future Enhancements

1. **Dynamic Search Spaces**: Search spaces could be adjusted based on market conditions
2. **Category-Specific Optimization**: Different optimization strategies per category
3. **Parameter Dependencies**: Handle dependencies between parameters
4. **Configuration Persistence**: Save/load optimized configurations
5. **Real-time Updates**: Update parameters during live trading