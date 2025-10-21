# Mode-Aware Data Fetching Integration

This document explains the integration of ares_launcher's mode system (full/blank/light) with the artifact_manager and BaseStep classes to provide automatic lookback period control based on execution mode.

## Overview

The integration allows all pipeline steps to automatically use the correct data lookback period based on the execution mode:

- **full**: 1460 days (4 years) - Production mode with complete dataset
- **blank**: 180 days (6 months) - Quick testing mode with moderate dataset  
- **light**: 20 days - Development mode with minimal dataset

**Key Feature**: The lookback period is calculated from the **latest available data point**, not from the current time. This ensures that steps always work with the most recent data available, regardless of when they are executed.

This eliminates the need for each step to manually implement mode-specific data fetching logic.

## Key Components

### 1. Enhanced ArtifactManager

The `ArtifactManager` class now includes mode-aware data fetching methods that automatically calculate lookback periods from the latest available data point:

```python
# Load data with mode-aware lookback
data = artifact_manager.load_data_with_mode(
    symbol="ETHUSDT",
    interval="15m", 
    mode="light"  # Automatically uses 20 days lookback
)

# Load klines with context (uses current execution mode)
data = artifact_manager.load_klines_with_mode(
    symbol="ETHUSDT",
    interval="15m"
)

# Get mode configuration
mode_config = artifact_manager.get_mode_config("light")
lookback_days = artifact_manager.get_mode_lookback_days("light")
```

### 2. Enhanced BaseStep

The `BaseStep` class provides convenient methods for mode-aware data loading:

```python
class MyStep(BaseStep):
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Set context with execution mode
        self._set_context(
            symbol=config.get('symbol'),
            exchange=config.get('exchange'),
            execution_mode=config.get('execution_mode', 'light')
        )
        
        # Load data using mode-aware fetching
        data = self._load_klines_with_mode(
            symbol="ETHUSDT",
            interval="15m",
            mode="light"  # Uses 20 days lookback
        )
        
        # Get mode information
        lookback_days = self._get_mode_lookback_days("light")
        mode_config = self._get_mode_config("light")
        
        return {'success': True, 'data': data}
```

## Date Calculation Strategy

The system uses a sophisticated multi-strategy approach to ensure lookback periods are calculated from the latest available data point:

### Strategy 1: Symbol-Specific Detection
- If symbol and interval are provided, check for the latest data for that specific combination
- Try processed data first, then raw data if processed is not available

### Strategy 2: Global Data Detection  
- If symbol-specific detection fails, scan all available datasets
- Find the most recent date across all available data

### Strategy 3: Fallback to Current Time
- Only used if no data is available at all
- Includes warning messages to alert users about potential data gaps

This ensures that:
- ✅ Steps always use the most recent data available
- ✅ Lookback periods are calculated from actual data, not current time
- ✅ The system gracefully handles missing or incomplete data
- ✅ Users are warned when falling back to current time

## Mode Configuration

Mode configurations are defined in `src/config/pipeline_modes.py`:

```python
# Full mode - Production
FULL_MODE_CONFIG = ModeConfiguration(
    name="full",
    description="Production mode - Complete training with full dataset",
    lookback_days=1460,  # 4 years
    lookback_years=4,
    intensity_percentage=1.0,
    computational_intensity="high",
    # ... other parameters
)

# Light mode - Development  
LIGHT_MODE_CONFIG = ModeConfiguration(
    name="light",
    description="Development mode - Minimal data with all features/models",
    lookback_days=20,  # 20 days
    lookback_years=0,
    intensity_percentage=0.025,
    computational_intensity="minimal",
    # ... other parameters
)

# Blank mode - Quick testing
BLANK_MODE_CONFIG = ModeConfiguration(
    name="blank", 
    description="Quick testing mode - All features/models with shorter lookback",
    lookback_days=180,  # 6 months
    lookback_years=0,
    intensity_percentage=0.1,
    computational_intensity="medium",
    # ... other parameters
)
```

## Usage Examples

### 1. Basic Mode-Aware Data Loading

```python
from src.utils.artifact_manager import ArtifactManager

# Create artifact manager
artifact_manager = ArtifactManager({
    'enable_compression': True,
    'enable_caching': True
})

# Set execution mode
artifact_manager.set_execution_mode('light')

# Load data (automatically uses 20 days lookback)
data = artifact_manager.load_klines_with_mode(
    symbol='ETHUSDT',
    interval='15m'
)
```

### 2. Step with Mode-Aware Data Fetching

```python
from src.training.steps.base_step import BaseStep

class FeatureGenerationStep(BaseStep):
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        execution_mode = config.get('execution_mode', 'light')
        
        # Set context with execution mode
        self._set_context(
            symbol=config.get('symbol'),
            exchange=config.get('exchange'),
            execution_mode=execution_mode
        )
        
        # Load data with mode-aware lookback
        data = self._load_klines_with_mode(
            symbol=config.get('symbol'),
            interval=config.get('timeframe', '15m'),
            mode=execution_mode
        )
        
        if data is None:
            return {'success': False, 'error': 'No data available'}
        
        # Process data...
        processed_data = process_features(data)
        
        # Save processed data
        self._save_dataframe(processed_data, 'processed_features')
        
        return {
            'success': True,
            'records_processed': len(processed_data),
            'lookback_days': self._get_mode_lookback_days(execution_mode)
        }
```

### 3. Launcher Integration

The ares_launcher automatically passes the execution mode to steps:

```bash
# Run step in light mode (20 days)
python ares_launcher.py feature_generation_step --symbol ETHUSDT --execution-mode light

# Run step in blank mode (180 days)  
python ares_launcher.py feature_generation_step --symbol ETHUSDT --execution-mode blank

# Run step in full mode (1460 days)
python ares_launcher.py feature_generation_step --symbol ETHUSDT --execution-mode full
```

## API Reference

### ArtifactManager Methods

#### `load_data_with_mode(symbol, interval, mode=None, data_type="raw", columns=None)`
Load data using mode-aware data fetching.

**Parameters:**
- `symbol` (str): Trading symbol
- `interval` (str): Data interval (e.g., "15m", "1h")
- `mode` (str, optional): Execution mode ("full", "blank", "light"). If None, uses current context mode.
- `data_type` (str): Data type ("raw" or "processed")
- `columns` (List[str], optional): List of columns to load

**Returns:**
- Loaded DataFrame or None

#### `load_klines_with_mode(symbol=None, interval="15m", mode=None, data_type="raw")`
Load klines data using mode-aware data fetching with context.

**Parameters:**
- `symbol` (str, optional): Trading symbol. If None, uses current context symbol.
- `interval` (str): Data interval (e.g., "15m", "1h")
- `mode` (str, optional): Execution mode ("full", "blank", "light"). If None, uses current context mode.
- `data_type` (str): Data type ("raw" or "processed")

**Returns:**
- Loaded DataFrame or None

#### `get_mode_lookback_days(mode=None)`
Get lookback days for the specified mode or current context mode.

**Parameters:**
- `mode` (str, optional): Execution mode ("full", "blank", "light"). If None, uses current context mode.

**Returns:**
- Number of lookback days for the mode

#### `get_mode_config(mode=None)`
Get configuration for the specified mode or current context mode.

**Parameters:**
- `mode` (str, optional): Execution mode ("full", "blank", "light"). If None, uses current context mode.

**Returns:**
- Mode configuration dictionary

#### `set_execution_mode(mode)`
Set the current execution mode for data fetching.

**Parameters:**
- `mode` (str): Execution mode ("full", "blank", "light")

**Raises:**
- `ValueError`: If mode is invalid
- `TypeError`: If mode is not a string

### BaseStep Methods

#### `_load_data_with_mode(symbol, interval, mode=None, data_type="raw", columns=None)`
Load data using mode-aware data fetching.

#### `_load_klines_with_mode(symbol=None, interval="15m", mode=None, data_type="raw")`
Load klines data using mode-aware data fetching with context.

#### `_get_mode_lookback_days(mode=None)`
Get lookback days for the specified mode or current context mode.

#### `_get_mode_config(mode=None)`
Get configuration for the specified mode or current context mode.

#### `_set_execution_mode(mode)`
Set the current execution mode for data fetching.

#### `_get_current_mode()`
Get the current execution mode.

## Benefits

1. **Automatic Compliance**: Steps automatically use the correct lookback period based on execution mode
2. **Latest Data Usage**: Lookback periods are calculated from the latest available data point, not current time
3. **Consistent Interface**: All steps use the same methods for mode-aware data loading
4. **Easy Migration**: Existing steps can be easily updated to use mode-aware data fetching
5. **Centralized Configuration**: Mode definitions are centralized in `pipeline_modes.py`
6. **Backward Compatibility**: Existing code continues to work without changes
7. **Robust Date Detection**: Multiple fallback strategies ensure we always find the latest available data

## Migration Guide

### For Existing Steps

1. **Update context setting** to include execution mode:
   ```python
   # Old
   self._set_context(symbol, exchange, information, direction, model)
   
   # New
   self._set_context(symbol, exchange, information, direction, model, execution_mode)
   ```

2. **Replace manual data loading** with mode-aware methods:
   ```python
   # Old
   data = self._load_klines(symbol, exchange, interval)
   
   # New
   data = self._load_klines_with_mode(symbol, interval, mode)
   ```

3. **Use mode information** for logging and metrics:
   ```python
   lookback_days = self._get_mode_lookback_days(mode)
   mode_config = self._get_mode_config(mode)
   ```

### For New Steps

1. Always use `_load_klines_with_mode()` instead of manual data loading
2. Set execution mode in context using `_set_context()`
3. Use mode configuration methods for consistent behavior

## Testing

Run the examples to test mode-aware data fetching:

```bash
# Test basic mode-aware data fetching
python src/examples/mode_aware_data_fetching_example.py

# Test lookback calculation from latest data point
python src/examples/test_lookback_from_latest_data.py
```

This will demonstrate:
- Loading data in different modes (light/blank/full)
- Automatic lookback period application from latest available data
- Mode configuration retrieval
- Error handling for missing data
- Verification that lookback periods are calculated from latest data, not current time

## Troubleshooting

### Common Issues

1. **No data available**: Ensure data exists for the requested symbol/interval combination
2. **Invalid mode**: Use only "full", "blank", or "light" as execution modes
3. **Context not set**: Call `_set_context()` before using mode-aware methods

### Debug Information

Enable debug logging to see mode-aware data fetching details:

```python
import logging
logging.getLogger("ares.step").setLevel(logging.DEBUG)
```

This will show:
- Mode configuration details
- Lookback period calculations
- Data loading progress
- Error details