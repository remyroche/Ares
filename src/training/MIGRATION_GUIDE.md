# Migration Guide: Using the New Simplified Training Components

This guide helps you migrate from the old training system to the new simplified and modular architecture.

## Overview of Changes

### Old Structure
```
src/training/
├── enhanced_training_manager.py (5600+ lines)
├── enhanced_training_manager_backup.py
├── enhanced_training_manager_enhanced.py
├── steps/ (mixed organization)
└── various duplicate files
```

### New Structure
```
src/training/
├── core/                    # Core components
├── steps/                   # Organized by function
│   ├── data_preparation/
│   ├── market_analysis/
│   ├── feature_engineering/
│   └── ...
├── utils/                   # Reusable utilities
└── examples/               # Usage examples
```

## Migration Steps

### 1. Update Imports

#### Old Import Pattern
```python
from src.training.enhanced_training_manager import EnhancedTrainingManager
from src.training.steps.step1_data_collection import DataCollectionStep
from src.training.steps.vectorized_advanced_feature_engineering import VectorizedAdvancedFeatureEngineering
```

#### New Import Pattern
```python
from src.training.core.training_manager import create_training_manager
from src.training.steps.data_preparation.step01_data_collection import DataCollectionStep
from src.training.utils.feature_engineering.technical_indicators import TechnicalIndicatorCalculator
```

### 2. Update Training Manager Usage

#### Old Usage
```python
# Old way - complex initialization
config = load_config()
manager = EnhancedTrainingManager(config)
await manager.initialize()

# Complex execution
result = await manager.execute_enhanced_training(
    enhanced_training_input={
        "symbol": "BTCUSDT",
        "exchange": "binance",
        # Many nested configurations
    }
)
```

#### New Usage
```python
# New way - simplified
config = {
    "symbol": "BTCUSDT",
    "exchange": "binance",
    "timeframe": "1m"
}

# Factory function handles initialization
manager = await create_training_manager(config)

# Simple execution
result = await manager.train(
    symbol="BTCUSDT",
    exchange="binance"
)
```

### 3. Update Step Implementations

#### Old Step Pattern
```python
class DataCollectionStep:
    def __init__(self, config):
        self.config = config
        # Complex initialization
    
    async def execute(self, training_input, pipeline_state):
        # Mixed validation and execution
        # No standard interface
```

#### New Step Pattern
```python
from src.training.base_step import BaseStep

class DataCollectionStep(BaseStep):
    def __init__(self, config):
        super().__init__(config, "01", "data_collection")
    
    def validate_inputs(self, training_input, pipeline_state):
        # Separate validation
        return is_valid, errors
    
    async def execute_logic(self, training_input, pipeline_state):
        # Clean execution logic
        return updated_state
    
    def validate_outputs(self, pipeline_state):
        # Output validation
        return is_valid, errors
```

### 4. Update Feature Engineering

#### Old Feature Engineering
```python
# Monolithic 6000+ line file
from src.training.steps.vectorized_advanced_feature_engineering import (
    VectorizedAdvancedFeatureEngineering
)

# Everything in one class
feature_eng = VectorizedAdvancedFeatureEngineering()
features = feature_eng.engineer_features(data)
```

#### New Feature Engineering
```python
# Modular components
from src.training.utils.feature_engineering.technical_indicators import TechnicalIndicatorCalculator
from src.training.utils.feature_engineering.wavelet_features import WaveletTransformAnalyzer
from src.training.utils.feature_engineering.resampling import OptimizedResampler

# Use specific components
indicator_calc = TechnicalIndicatorCalculator()
technical_features = indicator_calc.calculate_all_features(data)

wavelet_analyzer = WaveletTransformAnalyzer()
wavelet_features = wavelet_analyzer.extract_wavelet_features(data)
```

### 5. Update Configuration

#### Old Configuration
```python
config = {
    "enhanced_training_manager": {
        "enable_enhanced_features": True,
        "optimization_config": {
            # Deeply nested configuration
        }
    },
    "steps": {
        "step1": {
            # Step-specific config mixed with general config
        }
    }
}
```

#### New Configuration
```python
config = {
    # Top-level simple config
    "symbol": "BTCUSDT",
    "exchange": "binance",
    "timeframe": "1m",
    
    # Feature engineering settings
    "feature_engineering": {
        "enable_wavelets": True,
        "timeframes": ["5m", "15m", "1h"]
    },
    
    # Step-specific parameters
    "step_params": {
        "01": {"lookback_years": 2},
        "06": {"enable_wavelets": False}
    }
}
```

## Common Migration Scenarios

### Scenario 1: Running Full Pipeline

```python
# Old
manager = EnhancedTrainingManager(config)
await manager.initialize()
result = await manager.execute_enhanced_training(complex_input)

# New
manager = await create_training_manager(config)
result = await manager.train("BTCUSDT", "binance")
```

### Scenario 2: Running Specific Steps

```python
# Old - complex step management
# No standard way to run specific steps

# New
manager = await create_training_manager(config)
result = await manager.train(
    "BTCUSDT", 
    "binance",
    start_step="01",
    end_step="06"
)
```

### Scenario 3: Custom Feature Engineering

```python
# Old - modify the giant class
class CustomFeatureEngineering(VectorizedAdvancedFeatureEngineering):
    # Override methods in 6000+ line class

# New - use modular components
from src.training.utils.feature_engineering.technical_indicators import TechnicalIndicatorCalculator

class CustomIndicators(TechnicalIndicatorCalculator):
    def calculate_custom_features(self, data):
        # Add only what you need
        return custom_features
```

## Breaking Changes

1. **Import Paths**: All import paths have changed
2. **Class Names**: Some classes renamed for clarity
3. **Method Signatures**: Standardized interfaces
4. **Configuration Structure**: Flattened and simplified

## Backward Compatibility

For temporary backward compatibility, you can create adapters:

```python
# adapter.py
from src.training.core.training_manager import create_training_manager

class EnhancedTrainingManagerAdapter:
    """Adapter for backward compatibility."""
    
    def __init__(self, config):
        self.config = config
        self.manager = None
    
    async def initialize(self):
        self.manager = await create_training_manager(self.config)
    
    async def execute_enhanced_training(self, training_input):
        # Adapt old input format to new
        return await self.manager.train(
            training_input["symbol"],
            training_input["exchange"]
        )
```

## Benefits of Migration

1. **Maintainability**: Smaller, focused modules
2. **Testability**: Clear interfaces and dependencies
3. **Performance**: Modular loading, better caching
4. **Flexibility**: Easy to extend or modify specific components
5. **Clarity**: Clear step flow and dependencies

## Getting Help

- See `examples/simplified_pipeline_example.py` for usage examples
- Check `PIPELINE_DOCUMENTATION.md` for pipeline flow
- Review `MODULE_STRUCTURE.md` for organization

## Gradual Migration Strategy

1. **Phase 1**: Update imports and use adapters
2. **Phase 2**: Migrate configuration format
3. **Phase 3**: Update custom steps to use BaseStep
4. **Phase 4**: Refactor custom logic to use utilities
5. **Phase 5**: Remove adapters and old code

The new architecture is designed to be more maintainable, testable, and easier to understand while providing the same functionality as the old system.