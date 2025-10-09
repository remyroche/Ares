# Ares Launcher Integration Guide

## Overview

This guide explains how the feature lookback optimization and interactive feature generation systems are integrated with ares_launcher to ensure that the lookback period (20 days in "light" mode) is properly applied when loading data.

## 🎯 Key Integration Points

### 1. Data Loading Integration
- **AresLauncherDataLoader**: Centralized data loading that respects ares_launcher mode configuration
- **Automatic Mode Detection**: Detects execution mode from pipeline state
- **Date Range Calculation**: Automatically calculates start/end dates based on mode
- **Data Validation**: Ensures sufficient data is available for the requested mode

### 2. Mode-Specific Configuration
- **Light Mode**: 20 days lookback, optimized for development
- **Blank Mode**: 180 days lookback, optimized for testing
- **Full Mode**: 1460 days lookback, optimized for production

### 3. Parameter Adaptation
- **Feature Budgets**: Adjusted based on mode (fewer features in light mode)
- **Processing Parameters**: Workers, batch sizes, and timeouts adjusted per mode
- **Optimization Settings**: CV folds, trials, and other parameters optimized per mode

## 📁 Integration Files

```
src/utils/data/
└── ares_launcher_data_loader.py          # Centralized data loading utility

src/training/steps/pre_training/feature_lookback_optimization/
└── ares_launcher_integration.py          # Feature optimization integration

src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/
└── ares_launcher_integration.py          # Interactive feature generation integration
```

## 🚀 Usage Examples

### 1. Basic Data Loading

```python
from src.utils.data.ares_launcher_data_loader import AresLauncherDataLoader

# Initialize loader
loader = AresLauncherDataLoader()

# Load data in light mode (20 days)
data = loader.load_data_with_mode("ETHUSDT", "15m", "light")

# Load data in blank mode (180 days)
data = loader.load_data_with_mode("ETHUSDT", "15m", "blank")

# Load data in full mode (1460 days)
data = loader.load_data_with_mode("ETHUSDT", "15m", "full")
```

### 2. Feature Lookback Optimization Integration

```python
from src.training.steps.pre_training.feature_lookback_optimization.ares_launcher_integration import (
    AresLauncherFeatureLookbackOptimizer
)

# Initialize optimizer
optimizer = AresLauncherFeatureLookbackOptimizer()

# Pipeline state with mode information
pipeline_state = {
    'execution_mode': 'light',
    'symbol': 'ETHUSDT',
    'timeframe': '15m'
}

# Load data respecting ares_launcher mode
data = optimizer.load_data_for_optimization(
    symbol="ETHUSDT",
    timeframe="15m",
    pipeline_state=pipeline_state
)

# Validate data availability
is_available = optimizer.validate_data_for_optimization(
    symbol="ETHUSDT",
    timeframe="15m",
    pipeline_state=pipeline_state
)
```

### 3. Interactive Feature Generation Integration

```python
from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.ares_launcher_integration import (
    AresLauncherInteractiveFeatureGenerator
)

# Initialize generator
generator = AresLauncherInteractiveFeatureGenerator()

# Pipeline state with mode information
pipeline_state = {
    'execution_mode': 'light',
    'symbol': 'ETHUSDT',
    'timeframe': '15m'
}

# Load data respecting ares_launcher mode
data = generator.load_data_for_generation(
    symbol="ETHUSDT",
    timeframe="15m",
    pipeline_state=pipeline_state
)

# Get mode-specific parameters
parameters = generator.get_generation_parameters(pipeline_state)
print(f"Feature budget: {parameters['feature_budget_pre']}")
print(f"Interactions cap: {parameters['interactions_cap']}")
```

## 🔧 Mode Detection Logic

The integration automatically detects the execution mode from the pipeline state using the following priority:

1. **Explicit Mode**: `pipeline_state['execution_mode']` or `pipeline_state['mode']`
2. **Lookback Days**: Infers mode from `pipeline_state['lookback_days']`
3. **Intensity Percentage**: Infers mode from `pipeline_state['intensity_percentage']`
4. **Default**: Falls back to "light" mode

### Mode Detection Examples

```python
# Explicit mode
pipeline_state = {'execution_mode': 'light'}

# Inferred from lookback days
pipeline_state = {'lookback_days': 20}  # -> light mode

# Inferred from intensity
pipeline_state = {'intensity_percentage': 0.025}  # -> light mode
```

## 📊 Mode-Specific Parameters

### Light Mode (20 days)
- **Lookback Days**: 20
- **Feature Budget (pre)**: 60
- **Feature Budget (post)**: (15, 30)
- **Interactions Cap**: 8
- **Max Workers**: 4
- **Batch Size**: 1000
- **Intensity**: 2.5%

### Blank Mode (180 days)
- **Lookback Days**: 180
- **Feature Budget (pre)**: 100
- **Feature Budget (post)**: (25, 50)
- **Interactions Cap**: 12
- **Max Workers**: 6
- **Batch Size**: 1500
- **Intensity**: 10%

### Full Mode (1460 days)
- **Lookback Days**: 1460
- **Feature Budget (pre)**: 150
- **Feature Budget (post)**: (40, 80)
- **Interactions Cap**: 20
- **Max Workers**: 8
- **Batch Size**: 2000
- **Intensity**: 100%

## 🔄 Integration with Existing Components

### 1. Feature Lookback Optimization

The existing `FeatureLookbackOptimizationComponent` can be enhanced to use the ares launcher integration:

```python
# In feature_lookback_optimization.py
from .ares_launcher_integration import AresLauncherFeatureLookbackOptimizer

class FeatureLookbackOptimizationComponent:
    def __init__(self, config=None):
        # ... existing initialization ...
        self.ares_integration = AresLauncherFeatureLookbackOptimizer()
    
    async def execute(self, training_input, pipeline_state):
        # Use ares launcher integration for data loading
        data = self.ares_integration.load_data_for_optimization(
            symbol=pipeline_state.get('symbol', 'ETHUSDT'),
            timeframe=pipeline_state.get('timeframe', '15m'),
            pipeline_state=pipeline_state
        )
        
        # ... rest of optimization logic ...
```

### 2. Interactive Feature Generation

The existing `InteractiveFeatureGenerationComponent` can be enhanced similarly:

```python
# In interactive_feature_generation_component.py
from .ares_launcher_integration import AresLauncherInteractiveFeatureGenerator

class InteractiveFeatureGenerationComponent:
    def __init__(self, config=None):
        # ... existing initialization ...
        self.ares_integration = AresLauncherInteractiveFeatureGenerator()
    
    async def execute(self, training_input, pipeline_state):
        # Use ares launcher integration for data loading
        data = self.ares_integration.load_data_for_generation(
            symbol=pipeline_state.get('symbol', 'ETHUSDT'),
            timeframe=pipeline_state.get('timeframe', '15m'),
            pipeline_state=pipeline_state
        )
        
        # ... rest of generation logic ...
```

## 🧪 Testing the Integration

### 1. Test Data Loading

```python
from src.utils.data.ares_launcher_data_loader import AresLauncherDataLoader

def test_data_loading():
    loader = AresLauncherDataLoader()
    
    # Test all modes
    for mode in ['light', 'blank', 'full']:
        data = loader.load_data_with_mode("ETHUSDT", "15m", mode)
        print(f"{mode} mode: {len(data) if data is not None else 0} records")
```

### 2. Test Mode Detection

```python
from src.training.steps.pre_training.feature_lookback_optimization.ares_launcher_integration import AresLauncherFeatureLookbackOptimizer

def test_mode_detection():
    optimizer = AresLauncherFeatureLookbackOptimizer()
    
    # Test different pipeline states
    test_states = [
        {'execution_mode': 'light'},
        {'lookback_days': 20},
        {'intensity_percentage': 0.025}
    ]
    
    for state in test_states:
        mode = optimizer.detect_execution_mode(state)
        print(f"State: {state} -> Mode: {mode}")
```

### 3. Test Parameter Adaptation

```python
def test_parameter_adaptation():
    generator = AresLauncherInteractiveFeatureGenerator()
    
    for mode in ['light', 'blank', 'full']:
        pipeline_state = {'execution_mode': mode}
        params = generator.get_generation_parameters(pipeline_state)
        print(f"{mode} mode parameters: {params['feature_budget_pre']} features, {params['interactions_cap']} interactions")
```

## 🔍 Debugging and Monitoring

### 1. Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# The integration will provide detailed logging about:
# - Mode detection
# - Date range calculation
# - Data loading progress
# - Parameter adaptation
```

### 2. Monitor Data Loading

```python
# Check data availability before loading
loader = AresLauncherDataLoader()
is_available = loader.validate_data_availability("ETHUSDT", "15m", "light")
print(f"Data available: {is_available}")

# Get detailed data information
info = loader.get_available_data_info("ETHUSDT", "15m", "light")
print(f"Data info: {info}")
```

### 3. Validate Mode Configuration

```python
# Print mode summary
optimizer = AresLauncherFeatureLookbackOptimizer()
pipeline_state = {'execution_mode': 'light'}
optimizer.print_mode_summary(pipeline_state)
```

## 🚀 Production Deployment

### 1. Configuration Validation

Before deploying, ensure that:
- All modes are properly configured in `src/config/pipeline_modes.py`
- Data is available for all required timeframes
- Integration components are properly imported

### 2. Performance Monitoring

Monitor the following metrics:
- Data loading times per mode
- Memory usage during data loading
- Feature generation performance per mode
- Error rates and fallback behavior

### 3. Error Handling

The integration includes comprehensive error handling:
- Graceful fallback to available data when requested range is not available
- Clear error messages for debugging
- Automatic mode detection with fallbacks

## 📚 Additional Resources

### Configuration Files
- `src/config/pipeline_modes.py` - Mode definitions
- `src/launcher/ares_launcher.py` - Main launcher
- `src/launcher/configuration_manager.py` - Configuration management

### Integration Examples
- `src/utils/data/ares_launcher_data_loader.py` - Data loading examples
- `src/training/steps/pre_training/feature_lookback_optimization/ares_launcher_integration.py` - Optimization examples
- `src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/ares_launcher_integration.py` - Generation examples

### Testing
- Run the example scripts in each integration file
- Use the provided test functions to validate integration
- Monitor logs for proper mode detection and data loading

## ✅ Conclusion

The ares launcher integration ensures that:

1. **Consistent Data Loading**: All components use the same data loading logic
2. **Mode-Aware Processing**: Parameters are automatically adjusted based on execution mode
3. **Proper Lookback Periods**: 20-day lookback in light mode is consistently applied
4. **Robust Error Handling**: Graceful fallbacks and clear error messages
5. **Easy Integration**: Simple APIs for existing components

The integration maintains backward compatibility while providing enhanced functionality for ares_launcher users.