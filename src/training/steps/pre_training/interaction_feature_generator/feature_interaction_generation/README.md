# Optimized Interaction Feature Generation Pipeline

This module provides a fully wired, optimized interaction feature generation pipeline that integrates with the pre-training sub_pipeline architecture and ares_launcher. It features extensive logging, matrix operations optimization, M1 hardware acceleration, and comprehensive utility integration.

## 🚀 Key Features

### Complete Pipeline Integration
- **Feature Engineering Bank**: Gets features from the feature_engineering_roadmap/ bank
- **Lookback Optimization**: Selects optimal lookback periods using data-driven methods
- **Cross-timeframe Features**: Generates features across multiple timeframes
- **Interaction Features**: Creates 15 locked interactions with theory-first approach
- **Matrix Operations**: Vectorized computations using matrix_operations/ utilities
- **Hardware Acceleration**: M1 GPU and memory optimization

### Extensive Logging
- **tprint Integration**: Comprehensive logging throughout all functions and pipeline stages
- **Performance Monitoring**: Detailed timing and memory usage tracking
- **Stage-by-stage Progress**: Clear visibility into pipeline execution
- **Error Handling**: Detailed error messages and recovery information

### Utility Integration
- **common_operations.py**: Safe mathematical operations and data processing
- **math_validation.py**: Math-safe operations for robust calculations
- **matrix_operations/**: Vectorized computations and GPU acceleration
- **hardware/**: M1 GPU, memory, and CPU optimization
- **ml_common/**: ML utilities including Bayesian TPE optimization
- **data/**: Data loading, validation, and serialization utilities

## 📁 Module Structure

```
feature_interaction_generation/
├── optimized_interaction_orchestrator.py    # Main orchestrator
├── interactive_feature_generation_component.py  # Sub-pipeline integration
├── example_optimized_usage.py              # Usage examples
├── test_integration.py                     # Integration tests
├── README.md                               # This file
└── feature_engineering_roadmap/                    # Feature engineering components
    ├── assembly_dag.py                     # Assembly DAG orchestrator
    ├── feature_registry.py                 # Feature registry
    ├── lookback_selection.py               # Lookback optimization
    ├── transforms.py                       # Transform system
    └── interactions.py                     # Interaction engine
```

## 🔧 Core Components

### 1. OptimizedInteractionOrchestrator

The main orchestrator that coordinates the entire pipeline:

```python
from .optimized_interaction_orchestrator import (
    OptimizedInteractionOrchestrator, 
    OptimizedInteractionConfig,
    generate_optimized_interaction_features
)

# Create configuration
config = OptimizedInteractionConfig(
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="15m",
    feature_budget_pre=120,
    interactions_cap=15,
    enable_matrix_optimization=True,
    enable_hardware_optimization=True
)

# Generate features
result = await generate_optimized_interaction_features(
    training_input, pipeline_state, config
)
```

### 2. InteractiveFeatureGenerationComponent

Sub-pipeline integration component:

```python
from .interactive_feature_generation_component import (
    InteractiveFeatureGenerationComponent,
    InteractiveFeatureGenerationConfig,
    execute_interactive_feature_generation
)

# Create component
component = InteractiveFeatureGenerationComponent(config)

# Execute
result = await component.execute(training_input, pipeline_state)
```

### 3. Pipeline Stages

The pipeline executes in 9 stages:

1. **Initialization**: Validate inputs and setup
2. **Feature Engineering**: Generate parent features from market data
3. **Lookback Optimization**: Select optimal lookback periods
4. **Transform Application**: Apply transforms to parent features
5. **Interaction Generation**: Create interaction features
6. **Cross-timeframe Features**: Generate multi-timeframe features
7. **Final Assembly**: Combine and select final features
8. **Validation**: Validate generated features
9. **Completion**: Return results and cleanup

## 🎯 Usage Examples

### Basic Usage

```python
import asyncio
import pandas as pd
from .optimized_interaction_orchestrator import generate_optimized_interaction_features

async def main():
    # Create sample data
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    # Create targets
    targets = {1: data['close'].pct_change().shift(-1)}
    
    # Generate features
    result = await generate_optimized_interaction_features(
        {'data': data, 'targets': targets},
        {'symbol': 'ETHUSDT', 'exchange': 'binance', 'timeframe': '15m'}
    )
    
    print(f"Generated {len(result.feature_names)} features")
    print(f"Selected {len(result.selected_features)} features")
```

### Sub-pipeline Integration

```python
from ..sub_pipeline import PreTrainingSubPipeline, SubPipelineConfig, ExecutionMode

# Create sub-pipeline
pipeline = PreTrainingSubPipeline()

# Configure
config = SubPipelineConfig(
    mode=ExecutionMode.FULL,
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="15m",
    custom_params={
        'feature_budget_pre': 120,
        'interactions_cap': 15,
        'enable_matrix_optimization': True,
        'enable_hardware_optimization': True
    }
)

# Execute roadmap feature generation
result = await pipeline._execute_interactive_feature_generation(config)
```

## 🔧 Configuration Options

### OptimizedInteractionConfig

```python
@dataclass
class OptimizedInteractionConfig:
    # Basic configuration
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"
    data_dir: str = "historical_data"
    
    # Feature generation
    feature_budget_pre: int = 120
    feature_budget_post: Tuple[int, int] = (30, 60)
    interactions_cap: int = 15
    transforms_per_parent: int = 1
    lookback_ceiling_minutes: int = 120
    latency_budget_ms: int = 50
    
    # Optimization
    enable_matrix_optimization: bool = True
    enable_hardware_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 1000
    
    # Validation
    enable_validation: bool = True
    validation_threshold: float = 0.02
    
    # Logging
    verbose_logging: bool = True
    log_performance: bool = True
```

## 📊 Performance Features

### Matrix Operations Optimization
- **Vectorized Processing**: Uses matrix_operations/ for efficient computations
- **GPU Acceleration**: M1 GPU support for matrix operations
- **Batch Processing**: Optimized batch processing for large datasets
- **Memory Optimization**: Efficient memory usage patterns

### Hardware Acceleration
- **M1 GPU Manager**: Automatic GPU detection and utilization
- **Memory Optimizer**: M1-specific memory optimization
- **CPU Optimizer**: CPU-specific optimizations
- **Performance Monitoring**: Real-time performance metrics

### Parallel Processing
- **Multi-threading**: Parallel feature generation
- **Async Operations**: Non-blocking pipeline execution
- **Resource Management**: Efficient resource utilization
- **Scalability**: Configurable worker counts

## 🧪 Testing

### Running Tests

```bash
# Run integration tests
python test_integration.py

# Run specific test class
python -m pytest test_integration.py::TestOptimizedInteractionOrchestrator -v

# Run with coverage
python -m pytest test_integration.py --cov=. --cov-report=html
```

### Test Coverage

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Performance validation
- **Error Handling**: Error scenarios and recovery
- **Memory Tests**: Memory usage validation

## 📈 Performance Monitoring

### Built-in Metrics

The pipeline automatically tracks:
- **Execution Time**: Per-stage and total execution time
- **Memory Usage**: Peak and average memory consumption
- **Feature Counts**: Generated, selected, and interaction features
- **Hardware Usage**: CPU, GPU, and memory utilization
- **Quality Scores**: Data quality and feature quality metrics

### Logging Output

```
[2024-01-11 06:30:15.123] SUCCESS: 🚀 Starting optimized interaction feature generation
[2024-01-11 06:30:15.124] INFO: 🔧 Stage 1: Initialization
[2024-01-11 06:30:15.125] DEBUG: Validating inputs...
[2024-01-11 06:30:15.126] SUCCESS: ✅ Input validation passed
[2024-01-11 06:30:15.127] PERFORMANCE: Initialization took 0.003s
[2024-01-11 06:30:15.128] INFO: 🔧 Stage 2: Feature Engineering
[2024-01-11 06:30:15.129] DEBUG: Building parent features using assembly DAG...
[2024-01-11 06:30:15.130] SUCCESS: ✅ Generated 45 parent features
[2024-01-11 06:30:15.131] PERFORMANCE: Feature Engineering took 0.002s
...
[2024-01-11 06:30:15.200] SUCCESS: ✅ Feature generation completed in 0.077s
[2024-01-11 06:30:15.201] INFO: 📊 Generated 120 total features
[2024-01-11 06:30:15.202] INFO: 🎯 Selected 80 features
[2024-01-11 06:30:15.203] INFO: 🔗 Generated 15 interactions
[2024-01-11 06:30:15.204] INFO: ⏰ Generated 25 cross-timeframe features
[2024-01-11 06:30:15.205] INFO: 💾 Memory usage: 45.67 MB
[2024-01-11 06:30:15.206] INFO: ⏱️ Total execution time: 0.077s
```

## 🔗 Integration Points

### Sub-pipeline Integration
- **Component Factory**: Registered with component factory
- **Backward Compatibility**: Maintains existing interfaces
- **Configuration**: Seamless configuration passing
- **Error Handling**: Consistent error handling patterns

### Ares Launcher Integration
- **Pipeline State**: Compatible with pipeline state management
- **Artifacts**: Proper artifact management
- **Logging**: Integrated with system logging
- **Monitoring**: Performance and health monitoring

### Utility Integration
- **Common Operations**: Safe mathematical operations
- **Math Validation**: Robust calculation validation
- **Matrix Operations**: Vectorized computations
- **Hardware Utils**: M1 optimization
- **ML Common**: ML utilities and optimization
- **Data Utils**: Data loading and validation

## 🚀 Getting Started

### 1. Basic Setup

```python
# Import the main function
from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.optimized_interaction_orchestrator import generate_optimized_interaction_features

# Create your data
data = pd.DataFrame({...})  # Your market data
targets = {1: pd.Series(...)}  # Your targets

# Generate features
result = await generate_optimized_interaction_features(
    {'data': data, 'targets': targets},
    {'symbol': 'ETHUSDT', 'exchange': 'binance', 'timeframe': '15m'}
)
```

### 2. Advanced Configuration

```python
from .optimized_interaction_orchestrator import OptimizedInteractionConfig

# Create custom configuration
config = OptimizedInteractionConfig(
    symbol="BTCUSDT",
    exchange="binance",
    timeframe="5m",
    feature_budget_pre=200,
    interactions_cap=25,
    enable_matrix_optimization=True,
    enable_hardware_optimization=True,
    max_workers=8,
    verbose_logging=True
)

# Use with custom config
result = await generate_optimized_interaction_features(
    training_input, pipeline_state, config
)
```

### 3. Sub-pipeline Usage

```python
from ..sub_pipeline import PreTrainingSubPipeline, SubPipelineConfig, ExecutionMode

# Create and configure sub-pipeline
pipeline = PreTrainingSubPipeline()
config = SubPipelineConfig(
    mode=ExecutionMode.FULL,
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="15m",
    custom_params={
        'feature_budget_pre': 120,
        'interactions_cap': 15,
        'enable_matrix_optimization': True
    }
)

# Execute roadmap feature generation
result = await pipeline._execute_interactive_feature_generation(config)
```

## 📚 Additional Resources

- **Example Usage**: See `example_optimized_usage.py` for comprehensive examples
- **Integration Tests**: See `test_integration.py` for testing patterns
- **Feature Engineering**: See `feature_engineering_roadmap/` for component details
- **Sub-pipeline**: See `../sub_pipeline.py` for integration details

## 🤝 Contributing

When contributing to this module:

1. **Follow the logging patterns**: Use tprint for all logging
2. **Add performance monitoring**: Track execution time and memory usage
3. **Include error handling**: Comprehensive error handling and recovery
4. **Write tests**: Add tests for new functionality
5. **Update documentation**: Keep this README up to date

## 📄 License

This module is part of the Ares trading system and follows the same licensing terms.