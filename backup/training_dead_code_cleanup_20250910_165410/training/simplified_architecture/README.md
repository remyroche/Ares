# Simplified ML Pipeline Architecture

This directory contains the implementation of a simplified, maintainable ML pipeline architecture that addresses the complexity issues in the original training steps.

## Key Architectural Improvements

### 1. Dependency Injection (`dependency_injection.py`)
- **Purpose**: Replace hidden imports and complex dependency chains with explicit dependency management
- **Benefits**:
  - Clear visibility of component dependencies
  - Easy testing with mock implementations
  - No circular dependencies
  - Runtime configuration of components

### 2. Standard Interfaces (`standard_interfaces.py`)
- **Purpose**: All pipeline steps follow the same interface pattern
- **Benefits**:
  - Predictable behavior across all steps
  - Easy to add new steps
  - Consistent error handling and metrics
  - Simplified orchestration

### 3. Configuration-Driven Architecture (`config_driven_architecture.py`)
- **Purpose**: Move complexity from code to configuration files
- **Benefits**:
  - Change pipeline behavior without code changes
  - Easy A/B testing of different configurations
  - Version control for pipeline configurations
  - Clear documentation of pipeline structure

### 4. Modular Components (`modular_components.py`)
- **Purpose**: Each component has a single responsibility with abstract interfaces
- **Benefits**:
  - Easy to understand and maintain
  - Components can be tested in isolation
  - Reusable across different pipelines
  - Clear separation of concerns
  - **Extensible through abstract interfaces**:
    - `IExchangeDataSource` - Base for all exchange implementations
    - `IModelTrainer` - Base for all ML model trainers
    - Factory patterns for easy extension without modifying existing code

## Usage Examples

### Basic Pipeline Usage
```python
# 1. Create configuration file (config/pipeline.yaml)
name: ML_Trading_Pipeline
version: 1.0.0
global_settings:
  data_source:
    type: exchange
    exchange: binance  # Can use: binance, coinbase, kraken, etc.
  model:
    type: lightgbm    # Can use: lightgbm, xgboost, random_forest, neural_network
steps:
  data_loading:
    class_name: DataLoadingStep
    parameters:
      symbol: BTCUSDT
      timeframe: 1h

# 2. Run the pipeline
from src.training.simplified_architecture.integrated_example import IntegratedPipeline

pipeline = IntegratedPipeline("config/pipeline.yaml")
results = await pipeline.run()
```

### Adding New Exchange Support
```python
from src.training.simplified_architecture.modular_components import (
    BaseExchangeDataSource, ExchangeDataSourceFactory
)

class MyExchangeDataSource(BaseExchangeDataSource):
    @property
    def exchange_name(self) -> str:
        return "myexchange"
    
    async def fetch_data(self, symbol, start, end):
        # Your implementation here
        pass

# Register the new exchange
ExchangeDataSourceFactory.register_exchange('myexchange', MyExchangeDataSource)

# Now it can be used in configuration
config = {
    "data_source": {
        "type": "exchange",
        "exchange": "myexchange"
    }
}
```

### Adding New Model Support
```python
from src.training.simplified_architecture.modular_components import (
    BaseModelTrainer, ModelTrainerFactory
)

class MyModelTrainer(BaseModelTrainer):
    @property
    def model_type(self) -> str:
        return "mymodel"
    
    def train(self, X, y, validation_data=None):
        # Your implementation here
        pass

# Register the new model
ModelTrainerFactory.register_trainer('mymodel', MyModelTrainer)

# Now it can be used in configuration
config = {
    "model": {
        "type": "mymodel",
        "hyperparameters": {...}
    }
}
```

## Migration from Original Architecture

### Step 5 (Labeling) Migration:
```python
# Old: Complex dependency chain
if not triple_barrier_path.exists():
    self.logger.error(f"❌ Triple barrier labels not found")
    return False

# New: Explicit dependency injection
@inject(data_source='data_source', labeler='labeler')
class LabelingStep(BasePipelineStep):
    async def _execute_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        return await self.labeler.create_labels(data)
```

### Step 7 (Matrix Operations) Migration:
```python
# Old: Multiple decorators obscuring logic
@traced(span_name="execute_labeling")
@quality_gate
@with_enhanced_mlflow_logging
@validates()
@handles_errors
@cached
@log_execution_time
async def execute_labeling(self, ...):

# New: Single standard interface
class MatrixOperationsStep(BasePipelineStep):
    async def _execute_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        # Clear, simple logic
        return self.matrix_ops.process(data)
```

### Step 9 (HMM Training) Migration:
```python
# Old: Monolithic class with many responsibilities
class EnhancedHMMBasedTrainingStep:
    # 1000+ lines of mixed concerns

# New: Separated concerns
class HMMRegimeDetector:  # Only detects regimes
class HMMFeatureSelector:  # Only selects features
class HMMModelTrainer:    # Only trains models
```

### Step 10 (Neural Network) Migration:
```python
# Old: Complex transformer architecture
class MultiTimeframeHMMEncoder(nn.Module):
    # Complex attention mechanisms

# New: Simple, configurable architecture
def create_regime_model(config: dict):
    if config['complexity'] == 'simple':
        return FeatureBasedRegimeDetector(config)
    elif config['complexity'] == 'medium':
        return SimpleLSTMModel(config)
```

## Lookahead Bias Prevention

The new architecture includes built-in safeguards:

1. **Time-aware data splitting** with explicit gaps
2. **Feature validation** to detect forward-looking calculations
3. **Pipeline-level constraints** enforcing temporal ordering
4. **Automated testing** for lookahead bias detection

## Benefits Summary

1. **Maintainability**: Clear structure makes debugging and updates easier
2. **Testability**: Each component can be tested in isolation
3. **Flexibility**: Configuration-driven approach allows easy experimentation
4. **Performance**: Simpler architecture reduces overhead
5. **Reliability**: Explicit dependencies prevent hidden failures
6. **Transparency**: Clear data flow and decision points

## Next Steps

1. Migrate existing pipeline components to new architecture
2. Add comprehensive unit tests for each component
3. Create configuration templates for common use cases
4. Build monitoring dashboard for pipeline execution
5. Implement gradual rollout strategy for production