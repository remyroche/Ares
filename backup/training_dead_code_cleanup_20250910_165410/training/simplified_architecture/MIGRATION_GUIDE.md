# Migration Guide: From Monolithic to Simplified Architecture

## Overview

This guide provides step-by-step instructions for migrating from the current monolithic architecture to the new simplified, maintainable architecture. The migration addresses all the pain points identified in the current system.

## Migration Benefits

### Before Migration
- **Monolithic files**: `enhanced_training_manager.py` (3,079 lines)
- **Complex dependencies**: Hidden imports and circular dependencies
- **Hard to test**: Tightly coupled components
- **Difficult to extend**: Code changes required for new features
- **Poor maintainability**: Mixed concerns in single files

### After Migration
- **Modular components**: Single responsibility principle
- **Dependency injection**: Explicit service management
- **Easy testing**: Isolated, mockable components
- **Configuration-driven**: YAML-based pipeline configuration
- **Extensible**: Plugin architecture for new components

## Migration Steps

### Step 1: Set Up New Architecture

1. **Install new dependencies** (if any):
   ```bash
   pip install pyyaml  # For YAML configuration support
   ```

2. **Create configuration directory**:
   ```bash
   mkdir -p config/pipelines
   ```

3. **Copy new architecture files**:
   ```bash
   cp -r src/training/simplified_architecture/* src/training/
   ```

### Step 2: Create Your First Pipeline Configuration

Create a basic pipeline configuration file:

```yaml
# config/basic_pipeline.yaml
name: My_First_Pipeline
version: 1.0.0
description: Basic ML pipeline for testing

global_settings:
  data_source:
    type: file
    format: parquet
  model:
    type: lightgbm

steps:
  - name: data_loading
    class_name: DataCollectionStep
    parameters:
      source: data/raw/training_data.parquet
      required_columns: [open, high, low, close, volume]
    dependencies: []

  - name: feature_engineering
    class_name: FeatureEngineeringStep
    dependencies: [data_loading]
    parameters:
      feature_types: [technical, statistical]

  - name: model_training
    class_name: ModelTrainingStep
    dependencies: [feature_engineering]
    parameters:
      model_type: lightgbm
```

### Step 3: Run Your First Pipeline

```python
from src.training.simplified_architecture.enhanced_pipeline_orchestrator import create_pipeline

# Create and run pipeline
pipeline = create_pipeline("config/basic_pipeline.yaml")
result = await pipeline.run()

if result.is_success:
    print("Pipeline completed successfully!")
    print(f"Duration: {result.duration:.2f} seconds")
else:
    print(f"Pipeline failed: {result.errors}")
```

### Step 4: Migrate Existing Components

#### 4.1 Data Components Migration

**Before (Monolithic)**:
```python
# In enhanced_training_manager.py (lines 1-100)
class TrainingManager:
    def __init__(self, config):
        # 100+ lines of initialization
        self.data_manager = DataManager()
        self.feature_engineer = FeatureEngineer()
        # ... many more dependencies
```

**After (Simplified)**:
```python
# In migrated_components/data_components.py
class DataCollectionStep(BasePipelineStep, IDataStep):
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        # Clean, focused implementation
        data = await self.load_data(source, **kwargs)
        if not await self.validate_data(data):
            raise ValueError('Data validation failed')
        return {'data': data, 'metadata': {...}}
```

#### 4.2 Feature Engineering Migration

**Before**:
```python
# In step06_feature_engineering.py (38KB file)
class FeatureEngineeringStep:
    def __init__(self):
        # Complex initialization with many dependencies
        self.technical_indicators = TechnicalIndicators()
        self.statistical_features = StatisticalFeatures()
        # ... 1000+ lines of mixed concerns
```

**After**:
```python
# In migrated_components/feature_components.py
class FeatureEngineeringStep(BasePipelineStep, IFeatureStep):
    async def _execute_impl(self, **kwargs) -> pd.DataFrame:
        data = kwargs['data']
        features = await self.engineer_features(data)
        return features
```

#### 4.3 Training Components Migration

**Before**:
```python
# In step09_hmm_based_training.py (55KB file)
class HMMBasedTrainingStep:
    def __init__(self):
        # Complex initialization
        self.hmm_trainer = HMMTrainer()
        self.model_validator = ModelValidator()
        # ... 1000+ lines of mixed concerns
```

**After**:
```python
# In migrated_components/training_components.py
class HMMTrainingStep(BasePipelineStep, ITrainingStep):
    async def _execute_impl(self, **kwargs) -> Any:
        features = kwargs['features']
        labels = kwargs['labels']
        model = await self.train_model(features, labels)
        return model
```

### Step 5: Update Your Code

#### 5.1 Replace Direct Instantiation

**Before**:
```python
# Old way - direct instantiation
training_manager = TrainingManager(config)
result = await training_manager.execute_enhanced_training(input_data)
```

**After**:
```python
# New way - configuration-driven
pipeline = create_pipeline("config/my_pipeline.yaml")
result = await pipeline.run()
```

#### 5.2 Update Configuration

**Before**:
```python
# Hardcoded configuration
config = {
    'symbol': 'BTCUSDT',
    'exchange': 'binance',
    'timeframe': '1h',
    'lookback_days': 30,
    'model_type': 'lightgbm',
    'hyperparameters': {...}
}
```

**After**:
```yaml
# config/trading_pipeline.yaml
name: Trading_Pipeline
global_settings:
  data_source:
    type: exchange
    exchange: binance
  model:
    type: lightgbm
    hyperparameters:
      n_estimators: 1000
      learning_rate: 0.1

steps:
  - name: data_collection
    class_name: DataCollectionStep
    parameters:
      symbol: BTCUSDT
      timeframe: 1h
      lookback_days: 30
```

### Step 6: Testing Your Migration

#### 6.1 Run Unit Tests

```bash
cd src/training/simplified_architecture
python -m pytest tests/ -v
```

#### 6.2 Test Individual Components

```python
# Test data collection step
from src.training.simplified_architecture.migrated_components.data_components import DataCollectionStep
from src.training.simplified_architecture.enhanced_interfaces import StepConfig

config = StepConfig(
    name="test_data_collection",
    parameters={'source': 'data/test.parquet'}
)

step = DataCollectionStep(config)
result = await step.execute(source='data/test.parquet')
assert result.is_success
```

#### 6.3 Integration Testing

```python
# Test complete pipeline
pipeline = create_pipeline("config/test_pipeline.yaml")
result = await pipeline.run()

assert result.status == PipelineStatus.COMPLETED
assert len(result.step_results) > 0
```

## Configuration Examples

### Basic ML Pipeline

```yaml
name: Basic_ML_Pipeline
version: 1.0.0

global_settings:
  data_source:
    type: file
    format: parquet
  model:
    type: lightgbm

steps:
  - name: data_loading
    class_name: DataCollectionStep
    parameters:
      source: data/raw/training_data.parquet
      required_columns: [open, high, low, close, volume]
      min_rows: 1000

  - name: feature_engineering
    class_name: FeatureEngineeringStep
    dependencies: [data_loading]
    parameters:
      feature_types: [technical, statistical]
      technical_indicators: [sma, ema, rsi, macd]

  - name: model_training
    class_name: ModelTrainingStep
    dependencies: [feature_engineering]
    parameters:
      model_type: lightgbm
      hyperparameters:
        n_estimators: 1000
        learning_rate: 0.1
```

### Advanced Trading Pipeline

```yaml
name: Advanced_Trading_Pipeline
version: 2.0.0

global_settings:
  data_source:
    type: exchange
    exchange: binance
  model:
    type: ensemble
    models: [lightgbm, xgboost, neural_network]

steps:
  - name: data_collection
    class_name: DataCollectionStep
    parameters:
      symbol: BTCUSDT
      timeframe: 1h
      lookback_days: 90

  - name: hmm_regime_discovery
    class_name: HMMRegimeDiscoveryStep
    dependencies: [data_collection]
    parameters:
      n_components: 4
      max_iterations: 200

  - name: feature_engineering
    class_name: FeatureEngineeringStep
    dependencies: [hmm_regime_discovery]
    parameters:
      feature_types: [technical, statistical, wavelet, regime_based]

  - name: ensemble_training
    class_name: EnsembleTrainingStep
    dependencies: [feature_engineering]
    parameters:
      ensemble_method: stacking
      base_models:
        - type: lightgbm
          hyperparameters: {n_estimators: 2000}
        - type: xgboost
          hyperparameters: {n_estimators: 2000}
        - type: neural_network
          hyperparameters: {hidden_layers: [256, 128, 64]}
```

## Troubleshooting

### Common Issues

#### 1. Configuration Validation Errors

**Error**: `Configuration validation failed: Step 0: class_name is required`

**Solution**: Ensure all steps have a `class_name` field:
```yaml
steps:
  - name: data_loading
    class_name: DataCollectionStep  # This is required
    parameters: {...}
```

#### 2. Dependency Resolution Errors

**Error**: `Service 'step_data_loading' not registered`

**Solution**: Ensure step dependencies are correctly specified:
```yaml
steps:
  - name: data_loading
    class_name: DataCollectionStep
    dependencies: []  # No dependencies

  - name: feature_engineering
    class_name: FeatureEngineeringStep
    dependencies: [data_loading]  # Depends on data_loading
```

#### 3. Circular Dependency Errors

**Error**: `Circular dependency detected: step1 -> step2 -> step1`

**Solution**: Review and fix dependency chains:
```yaml
# Wrong - circular dependency
steps:
  - name: step1
    dependencies: [step2]
  - name: step2
    dependencies: [step1]

# Correct - linear dependency
steps:
  - name: step1
    dependencies: []
  - name: step2
    dependencies: [step1]
```

### Performance Issues

#### 1. Memory Usage

**Issue**: High memory usage during pipeline execution

**Solution**: Configure resource limits:
```yaml
steps:
  - name: data_loading
    class_name: DataCollectionStep
    resource_limits:
      max_memory_mb: 2048
    parameters:
      chunk_size: 10000  # Process data in chunks
```

#### 2. Execution Time

**Issue**: Pipeline takes too long to execute

**Solution**: Enable parallel execution:
```python
# Run with parallel execution
result = await pipeline.run(parallel_execution=True)
```

### Debugging

#### 1. Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = create_pipeline("config/pipeline.yaml")
result = await pipeline.run()
```

#### 2. Check Step Results

```python
result = await pipeline.run()

for step_name, step_result in result.step_results.items():
    print(f"Step: {step_name}")
    print(f"Status: {step_result.status}")
    print(f"Duration: {step_result.duration}")
    if step_result.error:
        print(f"Error: {step_result.error}")
    print(f"Metrics: {step_result.metrics}")
```

#### 3. Validate Configuration

```python
from src.training.simplified_architecture.enhanced_config_system import ConfigurationManager

config_manager = ConfigurationManager()
config = config_manager.load_config("config/pipeline.yaml")
errors = config_manager.validate_config(config)

if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

## Migration Checklist

### Pre-Migration
- [ ] Backup current implementation
- [ ] Document current pipeline behavior
- [ ] Identify critical components to migrate first
- [ ] Set up test environment

### During Migration
- [ ] Set up new architecture files
- [ ] Create basic pipeline configuration
- [ ] Migrate data components
- [ ] Migrate feature engineering components
- [ ] Migrate training components
- [ ] Migrate validation components
- [ ] Test each component individually
- [ ] Test complete pipeline

### Post-Migration
- [ ] Run comprehensive test suite
- [ ] Performance benchmarking
- [ ] Documentation updates
- [ ] Team training
- [ ] Production deployment

## Support

For questions or issues during migration:

1. **Check the test suite**: `tests/test_migrated_components.py`
2. **Review examples**: `config/` directory
3. **Read documentation**: `README.md` and `MIGRATION_PLAN.md`
4. **Run validation**: Use configuration validation tools

## Next Steps

After successful migration:

1. **Add new components**: Use the plugin architecture
2. **Optimize performance**: Configure resource limits and parallel execution
3. **Monitor pipelines**: Use built-in monitoring and metrics
4. **Scale up**: Deploy multiple pipeline instances
5. **Extend functionality**: Add new data sources, models, and validation methods

The new architecture provides a solid foundation for future development and makes it easy to add new features without modifying existing code.