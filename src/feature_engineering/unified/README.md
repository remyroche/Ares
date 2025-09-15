# Unified Feature Generation System

A comprehensive, unified feature generation system that provides a single source of truth for all feature generation across the Ares trading system, with full backwards compatibility and intelligent orchestration.

## 🚀 Key Features

- **Single Source of Truth**: Centralized feature generation with consistent interfaces
- **Backwards Compatibility**: Seamless integration with existing feature generation systems
- **Intelligent Orchestration**: Automatic dependency resolution and parallel processing
- **Performance Optimization**: Hardware-aware optimization and caching
- **Quality Assurance**: Comprehensive validation and quality metrics
- **Easy Extensibility**: Plugin architecture for custom feature generators
- **Comprehensive Testing**: Full test coverage with validation

## 📁 Architecture

```
src/feature_engineering/unified/
├── __init__.py                 # Main package exports
├── core.py                     # Core interfaces and base classes
├── registry.py                 # Feature generator registry system
├── orchestrator.py             # Unified feature orchestrator
├── compatibility.py            # Backwards compatibility layer
├── validation.py               # Validation and quality assurance
├── generators/                 # Concrete generator implementations
│   ├── __init__.py
│   ├── technical_indicators.py
│   ├── statistical_features.py
│   └── ...
├── tests/                      # Comprehensive test suite
│   ├── __init__.py
│   └── test_unified_system.py
├── example_usage.py            # Usage examples
├── migration_guide.md          # Migration guide
└── README.md                   # This file
```

## 🏗️ Core Components

### 1. FeatureGenerator Interface

The base interface that all feature generators must implement:

```python
from src.feature_engineering.unified import FeatureGenerator, FeatureGeneratorConfig

class MyFeatureGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureGeneratorConfig(
            name="my_generator",
            category=FeatureCategory.CUSTOM,
            priority=FeaturePriority.MEDIUM
        )
        super().__init__(config)
    
    async def initialize(self) -> bool:
        # Initialize your generator
        return True
    
    async def generate_features(self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> FeatureGenerationResult:
        # Generate your features
        return FeatureGenerationResult(success=True, features=features)
    
    def get_required_columns(self) -> List[str]:
        return ["open", "high", "low", "close", "volume"]
    
    def get_output_columns(self) -> List[str]:
        return ["my_feature_1", "my_feature_2"]
```

### 2. Feature Registry

Dynamic discovery and management of feature generators:

```python
from src.feature_engineering.unified import get_registry, register_feature_generator

# Register a generator
register_feature_generator("my_generator", MyFeatureGenerator, config)

# Get generator info
registry = get_registry()
generator_info = registry.get_generator("my_generator")

# List available generators
generators = registry.list_available_generators()
```

### 3. Feature Orchestrator

Intelligent coordination of feature generation:

```python
from src.feature_engineering.unified import FeatureOrchestrator, OrchestrationConfig

# Initialize orchestrator
config = OrchestrationConfig(
    enable_parallel_processing=True,
    max_parallel_generators=4,
    enable_validation=True
)

orchestrator = FeatureOrchestrator(config)
await orchestrator.initialize()

# Generate features
result = await orchestrator.generate_features(data)
```

### 4. Backwards Compatibility

Seamless integration with existing systems:

```python
from src.feature_engineering.unified import BackwardsCompatibilityLayer

# Initialize compatibility layer
compatibility = BackwardsCompatibilityLayer(orchestrator)
await compatibility.initialize()

# Use existing analyst system
result = await compatibility.generate_features_legacy(data, method="analyst")

# Wrap legacy functions
wrapped_generator = wrap_legacy_function(
    my_legacy_function,
    required_columns=["close"],
    output_columns=["legacy_feature"]
)
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from src.feature_engineering.unified import FeatureOrchestrator, OrchestrationConfig

async def main():
    # Initialize orchestrator
    config = OrchestrationConfig(enable_validation=True)
    orchestrator = FeatureOrchestrator(config)
    await orchestrator.initialize()
    
    # Generate features
    result = await orchestrator.generate_features(data)
    
    if result.success:
        features = result.features
        print(f"Generated {len(features.columns)} features")
    else:
        print(f"Failed: {result.errors}")

asyncio.run(main())
```

### Using Specific Generators

```python
# Use specific generators
result = await orchestrator.generate_features(
    data,
    generator_names=["technical_indicators", "statistical_features"]
)

# Use predefined pipeline
result = await orchestrator.generate_features(
    data,
    pipeline_name="basic_indicators"
)
```

### Custom Generator

```python
from src.feature_engineering.unified import FeatureGenerator, FeatureGeneratorConfig, FeatureCategory

class CustomGenerator(FeatureGenerator):
    # Implementation as shown above
    pass

# Register and use
register_feature_generator("custom", CustomGenerator, config)
result = await orchestrator.generate_features(data, generator_names=["custom"])
```

## 🔧 Configuration

### Orchestration Configuration

```python
config = OrchestrationConfig(
    enable_parallel_processing=True,      # Enable parallel processing
    max_parallel_generators=4,           # Max parallel generators
    enable_dependency_resolution=True,    # Resolve dependencies
    enable_performance_optimization=True, # Optimize performance
    enable_caching=True,                 # Enable caching
    cache_ttl_seconds=3600,             # Cache TTL
    enable_validation=True,              # Enable validation
    enable_quality_checks=True,          # Enable quality checks
    timeout_seconds=300,                 # Timeout for generation
    memory_limit_mb=2048,                # Memory limit
    retry_failed_generators=True,        # Retry failed generators
    max_retries=3,                       # Max retries
    retry_delay_seconds=1.0              # Retry delay
)
```

### Generator Configuration

```python
config = FeatureGeneratorConfig(
    name="my_generator",
    category=FeatureCategory.TECHNICAL_INDICATORS,
    priority=FeaturePriority.HIGH,
    enabled=True,
    parameters={
        "indicator_periods": [5, 10, 20, 50],
        "custom_param": "value"
    },
    dependencies=["other_generator"],
    output_columns=["feature_1", "feature_2"],
    validation_rules={
        "max_nan_ratio": 0.1,
        "min_variance": 1e-6
    },
    performance_targets={
        "max_duration_ms": 1000,
        "max_memory_mb": 100
    }
)
```

## 🔍 Validation and Quality

### Feature Validation

```python
from src.feature_engineering.unified import FeatureValidator

validator = FeatureValidator()
await validator.initialize()

# Validate features
validation_result = await validator.validate_features(result)

if validation_result.is_valid:
    print("Features are valid")
else:
    print(f"Errors: {validation_result.errors}")
    print(f"Warnings: {validation_result.warnings}")
```

### Quality Metrics

```python
from src.feature_engineering.unified import FeatureQualityMetrics

quality_calculator = FeatureQualityMetrics()
metrics = await quality_calculator.calculate_quality_metrics(result)

print(f"Completeness: {metrics.completeness:.2f}")
print(f"Consistency: {metrics.consistency:.2f}")
print(f"Stability: {metrics.stability:.2f}")
print(f"Performance: {metrics.performance:.2f}")
print(f"Overall Score: {metrics.overall_score:.2f}")
```

### Consistency Checking

```python
from src.feature_engineering.unified import FeatureConsistencyChecker

checker = FeatureConsistencyChecker()
checker.set_baseline(baseline_features)

is_consistent, details = await checker.check_consistency(current_features)
```

## 🔄 Migration Guide

### Gradual Migration (Recommended)

1. **Start with Backwards Compatibility**:
   ```python
   from src.feature_engineering.unified import BackwardsCompatibilityLayer
   
   compatibility = BackwardsCompatibilityLayer(orchestrator)
   await compatibility.initialize()
   
   # Use existing systems through compatibility layer
   result = await compatibility.generate_features_legacy(data, method="analyst")
   ```

2. **Wrap Legacy Functions**:
   ```python
   wrapped_generator = wrap_legacy_function(
       my_legacy_function,
       required_columns=["close"],
       output_columns=["legacy_feature"]
   )
   ```

3. **Gradually Replace**:
   ```python
   # Old way
   from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
   
   # New way
   from src.feature_engineering.unified import FeatureOrchestrator
   ```

### Direct Replacement

Replace existing calls directly:

```python
# Before
orchestrator = FeatureEngineeringOrchestrator(config)
features = await orchestrator.generate_all_features(klines_df, agg_trades_df, futures_df, sr_levels)

# After
orchestrator = FeatureOrchestrator(OrchestrationConfig())
await orchestrator.initialize()
result = await orchestrator.generate_features(klines_df)
features = result.features
```

## 📊 Performance Optimization

### Parallel Processing

The system automatically uses parallel processing when beneficial:

```python
config = OrchestrationConfig(
    enable_parallel_processing=True,
    max_parallel_generators=4
)
```

### Caching

Enable caching for repeated feature generation:

```python
config = OrchestrationConfig(
    enable_caching=True,
    cache_ttl_seconds=3600  # 1 hour cache
)
```

### Memory Management

Set memory limits for generators:

```python
config = FeatureGeneratorConfig(
    name="memory_intensive_generator",
    memory_limit_mb=512,
    timeout_seconds=60
)
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest src/feature_engineering/unified/tests/

# Run specific test
pytest src/feature_engineering/unified/tests/test_unified_system.py::TestFeatureGenerator

# Run with coverage
pytest --cov=src.feature_engineering.unified src/feature_engineering/unified/tests/
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end system testing
- **Validation Tests**: Feature validation testing
- **Performance Tests**: Performance and scalability testing
- **Compatibility Tests**: Backwards compatibility testing

## 📈 Monitoring and Debugging

### Performance Metrics

```python
# Get orchestrator metrics
metrics = orchestrator.get_performance_metrics()
print(f"Total generations: {metrics.get('total_generations', 0)}")
print(f"Last duration: {metrics.get('last_generation_duration', 0):.2f}s")

# Get generator metrics
generator_info = generator.get_info()
print(f"Generator performance: {generator_info['performance_metrics']}")
```

### Registry Information

```python
# Get registry statistics
stats = orchestrator.get_registry_stats()
print(f"Total generators: {stats['total_generators']}")
print(f"Enabled generators: {stats['enabled_generators']}")

# List available generators
generators = orchestrator.registry.list_available_generators()
print(f"Available generators: {generators}")
```

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.getLogger("FeatureOrchestrator").setLevel(logging.DEBUG)
logging.getLogger("FeatureGenerator").setLevel(logging.DEBUG)
```

## 🔧 Advanced Usage

### Custom Validation Rules

```python
from src.feature_engineering.unified import ValidationRule

def custom_validation_rule(result, generator, params):
    # Your custom validation logic
    return {"passed": True, "message": "Custom validation passed"}

rule = ValidationRule(
    name="custom_rule",
    description="Custom validation rule",
    check_function=custom_validation_rule,
    severity="warning"
)

validator.add_validation_rule(rule)
```

### Custom Pipelines

```python
# Create custom pipeline
success = orchestrator.create_custom_pipeline(
    "my_pipeline",
    ["generator1", "generator2", "generator3"]
)

# Use custom pipeline
result = await orchestrator.generate_features(data, pipeline_name="my_pipeline")
```

### Error Handling

```python
try:
    result = await orchestrator.generate_features(data)
    if result.success:
        features = result.features
    else:
        print(f"Generation failed: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 🎯 Best Practices

### 1. Use Appropriate Categories

```python
config = FeatureGeneratorConfig(
    name="my_generator",
    category=FeatureCategory.TECHNICAL_INDICATORS,  # Use appropriate category
    # ...
)
```

### 2. Set Proper Dependencies

```python
config = FeatureGeneratorConfig(
    name="advanced_generator",
    dependencies=["basic_generator", "statistical_generator"],
    # ...
)
```

### 3. Implement Proper Validation

```python
async def generate_features(self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> FeatureGenerationResult:
    # Validate input
    is_valid, errors = self.validate_input(data)
    if not is_valid:
        return FeatureGenerationResult(success=False, errors=errors)
    
    # Generate features
    features = self._generate_features(data)
    
    # Validate output
    is_valid, errors = self.validate_output(features)
    if not is_valid:
        return FeatureGenerationResult(success=False, features=features, errors=errors)
    
    return FeatureGenerationResult(success=True, features=features)
```

### 4. Use Performance Targets

```python
config = FeatureGeneratorConfig(
    name="my_generator",
    performance_targets={
        "max_duration_ms": 1000,  # 1 second max
        "max_memory_mb": 100      # 100MB max
    }
)
```

### 5. Handle Errors Gracefully

```python
try:
    result = await generator.generate_features(data)
    if not result.success:
        self.logger.warning(f"Generator {generator.config.name} failed: {result.errors}")
except Exception as e:
    self.logger.error(f"Unexpected error in generator {generator.config.name}: {e}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Generator not found**: Ensure the generator is registered
2. **Missing dependencies**: Check that all required generators are available
3. **Memory issues**: Reduce memory limits or use fewer parallel generators
4. **Timeout errors**: Increase timeout limits or optimize generator performance
5. **Validation failures**: Check input data quality and generator validation rules

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.getLogger("FeatureOrchestrator").setLevel(logging.DEBUG)
logging.getLogger("FeatureGenerator").setLevel(logging.DEBUG)
```

### Performance Profiling

Use the built-in performance monitoring:

```python
# Get detailed performance metrics
metrics = orchestrator.get_performance_metrics()
print(f"Performance metrics: {metrics}")

# Get generator-specific metrics
for generator in orchestrator.registry.get_enabled_generators():
    info = generator.get_info()
    print(f"{generator.config.name}: {info['performance_metrics']}")
```

## 📚 Examples

See `example_usage.py` for comprehensive examples including:

- Basic usage
- Custom generators
- Parallel processing
- Backwards compatibility
- Validation and quality
- Pipeline creation
- Error handling

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure backwards compatibility
5. Follow the established error handling patterns

## 📄 License

This project is part of the Ares trading system and follows the same licensing terms.

## 🆘 Support

For questions, issues, or contributions, please refer to the main Ares project documentation or create an issue in the project repository.

---

**The Unified Feature Generation System provides a robust, scalable, and maintainable solution for feature generation across your trading system. By following this guide, you can leverage the full power of the unified system while maintaining compatibility with existing implementations.**