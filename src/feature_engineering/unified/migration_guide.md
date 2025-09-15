# Unified Feature Generation System - Migration Guide

This guide helps you migrate from the scattered feature generation implementations to the unified system while maintaining full backwards compatibility.

## Overview

The unified feature generation system provides:
- **Single source of truth** for all feature generation
- **Backwards compatibility** with existing implementations
- **Intelligent orchestration** with dependency resolution
- **Performance optimization** with parallel processing
- **Quality assurance** with comprehensive validation
- **Easy extensibility** with plugin architecture

## Quick Start

### Basic Usage

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

if result.success:
    features = result.features
    print(f"Generated {len(features.columns)} features")
else:
    print(f"Feature generation failed: {result.errors}")
```

### Using Specific Pipelines

```python
# Use predefined pipeline
result = await orchestrator.generate_features(
    data, 
    pipeline_name="basic_indicators"
)

# Use specific generators
result = await orchestrator.generate_features(
    data,
    generator_names=["technical_indicators", "statistical_features"]
)
```

## Migration Strategies

### 1. Gradual Migration (Recommended)

Start by using the backwards compatibility layer to wrap existing implementations:

```python
from src.feature_engineering.unified import BackwardsCompatibilityLayer, FeatureOrchestrator

# Initialize orchestrator
orchestrator = FeatureOrchestrator(OrchestrationConfig())
await orchestrator.initialize()

# Create compatibility layer
compatibility = BackwardsCompatibilityLayer(orchestrator)
await compatibility.initialize()

# Use existing analyst system through compatibility layer
result = await compatibility.generate_features_legacy(
    data, 
    method="analyst"
)
```

### 2. Direct Replacement

Replace existing feature generation calls directly:

**Before:**
```python
from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator

orchestrator = FeatureEngineeringOrchestrator(config)
features = await orchestrator.generate_all_features(klines_df, agg_trades_df, futures_df, sr_levels)
```

**After:**
```python
from src.feature_engineering.unified import FeatureOrchestrator, OrchestrationConfig

orchestrator = FeatureOrchestrator(OrchestrationConfig())
await orchestrator.initialize()
result = await orchestrator.generate_features(klines_df)
features = result.features
```

### 3. Custom Generator Implementation

Create custom generators for specific needs:

```python
from src.feature_engineering.unified import FeatureGenerator, FeatureGeneratorConfig, FeatureCategory

class CustomFeatureGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureGeneratorConfig(
            name="custom_features",
            category=FeatureCategory.CUSTOM,
            enabled=True
        )
        super().__init__(config)
    
    async def initialize(self) -> bool:
        # Initialize your generator
        self._is_initialized = True
        return True
    
    async def generate_features(self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> FeatureGenerationResult:
        # Implement your feature generation logic
        features = pd.DataFrame()
        # ... your logic here ...
        
        return FeatureGenerationResult(
            success=True,
            features=features
        )
    
    def get_required_columns(self) -> List[str]:
        return ["open", "high", "low", "close", "volume"]
    
    def get_output_columns(self) -> List[str]:
        return ["custom_feature_1", "custom_feature_2"]

# Register and use
from src.feature_engineering.unified import register_feature_generator

generator = CustomFeatureGenerator()
register_feature_generator("custom", generator, generator.config)
```

## Backwards Compatibility

### Existing Analyst System

The unified system provides seamless integration with the existing analyst feature engineering:

```python
# Old way
from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator

# New way (with compatibility)
from src.feature_engineering.unified import BackwardsCompatibilityLayer

compatibility = BackwardsCompatibilityLayer(orchestrator)
result = await compatibility.generate_features_legacy(data, method="analyst")
```

### Existing ML Common System

```python
# Old way
from src.utils.ml_common.feature_selection import FeatureSelectionFramework

# New way (with compatibility)
result = await compatibility.generate_features_legacy(data, method="ml_common")
```

### Legacy Function Wrapping

Wrap existing functions without modification:

```python
from src.feature_engineering.unified import wrap_legacy_function

def my_legacy_feature_function(data: pd.DataFrame) -> pd.DataFrame:
    # Your existing function
    return features

# Wrap it
wrapped_generator = wrap_legacy_function(
    my_legacy_feature_function,
    required_columns=["open", "high", "low", "close", "volume"],
    output_columns=["my_feature_1", "my_feature_2"],
    name="my_legacy_features"
)

# Use it
result = await wrapped_generator.generate_features(data)
```

## Configuration

### Orchestration Configuration

```python
from src.feature_engineering.unified import OrchestrationConfig

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
from src.feature_engineering.unified import FeatureGeneratorConfig, FeatureCategory, FeaturePriority

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
    },
    memory_limit_mb=200,
    timeout_seconds=30
)
```

## Validation and Quality Assurance

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
    print(f"Validation errors: {validation_result.errors}")
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

# Set baseline
checker.set_baseline(baseline_features)

# Check consistency
is_consistent, details = await checker.check_consistency(current_features)

if is_consistent:
    print("Features are consistent with baseline")
else:
    print(f"Consistency issues: {details}")
```

## Performance Optimization

### Parallel Processing

The unified system automatically uses parallel processing when beneficial:

```python
# Enable parallel processing
config = OrchestrationConfig(
    enable_parallel_processing=True,
    max_parallel_generators=4
)

# Generators will run in parallel when possible
result = await orchestrator.generate_features(data)
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

## Error Handling

### Comprehensive Error Handling

The unified system provides comprehensive error handling:

```python
result = await orchestrator.generate_features(data)

if result.success:
    features = result.features
    print(f"Generated {len(features.columns)} features")
else:
    print("Feature generation failed:")
    for error in result.errors:
        print(f"  - {error}")
    
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
```

### Retry Logic

Failed generators are automatically retried:

```python
config = OrchestrationConfig(
    retry_failed_generators=True,
    max_retries=3,
    retry_delay_seconds=1.0
)
```

## Monitoring and Debugging

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

### Pipeline Information

```python
# List pipelines
pipelines = orchestrator.list_pipelines()
print(f"Available pipelines: {pipelines}")

# Get pipeline info
info = orchestrator.get_pipeline_info("basic_indicators")
print(f"Pipeline info: {info}")
```

## Best Practices

### 1. Use Appropriate Categories

Assign generators to appropriate categories for better organization:

```python
config = FeatureGeneratorConfig(
    name="my_generator",
    category=FeatureCategory.TECHNICAL_INDICATORS,  # Use appropriate category
    # ...
)
```

### 2. Set Proper Dependencies

Define dependencies between generators:

```python
config = FeatureGeneratorConfig(
    name="advanced_generator",
    dependencies=["basic_generator", "statistical_generator"],
    # ...
)
```

### 3. Implement Proper Validation

Always validate input and output in generators:

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

Set realistic performance targets:

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

Implement proper error handling:

```python
try:
    result = await generator.generate_features(data)
    if not result.success:
        self.logger.warning(f"Generator {generator.config.name} failed: {result.errors}")
except Exception as e:
    self.logger.error(f"Unexpected error in generator {generator.config.name}: {e}")
```

## Troubleshooting

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

## Conclusion

The unified feature generation system provides a robust, scalable, and maintainable solution for feature generation across your trading system. By following this migration guide, you can gradually transition from scattered implementations to a unified system while maintaining full backwards compatibility.

For more information, see the API documentation and examples in the `src/feature_engineering/unified/` directory.