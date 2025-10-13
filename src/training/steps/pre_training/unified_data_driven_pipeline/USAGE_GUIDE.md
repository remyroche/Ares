# Usage Guide: Unified Data-Driven Feature Pipeline

## Quick Start

### 1. **Basic Usage (Standalone)**
```python
from src.training.steps.pre_training.unified_data_driven_pipeline import process_features

# Load your data
data = pd.read_csv('market_data.csv')
targets = pd.read_csv('targets.csv')['returns']

# Process features (generates synthetic features if needed)
result = process_features(data, targets)

print(f"Selected {len(result.selected_features)} features")
print(f"Out-of-sample Sharpe: {result.out_of_sample_sharpe:.3f}")
```

### 2. **With Existing Feature Generation**
```python
from src.training.steps.pre_training.unified_data_driven_pipeline import process_with_integrated_pipeline

# Process with existing feature generation system
result = process_with_integrated_pipeline(
    data=data,
    targets=targets,
    feature_categories=['momentum', 'volatility', 'volume']
)

print(f"Generated and selected {len(result.selected_features)} features")
```

## Integration Patterns

### Pattern 1: **Feature Generation → Pipeline Selection**

Use the existing `src/feature_generation/` system to generate features, then use the unified pipeline for selection:

```python
# Step 1: Generate features using existing system
from src.feature_generation.core.factory import get_feature_bank

bank = get_feature_bank()
feature_data = {}

# Generate momentum features
momentum_generators = bank.get_generators_by_category('momentum')
for generator in momentum_generators[:10]:
    result = generator.generate(data)
    feature_data.update(result.features)

# Generate volatility features
volatility_generators = bank.get_generators_by_category('volatility')
for generator in volatility_generators[:10]:
    result = generator.generate(data)
    feature_data.update(result.features)

# Step 2: Select optimal features
from src.training.steps.pre_training.unified_data_driven_pipeline import process_features

features_df = pd.DataFrame(feature_data)
result = process_features(features_df, targets)
```

### Pattern 2: **Integrated Pipeline Class**

Use the integrated pipeline that handles both generation and selection:

```python
from src.training.steps.pre_training.unified_data_driven_pipeline import create_integrated_pipeline

# Create integrated pipeline
pipeline = create_integrated_pipeline()

# Process data (generates features and selects optimal subset)
result = pipeline.process(
    data=data,
    targets=targets,
    feature_categories=['momentum', 'volatility', 'volume', 'trend'],
    max_features_per_category=10
)

print(f"Selected {len(result.selected_features)} features")
print(f"Generation metadata: {result.generation_metadata}")
```

### Pattern 3: **Category-Specific Processing**

Process features by category with category-specific optimization:

```python
from src.training.steps.pre_training.unified_data_driven_pipeline import IntegratedFeaturePipeline

# Create pipeline with custom configuration
pipeline = IntegratedFeaturePipeline()

# Process by category
categories = ['momentum', 'volatility', 'volume']
results = {}

for category in categories:
    # Generate features for this category
    category_features = pipeline.feature_adapter.generate_features(data, [category])
    
    # Select optimal features for this category
    category_result = pipeline.pipeline.process(category_features, targets)
    
    results[category] = {
        'features': category_result.selected_features,
        'scores': category_result.objective_values,
        'count': len(category_result.selected_features)
    }

print("Category breakdown:")
for category, result in results.items():
    print(f"  {category}: {result['count']} features selected")
```

## Configuration Options

### 1. **Pipeline Configuration**

```python
from src.training.steps.pre_training.unified_data_driven_pipeline import create_high_performance_config

# High performance configuration
config = create_high_performance_config()
config.feature_selection.multi_objective.max_features = 30
config.feature_selection.multi_objective.objectives = {
    'out_of_sample_sharpe': 0.4,
    'drawdown': 0.3,
    'stability': 0.2,
    'diversity': 0.1
}

pipeline = create_integrated_pipeline(pipeline_config=config)
```

### 2. **Feature Generation Configuration**

```python
# Configure feature generation
feature_config = {
    'auto_optimization': True,
    'categories': ['momentum', 'volatility', 'volume'],
    'max_features_per_category': 15,
    'enable_vectorbt': True
}

pipeline = create_integrated_pipeline(feature_generation_config=feature_config)
```

### 3. **Time Series CV Configuration**

```python
from src.training.steps.pre_training.unified_data_driven_pipeline import UnifiedPipelineConfig

config = UnifiedPipelineConfig()
config.feature_selection.cv_config.n_splits = 5
config.feature_selection.cv_config.test_size = 0.2
config.feature_selection.cv_config.embargo_fraction = 0.05

pipeline = create_integrated_pipeline(pipeline_config=config)
```

## Advanced Usage

### 1. **Custom Feature Generation**

```python
from src.training.steps.pre_training.unified_data_driven_pipeline import FeatureGenerationAdapter

# Create custom feature adapter
adapter = FeatureGenerationAdapter(enable_existing_features=False)

# Generate custom features
custom_features = adapter.generate_features(
    data=data,
    categories=['momentum', 'volatility'],
    max_features_per_category=5
)

# Use with pipeline
from src.training.steps.pre_training.unified_data_driven_pipeline import process_features
result = process_features(custom_features, targets)
```

### 2. **Streaming Processing**

```python
from src.training.steps.pre_training.unified_data_driven_pipeline import create_integrated_pipeline

# Create pipeline for streaming
pipeline = create_integrated_pipeline()

# Initialize with sample data
sample_data = data.iloc[:100]
pipeline.feature_adapter.initialize(sample_data)

# Process streaming data
for i in range(0, len(data), 50):
    batch_data = data.iloc[i:i+50]
    batch_targets = targets.iloc[i:i+50]
    
    result = pipeline.process(batch_data, batch_targets)
    print(f"Batch {i//50}: {len(result.selected_features)} features selected")
```

### 3. **Performance Monitoring**

```python
# Enable performance monitoring
config = create_high_performance_config()
config.performance.enable_monitoring = True
config.performance.enable_profiling = True

pipeline = create_integrated_pipeline(pipeline_config=config)

# Process data
result = pipeline.process(data, targets)

# Get performance stats
stats = pipeline.pipeline.get_performance_stats()
print(f"Processing time: {stats['total_processing_time']:.2f}s")
print(f"Memory usage: {stats['memory_usage_mb']:.1f}MB")
```

## Error Handling

### 1. **Feature Generation Errors**

```python
try:
    result = process_with_integrated_pipeline(data, targets)
except Exception as e:
    print(f"Pipeline failed: {e}")
    
    # Fallback to synthetic features
    from src.training.steps.pre_training.unified_data_driven_pipeline import FeatureGenerationAdapter
    adapter = FeatureGenerationAdapter(enable_existing_features=False)
    features = adapter.generate_features(data)
    result = process_features(features, targets)
```

### 2. **Validation Errors**

```python
# Validate data before processing
if data.empty:
    raise ValueError("Data cannot be empty")

if targets is not None and len(targets) != len(data):
    raise ValueError("Targets length must match data length")

# Check for missing values
if data.isna().any().any():
    print("Warning: Missing values detected, filling with forward fill")
    data = data.fillna(method='ffill')
```

## Best Practices

### 1. **Data Preparation**
- Ensure data is time-ordered
- Handle missing values appropriately
- Validate data types and ranges
- Use appropriate time series CV parameters

### 2. **Feature Generation**
- Start with default categories
- Limit features per category to prevent explosion
- Use existing feature generation when available
- Fall back to synthetic features when needed

### 3. **Configuration**
- Start with default configuration
- Adjust objectives based on your strategy
- Use high-performance config for large datasets
- Monitor memory usage and processing time

### 4. **Validation**
- Always validate for leakage
- Check objective values make sense
- Monitor performance statistics
- Test with small datasets first

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Check if modules are available
   try:
       from src.feature_generation.core.factory import get_feature_bank
       print("Feature generation available")
   except ImportError:
       print("Feature generation not available, using synthetic features")
   ```

2. **Memory Issues**
   ```python
   # Use memory-efficient configuration
   from src.training.steps.pre_training.unified_data_driven_pipeline import create_memory_efficient_config
   config = create_memory_efficient_config()
   pipeline = create_integrated_pipeline(pipeline_config=config)
   ```

3. **Performance Issues**
   ```python
   # Use high-performance configuration
   from src.training.steps.pre_training.unified_data_driven_pipeline import create_high_performance_config
   config = create_high_performance_config()
   pipeline = create_integrated_pipeline(pipeline_config=config)
   ```

4. **No Features Selected**
   ```python
   # Check objective thresholds
   config.feature_selection.multi_objective.min_features = 1
   config.feature_selection.multi_objective.max_features = 50
   ```

## Examples

See the `examples/` directory for comprehensive examples:
- `usage_example.py` - Basic usage examples
- `integration_example.py` - Integration with existing feature generation

## API Reference

### Main Functions
- `process_features()` - Simple feature processing
- `process_with_integrated_pipeline()` - Integrated processing
- `create_integrated_pipeline()` - Create integrated pipeline
- `create_unified_pipeline()` - Create unified pipeline

### Main Classes
- `UnifiedDataDrivenPipeline` - Main pipeline orchestrator
- `IntegratedFeaturePipeline` - Integrated pipeline with feature generation
- `FeatureGenerationAdapter` - Adapter for feature generation
- `PurgedEmbargoedWalkForwardCV` - Time series cross-validation

### Configuration Classes
- `UnifiedPipelineConfig` - Main configuration
- `PurgedEmbargoedConfig` - CV configuration
- `MultiObjectiveConfig` - Feature selection configuration