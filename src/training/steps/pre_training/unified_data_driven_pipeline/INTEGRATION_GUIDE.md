# Integration Guide: Unified Pipeline with Existing Feature Generation

## Overview

This guide explains how to integrate the new **Unified Data-Driven Feature Pipeline** with the existing `src/feature_generation/` system to create a comprehensive, end-to-end feature engineering solution.

## Architecture Integration

### Current Systems
1. **`src/feature_generation/`** - General-purpose feature generation (100+ generators)
2. **`src/training/steps/pre_training/unified_data_driven_pipeline/`** - Data-driven selection and optimization

### Integration Strategy
The unified pipeline should **consume** features from the existing feature generation system and **optimize** their selection, rather than replacing the generation system.

## Integration Patterns

### 1. **Feature Generation → Pipeline Selection**

```python
# Step 1: Generate features using existing system
from src.feature_generation.core.factory import get_feature_bank
from src.feature_generation.core.auto_optimized_feature_generator import AutoOptimizedFeatureGenerator

# Get feature bank
bank = get_feature_bank()

# Generate features using existing generators
feature_data = {}
for category in ['momentum', 'volatility', 'volume', 'trend']:
    generators = bank.get_generators_by_category(category)
    for generator in generators:
        result = generator.generate(data)
        feature_data.update(result.features)

# Convert to DataFrame
features_df = pd.DataFrame(feature_data)

# Step 2: Use unified pipeline for selection
from src.training.steps.pre_training.unified_data_driven_pipeline import process_features

# Select optimal features
result = process_features(features_df, targets)
print(f"Selected {len(result.selected_features)} features from {len(features_df.columns)} candidates")
```

### 2. **Integrated Pipeline Class**

```python
from src.training.steps.pre_training.unified_data_driven_pipeline import UnifiedDataDrivenPipeline
from src.feature_generation.core.factory import get_feature_bank

class IntegratedFeaturePipeline(UnifiedDataDrivenPipeline):
    """Integrated pipeline that generates and selects features."""
    
    def __init__(self, config=None, feature_generation_config=None):
        super().__init__(config)
        self.feature_bank = get_feature_bank(feature_generation_config)
        self.generated_features = {}
    
    def generate_and_select_features(self, data, targets, feature_categories=None):
        """Generate features and select optimal subset."""
        
        # Step 1: Generate features using existing system
        tprint_info("Generating features using existing feature generation system")
        self.generated_features = self._generate_features(data, feature_categories)
        
        # Step 2: Select optimal features using unified pipeline
        tprint_info("Selecting optimal features using unified pipeline")
        result = self.process(self.generated_features, targets)
        
        return result
    
    def _generate_features(self, data, categories=None):
        """Generate features using existing feature generation system."""
        features = {}
        
        if categories is None:
            categories = ['momentum', 'volatility', 'volume', 'trend', 'oscillator']
        
        for category in categories:
            generators = self.feature_bank.get_generators_by_category(category)
            for generator in generators:
                try:
                    result = generator.generate(data)
                    features.update(result.features)
                except Exception as e:
                    tprint_warning(f"Feature generation failed for {generator.name}: {e}")
        
        return pd.DataFrame(features, index=data.index)
```

### 3. **Category-Specific Integration**

```python
from src.training.steps.pre_training.unified_data_driven_pipeline import create_unified_pipeline
from src.feature_generation.categories.momentum import MomentumFeatures
from src.feature_generation.categories.volatility import VolatilityFeatures

class CategoryAwarePipeline:
    """Pipeline that generates features by category and optimizes selection."""
    
    def __init__(self):
        self.pipeline = create_unified_pipeline()
        self.feature_generators = {
            'momentum': MomentumFeatures(),
            'volatility': VolatilityFeatures(),
            'volume': VolumeFeatures(),
            'trend': TrendFeatures()
        }
    
    def process_by_category(self, data, targets):
        """Process features by category with category-specific optimization."""
        results = {}
        
        for category, generator in self.feature_generators.items():
            tprint_info(f"Processing {category} features")
            
            # Generate features for this category
            category_features = generator.generate(data)
            
            # Select optimal features for this category
            category_result = self.pipeline.process(category_features, targets)
            
            results[category] = {
                'features': category_result.selected_features,
                'scores': category_result.objective_values,
                'count': len(category_result.selected_features)
            }
        
        return results
```

## Configuration Integration

### 1. **Unified Configuration**

```python
from src.training.steps.pre_training.unified_data_driven_pipeline import UnifiedPipelineConfig
from src.feature_generation.core.auto_optimization_config import AutoOptimizationConfig

class IntegratedConfig:
    """Configuration that combines both systems."""
    
    def __init__(self):
        # Unified pipeline config
        self.pipeline_config = UnifiedPipelineConfig()
        
        # Feature generation config
        self.feature_generation_config = {
            'auto_optimization': AutoOptimizationConfig(),
            'categories': ['momentum', 'volatility', 'volume', 'trend'],
            'max_features_per_category': 20,
            'enable_vectorbt': True
        }
    
    def get_pipeline_config(self):
        return self.pipeline_config
    
    def get_feature_generation_config(self):
        return self.feature_generation_config
```

### 2. **Category-Specific Optimization**

```python
# Configure different optimization strategies per category
category_configs = {
    'momentum': {
        'max_features': 15,
        'objectives': {'out_of_sample_sharpe': 0.4, 'stability': 0.3, 'diversity': 0.3}
    },
    'volatility': {
        'max_features': 10,
        'objectives': {'out_of_sample_sharpe': 0.5, 'drawdown': 0.3, 'stability': 0.2}
    },
    'volume': {
        'max_features': 8,
        'objectives': {'turnover': 0.4, 'stability': 0.3, 'diversity': 0.3}
    }
}
```

## Usage Examples

### Example 1: Basic Integration

```python
from src.training.steps.pre_training.unified_data_driven_pipeline import process_features
from src.feature_generation.core.factory import get_feature_bank

# Load your data
data = pd.read_csv('market_data.csv')
targets = pd.read_csv('targets.csv')['returns']

# Generate features using existing system
bank = get_feature_bank()
feature_data = {}

# Generate momentum features
momentum_generators = bank.get_generators_by_category('momentum')
for generator in momentum_generators[:10]:  # Limit to first 10
    result = generator.generate(data)
    feature_data.update(result.features)

# Generate volatility features
volatility_generators = bank.get_generators_by_category('volatility')
for generator in volatility_generators[:10]:
    result = generator.generate(data)
    feature_data.update(result.features)

# Convert to DataFrame
features_df = pd.DataFrame(feature_data)

# Select optimal features
result = process_features(features_df, targets)
print(f"Selected {len(result.selected_features)} features from {len(features_df.columns)} candidates")
```

### Example 2: Advanced Integration with Custom Pipeline

```python
from src.training.steps.pre_training.unified_data_driven_pipeline import create_unified_pipeline, create_high_performance_config
from src.feature_generation.core.factory import get_feature_bank

class AdvancedFeaturePipeline:
    def __init__(self):
        # High-performance configuration
        self.config = create_high_performance_config()
        self.pipeline = create_unified_pipeline(self.config)
        self.feature_bank = get_feature_bank()
    
    def process_with_categories(self, data, targets):
        """Process features with category-aware optimization."""
        
        # Generate features by category
        category_features = {}
        
        for category in ['momentum', 'volatility', 'volume', 'trend']:
            generators = self.feature_bank.get_generators_by_category(category)
            category_data = {}
            
            for generator in generators:
                try:
                    result = generator.generate(data)
                    category_data.update(result.features)
                except Exception as e:
                    print(f"Warning: {generator.name} failed: {e}")
            
            if category_data:
                category_features[category] = pd.DataFrame(category_data, index=data.index)
        
        # Combine all features
        all_features = pd.concat(category_features.values(), axis=1)
        
        # Select optimal features
        result = self.pipeline.process(all_features, targets)
        
        return {
            'selected_features': result.selected_features,
            'objective_values': result.objective_values,
            'category_breakdown': self._analyze_by_category(result.selected_features, category_features)
        }
    
    def _analyze_by_category(self, selected_features, category_features):
        """Analyze selected features by category."""
        breakdown = {}
        for category, features_df in category_features.items():
            category_selected = [f for f in selected_features if f in features_df.columns]
            breakdown[category] = {
                'selected': len(category_selected),
                'total': len(features_df.columns),
                'features': category_selected
            }
        return breakdown
```

### Example 3: Streaming Feature Generation

```python
from src.training.steps.pre_training.unified_data_driven_pipeline import create_unified_pipeline
from src.feature_generation.core.auto_optimized_feature_generator import AutoOptimizedFeatureGenerator

class StreamingFeaturePipeline:
    """Pipeline for streaming data with incremental feature generation."""
    
    def __init__(self, config=None):
        self.pipeline = create_unified_pipeline(config)
        self.feature_generators = {}
        self.feature_cache = {}
    
    def initialize_generators(self, data_sample):
        """Initialize feature generators with sample data."""
        from src.feature_generation.core.factory import get_feature_bank
        
        bank = get_feature_bank()
        
        # Initialize generators for each category
        for category in ['momentum', 'volatility', 'volume']:
            generators = bank.get_generators_by_category(category)
            self.feature_generators[category] = generators[:5]  # Limit to 5 per category
    
    def process_streaming_data(self, new_data, targets=None):
        """Process streaming data with incremental feature generation."""
        
        # Generate features for new data
        new_features = {}
        for category, generators in self.feature_generators.items():
            for generator in generators:
                try:
                    result = generator.generate(new_data)
                    new_features.update(result.features)
                except Exception as e:
                    print(f"Warning: {generator.name} failed: {e}")
        
        # Add to feature cache
        self.feature_cache.update(new_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(self.feature_cache, index=new_data.index)
        
        # Select features if targets provided
        if targets is not None:
            result = self.pipeline.process(features_df, targets)
            return result
        
        return features_df
```

## Best Practices

### 1. **Feature Generation Strategy**
- **Use existing generators** for feature creation
- **Use unified pipeline** for feature selection and optimization
- **Generate by category** for better organization
- **Limit generators per category** to prevent explosion

### 2. **Performance Optimization**
- **Enable VectorBT** in both systems
- **Use auto-optimization** in feature generation
- **Batch feature generation** when possible
- **Cache generated features** for reuse

### 3. **Memory Management**
- **Generate features in chunks** for large datasets
- **Use memory-efficient configurations**
- **Clear feature cache** periodically
- **Monitor memory usage** during processing

### 4. **Error Handling**
- **Wrap feature generation** in try-catch blocks
- **Log failed generators** for debugging
- **Continue processing** even if some generators fail
- **Validate generated features** before selection

## Migration Strategy

### Phase 1: Basic Integration
1. Use existing feature generation system
2. Apply unified pipeline for selection
3. Test with small datasets

### Phase 2: Advanced Integration
1. Create integrated pipeline classes
2. Implement category-specific optimization
3. Add streaming support

### Phase 3: Full Integration
1. Unified configuration system
2. Performance optimization
3. Comprehensive testing

## Conclusion

The integration between `src/feature_generation/` and the unified pipeline creates a powerful, comprehensive feature engineering system:

- **Feature Generation**: Use existing 100+ generators for feature creation
- **Feature Selection**: Use unified pipeline for optimal feature selection
- **Performance**: Leverage VectorBT optimization in both systems
- **Flexibility**: Support both batch and streaming processing
- **Maintainability**: Clear separation of concerns between generation and selection

This approach maximizes the value of both systems while providing a seamless, end-to-end feature engineering solution.