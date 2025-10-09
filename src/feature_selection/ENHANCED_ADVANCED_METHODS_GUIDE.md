# Enhanced Advanced Feature Selection Methods Guide

## 🚀 Overview

This guide covers the enhanced advanced feature selection methods with adaptive weighting, confidence scoring, native validation framework integration, and dynamic feature selection capabilities.

## 🧠 Enhanced Features

### 1. **Adaptive Weighting System** ⚖️
- **Performance-based weighting**: Weights are determined by each method's cross-validation performance
- **Dynamic updates**: Weights are updated based on recent performance
- **Smoothing**: Prevents rapid weight changes with configurable smoothing
- **Constraints**: Minimum and maximum weight limits to prevent dominance

### 2. **Confidence Scoring System** 🎯
- **Method agreement**: Features selected by multiple methods get higher confidence
- **Stability scoring**: Features with consistent selection across iterations
- **Performance weighting**: Features from better-performing methods get higher confidence
- **Consensus bonus**: Additional confidence for features selected by consensus

### 3. **Native Validation Framework** ✅
- **Built-in cross-validation**: Every selection automatically runs CV internally
- **Stability metrics**: Track feature selection stability across CV folds and bootstrap samples
- **Consensus scoring**: Features selected by multiple methods get higher confidence
- **Performance validation**: Validate that selected features improve downstream performance

### 4. **Dynamic Feature Selection** 🎛️
- **Multiple target specifications**: Absolute count, percentage, or performance threshold
- **Elbow method**: Automatically detect optimal number of features
- **Statistical thresholding**: Use statistical significance testing for cutoff determination
- **Performance curves**: Analyze performance vs. feature count relationships

## 🔧 Quick Start

### Basic Enhanced Selection

```python
from src.feature_selection import EnhancedAdvancedFeatureSelector, EnhancedAdvancedConfig

# Configure enhanced selection
config = EnhancedAdvancedConfig(
    enable_adaptive_weighting=True,
    enable_confidence_scoring=True,
    enable_native_validation=True,
    enable_dynamic_selection=True
)

# Create enhanced selector
selector = EnhancedAdvancedFeatureSelector(config)

# Select features with multiple target options
result = selector.select_features(
    X, y, 
    target_features=50,  # Absolute count
    # target_percentage=0.3,  # 30% of features
    # target_performance_threshold=0.8  # Performance threshold
)

print(f"Selected {len(result['selected_features'])} features")
print(f"Confidence scores: {result['confidence_scores']}")
print(f"Adaptive weights: {result['adaptive_weights']}")
```

### Enhanced Ensemble Selection

```python
from src.feature_selection import EnhancedEnsembleAdvancedSelector, EnhancedEnsembleConfig

# Configure enhanced ensemble
config = EnhancedEnsembleConfig(
    adaptive_weighting=AdaptiveWeightingConfig(
        enable_adaptive_weighting=True,
        performance_metric='r2',
        weight_smoothing=0.1
    ),
    confidence_scoring=ConfidenceScoringConfig(
        enable_confidence_scoring=True,
        agreement_threshold=0.5,
        consensus_bonus=0.2
    ),
    native_validation=NativeValidationConfig(
        enable_native_validation=True,
        cv_folds=5,
        enable_stability_metrics=True
    )
)

# Create enhanced ensemble selector
ensemble_selector = EnhancedEnsembleAdvancedSelector(config)

# Select features
result = ensemble_selector.select_features(X, y, target_percentage=0.4)

print(f"Ensemble selected {len(result['selected_features'])} features")
print(f"Individual results: {result['individual_results']}")
print(f"Validation results: {result['validation_results']}")
```

## ⚙️ Configuration Options

### Enhanced Advanced Configuration

```python
from src.feature_selection import (
    EnhancedAdvancedConfig,
    EnhancedEnsembleConfig,
    AdaptiveWeightingConfig,
    ConfidenceScoringConfig,
    NativeValidationConfig,
    DynamicFeatureSelectionConfig
)

# Main configuration
config = EnhancedAdvancedConfig(
    # Ensemble configuration
    ensemble_config=EnhancedEnsembleConfig(
        # Adaptive weighting
        adaptive_weighting=AdaptiveWeightingConfig(
            enable_adaptive_weighting=True,
            performance_metric='r2',
            weight_smoothing=0.1,
            min_weight=0.1,
            max_weight=0.8,
            weight_update_frequency=10
        ),
        
        # Confidence scoring
        confidence_scoring=ConfidenceScoringConfig(
            enable_confidence_scoring=True,
            agreement_threshold=0.5,
            consensus_bonus=0.2,
            stability_weight=0.3,
            performance_weight=0.4,
            agreement_weight=0.3
        ),
        
        # Native validation
        native_validation=NativeValidationConfig(
            enable_native_validation=True,
            cv_folds=5,
            cv_strategy='kfold',
            enable_stability_metrics=True,
            stability_n_bootstrap=10,
            enable_consensus_scoring=True
        ),
        
        # Dynamic selection
        dynamic_selection=DynamicFeatureSelectionConfig(
            enable_dynamic_selection=True,
            default_target_type='percentage',
            default_target_value=0.5,
            enable_elbow_method=True,
            enable_statistical_thresholding=True
        )
    ),
    
    # Method selection
    enable_auto_method_selection=True,
    method_selection_metrics=['r2', 'mse', 'stability'],
    
    # Performance monitoring
    enable_performance_monitoring=True,
    monitoring_interval=10,
    
    # Error handling
    enable_error_recovery=True,
    max_retry_attempts=3,
    retry_delay=1.0
)
```

### Adaptive Weighting Configuration

```python
adaptive_config = AdaptiveWeightingConfig(
    enable_adaptive_weighting=True,
    performance_metric='r2',  # 'r2', 'mse', 'accuracy', 'f1'
    weight_smoothing=0.1,  # Smoothing factor for weight updates
    min_weight=0.1,  # Minimum weight for any method
    max_weight=0.8,  # Maximum weight for any method
    weight_update_frequency=10,  # Update weights every N selections
    enable_weight_history=True,
    weight_decay=0.95  # Decay factor for historical weights
)
```

### Confidence Scoring Configuration

```python
confidence_config = ConfidenceScoringConfig(
    enable_confidence_scoring=True,
    agreement_threshold=0.5,  # Minimum agreement for high confidence
    consensus_bonus=0.2,  # Bonus for features selected by multiple methods
    stability_weight=0.3,  # Weight for stability in confidence calculation
    performance_weight=0.4,  # Weight for performance in confidence calculation
    agreement_weight=0.3,  # Weight for agreement in confidence calculation
    min_confidence=0.1,  # Minimum confidence score
    max_confidence=1.0  # Maximum confidence score
)
```

### Native Validation Configuration

```python
validation_config = NativeValidationConfig(
    enable_native_validation=True,
    cv_folds=5,
    cv_strategy='kfold',  # 'kfold', 'timeseries', 'stratified'
    enable_stability_metrics=True,
    stability_n_bootstrap=10,
    stability_threshold=0.8,
    enable_consensus_scoring=True,
    consensus_min_methods=2,
    enable_performance_validation=True,
    performance_validation_models=['linear', 'random_forest']
)
```

### Dynamic Selection Configuration

```python
dynamic_config = DynamicFeatureSelectionConfig(
    enable_dynamic_selection=True,
    default_target_type='percentage',  # 'absolute', 'percentage', 'performance_threshold'
    default_target_value=0.5,  # 50% of features by default
    enable_elbow_method=True,
    elbow_method_range=(5, 50),  # Range for elbow method
    enable_statistical_thresholding=True,
    statistical_significance_level=0.05
)
```

## 📊 Advanced Usage Examples

### Method Comparison with Enhanced Features

```python
# Compare all enhanced methods
comparison_result = selector.compare_methods(
    X, y, 
    target_percentage=0.3
)

print("Method Comparison Results:")
for method, result in comparison_result['results'].items():
    print(f"{method}:")
    print(f"  Selected: {result['n_selected']} features")
    print(f"  Execution time: {result['execution_time']:.3f}s")
    print(f"  Success: {result['success']}")
    
    if 'confidence_scores' in result:
        confidences = [scores['confidence_score'] for scores in result['confidence_scores'].values()]
        print(f"  Avg confidence: {np.mean(confidences):.3f}")
        print(f"  High confidence: {sum(1 for c in confidences if c >= 0.8)} features")
```

### Adaptive Weighting Insights

```python
# Get adaptive weighting insights
weighting_insights = selector.ensemble_selector.adaptive_weighting.get_weight_insights()

print("Adaptive Weighting Insights:")
print(f"Total selections: {weighting_insights['total_selections']}")
print(f"Weight updates: {weighting_insights['weight_updates']}")
print("Current weights:", weighting_insights['current_weights'])

for method, stability in weighting_insights['weight_stability'].items():
    print(f"{method} stability: {stability['stability_score']:.3f}")
    print(f"{method} trend: {stability['trend']}")
```

### Confidence Scoring Analysis

```python
# Get confidence scoring insights
confidence_insights = selector.ensemble_selector.confidence_scoring.get_confidence_insights(
    result['confidence_scores']
)

print("Confidence Scoring Insights:")
print(f"Total features: {confidence_insights['total_features']}")
print(f"High confidence: {confidence_insights['high_confidence_count']}")
print(f"Consensus features: {confidence_insights['consensus_count']}")
print(f"Avg confidence: {confidence_insights['avg_confidence']:.3f}")

# Get high confidence features
high_confidence_features = selector.ensemble_selector.confidence_scoring.get_high_confidence_features(
    result['confidence_scores'], threshold=0.8
)
print(f"High confidence features: {high_confidence_features}")
```

### Dynamic Selection with Elbow Method

```python
# Use elbow method for automatic feature count determination
result = selector.select_features(
    X, y,
    target_performance_threshold=0.8  # Will use elbow method
)

print(f"Target count determined: {result['target_info']['target_count']}")
print(f"Method used: {result['target_info']['method']}")

if 'performance_curve' in result['target_info']:
    curve = result['target_info']['performance_curve']
    print(f"Performance curve length: {len(curve)}")
    print(f"Performance curve: {curve}")
```

### Statistical Thresholding

```python
# Get statistical threshold information
if 'statistical_threshold' in result['target_info']:
    threshold_info = result['target_info']['statistical_threshold']
    print(f"Statistical threshold: {threshold_info['threshold']:.4f}")
    print(f"Significant features: {len(threshold_info['significant_features'])}")
    print(f"Method: {threshold_info['method']}")
    print(f"Significance level: {threshold_info['significance_level']}")
```

## 🔍 Performance Monitoring

### Get Comprehensive Statistics

```python
# Get performance statistics
stats = selector.get_performance_statistics()

print("Performance Statistics:")
print(f"Total selections: {stats['total_selections']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Avg selection time: {stats['avg_selection_time']:.3f}s")
print(f"Method usage: {stats['method_usage_ratio']}")

# Get ensemble statistics
ensemble_stats = stats['ensemble_statistics']
print(f"Ensemble weight updates: {ensemble_stats['adaptive_weighting']['weight_updates']}")
print(f"Confidence calculations: {ensemble_stats['confidence_scoring']['total_scorings']}")
print(f"Validation completions: {ensemble_stats['native_validation']['cv_completions']}")
```

### Get Enhanced Insights

```python
# Get comprehensive insights
insights = selector.get_enhanced_insights()

print("Enhanced Insights:")
print(f"Total selections: {insights['total_selections']}")
print(f"Success rate: {insights['success_rate']:.1%}")
print(f"Error recoveries: {insights['error_recoveries']}")

# Method usage analysis
print("Method Usage:")
for method, count in insights['method_usage'].items():
    print(f"  {method}: {count} selections")

# Ensemble insights
ensemble_insights = insights['ensemble_insights']
print(f"Ensemble insights: {ensemble_insights}")
```

### Method Recommendations

```python
# Get method recommendations based on data characteristics
recommendations = selector.get_method_recommendations(X, y)

print("Method Recommendations:")
print(f"Recommended method: {recommendations['recommended_method']}")
print("Reasoning:")
for reason in recommendations['reasoning']:
    print(f"  - {reason}")

print("Configuration suggestions:")
for key, value in recommendations['configuration_suggestions'].items():
    print(f"  {key}: {value}")
```

## 🎯 Use Cases

### High-Dimensional Data
```python
# For high-dimensional data, use pre-filtering and conservative selection
config = EnhancedAdvancedConfig(
    ensemble_config=EnhancedEnsembleConfig(
        dynamic_selection=DynamicFeatureSelectionConfig(
            default_target_type='percentage',
            default_target_value=0.1  # Select only 10% of features
        )
    )
)
```

### Small Sample Sizes
```python
# For small samples, use stability metrics and fewer CV folds
config = EnhancedAdvancedConfig(
    ensemble_config=EnhancedEnsembleConfig(
        native_validation=NativeValidationConfig(
            cv_folds=3,  # Fewer folds for small samples
            enable_stability_metrics=True
        )
    )
)
```

### Time Series Data
```python
# For time series, use temporal cross-validation
config = EnhancedAdvancedConfig(
    ensemble_config=EnhancedEnsembleConfig(
        native_validation=NativeValidationConfig(
            cv_strategy='timeseries'  # Use time series CV
        )
    )
)
```

### Critical Applications
```python
# For critical applications, use comprehensive validation
config = EnhancedAdvancedConfig(
    ensemble_config=EnhancedEnsembleConfig(
        native_validation=NativeValidationConfig(
            enable_native_validation=True,
            enable_stability_metrics=True,
            enable_consensus_scoring=True,
            enable_performance_validation=True
        ),
        confidence_scoring=ConfidenceScoringConfig(
            enable_confidence_scoring=True,
            agreement_threshold=0.7,  # Higher threshold for critical apps
            consensus_bonus=0.3
        )
    )
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Low Confidence Scores**
   ```python
   # Check method agreement
   confidence_insights = selector.ensemble_selector.confidence_scoring.get_confidence_insights(
       result['confidence_scores']
   )
   print("Method agreement analysis:", confidence_insights['method_agreement_analysis'])
   ```

2. **Unstable Weights**
   ```python
   # Check weight stability
   weighting_insights = selector.ensemble_selector.adaptive_weighting.get_weight_insights()
   for method, stability in weighting_insights['weight_stability'].items():
       print(f"{method} stability: {stability['stability_score']:.3f}")
   ```

3. **Poor Validation Results**
   ```python
   # Check validation details
   validation_results = result['validation_results']
   print("CV results:", validation_results['cross_validation'])
   print("Stability results:", validation_results['stability_metrics'])
   print("Consensus results:", validation_results['consensus_scores'])
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
from src.utils.tprint import tprint_debug
tprint_debug("Debug information will be shown")
```

## 📚 Complete Example

```python
import numpy as np
from src.feature_selection import (
    EnhancedAdvancedFeatureSelector,
    EnhancedAdvancedConfig,
    AdaptiveWeightingConfig,
    ConfidenceScoringConfig,
    NativeValidationConfig,
    DynamicFeatureSelectionConfig
)

# Generate sample data
X = np.random.rand(1000, 100)
y = np.random.rand(1000)

# Configure enhanced selection
config = EnhancedAdvancedConfig(
    ensemble_config=EnhancedEnsembleConfig(
        adaptive_weighting=AdaptiveWeightingConfig(
            enable_adaptive_weighting=True,
            performance_metric='r2',
            weight_smoothing=0.1
        ),
        confidence_scoring=ConfidenceScoringConfig(
            enable_confidence_scoring=True,
            agreement_threshold=0.5,
            consensus_bonus=0.2
        ),
        native_validation=NativeValidationConfig(
            enable_native_validation=True,
            cv_folds=5,
            enable_stability_metrics=True
        ),
        dynamic_selection=DynamicFeatureSelectionConfig(
            enable_dynamic_selection=True,
            default_target_type='percentage',
            default_target_value=0.3
        )
    )
)

# Create enhanced selector
selector = EnhancedAdvancedFeatureSelector(config)

# Select features
result = selector.select_features(X, y, target_percentage=0.3)

print(f"Selected {len(result['selected_features'])} features")
print(f"Method used: {result['method_used']}")
print(f"Execution time: {result['execution_time']:.3f}s")

# Analyze results
print("Confidence scores:")
for feature, scores in result['confidence_scores'].items():
    print(f"  {feature}: {scores['confidence_score']:.3f} (agreement: {scores['agreement_score']:.3f})")

print("Adaptive weights:")
for method, weight in result['adaptive_weights'].items():
    print(f"  {method}: {weight:.3f}")

# Get performance insights
insights = selector.get_enhanced_insights()
print(f"Success rate: {insights['success_rate']:.1%}")
print(f"Method usage: {insights['method_usage']}")
```

## 🎉 Conclusion

The enhanced advanced feature selection methods provide state-of-the-art capabilities with:

- **Adaptive weighting** for optimal method combination
- **Confidence scoring** for feature reliability assessment
- **Native validation** for robust performance estimation
- **Dynamic selection** for intelligent feature count determination
- **Comprehensive monitoring** for performance tracking

These enhancements make the feature selection process more robust, reliable, and suitable for production use in critical applications.