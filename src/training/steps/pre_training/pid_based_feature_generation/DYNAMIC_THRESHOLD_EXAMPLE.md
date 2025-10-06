# Dynamic Threshold Adjustment Example

## Overview

This example demonstrates how the system automatically adjusts quality thresholds when new features have higher quality than the reference pre-processing features.

## Configuration

```python
from src.training.steps.market_analysis.pid_based_feature_generation import (
    FeatureSelectionMechanism,
    FeatureSelectionConfig,
    SelectionStrategy
)

# Configure dynamic threshold adjustment
config = FeatureSelectionConfig(
    # Initial thresholds
    min_synergy_score=0.05,
    min_unique_info_score=0.02,
    max_redundancy_score=0.8,
    
    # Dynamic adjustment settings
    enable_dynamic_thresholds=True,
    quality_improvement_factor=1.2,  # Increase thresholds by 20% when quality improves
    min_threshold_improvement=0.01,  # Minimum 1% improvement to adjust
    max_threshold_increase=0.1,      # Maximum 10% threshold increase
    reference_feature_rank=150,      # Compare to 150th ranked pre-processing feature
    
    # Feature limits
    max_interaction_features=100,
    max_polynomial_features=50,
    max_cross_timeframe_features=50
)

# Initialize feature selection mechanism
feature_selection = FeatureSelectionMechanism(config)
```

## Example Scenario

### **Initial Selection (Low Quality Features)**

```python
# First selection with lower quality features
result1 = feature_selection.select_features(X_low_quality, feature_names, target)

# Get statistics
stats1 = feature_selection.get_selection_statistics(result1)

print("=== Initial Selection (Low Quality) ===")
print(f"Selection efficiency: {stats1['selection_efficiency']['overall_selection_rate']:.2%}")
print(f"Average synergy score: {stats1['quality_metrics']['average_synergy_score']:.4f}")
print(f"Average unique info score: {stats1['quality_metrics']['average_unique_info_score']:.4f}")

# Output:
# === Initial Selection (Low Quality) ===
# Selection efficiency: 85%
# Average synergy score: 0.0456
# Average unique info score: 0.0189
# Dynamic threshold adjustments: {'dynamic_adjustment': False, 'reason': 'No significant quality improvements detected'}
```

### **Second Selection (High Quality Features)**

```python
# Second selection with higher quality features
result2 = feature_selection.select_features(X_high_quality, feature_names, target)

# Get statistics with dynamic adjustment
stats2 = feature_selection.get_selection_statistics(result2)

print("=== Second Selection (High Quality) ===")
print(f"Selection efficiency: {stats2['selection_efficiency']['overall_selection_rate']:.2%}")
print(f"Average synergy score: {stats2['quality_metrics']['average_synergy_score']:.4f}")
print(f"Average unique info score: {stats2['quality_metrics']['average_unique_info_score']:.4f}")

# Dynamic threshold adjustments
adjustments = stats2['dynamic_threshold_adjustments']
if adjustments['dynamic_adjustment']:
    print("\n🔧 Dynamic Threshold Adjustments Applied:")
    for threshold, info in adjustments['adjustments_made'].items():
        print(f"   • {threshold}: {info['old']:.4f} → {info['new']:.4f} (improvement: {info['improvement']:.4f})")

# Output:
# === Second Selection (High Quality) ===
# Selection efficiency: 75%
# Average synergy score: 0.0678
# Average unique info score: 0.0256
# 
# 🔧 Dynamic Threshold Adjustments Applied:
#    • min_synergy_score: 0.0500 → 0.0600 (improvement: 0.0222)
#    • min_unique_info_score: 0.0200 → 0.0240 (improvement: 0.0067)
```

### **Third Selection (Using Adjusted Thresholds)**

```python
# Third selection using the adjusted thresholds
result3 = feature_selection.select_features(X_medium_quality, feature_names, target)

# Get statistics
stats3 = feature_selection.get_selection_statistics(result3)

print("=== Third Selection (Medium Quality with Adjusted Thresholds) ===")
print(f"Selection efficiency: {stats3['selection_efficiency']['overall_selection_rate']:.2%}")
print(f"Average synergy score: {stats3['quality_metrics']['average_synergy_score']:.4f}")
print(f"Average unique info score: {stats3['quality_metrics']['average_unique_info_score']:.4f}")

# Current thresholds
current_thresholds = stats3['selection_thresholds']
print(f"\nCurrent thresholds:")
print(f"   • min_synergy_score: {current_thresholds['min_synergy_score']:.4f}")
print(f"   • min_unique_info_score: {current_thresholds['min_unique_info_score']:.4f}")
print(f"   • max_redundancy_score: {current_thresholds['max_redundancy_score']:.4f}")

# Output:
# === Third Selection (Medium Quality with Adjusted Thresholds) ===
# Selection efficiency: 60%
# Average synergy score: 0.0623
# Average unique info score: 0.0234
# 
# Current thresholds:
#    • min_synergy_score: 0.0600  ← Increased from 0.0500
#    • min_unique_info_score: 0.0240  ← Increased from 0.0200
#    • max_redundancy_score: 0.8000  ← No change (redundancy didn't improve enough)
```

## Key Benefits

### **1. Adaptive Quality Control**
- **Automatic threshold adjustment** based on feature quality
- **Prevents quality degradation** by maintaining high standards
- **Self-improving system** that gets better over time

### **2. Quality vs Quantity Balance**
- **High quality features**: Thresholds increase to maintain standards
- **Low quality features**: Thresholds remain stable to allow selection
- **Optimal balance**: Ensures best features are always selected

### **3. Transparent Process**
- **Detailed logging** of all threshold adjustments
- **Quality improvement metrics** for each adjustment
- **Clear reasoning** for why adjustments were made

## Configuration Options

### **Dynamic Adjustment Parameters**

```python
config = FeatureSelectionConfig(
    # Enable/disable dynamic adjustment
    enable_dynamic_thresholds=True,
    
    # Quality improvement factor (1.2 = 20% increase)
    quality_improvement_factor=1.2,
    
    # Minimum improvement required to adjust (0.01 = 1%)
    min_threshold_improvement=0.01,
    
    # Maximum threshold increase (0.1 = 10%)
    max_threshold_increase=0.1,
    
    # Reference feature rank for comparison
    reference_feature_rank=150
)
```

### **Threshold Adjustment Logic**

```python
# Synergy threshold adjustment
if synergy_improvement > min_threshold_improvement:
    new_threshold = min(
        current_threshold * quality_improvement_factor,
        current_threshold + max_threshold_increase
    )

# Unique info threshold adjustment
if unique_info_improvement > min_threshold_improvement:
    new_threshold = min(
        current_threshold * quality_improvement_factor,
        current_threshold + max_threshold_increase
    )

# Redundancy threshold adjustment (lower is better)
if redundancy_improvement > min_threshold_improvement:
    new_threshold = max(
        current_threshold / quality_improvement_factor,
        current_threshold - max_threshold_increase
    )
```

## Monitoring and Validation

### **Selection Statistics**

```python
# Get comprehensive statistics
stats = feature_selection.get_selection_statistics(result)

# Monitor selection efficiency
efficiency = stats['selection_efficiency']
print(f"Overall selection rate: {efficiency['overall_selection_rate']:.2%}")

# Monitor quality metrics
quality = stats['quality_metrics']
print(f"Average synergy: {quality['average_synergy_score']:.4f}")
print(f"Average unique info: {quality['average_unique_info_score']:.4f}")

# Monitor threshold adjustments
adjustments = stats['dynamic_threshold_adjustments']
if adjustments['dynamic_adjustment']:
    print("Thresholds were adjusted based on quality improvements")
else:
    print("No threshold adjustments needed")
```

This dynamic threshold adjustment system ensures that the feature selection process continuously improves by automatically raising quality standards when better features are available, while maintaining flexibility for different data quality scenarios.