# Feature Selection Guide: How PID-Based Feature Selection Works

## Overview

The PID-based feature generation system uses **Partial Information Decomposition (PID)** to intelligently select the most relevant features for generating interaction, polynomial, and cross-timeframe features. This guide explains how the selection process works.

## How Feature Selection Works

### 1. **PID Analysis Process**

The feature selection mechanism uses the `partial_information_decompositor.py` to analyze relationships between features:

```python
# Core PID analysis
pid_result = self.pid_decompositor.decompose_information(X, target, feature_names)
```

**What it calculates:**
- **Synergy**: Information that emerges only when features are combined
- **Redundancy**: Information that is duplicated across features  
- **Unique Information**: Information that is unique to each feature

### 2. **Feature Selection Limits and Ranking**

**When we have more features than the limits (100/50/50), the system uses intelligent ranking:**

#### **Selection Process for Each Feature Type:**

1. **Sort by Score**: All features are sorted by their relevance score (synergy/unique info)
2. **Apply Threshold**: Only features above minimum threshold are considered
3. **Select Top N**: Take the highest-scoring features up to the limit
4. **Log Statistics**: Detailed logging shows selection efficiency

#### **Example: Interaction Features (Limit: 100)**
```
📊 Analyzing 1,000 feature pairs for interaction selection
📊 Interaction feature selection complete:
   • Total pairs analyzed: 1,000
   • Selected (synergy > 0.05): 100
   • Rejected (synergy ≤ 0.05): 900
   • Highest synergy score: 0.2341
📊 Reached interaction feature limit (100)
```

#### **Example: Polynomial Features (Limit: 50)**
```
📊 Analyzing 500 features for polynomial selection
📊 Polynomial feature selection complete:
   • Total features analyzed: 500
   • Selected (unique info > 0.02): 50
   • Rejected (unique info ≤ 0.02): 450
   • Highest unique information score: 0.1876
📊 Reached polynomial feature limit (50)
```

#### **Example: Cross-Timeframe Features (Limit: 50)**
```
📊 Identified 200 timeframe features
📊 Found 400 timeframe feature pairs
📊 Cross-timeframe feature selection complete:
   • Total timeframe pairs analyzed: 400
   • Selected cross-timeframe (synergy > 0.05): 50
   • Same timeframe pairs: 150
   • Rejected (synergy ≤ 0.05): 200
   • Highest cross-timeframe synergy score: 0.1987
📊 Reached cross-timeframe feature limit (50)
```

### 2. **Selection Criteria**

#### **Interaction Features (Up to 100)**
- **Selection Method**: Based on **synergy scores**
- **Threshold**: Features with synergy > `min_synergy_score` (default: 0.05)
- **Process**: 
  1. Calculate synergy between all feature pairs
  2. Sort by synergy score (highest first)
  3. Select top 100 pairs with synergy > threshold

```python
# Example synergy-based selection
synergy_items = sorted(pid_result.synergy.items(), key=lambda x: x[1], reverse=True)
for (feat1, feat2), synergy_score in synergy_items:
    if synergy_score > self.config.min_synergy_score:
        selected_features.append((feat1, feat2))
```

#### **Polynomial Features (Up to 50)**
- **Selection Method**: Based on **unique information scores**
- **Threshold**: Features with unique info > `min_unique_info_score` (default: 0.02)
- **Process**:
  1. Calculate unique information for each feature
  2. Sort by unique information score (highest first)
  3. Select top 50 features with unique info > threshold

```python
# Example unique information-based selection
unique_info_items = sorted(pid_result.unique_info.items(), key=lambda x: x[1], reverse=True)
for feature, unique_score in unique_info_items:
    if unique_score > self.config.min_unique_info_score:
        selected_features.append(feature)
```

#### **Cross-Timeframe Features (Up to 50)**
- **Selection Method**: Based on **synergy scores between different timeframes**
- **Threshold**: Features with synergy > `min_synergy_score` (default: 0.05)
- **Process**:
  1. Identify features from different timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d)
  2. Calculate synergy between timeframe features
  3. Select top 50 cross-timeframe pairs with synergy > threshold

```python
# Example cross-timeframe selection
timeframe_features = self._identify_timeframe_features(feature_names)
for (feat1, feat2), synergy_score in timeframe_synergy.items():
    if self._are_different_timeframes(feat1, feat2) and synergy_score > threshold:
        selected_features.append((feat1, feat2))
```

### 3. **Fallback Selection (When PID is Not Available)**

If PID analysis fails or target variable is not available, the system falls back to correlation-based selection:

#### **Correlation-Based Interaction Features**
- Calculate correlation matrix between all features
- Select pairs with correlation between `min_synergy_score` and `max_redundancy_score`

#### **Variance-Based Polynomial Features**
- Calculate variance for each feature
- Select features with highest variance (most informative)

#### **Timeframe-Based Cross-Timeframe Features**
- Identify features from different timeframes
- Select pairs with correlation between thresholds

## Configuration Parameters

### **Selection Thresholds**
```python
FeatureSelectionConfig(
    synergy_threshold=0.1,           # Minimum synergy for PID analysis
    redundancy_threshold=0.15,       # Maximum redundancy threshold
    unique_info_threshold=0.05,      # Minimum unique information
    
    min_synergy_score=0.05,          # Minimum synergy for selection
    min_unique_info_score=0.02,      # Minimum unique info for selection
    max_redundancy_score=0.8,        # Maximum redundancy for selection
)
```

### **Feature Limits**
```python
max_interaction_features=100,        # Maximum interaction features
max_polynomial_features=50,          # Maximum polynomial features
max_cross_timeframe_features=50,     # Maximum cross-timeframe features
```

## Selection Strategies

The system supports different selection strategies:

### **1. SYNERGY_BASED**
- Focuses on features with high synergy scores
- Best for finding complementary feature combinations

### **2. UNIQUE_INFO_BASED**
- Focuses on features with high unique information
- Best for finding features that provide distinct information

### **3. REDUNDANCY_BASED**
- Focuses on reducing redundancy
- Best for feature reduction and efficiency

### **4. COMBINED (Default)**
- Combines synergy, unique information, and redundancy criteria
- Provides balanced selection across all criteria

### **5. CORRELATION_BASED**
- Fallback strategy using correlation analysis
- Used when PID analysis is not available

## Quality Metrics and Selection Statistics

The system provides comprehensive quality metrics and selection statistics:

### **Basic Quality Metrics**
```python
result.average_synergy_score        # Average synergy of selected features
result.average_unique_info_score    # Average unique information
result.average_redundancy_score     # Average redundancy
result.selection_time               # Time taken for selection
```

### **Detailed Selection Statistics**
```python
# Get comprehensive selection statistics
stats = feature_selection.get_selection_statistics(result)

# Selection summary
stats['selection_summary'] = {
    'total_features_analyzed': 1000,
    'interaction_features_selected': 100,
    'polynomial_features_selected': 50,
    'cross_timeframe_features_selected': 50,
    'total_features_selected': 200
}

# Selection efficiency
stats['selection_efficiency'] = {
    'interaction_selection_rate': 1.0,      # 100/100 = 100%
    'polynomial_selection_rate': 1.0,       # 50/50 = 100%
    'cross_timeframe_selection_rate': 1.0,  # 50/50 = 100%
    'overall_selection_rate': 1.0           # 200/200 = 100%
}

# Quality metrics
stats['quality_metrics'] = {
    'average_synergy_score': 0.1234,
    'average_unique_info_score': 0.0567,
    'average_redundancy_score': 0.0890
}
```

### **Selection Efficiency Interpretation**

- **Selection Rate = 1.0 (100%)**: All available slots filled with high-quality features
- **Selection Rate = 0.5 (50%)**: Only half the slots filled, indicating strict quality thresholds
- **Selection Rate = 0.0 (0%)**: No features met quality thresholds, may need threshold adjustment

## Integration with Optimized Lookback Periods

The feature selection process integrates with optimized lookback periods from the previous step:

1. **OptimizedLookbackIntegration** runs FIRST
2. Applies optimized lookback periods to features
3. **FeatureSelectionMechanism** then selects from optimized features
4. **Feature generators** use selected features for generation

## Example Usage

```python
# Initialize feature selection
feature_selection = FeatureSelectionMechanism(FeatureSelectionConfig(
    max_interaction_features=100,
    max_polynomial_features=50,
    max_cross_timeframe_features=50,
    selection_strategy=SelectionStrategy.COMBINED
))

# Select features using PID analysis
result = feature_selection.select_features(
    X=feature_matrix,
    feature_names=feature_names,
    target=target_variable
)

# Use selected features
interaction_features = result.interaction_features
polynomial_features = result.polynomial_features
cross_timeframe_features = result.cross_timeframe_features
```

## Performance Considerations

- **Memory Optimization**: Uses matrix operations for efficient computation
- **GPU Acceleration**: Leverages Apple Silicon M1/M2/M3 GPU when available
- **Parallel Processing**: Processes feature pairs in parallel when possible
- **Fallback Mechanisms**: Graceful degradation when PID analysis fails

## Monitoring and Validation

The system provides comprehensive monitoring:

- **Selection Quality Scores**: Metrics for selection effectiveness
- **Performance Metrics**: Execution time and resource usage
- **Validation Results**: Quality checks on selected features
- **Error Handling**: Robust error handling with fallback strategies

This data-driven approach ensures that only the most relevant and informative features are selected for generation, leading to better model performance and more efficient computation.