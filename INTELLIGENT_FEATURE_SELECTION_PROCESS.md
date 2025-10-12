# Intelligent Feature Selection Process

## Overview

The intelligent feature selection process uses a sophisticated multi-stage approach to select 40-ish features from the full feature bank (200+ features) while ensuring diversity across categories and aspects.

## Process Flow

### Stage 1: Feature Generation & Initial Analysis

```
Input Data → Feature Bank → 200+ Features → Quality Filtering → Valid Features
```

**Steps:**
1. **Generate All Features**: Uses the full feature bank to create all available features
2. **Quality Filtering**: Removes features with:
   - Insufficient variance (< 1e-8)
   - Too much missing data (> 50%)
   - Constant values (nunique <= 1)
   - Infinite or NaN values

### Stage 2: Multi-Metric Scoring

For each valid feature, calculate:

```python
# Base score from variance and information content
base_score = log(1 + variance) * information_content

# Target correlation boost
correlation_boost = abs(correlation_with_target) * 0.3

# Uniqueness boost (inverse of max correlation with other features)
uniqueness_boost = uniqueness_score * 0.2

# Category weight (configurable per category)
category_weight = category_weights[category]

# Final score
total_score = (base_score + correlation_boost + uniqueness_boost) * category_weight
```

**Metrics Calculated:**
- **Variance**: Feature variability (higher = more informative)
- **Target Correlation**: Relevance to target variable (if provided)
- **Information Content**: Entropy-based measure of information density
- **Uniqueness Score**: Inverse of maximum correlation with other features
- **Category Weight**: Configurable importance weight per category

### Stage 3: Category-Based Grouping

Features are grouped by category and sorted by score:

```python
category_groups = {
    'momentum': [feature1, feature2, ...],  # Sorted by score
    'volatility': [feature3, feature4, ...],
    'trend': [feature5, feature6, ...],
    # ... 17+ categories
}
```

### Stage 4: Diversity-Constrained Selection

#### 4.1 Minimum Per Category
```python
for category, features in category_groups.items():
    min_required = 2  # Configurable (updated from 3)
    selected_from_category = features[:max(min_required, len(features))]
```

#### 4.2 Maximum Per Category
```python
max_allowed = min(4, len(features))  # Configurable (updated from 8)
selected_from_category = features[:max_allowed]
```

#### 4.3 Aspect Diversity Within Categories
```python
# Group by aspect type within category
aspect_groups = {
    'short_term': [feature1, feature2, ...],
    'medium_term': [feature3, feature4, ...],
    'long_term': [feature5, feature6, ...],
    'cross_timeframe': [feature7, feature8, ...]
}

# Select best feature from each aspect
for aspect, aspect_features in aspect_groups.items():
    best_feature = max(aspect_features, key=lambda x: x.score)
    selected_features.append(best_feature)
```

### Stage 5: Global Optimization

Fill remaining slots with highest-scoring features from any category:

```python
remaining_slots = target_feature_count - len(selected_features)
if remaining_slots > 0:
    unselected = [f for f in all_features if f not in selected_features]
    unselected.sort(key=lambda x: x.score, reverse=True)
    selected_features.extend(unselected[:remaining_slots])
```

## Category Weights

Default category weights (configurable):

```python
category_weights = {
    'momentum': 1.0,
    'volatility': 1.0,
    'trend': 1.0,
    'oscillator': 1.0,
    'volume': 1.0,
    'returns': 1.0,
    'cross_timeframe': 1.2,      # Higher weight
    'microstructure': 1.1,       # Slightly higher
    'entropy': 0.9,              # Slightly lower
    'support_resistance': 0.9,   # Slightly lower
    'candlestick_pattern': 0.8,  # Lower weight
    'time': 0.7,                 # Lower weight
    'order_flow': 1.0,
    'regime': 1.0,
    'acceleration': 1.0,
    'advanced_statistical': 1.0,
    'spectral_wavelet': 0.9
}
```

## Aspect Mapping

Different aspects within each category:

```python
aspect_mapping = {
    'momentum': ['short_term', 'medium_term', 'long_term', 'cross_timeframe'],
    'volatility': ['realized', 'implied', 'regime_based', 'cross_timeframe'],
    'trend': ['short_term', 'medium_term', 'long_term', 'regime_based'],
    'oscillator': ['momentum_based', 'trend_based', 'volume_based', 'price_based'],
    'volume': ['absolute', 'relative', 'momentum', 'pattern_based'],
    'returns': ['raw', 'normalized', 'risk_adjusted', 'regime_based'],
    'cross_timeframe': ['momentum', 'volatility', 'trend', 'volume'],
    'microstructure': ['bid_ask', 'order_flow', 'liquidity', 'execution'],
    'entropy': ['price', 'volume', 'information', 'regime'],
    'support_resistance': ['static', 'dynamic', 'volume_based', 'time_based'],
    'candlestick_pattern': ['reversal', 'continuation', 'indecision', 'volume_confirmation'],
    'time': ['intraday', 'daily', 'weekly', 'seasonal'],
    'order_flow': ['imbalance', 'pressure', 'aggression', 'liquidity'],
    'regime': ['volatility', 'trend', 'volume', 'market_state'],
    'acceleration': ['price', 'volume', 'momentum', 'volatility'],
    'advanced_statistical': ['higher_moments', 'distribution', 'dependence', 'regime'],
    'spectral_wavelet': ['frequency', 'time_frequency', 'decomposition', 'reconstruction']
}
```

## Quality Metrics

The system tracks several quality metrics:

```python
quality_metrics = {
    'average_score': float,           # Average score of selected features
    'average_variance': float,        # Average variance
    'average_correlation': float,     # Average correlation with target
    'average_information_content': float,  # Average information content
    'average_uniqueness': float,      # Average uniqueness score
    'category_diversity': int,        # Number of categories represented
    'aspect_diversity': int,          # Number of aspects represented
    'category_coverage': float        # Percentage of categories covered
}
```

## Early Termination Strategies

### 1. Quality-Based Early Termination
```python
if (current_interaction_count >= max_interactions and 
    high_quality_interactions >= high_quality_threshold):
    return None  # Stop generating more interactions
```

### 2. Feature Quality Checks
```python
# Skip features with insufficient data
if series.isna().sum() > len(series) * 0.5:
    return None

# Skip features with no variance
if series.nunique() <= 1:
    return None
```

### 3. Correlation-Based Filtering
```python
# Skip if features are too correlated
if max_correlation > correlation_threshold:
    return None
```

### 4. Quick Utility Scoring
```python
# Quick check before expensive calculations
quick_utility = calculate_quick_utility_score(series, targets)
if quick_utility < utility_threshold * 0.8:
    return None
```

### 5. Duplicate Detection
```python
# Skip if interaction is too similar to existing ones
if is_duplicate_interaction(series, utility_score):
    return None
```

## Performance Optimizations

### 1. VectorBT Integration
- Uses VectorBT for optimized rolling operations
- 2-5x speed improvement for statistical calculations
- GPU acceleration when available

### 2. Caching
- Caches computed interactions to avoid recalculation
- Intelligent cache management with size limits
- Cache hit rate tracking

### 3. Batch Processing
- Processes multiple interactions in parallel
- Memory-efficient chunked processing
- Automatic memory cleanup

### 4. Early Termination
- Multiple levels of early termination
- Quick utility scoring before expensive calculations
- Duplicate detection to avoid redundant work

## Configuration Options

```python
@dataclass
class FeatureSelectionConfig:
    target_feature_count: int = 40
    min_features_per_category: int = 2
    max_features_per_category: int = 4
    min_variance: float = 1e-8
    max_correlation_threshold: float = 0.95
    min_information_content: float = 0.1
    require_different_aspects: bool = True
    aspect_diversity_threshold: float = 0.3
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_vectorbt: bool = True
    category_weights: Dict[str, float] = None
```

## Example Output

```python
FeatureSelectionResult(
    selected_features=[
        FeatureScore(feature_name='rsi_14', category='momentum', aspect_type='short_term', score=0.85),
        FeatureScore(feature_name='volatility_20', category='volatility', aspect_type='realized', score=0.82),
        FeatureScore(feature_name='sma_50', category='trend', aspect_type='medium_term', score=0.78),
        # ... 37 more features
    ],
    category_distribution={
        'momentum': 4,
        'volatility': 4,
        'trend': 4,
        'oscillator': 3,
        'volume': 3,
        'returns': 3,
        'cross_timeframe': 5,
        'microstructure': 3,
        'entropy': 3,
        'support_resistance': 3,
        'candlestick_pattern': 3,
        'time': 3,
        'order_flow': 3,
        'regime': 3,
        'acceleration': 3,
        'advanced_statistical': 3,
        'spectral_wavelet': 3
    },
    aspect_distribution={
        'short_term': 8,
        'medium_term': 8,
        'long_term': 8,
        'cross_timeframe': 5,
        'realized': 4,
        'regime_based': 4,
        # ... more aspects
    },
    total_features_analyzed=247,
    selection_time=2.34,
    quality_metrics={
        'average_score': 0.76,
        'category_diversity': 17,
        'aspect_diversity': 24,
        'category_coverage': 1.0
    }
)
```

This intelligent selection process ensures that you get the most relevant, diverse, and high-quality features from your full feature bank while maintaining computational efficiency through early termination and optimization strategies.