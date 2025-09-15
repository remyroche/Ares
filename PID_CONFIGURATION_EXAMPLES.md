# PID Module Configuration & Artifact Generation Examples

## 🔧 **Configurable Thresholds Explained**

### **1. Synergy Threshold (`synergy_threshold`)**

**Purpose**: Controls which feature pairs are considered to have meaningful synergistic relationships.

```python
# Example configurations
config_low_synergy = PIDConfig(synergy_threshold=0.05)   # More sensitive, finds more interactions
config_high_synergy = PIDConfig(synergy_threshold=0.2)   # Less sensitive, only strong interactions
config_default = PIDConfig(synergy_threshold=0.1)        # Balanced approach
```

**Real-world example**:
```python
# If price_1m and volume_1m have synergy = 0.15
# With synergy_threshold=0.1: ✅ Interaction detected → Create price_1m_x_volume_1m
# With synergy_threshold=0.2: ❌ No interaction → Skip feature creation
```

**Impact on feature generation**:
- **Lower threshold (0.05)**: More interaction features, potential overfitting
- **Higher threshold (0.2)**: Fewer features, may miss subtle interactions
- **Recommended range**: 0.08 - 0.15

### **2. Redundancy Threshold (`redundancy_threshold`)**

**Purpose**: Identifies when features provide too much overlapping information.

```python
config_low_redundancy = PIDConfig(redundancy_threshold=0.1)   # Strict, removes more features
config_high_redundancy = PIDConfig(redundancy_threshold=0.3)  # Lenient, keeps more features
```

**Real-world example**:
```python
# If price_1m and price_5m have redundancy = 0.25
# With redundancy_threshold=0.2: ⚠️ High redundancy detected → Consider ratio features
# With redundancy_threshold=0.3: ✅ Acceptable redundancy → Keep both features
```

**Impact on feature selection**:
- **Lower threshold**: More aggressive redundancy removal
- **Higher threshold**: Allows more similar features
- **Recommended range**: 0.1 - 0.2

### **3. Unique Information Threshold (`unique_info_threshold`)**

**Purpose**: Ensures features provide meaningful individual contributions.

```python
config_strict_unique = PIDConfig(unique_info_threshold=0.08)   # Only features with high unique value
config_lenient_unique = PIDConfig(unique_info_threshold=0.02)  # Include features with low unique value
```

**Real-world example**:
```python
# If rsi_1m has unique_info = 0.06
# With unique_info_threshold=0.08: ❌ Removed (low unique contribution)
# With unique_info_threshold=0.05: ✅ Kept (sufficient unique contribution)
```

## 📊 **Polynomial Degree & Interaction Limits Explained**

### **1. Maximum Polynomial Degree (`max_polynomial_degree`)**

**Controls**: Complexity of polynomial relationships captured.

```python
# Example configurations
config_linear = PIDConfig(max_polynomial_degree=1)      # Only linear relationships
config_quadratic = PIDConfig(max_polynomial_degree=2)   # Up to quadratic (recommended)
config_cubic = PIDConfig(max_polynomial_degree=3)       # Up to cubic (complex)
config_high_order = PIDConfig(max_polynomial_degree=4)  # High-order (risky)
```

**Generated features for ['price', 'volume'] with degree=3**:
```python
# Degree 2 features:
"price_pow_2"           # price²
"volume_pow_2"          # volume²
"price_x_volume"        # price × volume

# Degree 3 features:
"price_pow_3"           # price³
"volume_pow_3"          # volume³
"price_x_volume_pow_2"  # price × volume²
"volume_x_price_pow_2"  # volume × price²
```

**Trade-offs**:
- **Degree 1**: Fast, prevents overfitting, but misses non-linear relationships
- **Degree 2**: Good balance, captures most important non-linearities
- **Degree 3+**: Captures complex relationships but risks overfitting

### **2. Interaction Feature Limits (`max_interaction_features`)**

**Controls**: Maximum number of interaction features generated to prevent feature explosion.

```python
config_conservative = PIDConfig(max_interaction_features=20)   # Few features, fast
config_moderate = PIDConfig(max_interaction_features=50)       # Balanced (default)
config_aggressive = PIDConfig(max_interaction_features=100)    # Many features, slower
```

**Feature generation process**:
1. Calculate synergy scores for all feature pairs
2. Sort pairs by synergy score (highest first)
3. For each significant pair, generate multiple interaction types:
   ```python
   interaction_types = [
       f"{feat1}_x_{feat2}",           # Multiplicative
       f"{feat1}_plus_{feat2}",        # Additive
       f"{feat1}_minus_{feat2}",       # Subtractive
       f"{feat1}_ratio_{feat2}",       # Ratio
       f"sqrt_{feat1}_x_{feat2}",      # Square root
       f"log_{feat1}_x_{feat2}",       # Logarithmic
       f"{feat1}_x_{feat2}_norm",      # Normalized
       f"{feat1}_rank_x_{feat2}_rank"  # Rank-based
   ]
   ```
4. Stop when `max_interaction_features` limit reached

## 📁 **Artifact Generation with Datetime**

### **Automatic Artifact Creation**

When PID analysis runs, it automatically creates timestamped artifacts:

```python
# Example usage - artifacts are created automatically
framework = FeatureSelectionFramework(config)
results = framework.run_comprehensive_feature_selection(
    X, y, feature_names,
    enable_pid_analysis=True
)

# Access artifact paths
pid_results = results['pid_results']
artifacts = pid_results['artifacts_generated']
print(f"Artifacts created: {artifacts}")
```

### **Generated Artifacts**

All artifacts include datetime in filename format: `YYYYMMDD_HHMMSS`

#### **1. Analysis Results** (`pid_analysis_20241215_143022.json`)
```json
{
  "analysis_metadata": {
    "timestamp": "2024-12-15T14:30:22",
    "execution_time": 2.45,
    "feature_pairs_analyzed": 105,
    "significant_interactions": 12,
    "config_used": {
      "synergy_threshold": 0.1,
      "redundancy_threshold": 0.15,
      "unique_info_threshold": 0.05,
      "max_polynomial_degree": 3,
      "max_interaction_features": 50
    }
  },
  "information_measures": {
    "redundancy": {
      "price_1m_price_5m": 0.23,
      "volume_1h_volume_4h": 0.18
    },
    "synergy": {
      "price_1m_volume_1m": 0.15,
      "rsi_1m_macd_1m": 0.12
    },
    "unique_info": {
      "price_1m": 0.08,
      "volume_1m": 0.06
    }
  },
  "generated_features": {
    "polynomial_features": ["price_1m_pow_2", "volume_1m_pow_2"],
    "interaction_features": ["price_1m_x_volume_1m", "rsi_1m_x_macd_1m"],
    "cross_timeframe_features": ["price_1m_to_5m_ratio"],
    "total_generated": 15
  }
}
```

#### **2. Expanded Feature Matrix** (`expanded_features_20241215_143022.parquet`)
- **Format**: Parquet for efficiency
- **Content**: Original + generated features as DataFrame
- **Metadata**: Includes creation info and PID config

#### **3. Feature Importance** (`feature_importance_20241215_143022.json`)
```json
{
  "timestamp": "2024-12-15T14:30:22",
  "feature_importance_scores": {
    "price_1m": 0.08,
    "volume_1m": 0.06,
    "price_1m_x_volume_1m": 0.12
  },
  "top_features": [
    ["price_1m_x_volume_1m", 0.12],
    ["price_1m", 0.08],
    ["volume_1m", 0.06]
  ]
}
```

#### **4. Interaction Summary** (`interaction_summary_20241215_143022.json`)
```json
{
  "timestamp": "2024-12-15T14:30:22",
  "significant_interactions": [
    {
      "features": "price_1m_volume_1m",
      "synergy_score": 0.15,
      "redundancy_score": 0.05
    }
  ],
  "feature_generation_summary": {
    "polynomial_features_count": 8,
    "interaction_features_count": 5,
    "cross_timeframe_features_count": 2,
    "total_new_features": 15
  }
}
```

### **Manual Artifact Creation**

You can also create artifacts manually:

```python
# Create comprehensive artifacts
decompositor = PartialInformationDecompositor(config)
pid_result = decompositor.decompose_information(X, y, feature_names)

# Generate all artifacts
artifacts = decompositor.create_comprehensive_artifact(
    X, y, feature_names, pid_result, 
    output_dir="custom_pid_artifacts"
)

# Or create individual artifacts
analysis_file = decompositor.save_analysis_results(pid_result)
matrix_file = decompositor.save_feature_matrix_artifact(X, feature_names, pid_result)
```

## 🎯 **Configuration Best Practices**

### **1. Domain-Specific Tuning**

```python
# For financial time series (crypto/forex)
crypto_config = PIDConfig(
    synergy_threshold=0.08,      # Lower - financial features often interact
    redundancy_threshold=0.12,   # Lower - remove redundant timeframes
    max_polynomial_degree=2,     # Conservative - avoid overfitting
    max_interaction_features=30  # Moderate - balance complexity
)

# For high-frequency trading
hft_config = PIDConfig(
    synergy_threshold=0.05,      # Very low - capture subtle interactions
    redundancy_threshold=0.08,   # Very low - remove redundant features
    max_polynomial_degree=3,     # Higher - capture complex relationships
    max_interaction_features=100 # More features - more opportunities
)

# For long-term analysis
longterm_config = PIDConfig(
    synergy_threshold=0.15,      # Higher - fewer spurious interactions
    redundancy_threshold=0.2,    # Higher - allow similar features
    max_polynomial_degree=2,     # Conservative
    max_interaction_features=20  # Fewer features
)
```

### **2. Performance Optimization**

```python
# For large datasets
large_data_config = PIDConfig(
    sample_size=5000,            # Sample for efficiency
    max_features_for_full_pid=15, # Limit full PID analysis
    max_interaction_features=25,  # Reduce feature explosion
    convergence_threshold=1e-4    # Faster convergence
)

# For real-time systems
realtime_config = PIDConfig(
    sample_size=1000,            # Small sample
    max_features_for_full_pid=10, # Very limited analysis
    max_interaction_features=15,  # Few features
    max_iterations=50            # Fast convergence
)
```

### **3. Feature Engineering Focus**

```python
# Focus on polynomial features
poly_config = PIDConfig(
    max_polynomial_degree=4,     # High degree
    max_interaction_features=20, # Fewer interactions
    synergy_threshold=0.12       # Higher threshold
)

# Focus on interactions
interaction_config = PIDConfig(
    max_polynomial_degree=2,     # Lower degree
    max_interaction_features=80, # Many interactions
    synergy_threshold=0.06       # Lower threshold
)

# Focus on cross-timeframe
timeframe_config = PIDConfig(
    cross_timeframe_threshold=0.08, # Lower threshold
    max_timeframe_lag=10,           # More lags
    max_polynomial_degree=2,        # Conservative
    max_interaction_features=40     # Moderate
)
```

The PID module now provides comprehensive configuration control and automatic artifact generation with datetime timestamps, making it easy to track and reproduce feature engineering experiments! 🚀