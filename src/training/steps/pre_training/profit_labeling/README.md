# Volatility-Aware Multi-Horizon Profit Labeling System

A comprehensive, data-driven profit labeling system that explicitly accounts for volatility and microstructure noise, optimized for creating strong labels that are learnable by ML models and generalize well.

## 🎯 Key Features

- **Volatility-Normalized Targets**: Uses σ-units instead of fixed percentages
- **Event-Based Bar Construction**: Reduces microstructure noise through better bar formation
- **Noise Gating**: Filters out labels when noise dominates signal (micro-range, variance ratio, signal-to-noise)
- **Multi-Target Scheme**: Data-driven selection of small/medium/high targets
- **Quality Scoring**: Comprehensive assessment of label quality
- **Adaptive Horizons**: Data-driven horizon selection based on first-passage time

## 🏗️ Architecture

The system is built with a modular architecture consisting of five main components:

### 1. Bar Construction (`bar_construction.py`)
- Event-based bars (dollar bars, volume bars) instead of time bars
- Outlier-robust OHLC computation using median prices
- Microstructure filtering to remove ultra-tight ranges
- Volume and duration filtering

### 2. Volatility Modeling (`volatility_modeling.py`)
- Realized volatility estimation using high-frequency returns
- ATR (Average True Range) calculation
- EWMA volatility for responsiveness without whipsaw
- Volatility unit definition with floor to avoid division blowups

### 3. Noise Gating (`noise_gating.py`)
- Minimum move vs. micro-range filtering
- Variance ratio test for microstructure detection
- Signal-to-noise ratio assessment

### 4. Quality Scoring (`quality_scoring.py`)
- Predictability assessment using baseline models
- Stability measurement across rolling folds
- Consistency evaluation via mutual information
- Balance assessment for class distribution
- SNR proxy calculation using information coefficient

### 5. Multi-Target Scheme (`multi_target_scheme.py`)
- Data-driven target selection within small/medium/high bands
- First-passage time (FPT) based horizon calculation
- Volatility-normalized target bands
- Quality-based target selection and filtering

## 🚀 Quick Start

### Basic Usage

```python
from src.training.steps.pre_training.profit_labeling import (
    VolatilityAwareMultiHorizonLabeler,
    VolatilityAwareConfig
)

# Create labeler with default configuration
labeler = VolatilityAwareMultiHorizonLabeler()

# Generate labels
result = labeler.generate_labels(market_data)

# Access results
print(f"Generated {len(result.labels)} label samples")
print(f"Number of targets: {result.n_targets}")
print(f"Processing time: {result.processing_time:.2f}s")
```

### Custom Configuration

```python
# Create custom configuration
config = VolatilityAwareConfig(
    min_data_points=1000,
    enable_caching=True,
    parallel_processing=True
)

# Customize bar construction
config.bar_construction = BarConstructionConfig(
    bar_type=BarType.DOLLAR,
    bar_size=500000.0,  # $500k bars
    enable_microstructure_filter=True
)

# Customize volatility modeling
config.volatility = VolatilityConfig(
    method=VolatilityMethod.COMBINED,
    rv_window=30,
    atr_window=20
)

# Create labeler with custom configuration
labeler = VolatilityAwareMultiHorizonLabeler(config)
result = labeler.generate_labels(market_data)
```

## 📊 Input Data Format

The system expects OHLCV market data with a datetime index:

```python
# Required columns: 'open', 'high', 'low', 'close', 'volume'
market_data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
}, index=pd.DatetimeIndex([...]))
```

## 📈 Output Format

The system returns a `LabelingResult` object containing:

- **`labels`**: DataFrame with hard labels (-1, 0, +1) for each target
- **`confidence_scores`**: DataFrame with confidence scores [0, 1] for each target
- **`eligibility_masks`**: DataFrame with eligibility masks for each target
- **`quality_scores`**: Dictionary mapping target names to quality metrics
- **`n_samples`**: Number of samples
- **`n_targets`**: Number of targets
- **`processing_time`**: Processing time in seconds

## 🔧 Configuration Options

### Bar Construction Configuration

```python
BarConstructionConfig(
    bar_type=BarType.DOLLAR,  # DOLLAR, VOLUME, TICK, TIME
    bar_size=1000000.0,       # Size for event-based bars
    enable_microstructure_filter=True,
    min_spread_ratio=0.0001,  # Minimum (high-low)/mid ratio
    min_volume_percentile=10.0,
    max_return_percentile=99.9
)
```

### Volatility Modeling Configuration

```python
VolatilityConfig(
    method=VolatilityMethod.COMBINED,  # REALIZED, ATR, EWMA, COMBINED
    rv_window=20,                      # Window for realized volatility
    atr_window=14,                     # Window for ATR calculation
    ewma_alpha=0.06,                   # EWMA decay factor
    volatility_floor=1e-6,             # Floor to avoid division blowups
    enable_smoothing=True
)
```

### Noise Gating Configuration

```python
NoiseGatingConfig(
    gate_type=NoiseGateType.COMBINED,  # COMBINED, MICRO_RANGE, VARIANCE_RATIO, etc.
    enable_micro_range_gating=True,
    min_move_ratio=1.5,                # Minimum k·σ_t / (α·mTR_t) ratio
    enable_variance_ratio_gating=True,
    enable_signal_noise_gating=True,
    min_snr_ratio=1.2                  # Minimum signal-to-noise ratio
)
```

### Quality Scoring Configuration

```python
QualityScoringConfig(
    baseline_models=['logistic', 'random_forest'],
    n_splits=5,                        # Number of CV splits
    min_auc_threshold=0.55,            # Minimum AUC threshold
    max_auc_std_threshold=0.03,        # Maximum AUC std threshold
    lqs_weights={                      # LQS score weights
        'predictability': 0.3,
        'stability': 0.2,
        'consistency': 0.2,
        'balance': 0.2,
        'snr_proxy': 0.1
    }
)
```

### Multi-Target Configuration

```python
MultiTargetConfig(
    small_band=(0.4, 0.8),            # k_s range
    medium_band=(0.8, 1.3),           # k_m range
    high_band=(1.3, 2.0),             # k_h range
    enable_asymmetry=True,
    asymmetry_ratios=[1.0, 1.25],
    max_targets_per_band=2,
    min_lqs_score=0.3,
    max_correlation_threshold=0.6
)
```

## 📊 Quality Metrics

The system provides comprehensive quality assessment:

### Core Quality Metrics

- **Predictability**: AUC/PR-AUC from baseline models
- **Stability**: Variance of AUC across folds, PSI
- **Consistency**: Mutual information between labels
- **Balance**: Class balance assessment
- **SNR Proxy**: Information coefficient between features and labels

### Composite Score

- **LQS (Label Quality Score)**: Weighted combination of all metrics

### Detailed Metrics

- AUC mean and standard deviation
- PR-AUC mean and standard deviation
- PSI score
- Flip rate
- Class balance
- Mutual information
- Information coefficient

## 🎯 Target Selection

The system uses a data-driven approach to select optimal targets:

1. **Generate Candidates**: Create targets within small/medium/high bands
2. **Calculate FPT Horizons**: Use first-passage time for adaptive horizons
3. **Assess Quality**: Evaluate each target using quality metrics
4. **Filter by Thresholds**: Remove targets below quality thresholds
5. **Ensure Orthogonality**: Select targets with low correlation
6. **Optimize Selection**: Use Bayesian optimization for parameter search

## 🔍 Examples

See `example_usage.py` for comprehensive examples including:

- Basic usage with default configuration
- Custom configuration for specific use cases
- Integration with ML pipelines
- Performance monitoring and optimization
- Batch processing for multiple datasets
- Quality analysis and visualization

## 🚀 Performance

The system is optimized for performance with:

- **Parallel Processing**: Multi-threaded execution where possible
- **Caching**: Intelligent caching of intermediate results
- **Memory Optimization**: Efficient memory usage for large datasets
- **M1 Optimization**: Specialized optimizations for Apple Silicon

## 🔧 Integration

### With Existing ML Pipelines

```python
# Generate labels
result = labeler.generate_labels(market_data)

# Prepare features
features = generate_features(market_data)

# Align data
common_index = features.index.intersection(result.labels.index)
X = features.loc[common_index]
y = result.labels.loc[common_index]

# Train model
model = RandomForestClassifier()
model.fit(X, y)
```

### With Existing Multi-Horizon Labeler

```python
# Replace existing labeler
from src.training.steps.pre_training.profit_labeling import VolatilityAwareMultiHorizonLabeler

# Use as drop-in replacement
labeler = VolatilityAwareMultiHorizonLabeler()
result = labeler.generate_labels(market_data)
```

## 📚 API Reference

### Main Classes

- `VolatilityAwareMultiHorizonLabeler`: Main labeling system
- `EventBasedBarConstructor`: Event-based bar construction
- `VolatilityModeler`: Volatility estimation and modeling
- `NoiseGatingFilter`: Noise gating and eligibility filtering
- `LabelQualityScorer`: Quality assessment and scoring
- `MultiTargetScheme`: Multi-target generation and selection

### Configuration Classes

- `VolatilityAwareConfig`: Main configuration
- `BarConstructionConfig`: Bar construction settings
- `VolatilityConfig`: Volatility modeling settings
- `NoiseGatingConfig`: Noise gating settings
- `QualityScoringConfig`: Quality scoring settings
- `MultiTargetConfig`: Multi-target scheme settings

### Result Classes

- `LabelingResult`: Main result container
- `BarConstructionResult`: Bar construction results
- `VolatilityResult`: Volatility modeling results
- `EligibilityResult`: Noise gating results
- `QualityMetrics`: Quality assessment results
- `TargetSelectionResult`: Multi-target selection results

## 🤝 Contributing

The system is designed to be modular and extensible. Key extension points:

1. **Custom Bar Types**: Add new bar construction methods
2. **Volatility Methods**: Implement new volatility estimation techniques
3. **Noise Gates**: Add new noise filtering methods
4. **Quality Metrics**: Implement additional quality assessment metrics
5. **Target Schemes**: Create new target selection strategies

## 📄 License

This module is part of the Ares Trading System and follows the same licensing terms.

## 🆘 Support

For questions, issues, or contributions, please refer to the main project documentation or contact the development team.