# Enhanced HMM Clustering for Market Analysis

A comprehensive Hidden Markov Model (HMM) clustering system for market regime detection, leveraging all common utilities for optimal performance and reliability.

## Features

### Core Capabilities
- **HMM Regime Detection**: Advanced regime discovery using Gaussian HMM
- **Multi-Timeframe Analysis**: Support for various timeframes (1h, 4h, 1d, etc.)
- **Feature Engineering**: Comprehensive technical indicator calculation
- **Feature Selection**: Intelligent feature selection using MRMR and other methods
- **Cross-Validation**: Temporal cross-validation with purged K-fold
- **Performance Optimization**: M1 hardware acceleration (GPU, CPU, Memory)

### Integration with Common Utilities
- **Data Processing**: Klines parquet management and data quality validation
- **Math Validation**: Safe mathematical operations and error handling
- **Matrix Operations**: Unified matrix operations for efficient computations
- **ML Common**: Cross-validation, feature selection, and HPO utilities
- **Hardware Optimization**: M1 GPU, CPU, and memory optimization
- **Serialization**: Universal serialization for model persistence

## Installation

```bash
# Install required dependencies
pip install numpy pandas scikit-learn hmmlearn matplotlib seaborn

# For M1 GPU acceleration (optional)
pip install torch torchvision

# For additional technical indicators (optional)
pip install ta-lib
```

## Quick Start

### Basic Usage

```python
from enhanced_hmm_clustering import run_hmm_clustering_analysis, HMMClusteringConfig

# Create configuration
config = HMMClusteringConfig(
    n_components=4,
    lookback_windows=[5, 10, 20, 50],
    technical_indicators=["rsi", "macd", "bollinger_bands", "atr"],
    use_gpu=True,
    use_memory_optimization=True
)

# Run analysis
result = run_hmm_clustering_analysis(
    symbol="BTCUSDT",
    interval="1h",
    config=config,
    save_results=True
)

# Access results
print(f"Regime labels: {result.regime_labels}")
print(f"Regime characteristics: {result.regime_characteristics}")
print(f"Performance metrics: {result.performance_metrics}")
```

### Using Configuration Presets

```python
from config import get_config_by_name, HMMClusteringConfigFactory

# Use predefined presets
crypto_config = get_config_by_name("crypto_btc_1h")
forex_config = get_config_by_name("forex_major_1h")
research_config = get_config_by_name("research")

# Or create custom configurations
custom_config = HMMClusteringConfigFactory.create_crypto_config(
    timeframe=TimeframeType.INTRADAY,
    market_volatility="high"
)
```

### Advanced Usage

```python
from enhanced_hmm_clustering import EnhancedHMMClustering

# Initialize clustering system
clustering = EnhancedHMMClustering(config)

# Load market data
data = clustering.load_market_data("BTCUSDT", "1h")

# Engineer features
features = clustering.engineer_features(data)

# Select optimal features
selected_features = clustering.select_features(features)

# Fit HMM model
result = clustering.fit_hmm_model(selected_features)

# Predict regimes for new data
new_regime_labels, new_regime_probs = clustering.predict_regimes(selected_features)
```

## Configuration Options

### HMM Parameters
- `n_components`: Number of regimes (default: 3)
- `covariance_type`: HMM covariance type (default: "full")
- `n_iter`: Maximum iterations (default: 100)
- `random_state`: Random seed (default: 42)

### Feature Engineering
- `lookback_windows`: List of lookback periods (default: [5, 10, 20, 50])
- `technical_indicators`: List of indicators to calculate
  - Available: "rsi", "macd", "bollinger_bands", "atr", "stochastic", "williams_r", "cci", "roc"

### Optimization
- `use_gpu`: Enable M1 GPU acceleration (default: True)
- `use_memory_optimization`: Enable memory optimization (default: True)
- `use_cpu_optimization`: Enable CPU optimization (default: True)

### Cross-Validation
- `cv_folds`: Number of CV folds (default: 5)
- `test_size`: Test set size (default: 0.2)
- `purged_cv`: Use purged cross-validation (default: True)

### Feature Selection
- `feature_selection_method`: Method for feature selection (default: "mrmr")
- `max_features`: Maximum number of features (default: 50)

## Available Configurations

### Market-Specific Presets
- **Crypto**: `crypto_btc_1h`, `crypto_eth_4h`, `crypto_daily`
- **Forex**: `forex_major_1h`, `forex_minor_4h`
- **Stocks**: `stocks_large_daily`, `stocks_small_1h`

### Specialized Presets
- **High Frequency**: `high_frequency` - For high-frequency trading analysis
- **Low Latency**: `low_latency` - For real-time processing
- **Research**: `research` - For comprehensive analysis and experimentation

## Technical Indicators

The system supports the following technical indicators:

### Momentum Indicators
- **RSI**: Relative Strength Index
- **Stochastic**: Stochastic Oscillator
- **Williams %R**: Williams Percent Range
- **ROC**: Rate of Change

### Trend Indicators
- **MACD**: Moving Average Convergence Divergence
- **ADX**: Average Directional Index

### Volatility Indicators
- **Bollinger Bands**: Price volatility bands
- **ATR**: Average True Range

### Volume Indicators
- **MFI**: Money Flow Index
- **CCI**: Commodity Channel Index

## Performance Optimization

### M1 Hardware Acceleration
- **GPU**: M1 Metal Performance Shaders (MPS) for matrix operations
- **CPU**: Optimized CPU operations for M1 architecture
- **Memory**: Efficient memory management and caching

### Matrix Operations
- Unified matrix operations for consistent performance
- Vectorized computations where possible
- Memory-efficient array handling

### Data Processing
- Optimized parquet file handling
- Efficient data quality validation
- Smart feature engineering

## Output and Results

### HMMClusteringResult Object
The main result object contains:

- `model`: Trained HMM model
- `regime_labels`: Array of regime assignments
- `regime_probabilities`: Array of regime probabilities
- `regime_characteristics`: Dictionary of regime properties
- `feature_importance`: Dictionary of feature importance scores
- `performance_metrics`: Dictionary of model performance metrics
- `config`: Configuration used for training
- `processing_time`: Total processing time in seconds
- `memory_usage`: Memory usage statistics

### Regime Characteristics
For each regime, the system provides:
- Count and percentage of data points
- Mean returns and volatility
- Technical indicator statistics
- Price characteristics

### Performance Metrics
- Regime stability
- Regime balance
- Average confidence
- Regime duration statistics

## Examples

### Single Symbol Analysis
```python
from example_usage import run_single_symbol_analysis

# Run analysis for BTCUSDT
result = run_single_symbol_analysis()
```

### Comprehensive Analysis
```python
from example_usage import run_comprehensive_analysis

# Run analysis across multiple symbols and timeframes
run_comprehensive_analysis()
```

### Custom Analysis
```python
from enhanced_hmm_clustering import EnhancedHMMClustering
from config import create_custom_config

# Create custom configuration
config = create_custom_config(
    n_components=5,
    lookback_windows=[3, 7, 14, 30],
    technical_indicators=["rsi", "macd", "atr"],
    max_features=25
)

# Run analysis
clustering = EnhancedHMMClustering(config)
# ... rest of analysis
```

## File Structure

```
market_analysis/hmm_clustering/
├── enhanced_hmm_clustering.py    # Main clustering implementation
├── config.py                     # Configuration system
├── example_usage.py             # Usage examples
├── README.md                    # This documentation
└── results/                     # Output directory
    ├── hmm_model_*.pkl         # Saved models
    ├── hmm_results_*.json      # Analysis results
    ├── regime_analysis.png     # Regime visualizations
    ├── feature_importance.png  # Feature importance plots
    └── comparison_analysis.png # Cross-symbol comparisons
```

## Dependencies

### Required
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- hmmlearn >= 0.2.7

### Optional
- torch >= 1.12.0 (for M1 GPU acceleration)
- ta-lib >= 0.4.24 (for additional technical indicators)
- matplotlib >= 3.5.0 (for visualizations)
- seaborn >= 0.11.0 (for enhanced plots)

## Common Utilities Integration

This implementation leverages the following common utilities:

- `src.utils.common_operations`: Core data operations and hardware management
- `src.utils.common_utilities`: DataFrame operations and validation
- `src.utils.math_validation`: Safe mathematical operations
- `src.utils.data.klines_parquet`: Market data management
- `src.utils.serialization_utils`: Model persistence
- `src.utils.matrix_operations`: Efficient matrix computations
- `src.utils.ml_common`: ML utilities (CV, feature selection, HPO)
- `src.utils.hardware`: M1 optimization utilities

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `max_features` or enable memory optimization
2. **GPU Issues**: Ensure M1 GPU support is properly installed
3. **Data Issues**: Check data quality and minimum data points
4. **Convergence Issues**: Increase `n_iter` or adjust `n_components`

### Performance Tips

1. Use appropriate configuration presets for your use case
2. Enable hardware optimization for better performance
3. Use feature selection to reduce dimensionality
4. Consider data quality requirements

## Contributing

When contributing to this module:

1. Follow the existing code structure and patterns
2. Add comprehensive docstrings and type hints
3. Include unit tests for new functionality
4. Update documentation as needed
5. Ensure compatibility with common utilities

## License

This module is part of the larger market analysis system and follows the same licensing terms.