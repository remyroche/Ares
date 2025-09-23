# NAS-Driven Clustering for Short-Term Trading

This module **replaces** the existing HMM clustering pipeline with enhanced Neural Architecture Search (NAS) driven clustering for short-term trading regime detection (5-30m timeframe) with micro-regime detection capabilities.

## 🎯 Key Features

- **NAS-Driven Regime Detection**: 10-15 different market states optimized for short-term trading
- **Short-Term Trading Optimization**: 5-30m timeframe with 15m minimum for actionable states
- **Micro-Regime Detection**: Subtle market changes that may not be captured by standard regime detection
- **Economic Significance**: Regime detection based on economically relevant market states
- **Trading Viability**: Assessment of regime quality for trading decisions
- **HMM Pipeline Replacement**: Complete replacement of existing HMM clustering pipeline
- **ML Model Training Support**: Enhanced features for DeepScale, LGBM, XGBoost, and other ML models

## 🏗️ Architecture

### Core Components

1. **NASClusterer**: Main clustering engine with NAS-driven architecture search
2. **NASFeatureExtractor**: Enhanced feature extraction optimized for regime detection
3. **MicroRegimeDetector**: Detection of subtle market changes
4. **NASOutputFormatter**: Pipeline-compatible output formatting

### Integration Components

1. **NASClusteringComponent**: HMM pipeline replacement component
2. **NASOrchestrator**: Main orchestrator for complete NAS clustering pipeline
3. **NASPipelineIntegration**: Seamless replacement of existing HMM pipeline

### Utility Components

1. **NASMetrics**: Comprehensive metrics for regime quality evaluation
2. **NASVisualizer**: Visualization tools for regime analysis
3. **NASValidator**: Validation tools for regime detection

## 🚀 Quick Start

### Basic Usage

```python
from src.training.steps.market_analysis.nas_clustering import NASOrchestrator

# Initialize NAS orchestrator
config = {
    'symbol': 'BTCUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'data_dir': 'historical_data'
}

orchestrator = NASOrchestrator(config)

# Run NAS clustering
results = await orchestrator.run_nas_clustering(
    data=market_data,
    timestamps=timestamps,
    symbol='BTCUSDT',
    exchange='binance',
    timeframe='15m'
)
```

### Advanced Usage

```python
from src.training.steps.market_analysis.nas_clustering import (
    NASClusterer, NASClusteringConfig, NASFeatureExtractor, MicroRegimeDetector
)

# Create NAS configuration
nas_config = NASClusteringConfig.create_short_term_trading_config()
nas_config.n_regimes = 12
nas_config.enable_micro_regime_detection = True
nas_config.economic_significance_threshold = 0.7
nas_config.trading_viability_threshold = 0.6

# Initialize components
clusterer = NASClusterer(nas_config)
feature_extractor = NASFeatureExtractor(nas_config.get_feature_config())
micro_regime_detector = MicroRegimeDetector(nas_config.get_micro_regime_config())

# Extract features
feature_result = feature_extractor.extract_features(market_data, timestamps)

# Detect micro-regimes
micro_regime_result = micro_regime_detector.detect_micro_regimes(
    market_data, timestamps, feature_result.features
)

# Perform NAS clustering
clustering_result = clusterer.cluster(
    market_data, timestamps, optimize_parameters=True, generate_report=True
)
```

## 📊 Output Format

The NAS clustering module provides enhanced output that **replaces** the existing HMM clustering pipeline with superior capabilities:

### Enhanced HMM-Replacement Fields

```python
{
    'success': True,
    'execution_time': 45.2,
    'timestamp': '2024-01-15T10:30:00Z',
    'method': 'nas_clustering',
    
    # Standard clustering results
    'labels': [0, 1, 2, 0, 1, ...],
    'cluster_centers': [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
    'statistics': {...},
    'quality_metrics': {...},
    'validation': {...},
    'metadata': {...},
    
    # Enhanced HMM-replacement fields
    'transition_matrix': [[0.8, 0.2], [0.3, 0.7], ...],
    'eigenvalues': [1.0, 0.8, 0.6, ...],
    'eigenvectors': [[0.7, 0.3], [0.4, 0.6], ...],
    'stationary_distribution': [0.6, 0.4, ...],
    'implied_timescales': [5.0, 10.0, ...],
    'msm_score': 0.85,
    'lag_time': 1
}
```

### NAS-Enhanced Fields

```python
{
    # NAS-specific fields
    'nas_architectures': {
        'volatility': {...},
        'trend': {...},
        'volume': {...},
        'momentum': {...},
        'hybrid': {...}
    },
    'nas_score': 0.82,
    'nas_architecture_type': 'hybrid',
    
    # Micro-regime fields
    'micro_regimes': {
        'regimes': [0, 1, 0, 2, ...],
        'types': ['breakout', 'consolidation', 'reversal', ...],
        'scores': [0.8, 0.6, 0.9, ...],
        'detection_accuracy': 0.75
    },
    
    # Economic significance
    'economic_significance_scores': [0.8, 0.7, 0.9, ...],
    'trading_viability_scores': [0.6, 0.8, 0.7, ...],
    
    # ML training data (DeepScale, LGBM, XGBoost, etc.)
    'ml_training_data': {
        'regime_sequences': [0, 1, 2, 0, ...],
        'regime_transitions': [[0.8, 0.2], [0.3, 0.7], ...],
        'economic_significance': [0.8, 0.7, 0.9, ...],
        'trading_viability': [0.6, 0.8, 0.7, ...],
        'micro_regime_sequences': [0, 1, 0, 2, ...],
        'micro_regime_types': ['breakout', 'consolidation', ...],
        'regime_features': {...},
        'transition_features': {...},
        'economic_features': {...},
        'trading_features': {...},
        'micro_regime_features': {...},
        'market_features': {...}
    }
}
```

## 🔧 Configuration

### NAS Clustering Configuration

```python
from src.training.steps.market_analysis.nas_clustering import NASClusteringConfig

# Create short-term trading configuration
config = NASClusteringConfig.create_short_term_trading_config()

# Customize configuration
config.n_regimes = 12  # Target number of regimes (10-15)
config.timeframe = "15m"  # Primary timeframe
config.micro_timeframe = "5m"  # Micro-regime detection timeframe
config.min_regime_duration = 15  # Minimum 15 minutes for actionable states
config.max_regime_duration = 180  # Maximum 3 hours for short-term trading
config.enable_micro_regime_detection = True
config.micro_regime_sensitivity = 0.7
config.economic_significance_threshold = 0.7
config.trading_viability_threshold = 0.6
config.regime_transition_cost = 0.05
```

### Feature Configuration

```python
# Feature extraction configuration
feature_config = {
    'timeframe': '15m',
    'micro_timeframe': '5m',
    'exclude_complex_features': True,  # Exclude polynomial, wavelet features
    'include_technical_indicators': True,
    'include_volume_features': True,
    'include_volatility_features': True,
    'include_momentum_features': True,
    'include_trend_features': True,
    'normalize_features': True
}
```

### Micro-Regime Configuration

```python
# Micro-regime detection configuration
micro_regime_config = {
    'enable_micro_regime_detection': True,
    'micro_regime_sensitivity': 0.7,
    'micro_regime_types': [
        'breakout', 'consolidation', 'reversal',
        'acceleration', 'volume_spike', 'volatility_spike'
    ],
    'micro_timeframe': '5m'
}
```

## 📈 Performance Metrics

### Standard Clustering Metrics

- **Silhouette Score**: Measures cluster separation and cohesion
- **Calinski-Harabasz Score**: Measures cluster quality
- **Davies-Bouldin Score**: Measures cluster compactness

### NAS-Specific Metrics

- **NAS Score**: Custom score combining multiple metrics
- **Economic Significance Score**: Measures economic relevance of regimes
- **Trading Viability Score**: Measures trading decision support
- **Regime Stability Score**: Measures regime persistence
- **Regime Separation Score**: Measures regime distinctiveness
- **Regime Consistency Score**: Measures regime internal consistency

### Micro-Regime Metrics

- **Micro-Regime Detection Accuracy**: Measures micro-regime detection quality
- **Micro-Regime Types**: Types of detected micro-regimes
- **Micro-Regime Scores**: Quality scores for micro-regimes

## 🎯 Regime Types

### Standard Regimes (10-15 states)

The NAS clustering detects 10-15 different market states optimized for short-term trading:

1. **High Volatility Bull**: Strong upward movement with high volatility
2. **High Volatility Bear**: Strong downward movement with high volatility
3. **Low Volatility Bull**: Steady upward movement with low volatility
4. **Low Volatility Bear**: Steady downward movement with low volatility
5. **Sideways High Volatility**: Range-bound with high volatility
6. **Sideways Low Volatility**: Range-bound with low volatility
7. **Breakout Bull**: Strong upward breakout
8. **Breakout Bear**: Strong downward breakout
9. **Reversal Bull**: Bullish reversal pattern
10. **Reversal Bear**: Bearish reversal pattern
11. **Consolidation**: Price consolidation phase
12. **Accumulation**: Accumulation phase
13. **Distribution**: Distribution phase
14. **Trending**: Strong directional trend
15. **Ranging**: Range-bound market

### Micro-Regimes (6 types)

The micro-regime detector identifies subtle market changes:

1. **Breakout**: Price breakout from consolidation
2. **Consolidation**: Price consolidation phase
3. **Reversal**: Price reversal pattern
4. **Acceleration**: Momentum acceleration
5. **Volume Spike**: Unusual volume activity
6. **Volatility Spike**: Unusual volatility activity

## 🔄 Pipeline Replacement

### HMM Clustering Replacement

The NAS clustering module **replaces** the existing HMM clustering pipeline with enhanced capabilities:

```python
# Replace HMM clustering with NAS clustering
from src.training.steps.market_analysis.nas_clustering import NASClusteringComponent

# Use as HMM replacement in existing pipeline
component = NASClusteringComponent(config)
result = await component.execute(data, pipeline_state)
```

### ML Model Training Support

The module provides enhanced features optimized for ML model training (DeepScale, LGBM, XGBoost, etc.):

```python
# Access ML training data
ml_training_data = result['ml_training_data']

# Regime sequences for ML training
regime_sequences = ml_training_data['regime_sequences']
micro_regime_sequences = ml_training_data['micro_regime_sequences']

# Enhanced features for ML models
regime_features = ml_training_data['regime_features']
transition_features = ml_training_data['transition_features']
economic_features = ml_training_data['economic_features']
trading_features = ml_training_data['trading_features']
micro_regime_features = ml_training_data['micro_regime_features']
market_features = ml_training_data['market_features']
```

## 📊 Visualization

### Regime Visualization

```python
from src.training.steps.market_analysis.nas_clustering import NASVisualizer

# Create visualizer
visualizer = NASVisualizer(config)

# Visualize regimes
visualizer.plot_regime_evolution(labels, timestamps)
visualizer.plot_regime_transitions(transition_matrix)
visualizer.plot_economic_significance(economic_scores)
visualizer.plot_trading_viability(trading_scores)
```

### Micro-Regime Visualization

```python
# Visualize micro-regimes
visualizer.plot_micro_regimes(micro_regime_result)
visualizer.plot_regime_quality_metrics(quality_metrics)
visualizer.plot_nas_architecture_performance(nas_architectures)
```

## 🧪 Testing

### Unit Tests

```python
# Test NAS clustering
python -m pytest tests/test_nas_clustering.py

# Test micro-regime detection
python -m pytest tests/test_micro_regime_detection.py

# Test feature extraction
python -m pytest tests/test_nas_feature_extraction.py
```

### Integration Tests

```python
# Test pipeline integration
python -m pytest tests/test_pipeline_integration.py

# Test output compatibility
python -m pytest tests/test_output_compatibility.py
```

## 📚 Examples

### Complete Pipeline Example

```python
import asyncio
from src.training.steps.market_analysis.nas_clustering import NASOrchestrator

async def main():
    # Initialize NAS orchestrator
    config = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'data_dir': 'historical_data',
        'nas_config': {
            'n_regimes': 12,
            'enable_micro_regime_detection': True,
            'economic_significance_threshold': 0.7,
            'trading_viability_threshold': 0.6
        }
    }
    
    orchestrator = NASOrchestrator(config)
    
    # Load market data
    market_data = load_market_data('BTCUSDT', 'binance', '15m')
    timestamps = market_data.index.values
    
    # Run NAS clustering
    results = await orchestrator.run_nas_clustering(
        data=market_data,
        timestamps=timestamps,
        symbol='BTCUSDT',
        exchange='binance',
        timeframe='15m'
    )
    
    # Save results
    orchestrator.save_results(results, 'output/nas_clustering_results')
    
    # Print summary
    print(f"✅ NAS clustering completed in {results['execution_time']:.2f}s")
    print(f"📊 Detected {len(np.unique(results['clustering_result'].labels))} regimes")
    print(f"🔍 Micro-regimes: {len(results['micro_regime_result'].micro_regime_types)}")
    print(f"💰 Economic significance: {np.mean(results['clustering_result'].economic_significance_scores):.3f}")
    print(f"📈 Trading viability: {np.mean(results['clustering_result'].trading_viability_scores):.3f}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🚀 Performance

### Expected Performance

- **Execution Time**: 30-60 seconds for 1000 samples
- **Memory Usage**: 2-4 GB for large datasets
- **Regime Detection Accuracy**: 85-95% for standard regimes
- **Micro-Regime Detection Accuracy**: 70-85% for micro-regimes
- **Economic Significance**: 70-90% for economically relevant regimes
- **Trading Viability**: 60-80% for trading-viable regimes

### Optimization

- **Hardware Acceleration**: GPU acceleration for large datasets
- **Matrix Optimization**: Optimized matrix operations for performance
- **Memory Management**: Efficient memory usage for large datasets
- **Parallel Processing**: Parallel processing for multiple regimes

## 🔧 Troubleshooting

### Common Issues

1. **No features extracted**: Check data format and feature configuration
2. **Low regime quality**: Adjust NAS architecture parameters
3. **Micro-regime detection failed**: Check micro-regime sensitivity settings
4. **Pipeline compatibility issues**: Verify output format configuration

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug information
results = await orchestrator.run_nas_clustering(
    data=market_data,
    timestamps=timestamps,
    symbol='BTCUSDT',
    exchange='binance',
    timeframe='15m',
    debug=True
)
```

## 📄 License

This module is part of the Ares trading system and follows the same licensing terms.

## 🤝 Contributing

Contributions are welcome! Please see the main project documentation for contribution guidelines.

## 📞 Support

For support and questions, please refer to the main project documentation or create an issue in the project repository.