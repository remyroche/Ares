# ML-Based Trading Indicators from Candle Patterns

This module provides a comprehensive system for generating trading indicators based on candlestick patterns using various machine learning models (LGBM, Random Forest, GRU, TFT).

## 🚀 Quick Start

```python
from src.feature_generation.categories.ml_indicator_integration import create_ml_indicator_system, ModelType, IndicatorType

# Create ML indicator system
system = create_ml_indicator_system()

# Train on historical data
training_results = system.train_system(data, symbol='BTCUSDT')

# Generate indicators
indicators = system.generate_indicators(data)
```

## 📋 Features

### Core Capabilities
- **Multiple ML Models**: Support for LGBM, Random Forest, GRU, and TFT
- **Candlestick Pattern Integration**: Uses existing pattern detection as input features
- **Market Context**: Combines patterns with volatility, volume, and momentum features
- **Real-time Generation**: Generate indicators on new data in real-time
- **Performance Evaluation**: Comprehensive backtesting and evaluation metrics
- **Model Persistence**: Save and load trained models

### Indicator Types
- **Directional Signal**: Buy/Sell/Hold signals (-1, 0, 1)
- **Strength Score**: Signal strength (0-1)
- **Confidence Level**: Prediction confidence (0-1)
- **Volatility Prediction**: Future volatility forecast
- **Price Target**: Price target prediction
- **Risk Score**: Risk assessment score

## 🏗️ Architecture

### System Components

1. **MLIndicatorGenerator**: Core generator class for traditional ML models
2. **NeuralIndicatorGenerator**: Neural network implementation (GRU/TFT)
3. **MLIndicatorTrainingPipeline**: Complete training pipeline with feature engineering
4. **MLIndicatorSystem**: Unified system interface

### Feature Engineering Pipeline

```
Raw OHLCV Data
    ↓
Candlestick Pattern Detection
    ↓
Market Context Features (Volatility, Volume, Momentum)
    ↓
Feature Combination & Selection
    ↓
ML Model Training
    ↓
Indicator Generation
```

## 📚 Usage Examples

### 1. Basic Usage

```python
from src.feature_generation.categories.ml_candle_pattern_indicators import create_ml_indicator_generator, ModelType

# Create generator
generator = create_ml_indicator_generator(
    model_type=ModelType.LIGHTGBM,
    indicator_types=[IndicatorType.DIRECTIONAL_SIGNAL, IndicatorType.STRENGTH_SCORE]
)

# Train on data
generator.train_models(data)

# Generate indicators
indicators = generator._generate_feature(data)
```

### 2. Advanced Configuration

```python
from src.feature_generation.categories.ml_candle_pattern_indicators import IndicatorConfig

# Custom configuration
config = IndicatorConfig(
    model_type=ModelType.RANDOM_FOREST,
    lookback_window=30,
    prediction_horizon=10,
    enable_market_context=True,
    confidence_threshold=0.8
)

generator = MLIndicatorGenerator(indicator_config=config)
```

### 3. Neural Networks

```python
from src.feature_generation.categories.ml_neural_indicators import create_neural_indicator_generator, NeuralConfig

# Neural network configuration
neural_config = NeuralConfig(
    hidden_size=128,
    num_layers=3,
    sequence_length=20,
    num_epochs=50
)

# Create neural generator
generator = create_neural_indicator_generator(
    model_type=ModelType.GRU,
    neural_config=neural_config
)
```

### 4. Complete Training Pipeline

```python
from src.feature_generation.categories.ml_indicator_training_pipeline import create_training_pipeline, TrainingConfig

# Training configuration
training_config = TrainingConfig(
    enable_feature_selection=True,
    max_features=30,
    enable_ensemble=True,
    enable_hyperparameter_optimization=True
)

# Create pipeline
pipeline = create_training_pipeline(training_config)

# Train all models
results = pipeline.train_all_models(data, symbol='BTCUSDT')
```

### 5. Real-time Trading System Integration

```python
class TradingSystem:
    def __init__(self):
        self.system = create_ml_indicator_system()
        self.is_trained = False
    
    def train(self, historical_data):
        self.system.train_system(historical_data, symbol='BTCUSDT')
        self.is_trained = True
    
    def generate_signal(self, current_data):
        if not self.is_trained:
            return None
        
        indicators = self.system.generate_indicators(current_data)
        return indicators['primary_indicator'].iloc[-1]
```

## 🔧 Configuration Options

### IndicatorConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | ModelType | LIGHTGBM | ML model type to use |
| `indicator_types` | List[IndicatorType] | [DIRECTIONAL_SIGNAL, STRENGTH_SCORE, CONFIDENCE_LEVEL] | Types of indicators to generate |
| `lookback_window` | int | 20 | Number of periods to look back |
| `prediction_horizon` | int | 5 | Number of periods ahead to predict |
| `enable_market_context` | bool | True | Include market context features |
| `confidence_threshold` | float | 0.7 | Minimum confidence for signals |

### NeuralConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_size` | int | 128 | Hidden layer size |
| `num_layers` | int | 2 | Number of layers |
| `sequence_length` | int | 20 | Input sequence length |
| `num_epochs` | int | 100 | Training epochs |
| `learning_rate` | float | 0.001 | Learning rate |
| `batch_size` | int | 32 | Batch size |

### TrainingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_feature_selection` | bool | True | Enable feature selection |
| `max_features` | int | 50 | Maximum number of features |
| `enable_ensemble` | bool | True | Enable ensemble methods |
| `enable_hyperparameter_optimization` | bool | True | Enable HPO |
| `hpo_trials` | int | 50 | Number of HPO trials |

## 📊 Performance Evaluation

### Built-in Metrics

- **Accuracy**: Classification accuracy for directional signals
- **Precision**: Precision score for signal classification
- **Recall**: Recall score for signal classification
- **F1-Score**: F1 score for signal classification
- **R² Score**: R-squared for regression indicators
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Maximum loss from peak
- **Win Rate**: Percentage of profitable trades

### Backtesting Integration

```python
# Simple backtesting example
def backtest_strategy(data, generator):
    indicators = generator._generate_feature(data)
    
    # Simple strategy: buy when indicator > 0.5, sell when < -0.5
    signals = np.where(indicators > 0.5, 1, 
                      np.where(indicators < -0.5, -1, 0))
    
    # Calculate returns
    returns = data['close'].pct_change()
    strategy_returns = signals * returns
    
    return strategy_returns.cumsum()
```

## 🚀 Advanced Features

### 1. Feature Engineering

The system automatically generates comprehensive features:

- **Candlestick Patterns**: Doji, hammer, engulfing, shooting star, etc.
- **Pattern Strength**: Consistency and reliability metrics
- **Market Context**: Volatility, volume, momentum features
- **Technical Indicators**: Moving averages, RSI, MACD, Bollinger Bands
- **Feature Interactions**: Polynomial and interaction features

### 2. Model Selection

```python
# Compare different models
from src.feature_generation.categories.ml_indicator_examples import MLIndicatorExamples

examples = MLIndicatorExamples()
comparison_results = examples.example_7_model_comparison()
```

### 3. Ensemble Methods

The system supports ensemble methods for improved performance:

- **Voting**: Majority vote across models
- **Stacking**: Meta-model learns to combine predictions
- **Blending**: Weighted combination of models

### 4. Real-time Updates

```python
# Real-time indicator generation
def update_indicators(system, new_data):
    indicators = system.generate_indicators(new_data)
    return indicators['primary_indicator'].iloc[-1]
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size for neural networks
   - Use feature selection to reduce dimensionality
   - Process data in smaller chunks

2. **Poor Performance**
   - Check data quality and format
   - Verify feature engineering pipeline
   - Try different model types
   - Adjust hyperparameters

3. **Training Failures**
   - Ensure data has required columns (OHLCV)
   - Check for missing values
   - Verify model configuration
   - Enable debug logging

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
generator = create_ml_indicator_generator()
generator.train_models(data)
```

## 📈 Performance Optimization

### 1. Feature Selection
- Use `enable_feature_selection=True`
- Set appropriate `max_features`
- Monitor feature importance scores

### 2. Model Selection
- Compare different model types
- Use ensemble methods
- Implement early stopping

### 3. Memory Management
- Use appropriate batch sizes
- Implement model checkpointing
- Monitor memory usage

## 🔄 Integration with Existing Systems

### VectorBT Integration

```python
import vectorbt as vbt

# Generate indicators
indicators = generator._generate_feature(data)

# Create VectorBT portfolio
portfolio = vbt.Portfolio.from_signals(
    data['close'], 
    entries=indicators > 0.5,
    exits=indicators < -0.5
)
```

### Trading System Integration

```python
class MLTradingStrategy:
    def __init__(self, symbol):
        self.symbol = symbol
        self.system = create_ml_indicator_system()
        self.is_trained = False
    
    def on_data(self, data):
        if not self.is_trained:
            self.system.train_system(data, symbol=self.symbol)
            self.is_trained = True
        
        indicators = self.system.generate_indicators(data)
        signal = indicators['primary_indicator'].iloc[-1]
        
        if signal > 0.5:
            self.buy()
        elif signal < -0.5:
            self.sell()
```

## 📚 Examples and Tutorials

### Complete Examples

Run the comprehensive examples:

```python
from src.feature_generation.categories.ml_indicator_examples import run_ml_indicator_examples

# Run all examples
results = run_ml_indicator_examples()
```

### Integration Guide

```python
from src.feature_generation.categories.ml_indicator_examples import create_integration_guide

# Get integration guide
guide = create_integration_guide()
print(guide)
```

## 🤝 Contributing

### Adding New Models

1. Extend the `ModelType` enum
2. Implement model-specific training logic
3. Add model to the factory
4. Update configuration options

### Adding New Indicators

1. Extend the `IndicatorType` enum
2. Implement indicator generation logic
3. Add evaluation metrics
4. Update documentation

## 📄 License

This module is part of the Ares trading system and follows the same licensing terms.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the examples and integration guide
3. Enable debug logging for detailed information
4. Check the performance metrics and evaluation results

---

**Note**: This system is designed for research and educational purposes. Always validate results with proper backtesting and risk management before using in live trading.