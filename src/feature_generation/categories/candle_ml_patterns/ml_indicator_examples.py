"""
ML Indicator Generator Examples and Integration Guide

This module provides comprehensive examples and integration patterns for using
ML-based trading indicators generated from candlestick patterns.

Key Features:
- Complete usage examples for all model types
- Integration with existing trading systems
- Performance evaluation and backtesting
- Real-time indicator generation
- Model comparison and selection
"""

import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Core imports
from .ml_candle_pattern_indicators import (
    MLIndicatorGenerator, IndicatorType, ModelType, IndicatorConfig,
    create_ml_indicator_generator
)
from .ml_indicator_training_pipeline import (
    MLIndicatorTrainingPipeline, TrainingConfig, create_training_pipeline
)
from .ml_neural_indicators import (
    NeuralIndicatorGenerator, NeuralConfig, create_neural_indicator_generator
)

# VectorBT imports for backtesting
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLIndicatorExamples:
    """Comprehensive examples for ML indicator generation."""
    
    def __init__(self):
        self.results = {}
        self.performance_metrics = {}
    
    def example_1_basic_usage(self):
        """Example 1: Basic usage with LightGBM model."""
        print("📚 Example 1: Basic ML Indicator Generation with LightGBM")
        print("=" * 60)
        
        # Create sample data
        data = self._create_sample_data(n_samples=1000)
        
        # Create indicator generator
        generator = create_ml_indicator_generator(
            model_type=ModelType.LIGHTGBM,
            indicator_types=[
                IndicatorType.DIRECTIONAL_SIGNAL,
                IndicatorType.STRENGTH_SCORE,
                IndicatorType.CONFIDENCE_LEVEL
            ]
        )
        
        # Train models
        print("🔧 Training models...")
        generator.train_models(data)
        
        # Generate indicators
        print("🔮 Generating indicators...")
        indicators = generator._generate_feature(data)
        
        # Display results
        self._display_indicator_results(indicators, "LightGBM")
        
        # Store results
        self.results['lightgbm'] = {
            'generator': generator,
            'indicators': indicators,
            'data': data
        }
        
        return generator, indicators
    
    def example_2_random_forest(self):
        """Example 2: Random Forest model with custom configuration."""
        print("\n📚 Example 2: Random Forest with Custom Configuration")
        print("=" * 60)
        
        # Create sample data
        data = self._create_sample_data(n_samples=1500)
        
        # Custom configuration
        indicator_config = IndicatorConfig(
            model_type=ModelType.RANDOM_FOREST,
            indicator_types=[
                IndicatorType.DIRECTIONAL_SIGNAL,
                IndicatorType.VOLATILITY_PREDICTION,
                IndicatorType.RISK_SCORE
            ],
            lookback_window=30,
            prediction_horizon=10,
            enable_market_context=True,
            enable_volume_features=True,
            confidence_threshold=0.8
        )
        
        # Create generator
        generator = MLIndicatorGenerator(indicator_config=indicator_config)
        
        # Train models
        print("🔧 Training Random Forest models...")
        generator.train_models(data)
        
        # Generate indicators
        print("🔮 Generating indicators...")
        indicators = generator._generate_feature(data)
        
        # Display results
        self._display_indicator_results(indicators, "Random Forest")
        
        # Store results
        self.results['random_forest'] = {
            'generator': generator,
            'indicators': indicators,
            'data': data
        }
        
        return generator, indicators
    
    def example_3_neural_networks(self):
        """Example 3: Neural network models (GRU/TFT)."""
        print("\n📚 Example 3: Neural Network Models (GRU/TFT)")
        print("=" * 60)
        
        try:
            # Create sample data
            data = self._create_sample_data(n_samples=2000)
            
            # Neural network configuration
            neural_config = NeuralConfig(
                hidden_size=128,
                num_layers=3,
                sequence_length=20,
                num_epochs=50,
                learning_rate=0.001,
                batch_size=64
            )
            
            # Create GRU generator
            print("🧠 Training GRU model...")
            gru_generator = create_neural_indicator_generator(
                model_type=ModelType.GRU,
                neural_config=neural_config
            )
            gru_generator.train_neural_models(data)
            gru_indicators = gru_generator._generate_feature(data)
            
            # Create TFT generator
            print("🧠 Training TFT model...")
            tft_generator = create_neural_indicator_generator(
                model_type=ModelType.TFT,
                neural_config=neural_config
            )
            tft_generator.train_neural_models(data)
            tft_indicators = tft_generator._generate_feature(data)
            
            # Display results
            self._display_indicator_results(gru_indicators, "GRU")
            self._display_indicator_results(tft_indicators, "TFT")
            
            # Store results
            self.results['gru'] = {
                'generator': gru_generator,
                'indicators': gru_indicators,
                'data': data
            }
            self.results['tft'] = {
                'generator': tft_generator,
                'indicators': tft_indicators,
                'data': data
            }
            
            return gru_generator, tft_generator
            
        except ImportError:
            print("❌ PyTorch not available, skipping neural network examples")
            return None, None
    
    def example_4_training_pipeline(self):
        """Example 4: Complete training pipeline with feature engineering."""
        print("\n📚 Example 4: Complete Training Pipeline")
        print("=" * 60)
        
        # Create sample data
        data = self._create_sample_data(n_samples=3000)
        
        # Training configuration
        training_config = TrainingConfig(
            enable_feature_selection=True,
            max_features=30,
            enable_ensemble=True,
            enable_hyperparameter_optimization=True,
            hpo_trials=20,
            enable_cross_validation=True,
            cv_folds=5
        )
        
        # Create training pipeline
        pipeline = create_training_pipeline(training_config)
        
        # Train all models
        print("🔧 Training all models with pipeline...")
        results = pipeline.train_all_models(data, symbol='BTCUSDT')
        
        # Display results
        print("\n📊 Training Results Summary:")
        for indicator_type, result in results.items():
            if isinstance(result, dict) and result.get('success', False):
                print(f"   ✅ {indicator_type}: Success")
                if 'test_score' in result:
                    print(f"      Test Score: {result['test_score']:.4f}")
                if 'metrics' in result:
                    metrics = result['metrics']
                    if 'accuracy' in metrics:
                        print(f"      Accuracy: {metrics['accuracy']:.4f}")
                    if 'r2' in metrics:
                        print(f"      R² Score: {metrics['r2']:.4f}")
            else:
                print(f"   ❌ {indicator_type}: Failed")
        
        # Store results
        self.results['pipeline'] = {
            'pipeline': pipeline,
            'results': results,
            'data': data
        }
        
        return pipeline, results
    
    def example_5_backtesting(self):
        """Example 5: Backtesting and performance evaluation."""
        print("\n📚 Example 5: Backtesting and Performance Evaluation")
        print("=" * 60)
        
        if not VECTORBT_AVAILABLE:
            print("❌ VectorBT not available, skipping backtesting example")
            return None
        
        # Create sample data
        data = self._create_sample_data(n_samples=2000)
        
        # Create multiple generators for comparison
        generators = {}
        
        # LightGBM
        generators['LightGBM'] = create_ml_indicator_generator(ModelType.LIGHTGBM)
        generators['LightGBM'].train_models(data)
        
        # Random Forest
        generators['RandomForest'] = create_ml_indicator_generator(ModelType.RANDOM_FOREST)
        generators['RandomForest'].train_models(data)
        
        # Generate indicators and backtest
        backtest_results = {}
        
        for name, generator in generators.items():
            print(f"🔮 Generating indicators with {name}...")
            indicators = generator._generate_feature(data)
            
            # Simple backtesting strategy
            backtest_result = self._simple_backtest(data, indicators)
            backtest_results[name] = backtest_result
            
            print(f"   📈 {name} Backtest Results:")
            print(f"      Total Return: {backtest_result['total_return']:.2%}")
            print(f"      Sharpe Ratio: {backtest_result['sharpe_ratio']:.2f}")
            print(f"      Max Drawdown: {backtest_result['max_drawdown']:.2%}")
            print(f"      Win Rate: {backtest_result['win_rate']:.2%}")
        
        # Store results
        self.results['backtesting'] = {
            'generators': generators,
            'backtest_results': backtest_results,
            'data': data
        }
        
        return backtest_results
    
    def example_6_real_time_generation(self):
        """Example 6: Real-time indicator generation."""
        print("\n📚 Example 6: Real-time Indicator Generation")
        print("=" * 60)
        
        # Create sample data
        data = self._create_sample_data(n_samples=1000)
        
        # Create generator
        generator = create_ml_indicator_generator(ModelType.LIGHTGBM)
        
        # Train on historical data
        print("🔧 Training on historical data...")
        generator.train_models(data.iloc[:800])  # Train on first 800 samples
        
        # Simulate real-time generation
        print("🔄 Simulating real-time generation...")
        real_time_indicators = []
        
        for i in range(800, len(data)):
            # Get recent data window
            recent_data = data.iloc[i-50:i+1]  # 50-period lookback
            
            # Generate indicator
            indicator = generator._generate_feature(recent_data)
            real_time_indicators.append(indicator.iloc[-1])  # Get latest value
            
            if i % 50 == 0:
                print(f"   Generated indicator for sample {i}: {indicator.iloc[-1]:.4f}")
        
        # Display results
        real_time_indicators = np.array(real_time_indicators)
        print(f"\n📊 Real-time Generation Results:")
        print(f"   Generated {len(real_time_indicators)} real-time indicators")
        print(f"   Mean: {real_time_indicators.mean():.4f}")
        print(f"   Std: {real_time_indicators.std():.4f}")
        print(f"   Min: {real_time_indicators.min():.4f}")
        print(f"   Max: {real_time_indicators.max():.4f}")
        
        # Store results
        self.results['real_time'] = {
            'generator': generator,
            'indicators': real_time_indicators,
            'data': data
        }
        
        return generator, real_time_indicators
    
    def example_7_model_comparison(self):
        """Example 7: Model comparison and selection."""
        print("\n📚 Example 7: Model Comparison and Selection")
        print("=" * 60)
        
        # Create sample data
        data = self._create_sample_data(n_samples=2000)
        
        # Test different model types
        model_types = [ModelType.LIGHTGBM, ModelType.RANDOM_FOREST]
        
        comparison_results = {}
        
        for model_type in model_types:
            print(f"🔧 Testing {model_type.value}...")
            
            # Create and train generator
            generator = create_ml_indicator_generator(model_type)
            generator.train_models(data)
            
            # Generate indicators
            indicators = generator._generate_feature(data)
            
            # Evaluate performance
            performance = self._evaluate_indicator_performance(data, indicators)
            comparison_results[model_type.value] = performance
            
            print(f"   📊 {model_type.value} Performance:")
            print(f"      Accuracy: {performance['accuracy']:.4f}")
            print(f"      Precision: {performance['precision']:.4f}")
            print(f"      Recall: {performance['recall']:.4f}")
            print(f"      F1-Score: {performance['f1_score']:.4f}")
        
        # Find best model
        best_model = max(comparison_results.items(), key=lambda x: x[1]['f1_score'])
        print(f"\n🏆 Best Model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")
        
        # Store results
        self.results['comparison'] = {
            'comparison_results': comparison_results,
            'best_model': best_model,
            'data': data
        }
        
        return comparison_results
    
    def run_all_examples(self):
        """Run all examples and generate comprehensive report."""
        print("🚀 Running All ML Indicator Examples")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run all examples
        self.example_1_basic_usage()
        self.example_2_random_forest()
        self.example_3_neural_networks()
        self.example_4_training_pipeline()
        self.example_5_backtesting()
        self.example_6_real_time_generation()
        self.example_7_model_comparison()
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        total_time = time.time() - start_time
        print(f"\n🎉 All examples completed in {total_time:.2f} seconds")
        
        return self.results
    
    def _create_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Create realistic sample OHLCV data."""
        np.random.seed(42)
        
        # Generate price series with trend and volatility
        base_price = 100.0
        trend = np.linspace(0, 0.1, n_samples)  # 10% trend over period
        noise = np.random.normal(0, 0.02, n_samples)
        returns = trend + noise
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples)
        }, index=pd.date_range('2020-01-01', periods=n_samples, freq='1min'))
        
        # Ensure OHLC constraints
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data
    
    def _display_indicator_results(self, indicators: pd.Series, model_name: str):
        """Display indicator results."""
        print(f"\n📊 {model_name} Indicator Results:")
        print(f"   Mean: {indicators.mean():.4f}")
        print(f"   Std: {indicators.std():.4f}")
        print(f"   Min: {indicators.min():.4f}")
        print(f"   Max: {indicators.max():.4f}")
        print(f"   Non-zero: {(indicators != 0).sum()}")
    
    def _simple_backtest(self, data: pd.DataFrame, indicators: pd.Series) -> Dict[str, float]:
        """Simple backtesting strategy."""
        # Simple strategy: buy when indicator > 0.5, sell when < -0.5
        signals = np.where(indicators > 0.5, 1, np.where(indicators < -0.5, -1, 0))
        
        # Calculate returns
        returns = data['close'].pct_change()
        strategy_returns = signals * returns
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min()
        win_rate = (strategy_returns > 0).mean()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def _evaluate_indicator_performance(self, data: pd.DataFrame, 
                                      indicators: pd.Series) -> Dict[str, float]:
        """Evaluate indicator performance."""
        # Create binary signals
        signals = np.where(indicators > 0.5, 1, 0)
        
        # Create target (future price direction)
        future_returns = data['close'].pct_change().shift(-1)
        targets = np.where(future_returns > 0, 1, 0)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Align signals and targets
        min_len = min(len(signals), len(targets))
        signals = signals[:min_len]
        targets = targets[:min_len]
        
        accuracy = accuracy_score(targets, signals)
        precision = precision_score(targets, signals, zero_division=0)
        recall = recall_score(targets, signals, zero_division=0)
        f1_score_val = f1_score(targets, signals, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score_val
        }
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive report of all examples."""
        print("\n📋 Comprehensive ML Indicator Report")
        print("=" * 80)
        
        # Summary statistics
        total_examples = len(self.results)
        successful_examples = sum(1 for result in self.results.values() 
                                if 'generator' in result or 'pipeline' in result)
        
        print(f"📊 Summary Statistics:")
        print(f"   Total Examples: {total_examples}")
        print(f"   Successful: {successful_examples}")
        print(f"   Success Rate: {successful_examples/total_examples:.1%}")
        
        # Model performance comparison
        if 'comparison' in self.results:
            print(f"\n🏆 Model Performance Comparison:")
            for model, performance in self.results['comparison']['comparison_results'].items():
                print(f"   {model}: F1-Score = {performance['f1_score']:.4f}")
        
        # Backtesting results
        if 'backtesting' in self.results:
            print(f"\n📈 Backtesting Results:")
            for model, result in self.results['backtesting']['backtest_results'].items():
                print(f"   {model}: Return = {result['total_return']:.2%}, "
                      f"Sharpe = {result['sharpe_ratio']:.2f}")
        
        print(f"\n✅ Comprehensive report generated successfully!")


def run_ml_indicator_examples():
    """Main function to run all ML indicator examples."""
    examples = MLIndicatorExamples()
    return examples.run_all_examples()


def create_integration_guide():
    """Create integration guide for ML indicators."""
    guide = """
# ML Indicator Generator Integration Guide

## Quick Start

### 1. Basic Usage
```python
from src.feature_generation.categories.ml_candle_pattern_indicators import create_ml_indicator_generator, ModelType, IndicatorType

# Create generator
generator = create_ml_indicator_generator(
    model_type=ModelType.LIGHTGBM,
    indicator_types=[IndicatorType.DIRECTIONAL_SIGNAL, IndicatorType.STRENGTH_SCORE]
)

# Train on historical data
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

### 4. Training Pipeline
```python
from src.feature_generation.categories.ml_indicator_training_pipeline import create_training_pipeline, TrainingConfig

# Training configuration
training_config = TrainingConfig(
    enable_feature_selection=True,
    enable_ensemble=True,
    enable_hyperparameter_optimization=True
)

# Create pipeline
pipeline = create_training_pipeline(training_config)

# Train all models
results = pipeline.train_all_models(data, symbol='BTCUSDT')
```

## Integration Patterns

### Real-time Trading System
```python
class TradingSystem:
    def __init__(self):
        self.generator = create_ml_indicator_generator(ModelType.LIGHTGBM)
        self.is_trained = False
    
    def train(self, historical_data):
        self.generator.train_models(historical_data)
        self.is_trained = True
    
    def generate_signal(self, current_data):
        if not self.is_trained:
            return None
        
        indicators = self.generator._generate_feature(current_data)
        return indicators.iloc[-1]  # Latest indicator value
```

### Backtesting Integration
```python
def backtest_strategy(data, generator):
    indicators = generator._generate_feature(data)
    
    # Simple strategy
    signals = np.where(indicators > 0.5, 1, 
                      np.where(indicators < -0.5, -1, 0))
    
    # Calculate returns
    returns = data['close'].pct_change()
    strategy_returns = signals * returns
    
    return strategy_returns.cumsum()
```

## Performance Optimization

### 1. Feature Selection
- Use `enable_feature_selection=True` in TrainingConfig
- Set `max_features` to limit dimensionality
- Monitor feature importance scores

### 2. Model Selection
- Compare different model types using `example_7_model_comparison()`
- Use ensemble methods for better performance
- Implement early stopping for neural networks

### 3. Memory Management
- Use appropriate batch sizes for neural networks
- Implement model checkpointing for large datasets
- Monitor memory usage during training

## Best Practices

1. **Data Quality**: Ensure clean, properly formatted OHLCV data
2. **Feature Engineering**: Combine candlestick patterns with market context
3. **Model Validation**: Use cross-validation and backtesting
4. **Performance Monitoring**: Track model performance over time
5. **Regular Retraining**: Update models with new data periodically

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce batch size or sequence length
2. **Poor Performance**: Check data quality and feature engineering
3. **Training Failures**: Verify data format and model configuration
4. **Slow Training**: Use GPU acceleration or reduce model complexity

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for troubleshooting
generator = create_ml_indicator_generator()
generator.train_models(data)
```
"""
    
    return guide


if __name__ == "__main__":
    # Run all examples
    results = run_ml_indicator_examples()
    
    # Print integration guide
    print("\n" + "="*80)
    print("INTEGRATION GUIDE")
    print("="*80)
    print(create_integration_guide())