"""
Example Usage of VectorBT-Enhanced Models

This script demonstrates how to use the enhanced PatchTST, GRU, and TFT models
with VectorBT integration for backtesting, financial metrics, and feature generation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced models
from src.models.vectorbt_enhanced_models import (
    create_patchtst_model,
    create_gru_model, 
    create_tft_model,
    create_all_models,
    VectorBTEnhancedModelInterface,
    UnifiedModelConfig,
    ModelType
)

# Import VectorBT configurations
try:
    from src.utils.ml_common.vectorbt_backtesting_engine import VectorBTBacktestConfig, BacktestMode
    from src.utils.ml_common.vectorbt_financial_metrics import FinancialMetricsConfig
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    logger.warning("VectorBT utils not available. Some features will be disabled.")


def generate_sample_data(n_samples: int = 1000, n_features: int = 10) -> tuple:
    """Generate sample time series data for demonstration."""
    np.random.seed(42)
    
    # Generate time series data
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='1H')
    
    # Generate features (OHLCV-like data)
    base_price = 100
    price_changes = np.random.normal(0, 0.01, n_samples)
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Create OHLCV data
    ohlcv_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)
    
    # Generate additional features
    feature_data = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    features_df = pd.DataFrame(feature_data, columns=feature_names, index=dates)
    
    # Generate target (price change)
    target = np.diff(prices) / prices[:-1]
    target = np.append(target, 0)  # Pad last value
    
    return ohlcv_data, features_df, target


def demonstrate_patchtst_model():
    """Demonstrate PatchTST model with VectorBT integration."""
    logger.info("🚀 Demonstrating PatchTST Model with VectorBT Integration")
    
    # Generate sample data
    ohlcv_data, features_df, target = generate_sample_data()
    
    # Create PatchTST model
    model = create_patchtst_model(
        sequence_length=24,
        hidden_size=64,
        enable_vectorbt=True,
        enable_vectorbt_backtesting=True,
        enable_vectorbt_metrics=True,
        enable_vectorbt_features=True
    )
    
    # Prepare data
    X = features_df.values
    y = target
    
    # Fit model
    logger.info("📊 Fitting PatchTST model...")
    model.fit(X, y)
    
    # Make predictions
    logger.info("🔮 Making predictions...")
    predictions = model.predict(X)
    
    # Generate VectorBT features
    logger.info("⚡ Generating VectorBT features...")
    vectorbt_features = model.generate_vectorbt_features(ohlcv_data)
    logger.info(f"Generated {vectorbt_features.shape[1]} VectorBT features")
    
    # Run VectorBT backtest if available
    if VECTORBT_AVAILABLE:
        logger.info("📈 Running VectorBT backtest...")
        # Convert predictions to signals
        signals = np.where(predictions > 0.001, 1, np.where(predictions < -0.001, -1, 0))
        
        backtest_results = model.run_vectorbt_backtest(
            signals=signals,
            prices=ohlcv_data['close'].values,
            timestamps=ohlcv_data.index,
            mode='cpu'
        )
        
        if backtest_results:
            logger.info("✅ Backtest completed successfully")
            logger.info(f"Total Return: {backtest_results['performance_metrics']['total_return']:.2%}")
            logger.info(f"Sharpe Ratio: {backtest_results['performance_metrics']['sharpe_ratio']:.3f}")
            logger.info(f"Max Drawdown: {backtest_results['performance_metrics']['max_drawdown']:.2%}")
    
    # Calculate VectorBT metrics
    if VECTORBT_AVAILABLE:
        logger.info("📊 Calculating VectorBT financial metrics...")
        portfolio_values = ohlcv_data['close'].values
        returns = ohlcv_data['close'].pct_change().dropna().values
        
        metrics = model.calculate_vectorbt_metrics(
            portfolio_values=portfolio_values,
            returns=returns,
            timestamps=ohlcv_data.index
        )
        
        if metrics:
            logger.info("✅ Financial metrics calculated successfully")
            logger.info(f"Volatility: {metrics['volatility']:.2%}")
            logger.info(f"Skewness: {metrics['skewness']:.3f}")
            logger.info(f"Kurtosis: {metrics['kurtosis']:.3f}")
    
    # Get VectorBT stats
    stats = model.get_vectorbt_stats()
    logger.info(f"📈 VectorBT Stats: {stats}")
    
    return model


def demonstrate_gru_model():
    """Demonstrate GRU model with VectorBT integration."""
    logger.info("🚀 Demonstrating GRU Model with VectorBT Integration")
    
    # Generate sample data
    ohlcv_data, features_df, target = generate_sample_data()
    
    # Create GRU model
    model = create_gru_model(
        sequence_length=24,
        hidden_size=64,
        enable_vectorbt=True,
        enable_vectorbt_backtesting=True,
        enable_vectorbt_metrics=True,
        enable_vectorbt_features=True
    )
    
    # Prepare data
    X = features_df.values
    y = target
    
    # Fit model
    logger.info("📊 Fitting GRU model...")
    model.fit(X, y)
    
    # Make predictions
    logger.info("🔮 Making predictions...")
    predictions = model.predict(X)
    
    # Generate VectorBT features
    logger.info("⚡ Generating VectorBT features...")
    vectorbt_features = model.generate_vectorbt_features(ohlcv_data)
    logger.info(f"Generated {vectorbt_features.shape[1]} VectorBT features")
    
    # Run VectorBT backtest if available
    if VECTORBT_AVAILABLE:
        logger.info("📈 Running VectorBT backtest...")
        # Convert predictions to signals
        signals = np.where(predictions > 0.001, 1, np.where(predictions < -0.001, -1, 0))
        
        backtest_results = model.run_vectorbt_backtest(
            signals=signals,
            prices=ohlcv_data['close'].values,
            timestamps=ohlcv_data.index,
            mode='cpu'
        )
        
        if backtest_results:
            logger.info("✅ Backtest completed successfully")
            logger.info(f"Total Return: {backtest_results['performance_metrics']['total_return']:.2%}")
            logger.info(f"Sharpe Ratio: {backtest_results['performance_metrics']['sharpe_ratio']:.3f}")
            logger.info(f"Max Drawdown: {backtest_results['performance_metrics']['max_drawdown']:.2%}")
    
    # Get VectorBT stats
    stats = model.get_vectorbt_stats()
    logger.info(f"📈 VectorBT Stats: {stats}")
    
    return model


def demonstrate_tft_model():
    """Demonstrate TFT model with VectorBT integration."""
    logger.info("🚀 Demonstrating TFT Model with VectorBT Integration")
    
    # Generate sample data
    ohlcv_data, features_df, target = generate_sample_data()
    
    # Create TFT model
    model = create_tft_model(
        sequence_length=24,
        hidden_size=64,
        enable_vectorbt=True,
        enable_vectorbt_backtesting=True,
        enable_vectorbt_metrics=True,
        enable_vectorbt_features=True
    )
    
    # Prepare data
    X = features_df.values
    y = target
    
    # Fit model
    logger.info("📊 Fitting TFT model...")
    model.fit(X, y)
    
    # Make predictions
    logger.info("🔮 Making predictions...")
    predictions = model.predict(X)
    
    # Generate VectorBT features
    logger.info("⚡ Generating VectorBT features...")
    vectorbt_features = model.generate_vectorbt_features(ohlcv_data)
    logger.info(f"Generated {vectorbt_features.shape[1]} VectorBT features")
    
    # Run VectorBT backtest if available
    if VECTORBT_AVAILABLE:
        logger.info("📈 Running VectorBT backtest...")
        # Convert predictions to signals
        signals = np.where(predictions > 0.001, 1, np.where(predictions < -0.001, -1, 0))
        
        backtest_results = model.run_vectorbt_backtest(
            signals=signals,
            prices=ohlcv_data['close'].values,
            timestamps=ohlcv_data.index,
            mode='cpu'
        )
        
        if backtest_results:
            logger.info("✅ Backtest completed successfully")
            logger.info(f"Total Return: {backtest_results['performance_metrics']['total_return']:.2%}")
            logger.info(f"Sharpe Ratio: {backtest_results['performance_metrics']['sharpe_ratio']:.3f}")
            logger.info(f"Max Drawdown: {backtest_results['performance_metrics']['max_drawdown']:.2%}")
    
    # Get VectorBT stats
    stats = model.get_vectorbt_stats()
    logger.info(f"📈 VectorBT Stats: {stats}")
    
    return model


def demonstrate_unified_interface():
    """Demonstrate the unified interface for all models."""
    logger.info("🚀 Demonstrating Unified Interface for All Models")
    
    # Generate sample data
    ohlcv_data, features_df, target = generate_sample_data()
    
    # Create all models
    models = create_all_models(
        sequence_length=24,
        hidden_size=64,
        enable_vectorbt=True,
        enable_vectorbt_backtesting=True,
        enable_vectorbt_metrics=True,
        enable_vectorbt_features=True
    )
    
    # Prepare data
    X = features_df.values
    y = target
    
    # Train and evaluate all models
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"📊 Training {model_name} model...")
        
        try:
            # Fit model
            model.fit(X, y)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Generate VectorBT features
            vectorbt_features = model.generate_vectorbt_features(ohlcv_data)
            
            # Store results
            results[model_name] = {
                'model': model,
                'predictions': predictions,
                'vectorbt_features': vectorbt_features,
                'vectorbt_stats': model.get_vectorbt_stats()
            }
            
            logger.info(f"✅ {model_name} model completed successfully")
            
        except Exception as e:
            logger.error(f"❌ {model_name} model failed: {e}")
            results[model_name] = {'error': str(e)}
    
    # Compare results
    logger.info("📊 Model Comparison Results:")
    for model_name, result in results.items():
        if 'error' not in result:
            stats = result['vectorbt_stats']
            logger.info(f"{model_name}: {stats}")
        else:
            logger.info(f"{model_name}: Error - {result['error']}")
    
    return results


def demonstrate_advanced_features():
    """Demonstrate advanced VectorBT features."""
    logger.info("🚀 Demonstrating Advanced VectorBT Features")
    
    # Generate sample data
    ohlcv_data, features_df, target = generate_sample_data()
    
    # Create model with advanced configuration
    model = create_patchtst_model(
        sequence_length=24,
        hidden_size=64,
        enable_vectorbt=True,
        enable_vectorbt_backtesting=True,
        enable_vectorbt_metrics=True,
        enable_vectorbt_features=True,
        enable_memory_optimization=True,
        enable_performance_monitoring=True,
        memory_limit_gb=4.0,
        enable_gpu=False,
        enable_parallel=True
    )
    
    # Prepare data
    X = features_df.values
    y = target
    
    # Fit model
    logger.info("📊 Fitting model with advanced features...")
    model.fit(X, y)
    
    # Demonstrate memory optimization
    logger.info("🧠 Memory optimization features:")
    memory_stats = model.get_vectorbt_stats()
    logger.info(f"Memory usage: {memory_stats.get('memory_usage_gb', 0):.2f} GB")
    logger.info(f"Memory utilization: {memory_stats.get('memory_utilization', 0):.1f}%")
    
    # Demonstrate performance monitoring
    logger.info("📊 Performance monitoring features:")
    logger.info(f"Total operations: {memory_stats.get('total_operations_monitored', 0)}")
    logger.info(f"Average duration: {memory_stats.get('average_operation_duration', 0):.3f}s")
    logger.info(f"Cache hit rate: {memory_stats.get('cache_hit_rate', 0):.1f}%")
    
    # Demonstrate feature generation
    logger.info("⚡ Advanced feature generation:")
    vectorbt_features = model.generate_vectorbt_features(ohlcv_data)
    logger.info(f"Generated {vectorbt_features.shape[1]} VectorBT features")
    logger.info(f"Feature names: {list(vectorbt_features.columns)}")
    
    return model


def main():
    """Main demonstration function."""
    logger.info("🎯 Starting VectorBT-Enhanced Models Demonstration")
    logger.info("=" * 60)
    
    try:
        # Demonstrate individual models
        logger.info("\n1. PatchTST Model Demonstration")
        logger.info("-" * 40)
        patchtst_model = demonstrate_patchtst_model()
        
        logger.info("\n2. GRU Model Demonstration")
        logger.info("-" * 40)
        gru_model = demonstrate_gru_model()
        
        logger.info("\n3. TFT Model Demonstration")
        logger.info("-" * 40)
        tft_model = demonstrate_tft_model()
        
        logger.info("\n4. Unified Interface Demonstration")
        logger.info("-" * 40)
        unified_results = demonstrate_unified_interface()
        
        logger.info("\n5. Advanced Features Demonstration")
        logger.info("-" * 40)
        advanced_model = demonstrate_advanced_features()
        
        logger.info("\n✅ All demonstrations completed successfully!")
        logger.info("=" * 60)
        
        # Summary
        logger.info("\n📊 Summary of Enhanced Models:")
        logger.info("• PatchTST: Transformer-based time series forecasting with VectorBT integration")
        logger.info("• GRU: Gated Recurrent Unit with VectorBT backtesting and metrics")
        logger.info("• TFT: Temporal Fusion Transformer with comprehensive VectorBT features")
        logger.info("• Unified Interface: Common API for all models with VectorBT capabilities")
        logger.info("• Advanced Features: Memory optimization, performance monitoring, feature generation")
        
        if VECTORBT_AVAILABLE:
            logger.info("\n🚀 VectorBT Integration Features:")
            logger.info("• High-performance backtesting engine")
            logger.info("• Comprehensive financial metrics calculation")
            logger.info("• Optimized feature generation")
            logger.info("• Memory management and optimization")
            logger.info("• Performance monitoring and statistics")
        else:
            logger.info("\n⚠️ VectorBT not available. Some features are disabled.")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()