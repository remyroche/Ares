"""
Test script for per-regime training integration.

This script tests the integration of per-regime ML model training into the
existing Analyst and Tactician training pipeline.
"""

import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.training.steps.model_training.per_regime_training_integration import (
    PerRegimeTrainingIntegration, PerRegimeTrainingResult,
    get_per_regime_integration, train_analyst_per_regime_models,
    train_tactician_per_regime_models, get_model_selector_for_trading
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_training_data(n_samples: int, timeframe: str) -> pd.DataFrame:
    """Generate sample training data for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Generate price data with some realistic patterns
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)  # 2% daily volatility
    
    # Add some regime-like patterns
    if n_samples > 200:
        # Add trending periods
        returns[100:150] += 0.01  # Uptrend
        returns[200:250] -= 0.01  # Downtrend
        returns[300:350] += 0.005  # Mild uptrend
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Generate realistic OHLC from close price
        volatility = abs(returns[i]) * 2
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume,
            'timestamp': datetime.now() - timedelta(minutes=(n_samples-i)*5 if timeframe == '5m' else (n_samples-i)*15)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # Add target columns
    df['target_long'] = (df['close'].shift(-1) > df['close']).astype(int)
    df['target_short'] = (df['close'].shift(-1) < df['close']).astype(int)
    
    # Remove last row where targets are NaN
    df = df[:-1]
    
    return df


def test_per_regime_integration_initialization():
    """Test per-regime training integration initialization."""
    logger.info("🧪 Testing per-regime training integration initialization...")
    
    try:
        # Test configuration
        config = {
            'n_regimes': 4,
            'timeframes': ['5m', '15m'],
            'model_types': ['random_forest', 'xgboost'],
            'enable_hpo': False,  # Disable for faster testing
            'enable_ensemble': True,
            'max_ensemble_models': 3
        }
        
        # Create integration
        integration = PerRegimeTrainingIntegration(config)
        
        # Initialize components
        success = integration.initialize_components()
        
        if success:
            logger.info("✅ Per-regime training integration initialization successful")
            return True
        else:
            logger.error("❌ Per-regime training integration initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Per-regime training integration initialization test failed: {e}")
        return False


def test_analyst_per_regime_training():
    """Test Analyst per-regime training."""
    logger.info("🧪 Testing Analyst per-regime training...")
    
    try:
        # Generate sample data
        training_data = generate_sample_training_data(1000, '15m')
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        target_columns = ['target_long', 'target_short']
        
        # Test training
        result = train_analyst_per_regime_models(
            training_data=training_data,
            feature_columns=feature_columns,
            target_columns=target_columns
        )
        
        if result.success:
            logger.info("✅ Analyst per-regime training successful")
            logger.info(f"   Regime models: {len(result.regime_models)}")
            logger.info(f"   Execution time: {result.execution_time:.2f}s")
            return True
        else:
            logger.error(f"❌ Analyst per-regime training failed: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Analyst per-regime training test failed: {e}")
        return False


def test_tactician_per_regime_training():
    """Test Tactician per-regime training."""
    logger.info("🧪 Testing Tactician per-regime training...")
    
    try:
        # Generate sample data
        training_data = generate_sample_training_data(500, '5m')
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        target_columns = ['target_long', 'target_short']
        
        # Test training
        result = train_tactician_per_regime_models(
            training_data=training_data,
            feature_columns=feature_columns,
            target_columns=target_columns
        )
        
        if result.success:
            logger.info("✅ Tactician per-regime training successful")
            logger.info(f"   Regime models: {len(result.regime_models)}")
            logger.info(f"   Execution time: {result.execution_time:.2f}s")
            return True
        else:
            logger.error(f"❌ Tactician per-regime training failed: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Tactician per-regime training test failed: {e}")
        return False


def test_model_selector_availability():
    """Test model selector availability for trading."""
    logger.info("🧪 Testing model selector availability...")
    
    try:
        # Get model selector
        model_selector = get_model_selector_for_trading()
        
        if model_selector is not None:
            logger.info("✅ Model selector available for trading")
            
            # Test system status
            status = model_selector.get_system_summary()
            logger.info(f"   System status: {status}")
            
            return True
        else:
            logger.error("❌ Model selector not available for trading")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model selector availability test failed: {e}")
        return False


def test_end_to_end_integration():
    """Test end-to-end integration."""
    logger.info("🧪 Testing end-to-end integration...")
    
    try:
        # Generate sample data
        analyst_data = generate_sample_training_data(1000, '15m')
        tactician_data = generate_sample_training_data(500, '5m')
        
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        target_columns = ['target_long', 'target_short']
        
        # Test Analyst training
        logger.info("🔄 Training Analyst per-regime models...")
        analyst_result = train_analyst_per_regime_models(
            training_data=analyst_data,
            feature_columns=feature_columns,
            target_columns=target_columns
        )
        
        if not analyst_result.success:
            logger.error("❌ Analyst training failed")
            return False
        
        # Test Tactician training
        logger.info("🔄 Training Tactician per-regime models...")
        tactician_result = train_tactician_per_regime_models(
            training_data=tactician_data,
            feature_columns=feature_columns,
            target_columns=target_columns
        )
        
        if not tactician_result.success:
            logger.error("❌ Tactician training failed")
            return False
        
        # Test model selector
        logger.info("🔄 Testing model selector...")
        model_selector = get_model_selector_for_trading()
        
        if model_selector is None:
            logger.error("❌ Model selector not available")
            return False
        
        # Test model selection
        logger.info("🔄 Testing model selection...")
        test_market_data = generate_sample_training_data(100, '5m')
        
        # This would normally be called from the trading system
        # For now, we'll just check if the selector is ready
        status = model_selector.get_system_summary()
        
        logger.info("✅ End-to-end integration successful")
        logger.info(f"   Analyst regimes: {len(analyst_result.regime_models)}")
        logger.info(f"   Tactician regimes: {len(tactician_result.regime_models)}")
        logger.info(f"   Model selector ready: {status.get('total_regimes', 0) > 0}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ End-to-end integration test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    logger.info("🚀 Starting per-regime training integration tests...")
    
    tests = [
        ("Per-Regime Integration Initialization", test_per_regime_integration_initialization),
        ("Analyst Per-Regime Training", test_analyst_per_regime_training),
        ("Tactician Per-Regime Training", test_tactician_per_regime_training),
        ("Model Selector Availability", test_model_selector_availability),
        ("End-to-End Integration", test_end_to_end_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Per-regime training integration is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the logs for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)