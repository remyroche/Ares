"""
Test script for trading model selection integration.

This script tests the complete integration of per-regime training with
the trading system, including model selection and real-time adaptation.
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.trading.model_selection.model_selector_service import (
    ModelSelectorService, ModelSelectionResult, TradingModelConfig
)
from src.trading.model_selection.trading_model_manager import (
    TradingModelManager, get_trading_model_manager, get_models_for_trading
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_market_data(n_samples: int, timeframe: str) -> pd.DataFrame:
    """Generate sample market data for testing."""
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
    return df


def test_model_selector_service():
    """Test model selector service initialization and functionality."""
    logger.info("🧪 Testing model selector service...")
    
    try:
        # Create configuration
        config = TradingModelConfig(
            analyst_models=['random_forest', 'xgboost'],
            tactician_models=['random_forest', 'xgboost'],
            n_regimes=4,
            primary_metric='f1_score',
            confidence_threshold=0.7,
            enable_ensemble=True,
            max_ensemble_models=3
        )
        
        # Initialize service
        service = ModelSelectorService(config)
        success = service.initialize()
        
        if not success:
            logger.error("❌ Model selector service initialization failed")
            return False
        
        # Test model selection
        market_data = generate_sample_market_data(100, '5m')
        result = service.select_models_for_trading(
            market_data=market_data,
            model_types=['random_forest', 'xgboost'],
            symbol='ETHUSDT',
            timeframe='5m'
        )
        
        if result.selected_models:
            logger.info("✅ Model selector service test successful")
            logger.info(f"   Selected models: {result.selected_models}")
            logger.info(f"   Regime ID: {result.regime_id}")
            logger.info(f"   Confidence: {result.confidence_score:.3f}")
            return True
        else:
            logger.error("❌ Model selection failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model selector service test failed: {e}")
        return False


def test_trading_model_manager():
    """Test trading model manager functionality."""
    logger.info("🧪 Testing trading model manager...")
    
    try:
        # Create configuration
        config = TradingModelConfig(
            analyst_models=['random_forest', 'xgboost'],
            tactician_models=['random_forest', 'xgboost'],
            n_regimes=4
        )
        
        # Initialize manager
        manager = TradingModelManager(config)
        success = manager.initialize()
        
        if not success:
            logger.error("❌ Trading model manager initialization failed")
            return False
        
        # Test model loading
        market_data = generate_sample_market_data(100, '5m')
        models = manager.get_models_for_trading(
            market_data=market_data,
            symbol='ETHUSDT',
            timeframe='5m'
        )
        
        if models:
            logger.info("✅ Trading model manager test successful")
            logger.info(f"   Loaded models: {list(models.keys())}")
            
            # Test performance tracking
            for model_type, model_info in models.items():
                manager.update_model_performance(
                    model_name=model_info['name'],
                    model_type=model_type,
                    regime_id=model_info['regime_id'],
                    predictions=np.random.randint(0, 2, 100),
                    actual_values=np.random.randint(0, 2, 100),
                    execution_time=0.1
                )
            
            # Get performance metrics
            metrics = manager.get_performance_metrics()
            logger.info(f"   Performance metrics: {len(metrics)} models tracked")
            
            return True
        else:
            logger.error("❌ Model loading failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Trading model manager test failed: {e}")
        return False


def test_model_selection_integration():
    """Test integration between model selector and model manager."""
    logger.info("🧪 Testing model selection integration...")
    
    try:
        # Test convenience function
        market_data = generate_sample_market_data(100, '5m')
        models = get_models_for_trading(
            market_data=market_data,
            symbol='ETHUSDT',
            timeframe='5m'
        )
        
        if models:
            logger.info("✅ Model selection integration test successful")
            logger.info(f"   Models loaded: {list(models.keys())}")
            
            # Test model properties
            for model_type, model_info in models.items():
                logger.info(f"   {model_type}: {model_info['name']} (weight: {model_info['weight']:.3f})")
            
            return True
        else:
            logger.error("❌ Model selection integration failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model selection integration test failed: {e}")
        return False


def test_performance_tracking():
    """Test performance tracking and adaptation."""
    logger.info("🧪 Testing performance tracking...")
    
    try:
        # Get model manager
        manager = get_trading_model_manager()
        
        # Simulate multiple trading cycles
        for i in range(5):
            market_data = generate_sample_market_data(100, '5m')
            models = manager.get_models_for_trading(
                market_data=market_data,
                symbol='ETHUSDT',
                timeframe='5m'
            )
            
            # Update performance for each model
            for model_type, model_info in models.items():
                # Simulate predictions and actual values
                predictions = np.random.randint(0, 2, 50)
                actual_values = np.random.randint(0, 2, 50)
                
                manager.update_model_performance(
                    model_name=model_info['name'],
                    model_type=model_type,
                    regime_id=model_info['regime_id'],
                    predictions=predictions,
                    actual_values=actual_values,
                    execution_time=0.1
                )
        
        # Get performance metrics
        metrics = manager.get_performance_metrics()
        
        if metrics:
            logger.info("✅ Performance tracking test successful")
            logger.info(f"   Tracked models: {len(metrics)}")
            
            for model_key, model_metrics in metrics.items():
                logger.info(f"   {model_key}: F1={model_metrics['f1_score']:.3f}, "
                          f"Accuracy={model_metrics['accuracy']:.3f}")
            
            return True
        else:
            logger.error("❌ Performance tracking failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Performance tracking test failed: {e}")
        return False


def test_system_status():
    """Test system status and monitoring."""
    logger.info("🧪 Testing system status...")
    
    try:
        # Get model manager
        manager = get_trading_model_manager()
        
        # Get system status
        status = manager.get_system_status()
        
        if status:
            logger.info("✅ System status test successful")
            logger.info(f"   Model selector ready: {status.get('model_selector_ready', False)}")
            logger.info(f"   Model loader ready: {status.get('model_loader_ready', False)}")
            logger.info(f"   Cached models: {status.get('cached_models', 0)}")
            logger.info(f"   Tracked models: {status.get('tracked_models', 0)}")
            
            return True
        else:
            logger.error("❌ System status test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ System status test failed: {e}")
        return False


def test_end_to_end_trading_simulation():
    """Test end-to-end trading simulation."""
    logger.info("🧪 Testing end-to-end trading simulation...")
    
    try:
        # Simulate trading session
        logger.info("🔄 Simulating trading session...")
        
        # Generate market data for different timeframes
        analyst_data = generate_sample_market_data(200, '15m')
        tactician_data = generate_sample_market_data(100, '5m')
        
        # Test Analyst model selection
        logger.info("🔄 Testing Analyst model selection...")
        analyst_models = get_models_for_trading(
            market_data=analyst_data,
            symbol='ETHUSDT',
            timeframe='15m'
        )
        
        if not analyst_models:
            logger.error("❌ Analyst model selection failed")
            return False
        
        # Test Tactician model selection
        logger.info("🔄 Testing Tactician model selection...")
        tactician_models = get_models_for_trading(
            market_data=tactician_data,
            symbol='ETHUSDT',
            timeframe='5m'
        )
        
        if not tactician_models:
            logger.error("❌ Tactician model selection failed")
            return False
        
        # Simulate model usage
        logger.info("🔄 Simulating model usage...")
        for model_type, model_info in {**analyst_models, **tactician_models}.items():
            model = model_info['model']
            if hasattr(model, 'predict'):
                # Generate predictions
                test_data = generate_sample_market_data(50, '5m')
                predictions = model.predict(test_data)
                
                # Update performance
                manager = get_trading_model_manager()
                manager.update_model_performance(
                    model_name=model_info['name'],
                    model_type=model_type,
                    regime_id=model_info['regime_id'],
                    predictions=predictions,
                    actual_values=np.random.randint(0, 2, len(predictions)),
                    execution_time=0.1
                )
        
        # Get final status
        manager = get_trading_model_manager()
        final_status = manager.get_system_status()
        
        logger.info("✅ End-to-end trading simulation successful")
        logger.info(f"   Analyst models: {len(analyst_models)}")
        logger.info(f"   Tactician models: {len(tactician_models)}")
        logger.info(f"   Total tracked models: {final_status.get('tracked_models', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ End-to-end trading simulation failed: {e}")
        return False


def main():
    """Run all trading integration tests."""
    logger.info("🚀 Starting trading model selection integration tests...")
    
    tests = [
        ("Model Selector Service", test_model_selector_service),
        ("Trading Model Manager", test_trading_model_manager),
        ("Model Selection Integration", test_model_selection_integration),
        ("Performance Tracking", test_performance_tracking),
        ("System Status", test_system_status),
        ("End-to-End Trading Simulation", test_end_to_end_trading_simulation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Trading model selection integration is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the logs for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)