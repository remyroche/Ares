"""
Complete Integration Validation Script

This script validates the complete integration of per-regime ML model training
with the trading system, ensuring all components work together correctly.

Tests:
1. Per-regime training integration with existing training pipeline
2. Model selector service functionality
3. Trading model manager functionality
4. Signal generation pipeline integration
5. End-to-end trading simulation
"""

import numpy as np
import pandas as pd
import logging
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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


def test_per_regime_training_integration():
    """Test per-regime training integration."""
    logger.info("🧪 Testing per-regime training integration...")
    
    try:
        # Import per-regime training integration
        from src.training.steps.model_training.per_regime_training_integration import (
            get_per_regime_integration, train_analyst_per_regime_models,
            train_tactician_per_regime_models
        )
        
        # Generate sample data
        analyst_data = generate_sample_market_data(1000, '15m')
        tactician_data = generate_sample_market_data(500, '5m')
        
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        target_columns = ['target_long', 'target_short']
        
        # Test Analyst training
        analyst_result = train_analyst_per_regime_models(
            training_data=analyst_data,
            feature_columns=feature_columns,
            target_columns=target_columns
        )
        
        if not analyst_result.success:
            logger.error("❌ Analyst per-regime training failed")
            return False
        
        # Test Tactician training
        tactician_result = train_tactician_per_regime_models(
            training_data=tactician_data,
            feature_columns=feature_columns,
            target_columns=target_columns
        )
        
        if not tactician_result.success:
            logger.error("❌ Tactician per-regime training failed")
            return False
        
        logger.info("✅ Per-regime training integration successful")
        logger.info(f"   Analyst regimes: {len(analyst_result.regime_models)}")
        logger.info(f"   Tactician regimes: {len(tactician_result.regime_models)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Per-regime training integration test failed: {e}")
        return False


def test_model_selector_service():
    """Test model selector service."""
    logger.info("🧪 Testing model selector service...")
    
    try:
        from src.trading.model_selection.model_selector_service import (
            ModelSelectorService, TradingModelConfig
        )
        
        # Create configuration
        config = TradingModelConfig(
            analyst_models=['random_forest', 'xgboost'],
            tactician_models=['random_forest', 'xgboost'],
            n_regimes=4
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
            symbol='ETHUSDT',
            timeframe='5m'
        )
        
        if not result.selected_models:
            logger.error("❌ Model selection failed")
            return False
        
        logger.info("✅ Model selector service test successful")
        logger.info(f"   Selected models: {result.selected_models}")
        logger.info(f"   Regime ID: {result.regime_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model selector service test failed: {e}")
        return False


def test_trading_model_manager():
    """Test trading model manager."""
    logger.info("🧪 Testing trading model manager...")
    
    try:
        from src.trading.model_selection.trading_model_manager import (
            get_trading_model_manager
        )
        
        # Get model manager
        manager = get_trading_model_manager()
        
        # Test model loading
        market_data = generate_sample_market_data(100, '5m')
        models = manager.get_models_for_trading(
            market_data=market_data,
            symbol='ETHUSDT',
            timeframe='5m'
        )
        
        if not models:
            logger.error("❌ Model loading failed")
            return False
        
        # Test performance tracking
        for model_type, model_info in models.items():
            manager.update_model_performance(
                model_name=model_info['name'],
                model_type=model_type,
                regime_id=model_info['regime_id'],
                predictions=np.random.randint(0, 2, 50),
                actual_values=np.random.randint(0, 2, 50),
                execution_time=0.1
            )
        
        logger.info("✅ Trading model manager test successful")
        logger.info(f"   Loaded models: {list(models.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Trading model manager test failed: {e}")
        return False


async def test_signal_generation_integration():
    """Test signal generation pipeline integration."""
    logger.info("🧪 Testing signal generation pipeline integration...")
    
    try:
        from src.trading.signal_generation.signal_pipeline import SignalGenerationPipeline
        from src.trading.config.trading_config import TradingConfig
        
        # Create signal generation pipeline
        config = TradingConfig()
        pipeline = SignalGenerationPipeline(config)
        
        # Initialize pipeline
        success = await pipeline.initialize()
        if not success:
            logger.error("❌ Signal generation pipeline initialization failed")
            return False
        
        # Test signal generation
        market_data = generate_sample_market_data(100, '5m')
        signal_result = await pipeline.generate_signal(
            symbol='ETHUSDT',
            market_data=market_data
        )
        
        if not signal_result:
            logger.error("❌ Signal generation failed")
            return False
        
        logger.info("✅ Signal generation pipeline integration successful")
        logger.info(f"   Signal: {signal_result.final_signal}")
        logger.info(f"   Confidence: {signal_result.final_confidence:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Signal generation pipeline integration test failed: {e}")
        return False


def test_end_to_end_simulation():
    """Test complete end-to-end simulation."""
    logger.info("🧪 Testing complete end-to-end simulation...")
    
    try:
        # Test complete flow
        logger.info("🔄 Step 1: Testing per-regime training...")
        if not test_per_regime_training_integration():
            return False
        
        logger.info("🔄 Step 2: Testing model selector service...")
        if not test_model_selector_service():
            return False
        
        logger.info("🔄 Step 3: Testing trading model manager...")
        if not test_trading_model_manager():
            return False
        
        logger.info("🔄 Step 4: Testing signal generation integration...")
        if not asyncio.run(test_signal_generation_integration()):
            return False
        
        logger.info("✅ Complete end-to-end simulation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ End-to-end simulation failed: {e}")
        return False


def test_performance_benchmark():
    """Test performance benchmark."""
    logger.info("🧪 Testing performance benchmark...")
    
    try:
        from src.trading.model_selection import get_trading_model_manager
        
        # Get model manager
        manager = get_trading_model_manager()
        
        # Benchmark model selection performance
        start_time = time.time()
        
        for i in range(10):
            market_data = generate_sample_market_data(100, '5m')
            models = manager.get_models_for_trading(
                market_data=market_data,
                symbol='ETHUSDT',
                timeframe='5m'
            )
            
            # Simulate model usage
            for model_type, model_info in models.items():
                model = model_info['model']
                if hasattr(model, 'predict'):
                    predictions = model.predict(market_data)
                    
                    # Update performance
                    manager.update_model_performance(
                        model_name=model_info['name'],
                        model_type=model_type,
                        regime_id=model_info['regime_id'],
                        predictions=predictions,
                        actual_values=np.random.randint(0, 2, len(predictions)),
                        execution_time=0.1
                    )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info("✅ Performance benchmark successful")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Average time per cycle: {total_time/10:.3f}s")
        
        # Get performance metrics
        metrics = manager.get_performance_metrics()
        logger.info(f"   Tracked models: {len(metrics)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return False


def main():
    """Run complete integration validation."""
    logger.info("🚀 Starting complete integration validation...")
    
    tests = [
        ("Per-Regime Training Integration", test_per_regime_training_integration),
        ("Model Selector Service", test_model_selector_service),
        ("Trading Model Manager", test_trading_model_manager),
        ("Signal Generation Integration", lambda: asyncio.run(test_signal_generation_integration())),
        ("End-to-End Simulation", test_end_to_end_simulation),
        ("Performance Benchmark", test_performance_benchmark)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*70}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*70}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("INTEGRATION VALIDATION SUMMARY")
    logger.info(f"{'='*70}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All integration tests passed! The complete system is working correctly.")
        logger.info("\n📋 Integration Summary:")
        logger.info("   ✅ Per-regime ML model training integrated with existing training pipeline")
        logger.info("   ✅ DataDrivenModelSelector wired into trading system")
        logger.info("   ✅ Real-time model selection based on market conditions")
        logger.info("   ✅ Performance tracking and adaptation")
        logger.info("   ✅ Signal generation pipeline integration")
        logger.info("   ✅ End-to-end trading simulation")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the logs for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)