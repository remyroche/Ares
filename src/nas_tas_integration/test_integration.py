"""
Test script for NAS/TAS integration pipeline.

This script tests the complete flow from market data input to signal emission
through NAS/TAS regime detection and model selection.
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

from src.nas_tas_integration.unified_regime_training_pipeline import (
    UnifiedRegimeTrainingPipeline, UnifiedTrainingConfig
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


def test_regime_detection():
    """Test NAS/TAS regime detection."""
    logger.info("🧪 Testing NAS/TAS regime detection...")
    
    try:
        # Generate sample data
        market_data = generate_sample_market_data(1000, '5m')
        
        # Create pipeline
        config = UnifiedTrainingConfig(
            timeframes=['5m'],
            n_regimes=4,
            model_types=['random_forest', 'xgboost'],
            enable_hpo=False,  # Disable for faster testing
            enable_ensemble=True
        )
        
        pipeline = UnifiedRegimeTrainingPipeline(config)
        pipeline.initialize_components()
        
        # Test regime detection
        regime_result = pipeline._detect_regimes_nas_tas(market_data, '5m')
        
        if regime_result.success:
            n_regimes = len(np.unique(regime_result.regime_predictions))
            logger.info(f"✅ Regime detection successful: {n_regimes} regimes detected")
            logger.info(f"   Economic significance: {np.mean(regime_result.economic_significance_scores):.3f}")
            logger.info(f"   Financial relevance: {np.mean(regime_result.financial_relevance_scores):.3f}")
            return True
        else:
            logger.error(f"❌ Regime detection failed: {regime_result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Regime detection test failed: {e}")
        return False


def test_per_regime_training():
    """Test per-regime model training."""
    logger.info("🧪 Testing per-regime model training...")
    
    try:
        # Generate sample data
        market_data = {
            '5m': generate_sample_market_data(1000, '5m'),
            '15m': generate_sample_market_data(500, '15m')
        }
        
        # Create pipeline
        config = UnifiedTrainingConfig(
            timeframes=['5m', '15m'],
            n_regimes=4,
            model_types=['random_forest', 'xgboost'],
            enable_hpo=False,  # Disable for faster testing
            enable_ensemble=True
        )
        
        pipeline = UnifiedRegimeTrainingPipeline(config)
        pipeline.initialize_components()
        
        # Test training
        results = pipeline.train_regime_models(market_data)
        
        if results:
            logger.info("✅ Per-regime training successful")
            for timeframe, result in results.items():
                if 'training_results' in result:
                    n_models = len(result['training_results'].get('models', {}))
                    logger.info(f"   {timeframe}: {n_models} regimes trained")
            return True
        else:
            logger.error("❌ Per-regime training failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Per-regime training test failed: {e}")
        return False


def test_model_selection():
    """Test data-driven model selection."""
    logger.info("🧪 Testing data-driven model selection...")
    
    try:
        # Create pipeline
        config = UnifiedTrainingConfig(
            timeframes=['5m'],
            n_regimes=4,
            model_types=['random_forest', 'xgboost', 'lightgbm'],
            enable_hpo=False,
            enable_ensemble=True,
            max_ensemble_models=3
        )
        
        pipeline = UnifiedRegimeTrainingPipeline(config)
        pipeline.initialize_components()
        
        # Test model selection
        available_models = ['random_forest_5m', 'xgboost_5m', 'lightgbm_5m']
        
        # Simulate some performance data
        for regime_id in range(4):
            for model_name in available_models:
                # Generate sample predictions and actual values
                predictions = np.random.randint(0, 2, 100)
                actual_values = np.random.randint(0, 2, 100)
                
                # Register performance
                pipeline.model_selector.register_model_performance(
                    regime_id=regime_id,
                    model_name=model_name,
                    predictions=predictions,
                    actual_values=actual_values,
                    execution_time=0.1
                )
        
        # Test model selection
        for regime_id in range(4):
            selected_model, ensemble_weights = pipeline.model_selector.select_model_for_regime(
                regime_id, available_models
            )
            
            logger.info(f"   Regime {regime_id}: {selected_model} (weights: {ensemble_weights})")
        
        logger.info("✅ Model selection test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model selection test failed: {e}")
        return False


def test_signal_generation():
    """Test signal generation pipeline."""
    logger.info("🧪 Testing signal generation pipeline...")
    
    try:
        # Generate sample data
        market_data = {
            '5m': generate_sample_market_data(1000, '5m'),
            '15m': generate_sample_market_data(500, '15m')
        }
        
        # Create pipeline
        config = UnifiedTrainingConfig(
            timeframes=['5m', '15m'],
            n_regimes=4,
            model_types=['random_forest', 'xgboost'],
            enable_hpo=False,
            enable_ensemble=True,
            enable_signal_generation=True
        )
        
        pipeline = UnifiedRegimeTrainingPipeline(config)
        pipeline.initialize_components()
        
        # Train models first
        training_results = pipeline.train_regime_models(market_data)
        
        if not training_results:
            logger.error("❌ Training failed, cannot test signal generation")
            return False
        
        # Test signal generation
        signals = pipeline.generate_signals(market_data)
        
        if signals:
            logger.info("✅ Signal generation successful")
            for timeframe, signal_data in signals.items():
                logger.info(f"   {timeframe}: {signal_data['signal']} (confidence: {signal_data['confidence']:.3f})")
            return True
        else:
            logger.error("❌ Signal generation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Signal generation test failed: {e}")
        return False


def test_end_to_end_flow():
    """Test complete end-to-end flow."""
    logger.info("🧪 Testing complete end-to-end flow...")
    
    try:
        # Generate sample data
        market_data = {
            '5m': generate_sample_market_data(1000, '5m'),
            '15m': generate_sample_market_data(500, '15m')
        }
        
        # Create pipeline
        config = UnifiedTrainingConfig(
            timeframes=['5m', '15m'],
            n_regimes=4,
            model_types=['random_forest', 'xgboost'],
            enable_hpo=False,
            enable_ensemble=True,
            enable_signal_generation=True
        )
        
        pipeline = UnifiedRegimeTrainingPipeline(config)
        
        # Initialize components
        if not pipeline.initialize_components():
            logger.error("❌ Pipeline initialization failed")
            return False
        
        # Test complete flow
        start_time = time.time()
        
        # Step 1: Train models
        logger.info("Step 1: Training regime models...")
        training_results = pipeline.train_regime_models(market_data)
        
        if not training_results:
            logger.error("❌ Training step failed")
            return False
        
        # Step 2: Generate signals
        logger.info("Step 2: Generating signals...")
        signals = pipeline.generate_signals(market_data)
        
        if not signals:
            logger.error("❌ Signal generation step failed")
            return False
        
        # Step 3: Get system status
        logger.info("Step 3: Checking system status...")
        status = pipeline.get_system_status()
        
        total_time = time.time() - start_time
        
        logger.info("✅ End-to-end flow test successful")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Timeframes processed: {len(training_results)}")
        logger.info(f"   Signals generated: {len(signals)}")
        logger.info(f"   System status: {status['pipeline_initialized']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ End-to-end flow test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    logger.info("🚀 Starting NAS/TAS integration tests...")
    
    tests = [
        ("Regime Detection", test_regime_detection),
        ("Per-Regime Training", test_per_regime_training),
        ("Model Selection", test_model_selection),
        ("Signal Generation", test_signal_generation),
        ("End-to-End Flow", test_end_to_end_flow)
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
        logger.info("🎉 All tests passed! NAS/TAS integration is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the logs for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)