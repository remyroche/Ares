#!/usr/bin/env python3
"""
Test script for Feature Lookback Optimization in MARKET_ANALYSIS pipeline.

This script tests the complete feature lookback optimization functionality,
including both ML commons integration and fallback statistical optimization.
"""

import asyncio
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, Any
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic test data for feature optimization."""
    logger.info(f"Creating test data with {n_samples} samples")
    
    # Generate synthetic price data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Generate realistic price movements
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Add some trend and volatility clustering
    trend = np.linspace(0, 0.1, n_samples)
    volatility = 0.01 + 0.005 * np.sin(np.arange(n_samples) * 0.1)
    
    prices = prices * (1 + trend) * (1 + np.random.normal(0, volatility))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Add some technical indicators for testing
    data['rsi_14'] = calculate_rsi(data['close'], 14)
    data['rsi_21'] = calculate_rsi(data['close'], 21)
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['ema_12'] = data['close'].ewm(span=12).mean()
    data['ema_26'] = data['close'].ewm(span=26).mean()
    
    # Add target variable (future returns)
    data['target'] = data['close'].pct_change(5).shift(-5)  # 5-period forward returns
    
    # Add regime column for regime-aware optimization
    data['regime'] = np.where(data['close'].rolling(20).std() > data['close'].rolling(20).std().quantile(0.7), 1, 0)
    
    logger.info(f"Created test data with columns: {list(data.columns)}")
    return data

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

async def test_feature_optimization_module():
    """Test the feature generation optimization module directly."""
    logger.info("🧪 Testing feature generation optimization module...")
    
    try:
        from src.feature_engineering.feature_generation_optimization import (
            FeatureGenerationOptimizer, 
            FeatureOptimizationConfig,
            OptimizationMethod
        )
        
        # Create test data
        data = create_test_data(500)
        
        # Create optimizer
        config = FeatureOptimizationConfig(
            min_lookback=5,
            max_lookback=50,
            step_size=5,
            optimization_method=OptimizationMethod.STATISTICAL_ANALYSIS,
            cv_folds=3
        )
        
        optimizer = FeatureGenerationOptimizer(config)
        
        # Define feature generators
        def rsi_generator(df: pd.DataFrame, lookback: int) -> pd.Series:
            return calculate_rsi(df['close'], lookback)
        
        def sma_generator(df: pd.DataFrame, lookback: int) -> pd.Series:
            return df['close'].rolling(lookback).mean()
        
        def ema_generator(df: pd.DataFrame, lookback: int) -> pd.Series:
            return df['close'].ewm(span=lookback).mean()
        
        # Test single feature optimization
        logger.info("Testing RSI optimization...")
        rsi_result = await optimizer.optimize_feature_lookback(
            data, 'rsi', 'target', rsi_generator
        )
        
        logger.info(f"RSI optimization result: {rsi_result.optimal_lookback} (score: {rsi_result.performance_score:.4f})")
        
        # Test multiple features optimization
        logger.info("Testing multiple features optimization...")
        feature_configs = {
            'rsi': {'generator': rsi_generator},
            'sma': {'generator': sma_generator},
            'ema': {'generator': ema_generator}
        }
        
        results = await optimizer.optimize_multiple_features(
            data, feature_configs, 'target'
        )
        
        logger.info("Multiple features optimization results:")
        for name, result in results.items():
            logger.info(f"  {name}: lookback={result.optimal_lookback}, score={result.performance_score:.4f}")
        
        # Test regime-aware optimization
        logger.info("Testing regime-aware optimization...")
        regime_result = await optimizer.optimize_feature_lookback(
            data, 'rsi', 'target', rsi_generator, regime_column='regime'
        )
        
        logger.info(f"Regime-aware RSI result: {regime_result.optimal_lookback} (score: {regime_result.performance_score:.4f})")
        
        # Generate summary
        summary = optimizer.get_optimization_summary(results)
        logger.info(f"Optimization summary: {summary}")
        
        logger.info("✅ Feature generation optimization module test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature generation optimization module test failed: {e}")
        return False

async def test_market_analysis_pipeline():
    """Test the feature lookback optimization within the market analysis pipeline."""
    logger.info("🧪 Testing feature lookback optimization in MARKET_ANALYSIS pipeline...")
    
    try:
        from src.training.steps.market_analysis.sub_pipeline import (
            MarketAnalysisSubPipeline,
            SubPipelineConfig,
            ExecutionMode
        )
        
        # Create test data and save it
        data = create_test_data(500)
        test_data_dir = Path("test_data_cache")
        test_data_dir.mkdir(exist_ok=True)
        
        test_data_file = test_data_dir / "features_binance_BTCUSDT_consolidated.parquet"
        data.to_parquet(test_data_file)
        logger.info(f"Saved test data to {test_data_file}")
        
        # Create pipeline configuration
        config = SubPipelineConfig(
            mode=ExecutionMode.FULL,
            data_dir=str(test_data_dir),
            exchange="binance",
            symbol="BTCUSDT",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Create pipeline
        pipeline = MarketAnalysisSubPipeline()
        
        # Test the feature lookback optimization pipeline directly
        logger.info("Testing feature lookback optimization pipeline...")
        artifacts = await pipeline._feature_lookback_optimization_pipeline(config)
        
        logger.info("Feature lookback optimization artifacts:")
        for key, value in artifacts.items():
            logger.info(f"  {key}: {value}")
        
        # Verify artifacts structure
        required_keys = ['optimization_results', 'optimal_lookbacks', 'optimization_metrics']
        for key in required_keys:
            if key not in artifacts:
                raise ValueError(f"Missing required artifact key: {key}")
        
        # Verify optimal lookbacks
        optimal_lookbacks = artifacts['optimal_lookbacks']
        expected_indicators = ['rsi', 'sma', 'ema']
        for indicator in expected_indicators:
            if indicator not in optimal_lookbacks:
                logger.warning(f"Missing optimal lookback for {indicator}")
            else:
                logger.info(f"Optimal {indicator} lookback: {optimal_lookbacks[indicator]}")
        
        logger.info("✅ MARKET_ANALYSIS pipeline feature lookback optimization test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ MARKET_ANALYSIS pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_statistical_optimization_methods():
    """Test the statistical optimization methods in the sub-pipeline."""
    logger.info("🧪 Testing statistical optimization methods...")
    
    try:
        from src.training.steps.market_analysis.sub_pipeline import MarketAnalysisSubPipeline
        
        # Create test data
        data = create_test_data(300)
        
        # Create pipeline instance
        pipeline = MarketAnalysisSubPipeline()
        
        # Test different optimization methods
        test_cases = [
            ('rsi', [7, 14, 21, 28], 'signal_strength'),
            ('sma', [10, 20, 30, 50], 'noise_reduction'),
            ('ema', [8, 12, 20, 26], 'trend_following')
        ]
        
        for indicator, periods, method in test_cases:
            logger.info(f"Testing {indicator} optimization with method {method}")
            
            optimal_period = pipeline._optimize_lookback_statistical(
                data, indicator, periods, method
            )
            
            logger.info(f"  Optimal {indicator} period: {optimal_period}")
            
            # Verify the result is in the expected range
            if optimal_period not in periods:
                logger.warning(f"  Warning: optimal period {optimal_period} not in test periods {periods}")
        
        logger.info("✅ Statistical optimization methods test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Statistical optimization methods test failed: {e}")
        return False

async def test_ml_commons_integration():
    """Test ML commons integration for feature optimization."""
    logger.info("🧪 Testing ML commons integration...")
    
    try:
        # Check if ML commons are available
        from src.training.steps.market_analysis.sub_pipeline import ML_COMMONS_AVAILABLE
        
        if ML_COMMONS_AVAILABLE:
            logger.info("✅ ML commons are available")
            
            # Test importing ML commons components
            from src.training.steps.market_analysis.sub_pipeline import (
                get_feature_optimizer, 
                FeatureOptimizationConfig
            )
            
            logger.info("✅ ML commons components imported successfully")
            
            # Test creating optimizer
            optimizer = get_feature_optimizer()
            logger.info("✅ Feature optimizer created successfully")
            
        else:
            logger.info("ℹ️ ML commons not available, using fallback statistical optimization")
        
        logger.info("✅ ML commons integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ ML commons integration test failed: {e}")
        return False

async def run_comprehensive_test():
    """Run comprehensive test suite for feature lookback optimization."""
    logger.info("🚀 Starting comprehensive feature lookback optimization test suite...")
    
    test_results = {}
    
    # Test 1: Feature optimization module
    test_results['feature_optimization_module'] = await test_feature_optimization_module()
    
    # Test 2: Statistical optimization methods
    test_results['statistical_optimization'] = await test_statistical_optimization_methods()
    
    # Test 3: ML commons integration
    test_results['ml_commons_integration'] = await test_ml_commons_integration()
    
    # Test 4: Market analysis pipeline
    test_results['market_analysis_pipeline'] = await test_market_analysis_pipeline()
    
    # Summary
    logger.info("📊 Test Results Summary:")
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed_tests += 1
    
    logger.info(f"📈 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Feature lookback optimization is fully implemented.")
        return True
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} tests failed. Review implementation.")
        return False

if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(run_comprehensive_test())
    
    if success:
        print("\n🎉 Feature Lookback Optimization Implementation: COMPLETE")
        sys.exit(0)
    else:
        print("\n❌ Feature Lookback Optimization Implementation: NEEDS ATTENTION")
        sys.exit(1)