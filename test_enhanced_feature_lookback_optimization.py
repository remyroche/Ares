#!/usr/bin/env python3
"""
Enhanced Test script for Feature Lookback Optimization Component.

This script tests the enhanced feature lookback optimization functionality
with comprehensive validation, error handling, and reporting.
"""

import asyncio
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, Any
import logging
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_comprehensive_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create comprehensive synthetic test data for feature optimization."""
    logger.info(f"Creating comprehensive test data with {n_samples} samples")
    
    # Generate synthetic price data with realistic patterns
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Generate realistic price movements with trends and volatility
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Add trend and volatility clustering
    trend = np.linspace(0, 0.1, n_samples)
    volatility = 0.01 + 0.005 * np.sin(np.arange(n_samples) * 0.1)
    
    prices = prices * (1 + trend) * (1 + np.random.normal(0, volatility))
    
    # Create OHLCV data with realistic relationships
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
    
    # Add technical indicators
    data['rsi_14'] = calculate_rsi(data['close'], 14)
    data['rsi_21'] = calculate_rsi(data['close'], 21)
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['ema_12'] = data['close'].ewm(span=12).mean()
    data['ema_26'] = data['close'].ewm(span=26).mean()
    
    # Add target variable (future returns)
    data['target'] = data['close'].pct_change(5).shift(-5)
    
    # Add regime column for regime-aware optimization
    data['regime'] = np.where(
        data['close'].rolling(20).std() > data['close'].rolling(20).std().quantile(0.7), 1, 0
    )
    
    logger.info(f"Created test data with columns: {list(data.columns)}")
    return data

def create_corrupted_test_data(n_samples: int = 500) -> pd.DataFrame:
    """Create test data with various data quality issues."""
    logger.info(f"Creating corrupted test data with {n_samples} samples")
    
    # Start with good data
    data = create_comprehensive_test_data(n_samples)
    
    # Introduce data quality issues
    # 1. Missing values
    data.loc[100:110, 'volume'] = np.nan
    data.loc[200:205, 'close'] = np.nan
    
    # 2. Infinite values
    data.loc[300:302, 'high'] = np.inf
    data.loc[303:305, 'low'] = -np.inf
    
    # 3. Negative prices (invalid)
    data.loc[400:402, 'close'] = -10.0
    
    # 4. Inconsistent high/low
    data.loc[450:452, 'high'] = data.loc[450:452, 'low'] - 1
    
    logger.info("Created corrupted test data with various quality issues")
    return data

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

async def test_enhanced_component():
    """Test the enhanced feature lookback optimization component."""
    logger.info("🧪 Testing enhanced Feature Lookback Optimization Component...")
    
    try:
        from src.training.steps.market_analysis.components.feature_lookback_optimization import (
            FeatureLookbackOptimizationComponent,
            OptimizationStatus,
            OptimizationMetrics,
            ValidationResult
        )
        from src.training.steps.market_analysis.components.base_component import ComponentConfig
        
        # Test 1: Component initialization
        logger.info("Test 1: Component initialization")
        config = ComponentConfig(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="30m"
        )
        component = FeatureLookbackOptimizationComponent(config)
        
        assert component.optimization_status == OptimizationStatus.PENDING
        assert component.metrics is None
        assert component.validation_result is None
        logger.info("✅ Component initialization test passed")
        
        # Test 2: Data validation with good data
        logger.info("Test 2: Data validation with good data")
        good_data = create_comprehensive_test_data(500)
        validation_result = component._validate_input_data(good_data)
        
        assert validation_result.is_valid == True
        assert validation_result.data_quality_score > 0.8
        assert len(validation_result.errors) == 0
        logger.info(f"✅ Good data validation passed (quality score: {validation_result.data_quality_score:.3f})")
        
        # Test 3: Data validation with corrupted data
        logger.info("Test 3: Data validation with corrupted data")
        corrupted_data = create_corrupted_test_data(300)
        validation_result = component._validate_input_data(corrupted_data)
        
        assert validation_result.is_valid == False
        assert len(validation_result.errors) > 0
        assert len(validation_result.warnings) > 0
        logger.info(f"✅ Corrupted data validation passed (errors: {len(validation_result.errors)}, warnings: {len(validation_result.warnings)})")
        
        # Test 4: Pipeline state validation
        logger.info("Test 4: Pipeline state validation")
        
        # Test with missing triple barrier labeling
        empty_pipeline_state = {}
        validation_result = component._validate_pipeline_state(empty_pipeline_state)
        assert validation_result.is_valid == False
        assert "No triple barrier labeling results" in validation_result.errors[0]
        
        # Test with valid pipeline state
        valid_pipeline_state = {
            'triple_barrier_labeling_result': {
                'labels': [1, 0, 1, 0],
                'barriers': {'upper': 0.02, 'lower': -0.02},
                'metadata': {'method': 'triple_barrier'}
            },
            'regime_data_splitting_result': {
                'regimes': [0, 1, 0, 1],
                'metadata': {'method': 'hmm'}
            }
        }
        validation_result = component._validate_pipeline_state(valid_pipeline_state)
        assert validation_result.is_valid == True
        logger.info("✅ Pipeline state validation test passed")
        
        # Test 5: Performance monitoring
        logger.info("Test 5: Performance monitoring")
        component._monitor_performance('test_operation')
        assert len(component.performance_monitor['execution_times']['test_operation']) > 0
        logger.info("✅ Performance monitoring test passed")
        
        # Test 6: Optimization metrics creation
        logger.info("Test 6: Optimization metrics creation")
        optimization_results = {
            'best_lookback_period': 20,
            'best_score': 0.75,
            'optimization_method': 'genetic_algorithm'
        }
        optimized_features = {
            'rsi': {'lookback': 14, 'score': 0.7},
            'sma': {'lookback': 20, 'score': 0.6},
            'ema': {'lookback': 12, 'score': 0.65}
        }
        optimization_metrics = {'convergence_iterations': 50}
        optimization_result = {'optimization_time': 10.5}
        
        metrics = component._create_optimization_metrics(
            optimization_results, optimized_features, optimization_metrics, optimization_result
        )
        
        assert isinstance(metrics, OptimizationMetrics)
        assert metrics.best_lookback_period == 20
        assert metrics.best_score == 0.75
        assert metrics.total_features_optimized == 3
        logger.info("✅ Optimization metrics creation test passed")
        
        # Test 7: Report generation
        logger.info("Test 7: Report generation")
        component.metrics = metrics
        component.start_time = time.time() - 10.0
        component.optimization_status = OptimizationStatus.COMPLETED
        
        report = component._generate_optimization_report()
        
        assert 'summary' in report
        assert 'performance' in report
        assert 'quality_metrics' in report
        assert 'recommendations' in report
        assert len(report['recommendations']) > 0
        logger.info("✅ Report generation test passed")
        
        # Test 8: Full execution with good data
        logger.info("Test 8: Full execution with good data")
        result = await component.execute(good_data, valid_pipeline_state)
        
        assert result.success == True
        assert 'feature_lookback_optimization_result' in result.artifacts
        assert result.execution_time > 0
        assert result.metadata['optimization_status'] == 'completed'
        logger.info(f"✅ Full execution test passed (execution time: {result.execution_time:.2f}s)")
        
        # Test 9: Full execution with corrupted data (should fail gracefully)
        logger.info("Test 9: Full execution with corrupted data")
        result = await component.execute(corrupted_data, valid_pipeline_state)
        
        assert result.success == False
        assert 'validation_errors' in result.metadata
        assert result.metadata['optimization_status'] == 'failed'
        logger.info("✅ Corrupted data execution test passed (failed gracefully)")
        
        # Test 10: Artifact validation
        logger.info("Test 10: Artifact validation")
        if result.success:
            artifacts = result.artifacts
            main_artifact = artifacts['feature_lookback_optimization_result']
            
            required_keys = [
                'optimization_results', 'optimized_features', 'optimization_metrics',
                'optimization_summary', 'detailed_report', 'validation_results',
                'performance_metrics', 'metadata'
            ]
            
            for key in required_keys:
                assert key in main_artifact, f"Missing required key: {key}"
            
            logger.info("✅ Artifact validation test passed")
        
        logger.info("🎉 All enhanced component tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_error_scenarios():
    """Test various error scenarios and edge cases."""
    logger.info("🧪 Testing error scenarios and edge cases...")
    
    try:
        from src.training.steps.market_analysis.components.feature_lookback_optimization import (
            FeatureLookbackOptimizationComponent
        )
        from src.training.steps.market_analysis.components.base_component import ComponentConfig
        
        component = FeatureLookbackOptimizationComponent(ComponentConfig())
        
        # Test 1: None data
        logger.info("Test 1: None data handling")
        validation_result = component._validate_input_data(None)
        assert validation_result.is_valid == False
        assert "Input data is None" in validation_result.errors
        logger.info("✅ None data handling test passed")
        
        # Test 2: Empty DataFrame
        logger.info("Test 2: Empty DataFrame handling")
        empty_df = pd.DataFrame()
        validation_result = component._validate_input_data(empty_df)
        assert validation_result.is_valid == False
        assert "Input data is empty" in validation_result.errors
        logger.info("✅ Empty DataFrame handling test passed")
        
        # Test 3: Non-DataFrame data
        logger.info("Test 3: Non-DataFrame data handling")
        validation_result = component._validate_input_data("not a dataframe")
        assert validation_result.is_valid == False
        assert "Input data must be a pandas DataFrame" in validation_result.errors
        logger.info("✅ Non-DataFrame data handling test passed")
        
        # Test 4: Missing required columns
        logger.info("Test 4: Missing required columns handling")
        incomplete_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100),
            'price': np.random.randn(100)
        })
        validation_result = component._validate_input_data(incomplete_data)
        assert validation_result.is_valid == False
        assert "Missing required columns" in validation_result.errors[0]
        logger.info("✅ Missing columns handling test passed")
        
        # Test 5: Data with all NaN values
        logger.info("Test 5: All NaN values handling")
        nan_data = pd.DataFrame({
            'open': [np.nan] * 100,
            'high': [np.nan] * 100,
            'low': [np.nan] * 100,
            'close': [np.nan] * 100,
            'volume': [np.nan] * 100
        })
        validation_result = component._validate_input_data(nan_data)
        assert validation_result.data_completeness == 0.0
        assert validation_result.data_quality_score < 0.6
        logger.info("✅ All NaN values handling test passed")
        
        logger.info("🎉 All error scenario tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error scenario test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_metrics():
    """Test performance monitoring and metrics collection."""
    logger.info("🧪 Testing performance metrics and monitoring...")
    
    try:
        from src.training.steps.market_analysis.components.feature_lookback_optimization import (
            FeatureLookbackOptimizationComponent
        )
        from src.training.steps.market_analysis.components.base_component import ComponentConfig
        
        component = FeatureLookbackOptimizationComponent(ComponentConfig())
        
        # Test performance monitoring
        initial_memory = len(component.performance_monitor['memory_usage'])
        initial_cpu = len(component.performance_monitor['cpu_usage'])
        
        # Simulate some operations
        for i in range(5):
            component._monitor_performance(f'operation_{i}')
        
        # Check that metrics were collected
        assert len(component.performance_monitor['memory_usage']) > initial_memory
        assert len(component.performance_monitor['cpu_usage']) > initial_cpu
        assert len(component.performance_monitor['execution_times']) > 0
        
        logger.info("✅ Performance monitoring test passed")
        
        # Test stability score calculation
        optimized_features = {
            'feature1': {'lookback': 10, 'score': 0.8},
            'feature2': {'lookback': 12, 'score': 0.7},
            'feature3': {'lookback': 14, 'score': 0.9}
        }
        
        stability_score = component._calculate_stability_score(optimized_features)
        assert 0.0 <= stability_score <= 1.0
        logger.info(f"✅ Stability score calculation test passed (score: {stability_score:.3f})")
        
        # Test validation score calculation
        optimization_results = {
            'best_lookback_period': 20,
            'best_score': 0.75
        }
        
        validation_score = component._calculate_validation_score(optimization_results, optimized_features)
        assert 0.0 <= validation_score <= 1.0
        logger.info(f"✅ Validation score calculation test passed (score: {validation_score:.3f})")
        
        logger.info("🎉 All performance metrics tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_comprehensive_enhanced_test():
    """Run comprehensive test suite for enhanced feature lookback optimization."""
    logger.info("🚀 Starting comprehensive enhanced feature lookback optimization test suite...")
    
    test_results = {}
    
    # Test 1: Enhanced component functionality
    test_results['enhanced_component'] = await test_enhanced_component()
    
    # Test 2: Error scenarios
    test_results['error_scenarios'] = await test_error_scenarios()
    
    # Test 3: Performance metrics
    test_results['performance_metrics'] = await test_performance_metrics()
    
    # Summary
    logger.info("📊 Enhanced Test Results Summary:")
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed_tests += 1
    
    logger.info(f"📈 Overall: {passed_tests}/{total_tests} enhanced tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All enhanced tests passed! Feature lookback optimization is fully enhanced.")
        return True
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} enhanced tests failed. Review implementation.")
        return False

if __name__ == "__main__":
    import time
    
    # Run the comprehensive enhanced test
    success = asyncio.run(run_comprehensive_enhanced_test())
    
    if success:
        print("\n🎉 Enhanced Feature Lookback Optimization Implementation: COMPLETE")
        sys.exit(0)
    else:
        print("\n❌ Enhanced Feature Lookback Optimization Implementation: NEEDS ATTENTION")
        sys.exit(1)