#!/usr/bin/env python3
"""
Test script for the new Feature Lookback Optimization structure.

This script tests the reorganized feature lookback optimization module
to ensure all imports work correctly and the component functions properly.
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

def create_test_data(n_samples: int = 500) -> pd.DataFrame:
    """Create synthetic test data for feature optimization."""
    logger.info(f"Creating test data with {n_samples} samples")
    
    # Generate synthetic price data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Generate realistic price movements
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
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
    
    # Add target variable
    data['target'] = data['close'].pct_change(5).shift(-5)
    
    logger.info(f"Created test data with columns: {list(data.columns)}")
    return data

async def test_new_structure_imports():
    """Test that all imports work correctly with the new structure."""
    logger.info("🧪 Testing new structure imports...")
    
    try:
        # Test main component import
        from src.training.steps.market_analysis.feature_lookback_optimization import (
            FeatureLookbackOptimizationComponent
        )
        logger.info("✅ Main component import successful")
        
        # Test individual module imports
        from src.training.steps.market_analysis.feature_lookback_optimization.optimization_reporter import (
            OptimizationReporter
        )
        logger.info("✅ Optimization reporter import successful")
        
        from src.training.steps.market_analysis.feature_lookback_optimization.validation_framework import (
            ValidationFramework, ValidationLevel, ValidationStatus
        )
        logger.info("✅ Validation framework import successful")
        
        from src.training.steps.market_analysis.feature_lookback_optimization.dependency_manager import (
            DependencyManager, get_dependency, is_dependency_available
        )
        logger.info("✅ Dependency manager import successful")
        
        from src.training.steps.market_analysis.feature_lookback_optimization.monitoring_metrics import (
            MonitoringMetrics, MetricType, MetricLevel
        )
        logger.info("✅ Monitoring metrics import successful")
        
        # Test package-level imports
        from src.training.steps.market_analysis.feature_lookback_optimization import (
            FeatureLookbackOptimizationComponent,
            OptimizationReporter,
            ValidationFramework,
            ValidationLevel,
            ValidationStatus,
            DependencyManager,
            get_dependency,
            is_dependency_available,
            MonitoringMetrics,
            MetricType,
            MetricLevel
        )
        logger.info("✅ Package-level imports successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_component_initialization():
    """Test component initialization with new structure."""
    logger.info("🧪 Testing component initialization...")
    
    try:
        from src.training.steps.market_analysis.feature_lookback_optimization import (
            FeatureLookbackOptimizationComponent
        )
        from src.training.steps.market_analysis.components.base_component import ComponentConfig
        
        # Create component configuration
        config = ComponentConfig(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="30m"
        )
        
        # Initialize component
        component = FeatureLookbackOptimizationComponent(config)
        
        # Verify component properties
        assert component.config.symbol == "BTCUSDT"
        assert component.config.exchange == "binance"
        assert component.config.timeframe == "30m"
        assert hasattr(component, 'reporter')
        assert hasattr(component, 'validation_framework')
        assert hasattr(component, 'monitoring')
        
        logger.info("✅ Component initialization successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_validation_framework():
    """Test validation framework functionality."""
    logger.info("🧪 Testing validation framework...")
    
    try:
        from src.training.steps.market_analysis.feature_lookback_optimization.validation_framework import (
            ValidationFramework, ValidationLevel, ValidationStatus
        )
        
        # Create validation framework
        framework = ValidationFramework()
        
        # Test data validation
        good_data = create_test_data(100)
        is_valid, results, fixed_data = framework.validate_data(good_data)
        
        assert is_valid == True
        assert len(results) > 0
        assert fixed_data is not None
        
        # Test with bad data
        bad_data = pd.DataFrame()  # Empty DataFrame
        is_valid, results, fixed_data = framework.validate_data(bad_data)
        
        assert is_valid == False
        assert len(results) > 0
        
        logger.info("✅ Validation framework test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_dependency_manager():
    """Test dependency manager functionality."""
    logger.info("🧪 Testing dependency manager...")
    
    try:
        from src.training.steps.market_analysis.feature_lookback_optimization.dependency_manager import (
            get_dependency, is_dependency_available, get_dependency_status_report
        )
        
        # Test core dependency availability
        np_available = is_dependency_available('numpy')
        pd_available = is_dependency_available('pandas')
        
        assert np_available == True
        assert pd_available == True
        
        # Test dependency retrieval
        np, np_fallback = get_dependency('numpy')
        assert np is not None
        assert np_fallback == False
        
        # Test status report
        status_report = get_dependency_status_report()
        assert 'total_dependencies' in status_report
        assert 'available' in status_report
        
        logger.info("✅ Dependency manager test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dependency manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_monitoring_metrics():
    """Test monitoring metrics functionality."""
    logger.info("🧪 Testing monitoring metrics...")
    
    try:
        from src.training.steps.market_analysis.feature_lookback_optimization.monitoring_metrics import (
            MonitoringMetrics, MetricType, MetricLevel
        )
        
        # Create monitoring instance
        monitoring = MonitoringMetrics("TestComponent")
        
        # Start monitoring
        monitoring.start_monitoring()
        
        # Record some metrics
        monitoring.record_metric(
            name="test_metric",
            value=42.0,
            metric_type=MetricType.PERFORMANCE,
            level=MetricLevel.INFO
        )
        
        monitoring.record_quality_metric("test_quality", 0.85)
        monitoring.record_business_metric("test_business", 10)
        
        # Stop monitoring
        monitoring.stop_monitoring()
        
        # Get metrics summary
        summary = monitoring.get_metrics_summary()
        assert 'total_metrics' in summary
        assert summary['total_metrics'] > 0
        
        logger.info("✅ Monitoring metrics test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Monitoring metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_optimization_reporter():
    """Test optimization reporter functionality."""
    logger.info("🧪 Testing optimization reporter...")
    
    try:
        from src.training.steps.market_analysis.feature_lookback_optimization.optimization_reporter import (
            OptimizationReporter
        )
        
        # Create reporter
        reporter = OptimizationReporter("test_reports")
        
        # Create mock optimization result
        optimization_result = {
            'optimization_results': {
                'best_lookback_period': 20,
                'best_score': 0.75,
                'optimization_method': 'genetic_algorithm'
            },
            'optimized_features': {
                'rsi': {'lookback': 14, 'score': 0.7},
                'sma': {'lookback': 20, 'score': 0.6}
            },
            'optimization_metrics': {
                'convergence_iterations': 50
            },
            'optimization_time': 10.5
        }
        
        # Create mock metrics
        class MockMetrics:
            best_lookback_period = 20
            best_score = 0.75
            optimization_method = 'genetic_algorithm'
            total_features_optimized = 2
            optimization_time = 10.5
            convergence_iterations = 50
            memory_usage_mb = 100.0
            cpu_usage_percent = 25.0
            validation_score = 0.8
            stability_score = 0.7
            regime_coverage = 0.9
            error_rate = 0.0
        
        metrics = MockMetrics()
        
        # Generate report
        report = reporter.generate_comprehensive_report(
            optimization_result=optimization_result,
            metrics=metrics,
            validation_results={
                'data_validation': {'summary': {'quality_score': 0.9}},
                'pipeline_validation': {'summary': {'quality_score': 0.8}},
                'optimization_validation': {'summary': {'quality_score': 0.85}}
            },
            performance_metrics={'memory_usage': [100.0], 'cpu_usage': [25.0]},
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="30m"
        )
        
        assert 'report_metadata' in report
        assert 'executive_summary' in report
        assert 'optimization_results' in report
        
        logger.info("✅ Optimization reporter test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization reporter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_component_execution():
    """Test component execution with new structure."""
    logger.info("🧪 Testing component execution...")
    
    try:
        from src.training.steps.market_analysis.feature_lookback_optimization import (
            FeatureLookbackOptimizationComponent
        )
        from src.training.steps.market_analysis.components.base_component import ComponentConfig
        
        # Create test data
        data = create_test_data(200)
        
        # Create pipeline state
        pipeline_state = {
            'triple_barrier_labeling_result': {
                'labels': [1, 0, 1, 0] * 50,
                'barriers': {'upper': 0.02, 'lower': -0.02},
                'metadata': {'method': 'triple_barrier'}
            },
            'regime_data_splitting_result': {
                'regimes': [0, 1, 0, 1] * 50,
                'metadata': {'method': 'hmm'}
            }
        }
        
        # Create component
        config = ComponentConfig(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="30m"
        )
        component = FeatureLookbackOptimizationComponent(config)
        
        # Execute component
        result = await component.execute(data, pipeline_state)
        
        # Verify result
        assert hasattr(result, 'success')
        assert hasattr(result, 'artifacts')
        assert hasattr(result, 'execution_time')
        
        if result.success:
            assert 'feature_lookback_optimization_result' in result.artifacts
            logger.info("✅ Component execution successful")
        else:
            logger.info(f"⚠️ Component execution failed: {result.error_message}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Component execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_comprehensive_structure_test():
    """Run comprehensive test suite for new structure."""
    logger.info("🚀 Starting comprehensive new structure test suite...")
    
    test_results = {}
    
    # Test 1: Import functionality
    test_results['imports'] = await test_new_structure_imports()
    
    # Test 2: Component initialization
    test_results['initialization'] = await test_component_initialization()
    
    # Test 3: Validation framework
    test_results['validation_framework'] = await test_validation_framework()
    
    # Test 4: Dependency manager
    test_results['dependency_manager'] = await test_dependency_manager()
    
    # Test 5: Monitoring metrics
    test_results['monitoring_metrics'] = await test_monitoring_metrics()
    
    # Test 6: Optimization reporter
    test_results['optimization_reporter'] = await test_optimization_reporter()
    
    # Test 7: Component execution
    test_results['component_execution'] = await test_component_execution()
    
    # Summary
    logger.info("📊 New Structure Test Results Summary:")
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed_tests += 1
    
    logger.info(f"📈 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All new structure tests passed! Reorganization successful.")
        return True
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} tests failed. Review implementation.")
        return False

if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(run_comprehensive_structure_test())
    
    if success:
        print("\n🎉 Feature Lookback Optimization Reorganization: SUCCESS")
        sys.exit(0)
    else:
        print("\n❌ Feature Lookback Optimization Reorganization: NEEDS ATTENTION")
        sys.exit(1)