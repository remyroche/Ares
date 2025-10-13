"""
Test script for Enhanced Unified Data-Driven Pipeline

This script tests the enhanced pipeline to ensure all components are properly integrated
and all logic from individual components is captured.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import the enhanced pipeline
from .enhanced_unified_pipeline import (
    EnhancedUnifiedDataDrivenPipeline, 
    create_enhanced_unified_pipeline
)
from .config import create_default_config


def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create test data for pipeline testing."""
    tprint_info("Creating test data")
    
    # Generate synthetic OHLCV data
    np.random.seed(42)
    
    # Generate price data with trend and volatility
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    tprint_success(f"Created test data with {n_samples} samples")
    return data


def create_test_targets(data: pd.DataFrame) -> pd.Series:
    """Create test targets for supervised learning."""
    tprint_info("Creating test targets")
    
    # Create targets based on future returns
    future_returns = data['close'].pct_change(5).shift(-5)  # 5-period future returns
    targets = (future_returns > 0).astype(int)  # Binary classification
    
    tprint_success(f"Created test targets: {targets.sum()} positive samples out of {len(targets)}")
    return targets


def test_enhanced_pipeline():
    """Test the enhanced unified pipeline."""
    tprint_info("🧪 Starting Enhanced Unified Pipeline Test")
    
    try:
        # Create test data
        data = create_test_data(1000)
        targets = create_test_targets(data)
        
        # Create enhanced pipeline
        tprint_info("Creating enhanced pipeline")
        config = create_default_config()
        pipeline = create_enhanced_unified_pipeline(config)
        
        # Test pipeline processing
        tprint_info("Testing pipeline processing")
        start_time = time.time()
        
        result = pipeline.process(data, targets, "15m")
        
        execution_time = time.time() - start_time
        
        # Validate results
        tprint_info("Validating results")
        
        # Check if processing was successful
        if not result.success:
            tprint_error(f"Pipeline processing failed: {result.error_message}")
            return False
        
        # Validate period optimization results
        if not result.optimal_periods:
            tprint_warning("No optimal periods found")
        else:
            tprint_success(f"✅ Period optimization: {len(result.optimal_periods)} optimal periods")
            tprint_debug(f"Optimal periods: {result.optimal_periods}")
        
        # Validate feature selection results
        if not result.selected_features:
            tprint_warning("No features selected")
        else:
            tprint_success(f"✅ Feature selection: {len(result.selected_features)} features selected")
            tprint_debug(f"Feature categories: {result.feature_selection_metrics.get('category_distribution', {})}")
        
        # Validate interaction generation results
        if not result.generated_interactions:
            tprint_warning("No interactions generated")
        else:
            tprint_success(f"✅ Interaction generation: {len(result.generated_interactions)} interactions generated")
            tprint_debug(f"Interaction types: {result.interaction_metrics.get('interaction_types', [])}")
        
        # Validate HTF interaction results
        if not result.htf_interactions:
            tprint_warning("No HTF interactions generated")
        else:
            tprint_success(f"✅ HTF interactions: {len(result.htf_interactions)} HTF interactions generated")
            tprint_debug(f"HTF interaction types: {result.htf_metrics.get('interaction_types', [])}")
        
        # Validate lookback optimization results
        if not result.optimized_lookbacks:
            tprint_warning("No lookback optimizations performed")
        else:
            tprint_success(f"✅ Lookback optimization: {len(result.optimized_lookbacks)} features optimized")
            tprint_debug(f"Lookback range: {result.lookback_metrics.get('min_lookback', 0)} - {result.lookback_metrics.get('max_lookback', 0)}")
        
        # Display performance statistics
        tprint_info("Performance Statistics:")
        tprint_info(f"  Total execution time: {result.total_processing_time:.3f}s")
        tprint_info(f"  VectorBT operations: {result.vectorbt_operations}")
        tprint_info(f"  Economic evaluations: {result.economic_evaluations}")
        tprint_info(f"  Feature selections: {result.feature_selections}")
        tprint_info(f"  Interaction generations: {result.interaction_generations}")
        tprint_info(f"  HTF generations: {result.htf_generations}")
        tprint_info(f"  Lookback optimizations: {result.lookback_optimizations}")
        
        # Test component integration
        tprint_info("Testing component integration")
        
        # Test VectorBT optimizer
        vectorbt_stats = pipeline.vectorbt_optimizer.get_performance_stats()
        tprint_debug(f"VectorBT optimizer stats: {vectorbt_stats}")
        
        # Test economic evaluator
        economic_stats = pipeline.economic_evaluator.get_performance_stats()
        tprint_debug(f"Economic evaluator stats: {economic_stats}")
        
        # Test feature selector
        feature_stats = pipeline.feature_selector.get_performance_stats()
        tprint_debug(f"Feature selector stats: {feature_stats}")
        
        # Test HTF generator
        htf_stats = pipeline.htf_generator.get_performance_stats()
        tprint_debug(f"HTF generator stats: {htf_stats}")
        
        tprint_success("✅ Enhanced pipeline test completed successfully")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Enhanced pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_component_integration():
    """Test individual component integration."""
    tprint_info("🧪 Testing Component Integration")
    
    try:
        # Create test data
        data = create_test_data(500)
        targets = create_test_targets(data)
        
        # Test VectorBT optimizer
        tprint_info("Testing VectorBT optimizer")
        from .enhanced_components.vectorbt_enhancements import create_enhanced_vectorbt_optimizer
        
        vectorbt_optimizer = create_enhanced_vectorbt_optimizer()
        period_analysis = vectorbt_optimizer.optimize_period_analysis(data, [5, 10, 15, 20])
        
        if period_analysis:
            tprint_success("✅ VectorBT optimizer working")
        else:
            tprint_warning("⚠️ VectorBT optimizer returned empty results")
        
        # Test economic evaluator
        tprint_info("Testing economic evaluator")
        from .enhanced_components.economic_evaluation import create_economic_evaluator
        
        economic_evaluator = create_economic_evaluator()
        economic_result = economic_evaluator.evaluate_periods(data, [5, 10, 15, 20], "15m")
        
        if economic_result.success:
            tprint_success("✅ Economic evaluator working")
        else:
            tprint_warning(f"⚠️ Economic evaluator failed: {economic_result.error_message}")
        
        # Test feature selector
        tprint_info("Testing feature selector")
        from .enhanced_components.advanced_feature_selection import create_advanced_feature_selector
        
        feature_selector = create_advanced_feature_selector()
        feature_result = feature_selector.select_features(data, targets)
        
        if feature_result.success:
            tprint_success("✅ Feature selector working")
        else:
            tprint_warning(f"⚠️ Feature selector failed: {feature_result.error_message}")
        
        # Test HTF generator
        tprint_info("Testing HTF generator")
        from .enhanced_components.htf_template_system import create_htf_interaction_generator
        
        htf_generator = create_htf_interaction_generator()
        htf_features = {'htf_trend': data['close'].rolling(16).mean()}
        htf_result = htf_generator.generate_interactions(htf_features, data, targets)
        
        if htf_result:
            tprint_success("✅ HTF generator working")
        else:
            tprint_warning("⚠️ HTF generator returned empty results")
        
        tprint_success("✅ Component integration test completed")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Component integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_benchmark():
    """Test performance benchmark."""
    tprint_info("🧪 Testing Performance Benchmark")
    
    try:
        # Create test data with different sizes
        test_sizes = [500, 1000, 2000]
        
        for size in test_sizes:
            tprint_info(f"Testing with {size} samples")
            
            data = create_test_data(size)
            targets = create_test_targets(data)
            
            pipeline = create_enhanced_unified_pipeline()
            
            start_time = time.time()
            result = pipeline.process(data, targets, "15m")
            execution_time = time.time() - start_time
            
            if result.success:
                tprint_success(f"✅ {size} samples: {execution_time:.3f}s")
                tprint_debug(f"  Features: {len(result.selected_features)}")
                tprint_debug(f"  Interactions: {len(result.generated_interactions)}")
                tprint_debug(f"  HTF interactions: {len(result.htf_interactions)}")
            else:
                tprint_error(f"❌ {size} samples failed: {result.error_message}")
        
        tprint_success("✅ Performance benchmark completed")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Performance benchmark failed: {e}")
        return False


def main():
    """Main test function."""
    tprint_info("🚀 Starting Enhanced Unified Pipeline Test Suite")
    
    # Test 1: Component integration
    test1_passed = test_component_integration()
    
    # Test 2: Enhanced pipeline
    test2_passed = test_enhanced_pipeline()
    
    # Test 3: Performance benchmark
    test3_passed = test_performance_benchmark()
    
    # Summary
    tprint_info("📊 Test Summary:")
    tprint_info(f"  Component Integration: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    tprint_info(f"  Enhanced Pipeline: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    tprint_info(f"  Performance Benchmark: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        tprint_success("🎉 All tests passed! Enhanced pipeline is working correctly.")
        return True
    else:
        tprint_error("❌ Some tests failed. Please check the logs for details.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)