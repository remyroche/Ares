
"""
Integration Test for Feature Generation Optimizations

This test verifies that all optimization utilities are properly integrated
throughout the feature generation system.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_optimization_integration():
    """Test that optimization utilities are integrated throughout the system."""
    logger.info("🧪 Testing optimization integration...")
    
    # Create test data
    data = pd.DataFrame({
        'open': np.random.randn(1000) + 100,
        'high': np.random.randn(1000) + 101,
        'low': np.random.randn(1000) + 99,
        'close': np.random.randn(1000) + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Test optimized feature factory
    try:
        from src.feature_generation.utils.optimized_feature_factory import get_optimized_feature_factory
        
        factory = get_optimized_feature_factory()
        
        # Test feature generation
        features = factory.generate_features_optimized(
            data=data,
            categories=['momentum', 'volatility'],
            target_column='close'
        )
        
        logger.info(f"✅ Optimized factory test passed: {len(features.columns)} features generated")
        
        # Test DataFrame optimization
        optimized_data = factory.optimize_dataframe(data)
        logger.info(f"✅ DataFrame optimization test passed")
        
        # Test performance report
        report = factory.get_performance_report()
        logger.info(f"✅ Performance reporting test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization integration test failed: {e}")
        return False

def test_category_optimizations():
    """Test that individual categories use optimization utilities."""
    logger.info("🧪 Testing category optimizations...")
    
    try:
        from src.feature_generation.categories.momentum import MomentumFeatureGenerator
        from src.feature_generation.categories.volatility import VolatilityFeatureGenerator
        
        # Test momentum generator
        momentum_gen = MomentumFeatureGenerator()
        data = pd.DataFrame({
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Test optimization methods
        optimized_data = momentum_gen.optimize_dataframe_processing(data)
        logger.info("✅ Momentum generator optimization test passed")
        
        # Test volatility generator
        volatility_gen = VolatilityFeatureGenerator()
        optimized_data = volatility_gen.optimize_dataframe_processing(data)
        logger.info("✅ Volatility generator optimization test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Category optimization test failed: {e}")
        return False

def run_integration_tests():
    """Run all integration tests."""
    logger.info("🚀 Starting Feature Generation Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Optimization Integration", test_optimization_integration),
        ("Category Optimizations", test_category_optimizations)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:30} : {status}")
    
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    logger.info("=" * 60)
    
    return results

if __name__ == "__main__":
    results = run_integration_tests()
    
    # Exit with error code if any tests failed
    failed_tests = [name for name, result in results.items() if not result]
    if failed_tests:
        logger.error(f"❌ {len(failed_tests)} test(s) failed: {failed_tests}")
        sys.exit(1)
    else:
        logger.info("🎉 All integration tests passed!")
        sys.exit(0)
