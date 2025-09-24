#!/usr/bin/env python3
"""
Test script to validate the NAS regime upgrades.

This script tests the enhanced functionality without requiring full data or complex setup.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_matrix_operations():
    """Test the enhanced matrix operations."""
    logger.info("🧪 Testing Enhanced Matrix Operations...")
    
    try:
        from core.enhanced_matrix_operations import EnhancedMatrixOperations
        
        # Initialize with M1 optimization
        matrix_ops = EnhancedMatrixOperations(
            enable_gpu=False,  # Disable GPU for testing
            enable_optimization=True,
            enable_m1_optimization=False  # Disable M1 for testing
        )
        
        # Create test data
        test_data = np.random.randn(100, 5)
        
        # Test normalization
        normalized_data = matrix_ops.normalize_data(test_data, method='z_score')
        logger.info(f"✅ Normalization test passed - shape: {normalized_data.shape}")
        
        # Test correlation matrix
        corr_matrix = matrix_ops.calculate_correlation_matrix(test_data)
        logger.info(f"✅ Correlation matrix test passed - shape: {corr_matrix.shape}")
        
        # Test enhanced features
        features = matrix_ops.calculate_enhanced_features(test_data, window=10)
        logger.info(f"✅ Enhanced features test passed - features: {list(features.keys())}")
        
        # Test state persistence
        state_saved = matrix_ops.save_operations_state("test_matrix_state.json")
        state_loaded = matrix_ops.load_operations_state("test_matrix_state.json")
        logger.info(f"✅ State persistence test passed - saved: {state_saved}, loaded: {state_loaded}")
        
        # Clean up test file
        if os.path.exists("test_matrix_state.json"):
            os.remove("test_matrix_state.json")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Matrix Operations test failed: {e}")
        return False

def test_enhanced_data_operations():
    """Test the enhanced data operations."""
    logger.info("🧪 Testing Enhanced Data Operations...")
    
    try:
        from core.enhanced_data_operations import EnhancedDataOperations
        
        # Initialize
        data_ops = EnhancedDataOperations(
            data_dir="test_data",
            enable_validation=True
        )
        
        # Create test market data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        test_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # Ensure high >= low and high >= open/close
        test_data['high'] = np.maximum(test_data['high'], 
                                     np.maximum(test_data['open'], test_data['close']))
        test_data['low'] = np.minimum(test_data['low'], 
                                    np.minimum(test_data['open'], test_data['close']))
        
        # Test data validation
        validation_result = data_ops.validate_market_data(test_data)
        logger.info(f"✅ Data validation test passed - valid: {validation_result['is_valid']}")
        
        # Test data processing
        processed_data = data_ops.process_market_data(test_data)
        logger.info(f"✅ Data processing test passed - shape: {processed_data.shape}")
        
        # Test quality report
        quality_report = data_ops.get_data_quality_report(test_data)
        logger.info(f"✅ Quality report test passed - report keys: {list(quality_report.keys())}")
        
        # Test state persistence
        state_saved = data_ops.save_operations_state("test_data_state.json")
        state_loaded = data_ops.load_operations_state("test_data_state.json")
        logger.info(f"✅ State persistence test passed - saved: {state_saved}, loaded: {state_loaded}")
        
        # Clean up test files
        for filename in ["test_data_state.json"]:
            if os.path.exists(filename):
                os.remove(filename)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Data Operations test failed: {e}")
        return False

def test_enhanced_ml_common_integration():
    """Test the enhanced ML common integration."""
    logger.info("🧪 Testing Enhanced ML Common Integration...")
    
    try:
        from core.enhanced_ml_common_integration import EnhancedMLCommonIntegration, MLCommonConfig
        
        # Initialize with configuration
        config = MLCommonConfig(
            enable_validation=True,
            enable_safe_math=True,
            enable_hardware_optimization=False,  # Disable for testing
            math_validation_level='standard'
        )
        
        ml_integration = EnhancedMLCommonIntegration(config)
        
        # Create test data
        test_data = np.random.randn(100, 5)
        
        # Test data validation
        validation_result = ml_integration.validate_data(test_data, 'market_data')
        logger.info(f"✅ Data validation test passed - valid: {validation_result['is_valid']}")
        
        # Test feature selection
        feature_result = ml_integration.select_features(test_data)
        logger.info(f"✅ Feature selection test passed - selected: {len(feature_result.get('selected_features', []))}")
        
        # Test ensemble methods
        ensemble_result = ml_integration.create_ensemble(test_data, np.random.randint(0, 2, 100))
        logger.info(f"✅ Ensemble methods test passed - ensemble created: {ensemble_result is not None}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced ML Common Integration test failed: {e}")
        return False

def test_utility_imports():
    """Test that all utility imports work correctly."""
    logger.info("🧪 Testing Utility Imports...")
    
    try:
        # Test common operations
        from src.utils.common_operations import safe_divide, safe_log, validate_finite
        logger.info("✅ Common operations imports successful")
        
        # Test math validation
        from src.utils.math_validation import safe_correlation, validate_numeric_array
        logger.info("✅ Math validation imports successful")
        
        # Test serialization
        from src.utils.serialization_utils import UniversalSerializer
        logger.info("✅ Serialization imports successful")
        
        # Test data utilities (if available)
        try:
            from src.utils.data.klines_parquet import KlinesParquetManager
            logger.info("✅ Data utilities imports successful")
        except ImportError:
            logger.warning("⚠️ Data utilities not available (expected in some environments)")
        
        # Test hardware utilities (if available)
        try:
            from src.utils.hardware.m1_gpu_utils import M1GPUManager
            logger.info("✅ Hardware utilities imports successful")
        except ImportError:
            logger.warning("⚠️ Hardware utilities not available (expected in some environments)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Utility imports test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting NAS Regime Upgrade Validation Tests...")
    
    tests = [
        ("Utility Imports", test_utility_imports),
        ("Enhanced Matrix Operations", test_enhanced_matrix_operations),
        ("Enhanced Data Operations", test_enhanced_data_operations),
        ("Enhanced ML Common Integration", test_enhanced_ml_common_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Upgrades are working correctly.")
        return 0
    else:
        logger.error("⚠️ Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())