#!/usr/bin/env python3
"""
Test script for enhanced TAS regime system integration.

This script tests the integration of utility tools with the TAS regime system.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_utility_tools_import():
    """Test that utility tools can be imported."""
    try:
        from src.utils.common_operations import CommonUtilities, safe_dataframe_operation
        from src.utils.math_validation import MathValidation, safe_divide
        from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
        from src.utils.serialization_utils import UniversalSerializer
        from src.utils.data.klines_parquet import get_klines_manager
        
        logger.info("✅ All utility tools imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Utility tools import failed: {e}")
        return False

def test_enhanced_tas_engine():
    """Test enhanced TAS engine initialization."""
    try:
        from src.training.steps.market_analysis.tas_regime.core.tas_engine import (
            TreeArchitectureSearchEngine, TASEngineConfig
        )
        
        # Create configuration
        config = TASEngineConfig(
            enable_hardware_optimization=True,
            enable_meta_learning=True,
            enable_uncertainty_estimation=True,
            enable_regime_analysis=True,
            enable_real_time_adaptation=True
        )
        
        # Initialize engine
        engine = TreeArchitectureSearchEngine(config)
        
        logger.info("✅ Enhanced TAS engine initialized successfully")
        logger.info(f"   Utility status: {engine._get_utility_status()}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced TAS engine initialization failed: {e}")
        return False

def test_enhanced_regime_detector():
    """Test enhanced regime detector initialization."""
    try:
        from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import (
            TASRegimeDetector, TASRegimeConfig
        )
        
        # Create configuration
        config = TASRegimeConfig(
            n_regimes=3,
            enable_economic_evaluation=True,
            enable_uncertainty_quantification=True,
            enable_multi_scale_analysis=True
        )
        
        # Initialize detector
        detector = TASRegimeDetector(config)
        
        logger.info("✅ Enhanced regime detector initialized successfully")
        logger.info(f"   Enhanced utility status: {detector._get_enhanced_utility_status()}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced regime detector initialization failed: {e}")
        return False

def test_enhanced_backtesting_engine():
    """Test enhanced backtesting engine initialization."""
    try:
        from src.training.steps.market_analysis.tas_regime.backtesting.backtesting_engine import (
            BacktestingEngine, BacktestingConfig
        )
        
        # Create configuration
        config = BacktestingConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000.0,
            enable_regime_aware_backtesting=True
        )
        
        # Initialize engine
        engine = BacktestingEngine(config)
        
        logger.info("✅ Enhanced backtesting engine initialized successfully")
        logger.info(f"   Enhanced utility status: {engine._get_enhanced_utility_status()}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced backtesting engine initialization failed: {e}")
        return False

def test_utility_tools_functionality():
    """Test utility tools functionality."""
    try:
        from src.utils.common_operations import CommonUtilities, safe_dataframe_operation
        from src.utils.math_validation import MathValidation, safe_divide
        from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
        
        # Test common utilities
        common_utils = CommonUtilities()
        logger.info("✅ Common utilities initialized")
        
        # Test math validation
        math_validator = MathValidation()
        result = math_validator.safe_divide(10, 2)
        assert result == 5.0, f"Expected 5.0, got {result}"
        logger.info("✅ Math validation working")
        
        # Test matrix operations
        matrix_ops = get_unified_matrix_operations()
        logger.info("✅ Matrix operations initialized")
        
        # Test DataFrame operations
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result_df = safe_dataframe_operation(df, lambda x: x * 2)
        assert len(result_df) == 3, f"Expected 3 rows, got {len(result_df)}"
        logger.info("✅ DataFrame operations working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Utility tools functionality test failed: {e}")
        return False

def test_data_quality_checks():
    """Test data quality checks."""
    try:
        from src.utils.common_operations import (
            calculate_data_quality_metrics, create_data_quality_report,
            guard_dataframe_nulls, optimize_dataframe_dtypes
        )
        
        # Create test data with some issues
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [1.1, 2.2, 3.3, 4.4, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Test data quality metrics
        metrics = calculate_data_quality_metrics(df)
        assert 'total_rows' in metrics, "Missing total_rows in metrics"
        logger.info("✅ Data quality metrics calculated")
        
        # Test data quality report
        report = create_data_quality_report(df)
        assert 'basic_info' in report, "Missing basic_info in report"
        logger.info("✅ Data quality report created")
        
        # Test null handling
        df_cleaned = guard_dataframe_nulls(df)
        assert len(df_cleaned) == len(df), "DataFrame length changed unexpectedly"
        logger.info("✅ Null handling working")
        
        # Test dtype optimization
        df_optimized = optimize_dataframe_dtypes(df_cleaned)
        assert len(df_optimized) == len(df_cleaned), "DataFrame length changed unexpectedly"
        logger.info("✅ Dtype optimization working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data quality checks test failed: {e}")
        return False

def test_math_validation():
    """Test math validation functionality."""
    try:
        from src.utils.math_validation import (
            safe_divide, safe_log, safe_sqrt, safe_power,
            validate_finite, validate_positive, validate_range,
            safe_correlation, safe_covariance, safe_mean, safe_std
        )
        
        # Test safe operations
        assert safe_divide(10, 2) == 5.0, "Safe divide failed"
        assert safe_divide(10, 0) == 0.0, "Safe divide with zero failed"
        logger.info("✅ Safe divide working")
        
        assert safe_log(2.718) == 1.0, "Safe log failed"
        assert safe_log(-1) == 0.0, "Safe log with negative failed"
        logger.info("✅ Safe log working")
        
        assert safe_sqrt(4) == 2.0, "Safe sqrt failed"
        assert safe_sqrt(-1) == 0.0, "Safe sqrt with negative failed"
        logger.info("✅ Safe sqrt working")
        
        # Test validation
        assert validate_finite(5.0) == 5.0, "Validate finite failed"
        assert validate_positive(5.0) == 5.0, "Validate positive failed"
        assert validate_range(5.0, 0, 10) == 5.0, "Validate range failed"
        logger.info("✅ Math validation working")
        
        # Test statistical operations
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        
        assert safe_correlation(x, y) > 0.9, "Safe correlation failed"
        assert safe_mean(x) == 3.0, "Safe mean failed"
        assert safe_std(x) > 0, "Safe std failed"
        logger.info("✅ Statistical operations working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Math validation test failed: {e}")
        return False

def test_matrix_operations():
    """Test matrix operations."""
    try:
        from src.utils.matrix_operations.unified_operations import (
            get_unified_matrix_operations, safe_matrix_multiply,
            safe_correlation_matrix, safe_matrix_inverse
        )
        
        # Test matrix operations initialization
        matrix_ops = get_unified_matrix_operations()
        logger.info("✅ Matrix operations initialized")
        
        # Test matrix multiplication
        A = np.random.randn(100, 100)
        B = np.random.randn(100, 100)
        C = safe_matrix_multiply(A, B)
        assert C.shape == (100, 100), f"Matrix multiplication shape failed: {C.shape}"
        logger.info("✅ Matrix multiplication working")
        
        # Test correlation matrix
        data = np.random.randn(100, 10)
        corr_matrix = safe_correlation_matrix(data)
        assert corr_matrix.shape == (10, 10), f"Correlation matrix shape failed: {corr_matrix.shape}"
        logger.info("✅ Correlation matrix working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Matrix operations test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting enhanced TAS regime system integration tests")
    
    tests = [
        ("Utility Tools Import", test_utility_tools_import),
        ("Enhanced TAS Engine", test_enhanced_tas_engine),
        ("Enhanced Regime Detector", test_enhanced_regime_detector),
        ("Enhanced Backtesting Engine", test_enhanced_backtesting_engine),
        ("Utility Tools Functionality", test_utility_tools_functionality),
        ("Data Quality Checks", test_data_quality_checks),
        ("Math Validation", test_math_validation),
        ("Matrix Operations", test_matrix_operations)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running test: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    logger.info(f"\n📊 Test Results:")
    logger.info(f"   ✅ Passed: {passed}")
    logger.info(f"   ❌ Failed: {failed}")
    logger.info(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Enhanced TAS regime system is working correctly.")
        return True
    else:
        logger.error(f"⚠️ {failed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)