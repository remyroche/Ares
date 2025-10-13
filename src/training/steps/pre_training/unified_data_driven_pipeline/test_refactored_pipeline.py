"""
Test script for the refactored unified data-driven pipeline.

This script tests the refactored pipeline to ensure all functionality works correctly
with the integrated utilities from src/utils/.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.core.unified_pipeline import (
        UnifiedDataDrivenPipeline, create_unified_pipeline, process_features
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.core.config import (
        create_default_config, create_fast_config, create_memory_efficient_config
    )
    from src.utils.tprint import tprint_info, tprint_success, tprint_error, tprint_warning
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def create_test_data(n_samples=1000, n_features=20):
    """Create test data for pipeline testing."""
    np.random.seed(42)
    
    # Create synthetic time series data
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Generate features with different characteristics
    data = {}
    
    # Price-like features (random walk)
    for i in range(5):
        data[f'price_{i}'] = np.cumsum(np.random.randn(n_samples) * 0.01) + 100
    
    # Volatility features
    for i in range(3):
        data[f'vol_{i}'] = np.abs(np.random.randn(n_samples) * 0.02)
    
    # Momentum features
    for i in range(4):
        data[f'momentum_{i}'] = np.random.randn(n_samples) * 0.1
    
    # Mean reversion features
    for i in range(3):
        data[f'mean_reversion_{i}'] = np.random.randn(n_samples) * 0.05
    
    # Volume features
    for i in range(3):
        data[f'volume_{i}'] = np.random.exponential(1000, n_samples)
    
    # Time-based features
    data['day_of_week'] = dates.dayofweek
    data['month'] = dates.month
    data['quarter'] = dates.quarter
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    for col in df.columns[:5]:  # Add missing values to first 5 columns
        df.loc[df.index[missing_indices], col] = np.nan
    
    # Create target variable (returns)
    returns = df['price_0'].pct_change().dropna()
    targets = returns.shift(-1).dropna()  # Next day returns
    
    # Align data and targets
    common_idx = df.index.intersection(targets.index)
    df = df.loc[common_idx]
    targets = targets.loc[common_idx]
    
    return df, targets


def test_pipeline_initialization():
    """Test pipeline initialization with different configurations."""
    tprint_info("Testing pipeline initialization...")
    
    try:
        # Test default configuration
        pipeline = create_unified_pipeline()
        tprint_success("✓ Default pipeline initialization successful")
        
        # Test with custom configuration
        config = create_fast_config()
        pipeline_custom = create_unified_pipeline(config)
        tprint_success("✓ Custom configuration pipeline initialization successful")
        
        # Test memory efficient configuration
        config_mem = create_memory_efficient_config()
        pipeline_mem = create_unified_pipeline(config_mem)
        tprint_success("✓ Memory efficient pipeline initialization successful")
        
        return True
        
    except Exception as e:
        tprint_error(f"✗ Pipeline initialization failed: {e}")
        return False


def test_data_processing():
    """Test data processing functionality."""
    tprint_info("Testing data processing...")
    
    try:
        # Create test data
        data, targets = create_test_data(500, 15)
        tprint_info(f"Created test data: {data.shape}, targets: {len(targets)}")
        
        # Test pipeline
        pipeline = create_unified_pipeline(create_fast_config())
        
        # Test data validation
        pipeline._validate_inputs(data, targets, list(data.columns[:10]))
        tprint_success("✓ Data validation successful")
        
        # Test data preparation
        processed_data, processed_targets = pipeline._prepare_data(data, targets, list(data.columns[:10]))
        tprint_success(f"✓ Data preparation successful: {processed_data.shape}")
        
        return True
        
    except Exception as e:
        tprint_error(f"✗ Data processing failed: {e}")
        return False


def test_performance_monitoring():
    """Test performance monitoring functionality."""
    tprint_info("Testing performance monitoring...")
    
    try:
        pipeline = create_unified_pipeline(create_fast_config())
        
        # Test performance stats
        stats = pipeline.get_performance_stats()
        tprint_success(f"✓ Performance stats retrieved: {len(stats)} categories")
        
        # Test reset functionality
        pipeline.reset_performance_stats()
        tprint_success("✓ Performance stats reset successful")
        
        return True
        
    except Exception as e:
        tprint_error(f"✗ Performance monitoring failed: {e}")
        return False


def test_error_handling():
    """Test error handling functionality."""
    tprint_info("Testing error handling...")
    
    try:
        pipeline = create_unified_pipeline(create_fast_config())
        
        # Test with invalid data
        try:
            pipeline._validate_inputs(None, None, None)
            tprint_warning("✗ Should have raised error for None data")
            return False
        except Exception:
            tprint_success("✓ Error handling for None data works correctly")
        
        # Test with empty data
        try:
            empty_df = pd.DataFrame()
            pipeline._validate_inputs(empty_df, None, None)
            tprint_warning("✗ Should have raised error for empty data")
            return False
        except Exception:
            tprint_success("✓ Error handling for empty data works correctly")
        
        # Test error handler functionality
        error_handler = pipeline.error_handler
        result = error_handler.safe_execute(lambda: 1/0, default="error_handled")
        if result == "error_handled":
            tprint_success("✓ Safe execution with error handling works correctly")
        else:
            tprint_warning("✗ Safe execution not working as expected")
            return False
        
        return True
        
    except Exception as e:
        tprint_error(f"✗ Error handling test failed: {e}")
        return False


def test_utility_integration():
    """Test integration of utilities from src/utils/."""
    tprint_info("Testing utility integration...")
    
    try:
        pipeline = create_unified_pipeline(create_fast_config())
        
        # Test that utilities are properly initialized
        assert hasattr(pipeline, 'error_handler'), "Error handler not initialized"
        assert hasattr(pipeline, 'data_processor'), "Data processor not initialized"
        assert hasattr(pipeline, 'performance_monitor'), "Performance monitor not initialized"
        assert hasattr(pipeline, 'unified_monitor'), "Unified monitor not initialized"
        
        tprint_success("✓ All utilities properly initialized")
        
        # Test data processor functionality
        data, _ = create_test_data(100, 5)
        cleaned_data = pipeline.data_processor.clean_data(data)
        tprint_success(f"✓ Data processor works: {cleaned_data.shape}")
        
        # Test memory optimization
        if pipeline.config.vectorization.memory_efficient:
            optimized_data = pipeline.data_processor.memory_optimize_dataframe(data)
            tprint_success(f"✓ Memory optimization works: {optimized_data.shape}")
        
        return True
        
    except Exception as e:
        tprint_error(f"✗ Utility integration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    tprint_info("Starting comprehensive pipeline testing...")
    
    tests = [
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Data Processing", test_data_processing),
        ("Performance Monitoring", test_performance_monitoring),
        ("Error Handling", test_error_handling),
        ("Utility Integration", test_utility_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        tprint_info(f"\n--- Running {test_name} Test ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            tprint_error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    tprint_info("\n" + "="*50)
    tprint_info("TEST SUMMARY")
    tprint_info("="*50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        tprint_info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    tprint_info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        tprint_success("🎉 All tests passed! Pipeline refactoring successful.")
        return True
    else:
        tprint_error(f"❌ {total - passed} tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)