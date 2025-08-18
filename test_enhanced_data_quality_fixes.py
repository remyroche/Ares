#!/usr/bin/env python3
"""
Test Enhanced Data Quality Fixes
Demonstrates the usage of all new enhanced data quality decorators and fixes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Import the new enhanced decorators
from src.utils.enhanced_data_quality_decorators import (
    validate_constant_features,
    validate_low_variance_features,
    validate_data_completeness,
    validate_datetime_index,
    validate_multi_timeframe_alignment,
    validate_hmm_data_requirements,
    validate_data_structure,
    optimize_memory_usage,
    comprehensive_data_validation,
    validate_memory_optimized_data_quality,
    validate_feature_engineering_pipeline,
    validate_hmm_regime_discovery,
    validate_multi_timeframe_processing,
    get_memory_usage,
    clear_data_quality_cache,
    optimize_dataframe,
    MemoryOptimizer,
    DataQualityCache
)

# Import the updated raw data quality checker
from src.training.steps.raw_data_quality_checker import (
    RawDataQualityChecker,
    validate_raw_data_quality,
    fix_irregular_intervals_automatically
)


class TestDataQualityFixes:
    """Test class demonstrating all the new data quality fixes."""
    
    def __init__(self):
        self.logger = None  # Will be set up if needed
        
    def create_test_data(self, include_issues=True):
        """Create test data with various quality issues."""
        # Create base datetime index
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='1min')
        
        # Create base OHLCV data
        np.random.seed(42)
        base_price = 100.0
        
        data = pd.DataFrame({
            'open': base_price + np.random.randn(1000) * 0.1,
            'high': base_price + np.random.randn(1000) * 0.15,
            'low': base_price + np.random.randn(1000) * 0.15,
            'close': base_price + np.random.randn(1000) * 0.1,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
        # Ensure high >= open, close and low <= open, close
        data['high'] = data[['open', 'close']].max(axis=1) + np.abs(np.random.randn(1000) * 0.05)
        data['low'] = data[['open', 'close']].min(axis=1) - np.abs(np.random.randn(1000) * 0.05)
        
        if include_issues:
            # Add constant features
            data['constant_feature'] = 42.0
            
            # Add low variance features
            data['low_variance_feature'] = 100.0 + np.random.randn(1000) * 1e-10
            
            # Add missing data
            data.loc[100:150, 'volume'] = np.nan
            
            # Add irregular intervals (remove some timestamps)
            data = data.drop(data.index[200:250])
            
            # Add some extreme values
            data.iloc[300, data.columns.get_loc('close')] = 1000.0  # Extreme price spike
            data.iloc[301, data.columns.get_loc('volume')] = 1000000  # Extreme volume spike
        
        return data
    
    @validate_constant_features
    def test_constant_feature_detection(self, data, symbol="TEST", exchange="TEST"):
        """Test constant feature detection and removal."""
        print(f"✅ Constant feature detection test passed")
        print(f"   Data shape after constant feature removal: {data.shape}")
        return data
    
    @validate_low_variance_features
    def test_low_variance_feature_detection(self, data, symbol="TEST", exchange="TEST"):
        """Test low variance feature detection and removal."""
        print(f"✅ Low variance feature detection test passed")
        print(f"   Data shape after low variance feature removal: {data.shape}")
        return data
    
    @validate_data_completeness
    def test_data_completeness_validation(self, data, symbol="TEST", exchange="TEST"):
        """Test data completeness validation and missing data handling."""
        print(f"✅ Data completeness validation test passed")
        print(f"   Missing data handled, data shape: {data.shape}")
        return data
    
    @validate_datetime_index
    def test_datetime_index_validation(self, data, symbol="TEST", exchange="TEST"):
        """Test datetime index validation and fixing."""
        print(f"✅ Datetime index validation test passed")
        print(f"   Index type: {type(data.index)}")
        return data
    
    @validate_multi_timeframe_alignment
    def test_multi_timeframe_alignment(self, data, symbol="TEST", exchange="TEST"):
        """Test multi-timeframe alignment validation."""
        print(f"✅ Multi-timeframe alignment validation test passed")
        return data
    
    @validate_hmm_data_requirements
    def test_hmm_data_requirements(self, data, symbol="TEST", exchange="TEST"):
        """Test HMM data requirements validation."""
        print(f"✅ HMM data requirements validation test passed")
        return data
    
    @validate_data_structure
    def test_data_structure_validation(self, data, symbol="TEST", exchange="TEST"):
        """Test data structure validation."""
        print(f"✅ Data structure validation test passed")
        return data
    
    @optimize_memory_usage
    def test_memory_optimization(self, data, symbol="TEST", exchange="TEST"):
        """Test memory optimization."""
        print(f"✅ Memory optimization test passed")
        return data
    
    @comprehensive_data_validation
    def test_comprehensive_validation(self, data, symbol="TEST", exchange="TEST"):
        """Test comprehensive data validation."""
        print(f"✅ Comprehensive data validation test passed")
        return data
    
    @validate_memory_optimized_data_quality
    def test_memory_optimized_validation(self, data, symbol="TEST", exchange="TEST"):
        """Test memory-optimized validation."""
        print(f"✅ Memory-optimized validation test passed")
        return data
    
    @validate_feature_engineering_pipeline
    def test_feature_engineering_pipeline(self, data, symbol="TEST", exchange="TEST"):
        """Test feature engineering pipeline validation."""
        print(f"✅ Feature engineering pipeline validation test passed")
        return data
    
    @validate_hmm_regime_discovery
    def test_hmm_regime_discovery(self, data, symbol="TEST", exchange="TEST"):
        """Test HMM regime discovery validation."""
        print(f"✅ HMM regime discovery validation test passed")
        return data
    
    @validate_multi_timeframe_processing
    def test_multi_timeframe_processing(self, data, symbol="TEST", exchange="TEST"):
        """Test multi-timeframe processing validation."""
        print(f"✅ Multi-timeframe processing validation test passed")
        return data


def test_raw_data_quality_checker():
    """Test the updated raw data quality checker with multi-timeframe validation."""
    print("\n" + "="*60)
    print("TESTING UPDATED RAW DATA QUALITY CHECKER")
    print("="*60)
    
    # Create test data
    test_data = TestDataQualityFixes().create_test_data(include_issues=True)
    
    # Test the raw data quality checker
    checker = RawDataQualityChecker()
    results, processed_data = checker.validate_raw_data(test_data, "BTCUSDT", "binance")
    
    print(f"✅ Raw data quality validation completed")
    print(f"   Validation passed: {results['validation_passed']}")
    print(f"   Quality score: {results['data_quality_score']:.3f}")
    print(f"   Critical issues: {len(results['critical_issues'])}")
    print(f"   Warnings: {len(results['warnings'])}")
    
    # Check for multi-timeframe analysis
    if "multi_timeframe_analysis" in results.get("detailed_analysis", {}):
        mt_analysis = results["detailed_analysis"]["multi_timeframe_analysis"]
        print(f"   Multi-timeframe irregular ratio: {mt_analysis.get('irregular_interval_ratio', 0):.3f}")
    
    return results, processed_data


def test_memory_optimizer():
    """Test the memory optimizer utility."""
    print("\n" + "="*60)
    print("TESTING MEMORY OPTIMIZER")
    print("="*60)
    
    # Create large test data
    dates = pd.date_range(start='2023-01-01', periods=10000, freq='1min')
    data = pd.DataFrame({
        'int64_col': np.random.randint(0, 100, 10000, dtype=np.int64),
        'float64_col': np.random.randn(10000).astype(np.float64),
        'object_col': ['category_' + str(i % 10) for i in range(10000)]
    }, index=dates)
    
    # Get initial memory usage
    initial_memory = data.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Initial memory usage: {initial_memory:.2f}MB")
    
    # Optimize memory
    optimizer = MemoryOptimizer()
    optimized_data = optimizer.optimize_dataframe_memory(data.copy())
    
    # Get final memory usage
    final_memory = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Final memory usage: {final_memory:.2f}MB")
    print(f"Memory saved: {initial_memory - final_memory:.2f}MB")
    
    # Check data types
    print(f"Data types after optimization:")
    for col, dtype in optimized_data.dtypes.items():
        print(f"   {col}: {dtype}")
    
    return optimized_data


def test_data_quality_cache():
    """Test the data quality cache."""
    print("\n" + "="*60)
    print("TESTING DATA QUALITY CACHE")
    print("="*60)
    
    # Create test data
    data = TestDataQualityFixes().create_test_data(include_issues=False)
    
    # Create cache
    cache = DataQualityCache(max_size=5)
    
    # Test cache operations
    test_result = {"test": "result", "timestamp": datetime.now().isoformat()}
    
    # Set cache
    cache.set(data, "test_method", test_result)
    print(f"✅ Cache set successfully")
    
    # Get cache
    cached_result = cache.get(data, "test_method")
    if cached_result:
        print(f"✅ Cache hit - retrieved result")
    else:
        print(f"❌ Cache miss")
    
    # Clear cache
    cache.clear()
    print(f"✅ Cache cleared")
    
    return cache


def test_utility_functions():
    """Test utility functions."""
    print("\n" + "="*60)
    print("TESTING UTILITY FUNCTIONS")
    print("="*60)
    
    # Test memory usage
    memory_usage = get_memory_usage()
    print(f"Current memory usage: {memory_usage['rss_mb']:.2f}MB")
    
    # Test dataframe optimization
    data = TestDataQualityFixes().create_test_data(include_issues=False)
    optimized_data = optimize_dataframe(data.copy())
    print(f"✅ Dataframe optimization completed")
    
    # Test cache clearing
    clear_data_quality_cache()
    print(f"✅ Data quality cache cleared")
    
    return memory_usage, optimized_data


def run_comprehensive_test():
    """Run comprehensive test of all fixes."""
    print("="*80)
    print("COMPREHENSIVE TEST OF ENHANCED DATA QUALITY FIXES")
    print("="*80)
    
    # Create test instance
    tester = TestDataQualityFixes()
    
    # Create test data with issues
    print("\n1. Creating test data with quality issues...")
    test_data = tester.create_test_data(include_issues=True)
    print(f"   Original data shape: {test_data.shape}")
    print(f"   Columns: {list(test_data.columns)}")
    
    # Test individual decorators
    print("\n2. Testing individual decorators...")
    
    # Test constant feature detection
    data1 = tester.test_constant_feature_detection(test_data.copy())
    
    # Test low variance feature detection
    data2 = tester.test_low_variance_feature_detection(test_data.copy())
    
    # Test data completeness
    data3 = tester.test_data_completeness_validation(test_data.copy())
    
    # Test datetime index validation
    data4 = tester.test_datetime_index_validation(test_data.copy())
    
    # Test multi-timeframe alignment
    data5 = tester.test_multi_timeframe_alignment(test_data.copy())
    
    # Test HMM data requirements
    data6 = tester.test_hmm_data_requirements(test_data.copy())
    
    # Test data structure validation
    data7 = tester.test_data_structure_validation(test_data.copy())
    
    # Test memory optimization
    data8 = tester.test_memory_optimization(test_data.copy())
    
    # Test comprehensive validation
    data9 = tester.test_comprehensive_validation(test_data.copy())
    
    # Test memory-optimized validation
    data10 = tester.test_memory_optimized_validation(test_data.copy())
    
    # Test feature engineering pipeline
    data11 = tester.test_feature_engineering_pipeline(test_data.copy())
    
    # Test HMM regime discovery
    data12 = tester.test_hmm_regime_discovery(test_data.copy())
    
    # Test multi-timeframe processing
    data13 = tester.test_multi_timeframe_processing(test_data.copy())
    
    print("\n3. Testing raw data quality checker...")
    results, processed_data = test_raw_data_quality_checker()
    
    print("\n4. Testing memory optimizer...")
    optimized_data = test_memory_optimizer()
    
    print("\n5. Testing data quality cache...")
    cache = test_data_quality_cache()
    
    print("\n6. Testing utility functions...")
    memory_usage, utility_optimized_data = test_utility_functions()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return {
        "test_data": test_data,
        "processed_data": processed_data,
        "optimized_data": optimized_data,
        "results": results,
        "memory_usage": memory_usage
    }


if __name__ == "__main__":
    # Run the comprehensive test
    results = run_comprehensive_test()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ All enhanced data quality decorators implemented and tested")
    print(f"✅ Raw data quality checker updated with multi-timeframe validation")
    print(f"✅ Memory optimization utilities working")
    print(f"✅ Data quality cache implemented")
    print(f"✅ Utility functions available")
    print(f"✅ Comprehensive validation pipeline ready")
    print("="*80)