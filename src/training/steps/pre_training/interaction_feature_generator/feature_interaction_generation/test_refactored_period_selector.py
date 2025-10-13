"""
Test script for the refactored DataDrivenPeriodSelector implementation.

This script tests the new focused architecture and verifies that the refactoring
maintains backward compatibility while improving code organization.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_points: int = 1000, timeframe_minutes: int = 15) -> pd.DataFrame:
    """Create sample financial data for testing."""
    # Generate timestamps
    start_time = datetime.now() - timedelta(minutes=n_points * timeframe_minutes)
    timestamps = [start_time + timedelta(minutes=i * timeframe_minutes) for i in range(n_points)]
    
    # Generate sample price data with some patterns
    np.random.seed(42)  # For reproducible results
    
    # Base price with trend
    base_price = 100 + np.cumsum(np.random.normal(0, 0.5, n_points))
    
    # Add some volatility clusters
    volatility = np.ones(n_points) * 0.5
    volatility[200:300] = 2.0  # High volatility cluster
    volatility[600:700] = 2.0  # Another high volatility cluster
    
    # Add some cyclical patterns
    cycle1 = 5 * np.sin(np.arange(n_points) * 2 * np.pi / 50)  # 50-period cycle
    cycle2 = 3 * np.sin(np.arange(n_points) * 2 * np.pi / 100)  # 100-period cycle
    
    # Generate OHLCV data
    close_prices = base_price + cycle1 + cycle2 + np.random.normal(0, volatility)
    
    # Generate volume with some spikes
    volume = np.random.lognormal(10, 0.5, n_points)
    volume[200:210] *= 3  # Volume spike
    volume[600:610] *= 3  # Another volume spike
    
    # Create DataFrame
    data = pd.DataFrame({
        'close': close_prices,
        'volume': volume,
        'open': close_prices + np.random.normal(0, 0.1, n_points),
        'high': close_prices + np.abs(np.random.normal(0, 0.2, n_points)),
        'low': close_prices - np.abs(np.random.normal(0, 0.2, n_points))
    }, index=pd.DatetimeIndex(timestamps))
    
    return data

def test_basic_functionality():
    """Test basic functionality of the refactored implementation."""
    print("🧪 Testing basic functionality...")
    
    try:
        from data_driven_periods import DataDrivenPeriodSelector, get_data_driven_periods
        
        # Create sample data
        data = create_sample_data(1000, 15)
        print(f"📊 Created sample data: {data.shape}")
        
        # Test main class
        selector = DataDrivenPeriodSelector(max_periods=5)
        result = selector.select_optimal_periods(data, target_timeframe="15m")
        
        print(f"✅ Selected periods: {result.optimal_periods}")
        print(f"✅ Confidence score: {result.confidence_score:.3f}")
        print(f"✅ Categories: {result.period_categories}")
        
        # Test convenience function
        periods = get_data_driven_periods(data, target_timeframe="15m", max_periods=5)
        print(f"✅ Convenience function periods: {periods}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_focused_classes():
    """Test the individual focused classes."""
    print("\n🧪 Testing focused classes...")
    
    try:
        from data_driven_periods import PeriodAnalyzer, PeriodValidator, PeriodSelector
        
        # Create sample data
        data = create_sample_data(500, 15)
        
        # Test PeriodAnalyzer
        print("📊 Testing PeriodAnalyzer...")
        analyzer = PeriodAnalyzer(enable_vectorbt=False)  # Disable for simpler testing
        characteristics = analyzer.analyze_data_characteristics(data)
        print(f"✅ Analyzer characteristics: {list(characteristics.keys())}")
        
        # Test PeriodValidator
        print("📊 Testing PeriodValidator...")
        validator = PeriodValidator(min_period=2, max_period=100, max_periods=5)
        candidate_periods = [5, 10, 20, 50, 100]
        filtered_periods = validator.filter_periods(candidate_periods, characteristics)
        print(f"✅ Filtered periods: {filtered_periods}")
        
        ranked_periods = validator.rank_periods(filtered_periods, data, characteristics)
        print(f"✅ Ranked periods: {ranked_periods}")
        
        # Test PeriodSelector
        print("📊 Testing PeriodSelector...")
        selector = PeriodSelector(max_periods=5, enable_vectorbt=False)
        result = selector.select_optimal_periods(data, target_timeframe="15m")
        print(f"✅ Selector periods: {result.optimal_periods}")
        
        return True
        
    except Exception as e:
        print(f"❌ Focused classes test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    print("\n🧪 Testing performance monitoring...")
    
    try:
        from data_driven_periods import DataDrivenPeriodSelector
        
        # Create sample data
        data = create_sample_data(1000, 15)
        
        # Test with performance monitoring
        selector = DataDrivenPeriodSelector(max_periods=5)
        
        # Run analysis
        result = selector.select_optimal_periods(data, target_timeframe="15m")
        
        # Get performance stats
        stats = selector.get_performance_stats()
        print(f"✅ Performance stats: {len(stats)} metrics")
        print(f"✅ Total operations: {stats.get('total_operations', 0)}")
        print(f"✅ Cache hit rate: {stats.get('cache_hit_rate', 0):.1f}%")
        
        # Test cache functionality
        selector.enable_cache(True, max_size=10)
        cache_stats = selector.get_cache_stats()
        print(f"✅ Cache enabled: {cache_stats['cache_enabled']}")
        print(f"✅ Cache size: {cache_stats['cache_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and validation."""
    print("\n🧪 Testing error handling...")
    
    try:
        from data_driven_periods import DataDrivenPeriodSelector, ValidationError, AnalysisError
        
        selector = DataDrivenPeriodSelector(max_periods=5)
        
        # Test with invalid data
        try:
            selector.select_optimal_periods(pd.DataFrame(), target_timeframe="15m")
            print("❌ Should have failed with empty DataFrame")
            return False
        except (ValidationError, AnalysisError) as e:
            print(f"✅ Correctly caught error for empty DataFrame: {type(e).__name__}")
        
        # Test with insufficient data
        try:
            small_data = create_sample_data(50, 15)  # Less than min_data_points
            selector.select_optimal_periods(small_data, target_timeframe="15m")
            print("❌ Should have failed with insufficient data")
            return False
        except (ValidationError, AnalysisError) as e:
            print(f"✅ Correctly caught error for insufficient data: {type(e).__name__}")
        
        # Test with valid data
        data = create_sample_data(200, 15)
        result = selector.select_optimal_periods(data, target_timeframe="15m")
        print(f"✅ Valid data processed successfully: {result.optimal_periods}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_backward_compatibility():
    """Test backward compatibility with existing code."""
    print("\n🧪 Testing backward compatibility...")
    
    try:
        from data_driven_periods import (
            DataDrivenPeriodSelector, 
            get_data_driven_periods,
            get_data_driven_periods_with_stats,
            benchmark_period_selector
        )
        
        # Create sample data
        data = create_sample_data(1000, 15)
        
        # Test all convenience functions
        periods1 = get_data_driven_periods(data, target_timeframe="15m", max_periods=5)
        print(f"✅ get_data_driven_periods: {periods1}")
        
        periods2, stats = get_data_driven_periods_with_stats(data, target_timeframe="15m", max_periods=5)
        print(f"✅ get_data_driven_periods_with_stats: {periods2}")
        print(f"✅ Stats keys: {list(stats.keys())[:5]}...")
        
        # Test benchmarking (with fewer trials for speed)
        benchmark_results = benchmark_period_selector(data, target_timeframe="15m", max_periods=5, trials=2)
        print(f"✅ Benchmark results: {list(benchmark_results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting refactored DataDrivenPeriodSelector tests...\n")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Focused Classes", test_focused_classes),
        ("Performance Monitoring", test_performance_monitoring),
        ("Error Handling", test_error_handling),
        ("Backward Compatibility", test_backward_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Refactoring successful.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)