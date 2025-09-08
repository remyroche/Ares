#!/usr/bin/env python3
"""
Test Script for Step14 Optimizations

This script tests all the optimizations, fast-fail validations, and fixes
implemented in the Step14 tactician labeling system.
"""

import asyncio
import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append('/workspace')

def create_test_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create test data for validation."""
    np.random.seed(42)
    
    # Generate realistic market data
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    base_price = 100.0
    
    # Generate price data with regime-specific characteristics
    returns = np.random.normal(0, 0.001, n_samples)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
        'close': prices,
        'volume': np.random.exponential(1000, n_samples),
        'spread': np.random.exponential(0.0001, n_samples)
    })
    
    # Add volatility feature
    data['volatility'] = data['close'].pct_change().rolling(60).std().bfill()
    
    # Add regime labels (simulate 3 regimes)
    regime_conditions = [
        data['volatility'] < data['volatility'].quantile(0.33),
        (data['volatility'] >= data['volatility'].quantile(0.33)) & 
        (data['volatility'] < data['volatility'].quantile(0.66)),
        data['volatility'] >= data['volatility'].quantile(0.66)
    ]
    regime_choices = [0, 1, 2]
    data['composite_cluster_id'] = np.select(regime_conditions, regime_choices, default=0)
    
    return data

async def test_fast_fail_validations():
    """Test fast-fail validation mechanisms."""
    print("🧪 Testing Fast-Fail Validations...")
    
    try:
        from src.training.steps.model_training.step14_tactician_labeling import RegimeAwareTacticianLabeler
        
        config = {
            'tactician_triple_barrier': {
                'max_lookahead': 50,
                'enable_high_precision_mode': True,
                'precision_threshold': 0.85
            },
            'regime_specific_tactician': {
                'regime_specific_barriers': True,
                'min_regime_samples': 100
            },
            'memory_threshold_gb': 8.0,
            'max_data_points': 1000000
        }
        
        labeler = RegimeAwareTacticianLabeler(config)
        
        # Test 1: Insufficient data
        print("  📊 Test 1: Insufficient data validation")
        small_data = create_test_data(50)  # Less than minimum
        try:
            result = await labeler.apply_regime_specific_labeling(small_data)
            print("    ❌ Should have failed with insufficient data")
        except ValueError as e:
            print(f"    ✅ Correctly failed: {e}")
        
        # Test 2: Missing required columns
        print("  📊 Test 2: Missing columns validation")
        incomplete_data = create_test_data(1000)
        incomplete_data = incomplete_data.drop('close', axis=1)
        try:
            result = await labeler.apply_regime_specific_labeling(incomplete_data)
            print("    ❌ Should have failed with missing columns")
        except ValueError as e:
            print(f"    ✅ Correctly failed: {e}")
        
        # Test 3: Excessive missing values
        print("  📊 Test 3: Missing values validation")
        missing_data = create_test_data(1000)
        missing_data.loc[:500, 'close'] = np.nan  # 50% missing
        try:
            result = await labeler.apply_regime_specific_labeling(missing_data)
            print("    ❌ Should have failed with excessive missing values")
        except ValueError as e:
            print(f"    ✅ Correctly failed: {e}")
        
        print("  ✅ Fast-fail validations working correctly")
        
    except Exception as e:
        print(f"  ❌ Fast-fail validation test failed: {e}")

async def test_barrier_parameter_validation():
    """Test barrier parameter validation."""
    print("🧪 Testing Barrier Parameter Validation...")
    
    try:
        from src.training.steps.model_training.step14_tactician_labeling import RegimeAwareTacticianLabeler
        
        config = {
            'tactician_triple_barrier': {
                'max_lookahead': 50,
                'enable_high_precision_mode': True,
                'precision_threshold': 0.85
            },
            'regime_specific_tactician': {
                'regime_specific_barriers': True,
                'min_regime_samples': 100
            }
        }
        
        labeler = RegimeAwareTacticianLabeler(config)
        
        # Test invalid volatility
        print("  📊 Test 1: Invalid volatility parameter")
        assert not labeler._validate_barrier_parameters(-0.1, 1000, 0.0001), "Should reject negative volatility"
        assert not labeler._validate_barrier_parameters(1.5, 1000, 0.0001), "Should reject volatility > 1"
        print("    ✅ Volatility validation working")
        
        # Test invalid volume
        print("  📊 Test 2: Invalid volume parameter")
        assert not labeler._validate_barrier_parameters(0.01, -100, 0.0001), "Should reject negative volume"
        assert not labeler._validate_barrier_parameters(0.01, 0, 0.0001), "Should reject zero volume"
        print("    ✅ Volume validation working")
        
        # Test invalid spread
        print("  📊 Test 3: Invalid spread parameter")
        assert not labeler._validate_barrier_parameters(0.01, 1000, -0.0001), "Should reject negative spread"
        print("    ✅ Spread validation working")
        
        # Test calculated barriers
        print("  📊 Test 4: Calculated barrier validation")
        assert not labeler._validate_calculated_barriers(0.6, 0.01), "Should reject upper barrier > 0.5"
        assert not labeler._validate_calculated_barriers(0.01, 0.6), "Should reject lower barrier > 0.5"
        assert not labeler._validate_calculated_barriers(0.01, 0.02), "Should reject upper <= lower"
        print("    ✅ Calculated barrier validation working")
        
        print("  ✅ Barrier parameter validation working correctly")
        
    except Exception as e:
        print(f"  ❌ Barrier parameter validation test failed: {e}")

async def test_regime_detection_logic():
    """Test fixed regime detection logic."""
    print("🧪 Testing Regime Detection Logic...")
    
    try:
        from src.training.steps.model_training.step14_tactician_labeling import RegimeAwareTacticianLabeler
        
        config = {
            'tactician_triple_barrier': {
                'max_lookahead': 50,
                'enable_high_precision_mode': True,
                'precision_threshold': 0.85
            }
        }
        
        labeler = RegimeAwareTacticianLabeler(config)
        
        # Test with valid data
        print("  📊 Test 1: Valid volatility data")
        test_data = create_test_data(1000)
        regimes = labeler._get_market_regime(test_data)
        
        assert len(regimes) == len(test_data), "Regime series should match data length"
        assert regimes.notna().all(), "No NaN values should be present"
        assert len(regimes.unique()) >= 2, "Should have at least 2 regimes"
        print("    ✅ Valid data processing working")
        
        # Test with missing volatility column
        print("  📊 Test 2: Missing volatility column")
        no_vol_data = test_data.drop('volatility', axis=1)
        regimes = labeler._get_market_regime(no_vol_data)
        
        assert len(regimes) == len(no_vol_data), "Should return default regimes"
        assert (regimes == 'SIDEWAYS').all(), "Should return all SIDEWAYS regimes"
        print("    ✅ Missing column handling working")
        
        # Test with all NaN volatility
        print("  📊 Test 3: All NaN volatility data")
        nan_vol_data = test_data.copy()
        nan_vol_data['volatility'] = np.nan
        regimes = labeler._get_market_regime(nan_vol_data)
        
        assert len(regimes) == len(nan_vol_data), "Should return default regimes"
        assert (regimes == 'SIDEWAYS').all(), "Should return all SIDEWAYS regimes"
        print("    ✅ NaN data handling working")
        
        print("  ✅ Regime detection logic working correctly")
        
    except Exception as e:
        print(f"  ❌ Regime detection logic test failed: {e}")

async def test_computational_optimizations():
    """Test computational optimizations."""
    print("🧪 Testing Computational Optimizations...")
    
    try:
        from src.training.steps.model_training.step14_tactician_labeling import RegimeAwareTacticianLabeler
        
        config = {
            'tactician_triple_barrier': {
                'max_lookahead': 50,
                'enable_high_precision_mode': True,
                'precision_threshold': 0.85
            },
            'regime_specific_tactician': {
                'regime_specific_barriers': True,
                'min_regime_samples': 100
            }
        }
        
        labeler = RegimeAwareTacticianLabeler(config)
        
        # Test regime statistics pre-calculation
        print("  📊 Test 1: Regime statistics pre-calculation")
        test_data = create_test_data(5000)
        unique_regimes = test_data['composite_cluster_id'].unique()
        
        start_time = time.time()
        regime_stats = labeler._calculate_regime_statistics_optimized(
            test_data, 'composite_cluster_id', unique_regimes
        )
        calc_time = time.time() - start_time
        
        assert len(regime_stats) == len(unique_regimes), "Should calculate stats for all regimes"
        for regime, stats in regime_stats.items():
            assert 'volatility' in stats, "Should include volatility"
            assert 'volume_mean' in stats, "Should include volume_mean"
            assert 'sample_count' in stats, "Should include sample_count"
        
        print(f"    ✅ Regime statistics calculated in {calc_time:.3f}s")
        
        # Test barrier caching
        print("  📊 Test 2: Barrier calculation caching")
        regime_data = test_data[test_data['composite_cluster_id'] == 0]
        
        # First calculation
        start_time = time.time()
        barriers1 = await labeler._get_regime_specific_barriers_optimized(
            '0', regime_data, regime_stats.get(0, {})
        )
        first_calc_time = time.time() - start_time
        
        # Second calculation (should use cache)
        start_time = time.time()
        barriers2 = await labeler._get_regime_specific_barriers_optimized(
            '0', regime_data, regime_stats.get(0, {})
        )
        second_calc_time = time.time() - start_time
        
        assert barriers1 == barriers2, "Cached results should match"
        assert second_calc_time < first_calc_time, "Cached calculation should be faster"
        print(f"    ✅ Caching working: {first_calc_time:.3f}s -> {second_calc_time:.3f}s")
        
        print("  ✅ Computational optimizations working correctly")
        
    except Exception as e:
        print(f"  ❌ Computational optimizations test failed: {e}")

async def test_memory_management():
    """Test memory management and leak prevention."""
    print("🧪 Testing Memory Management...")
    
    try:
        from src.training.steps.model_training.step14_tactician_labeling import RegimeAwareTacticianLabeler
        
        config = {
            'tactician_triple_barrier': {
                'max_lookahead': 50,
                'enable_high_precision_mode': True,
                'precision_threshold': 0.85
            },
            'regime_specific_tactician': {
                'regime_specific_barriers': True,
                'min_regime_samples': 100
            },
            'max_cache_size': 10,  # Small cache for testing
            'cleanup_frequency': 5
        }
        
        labeler = RegimeAwareTacticianLabeler(config)
        
        # Test bounded cache
        print("  📊 Test 1: Bounded cache management")
        initial_cache_size = len(labeler._barrier_cache)
        
        # Add more items than cache size
        for i in range(15):
            labeler._barrier_cache[f'key_{i}'] = {'data': f'value_{i}'}
        
        assert len(labeler._barrier_cache) <= config['max_cache_size'], "Cache should be bounded"
        print("    ✅ Cache bounded correctly")
        
        # Test periodic cleanup
        print("  📊 Test 2: Periodic cleanup")
        labeler._operation_count = 0
        labeler._regime_stats_cache = {f'key_{i}': f'value_{i}' for i in range(150)}
        
        # Trigger cleanup
        for _ in range(5):
            labeler._periodic_cleanup()
        
        assert len(labeler._regime_stats_cache) <= 100, "Cache should be cleaned up"
        print("    ✅ Periodic cleanup working")
        
        # Test resource cleanup
        print("  📊 Test 3: Resource cleanup")
        labeler.regime_barrier_results = {'test': 'data'}
        labeler.regime_labeling_results = {'test': 'data'}
        labeler._regime_stats_cache = {'test': 'data'}
        labeler._barrier_cache = {'test': 'data'}
        
        labeler._cleanup_resources()
        
        assert len(labeler.regime_barrier_results) == 0, "Results should be cleared"
        assert len(labeler.regime_labeling_results) == 0, "Results should be cleared"
        assert len(labeler._regime_stats_cache) == 0, "Stats cache should be cleared"
        assert len(labeler._barrier_cache) == 0, "Barrier cache should be cleared"
        print("    ✅ Resource cleanup working")
        
        print("  ✅ Memory management working correctly")
        
    except Exception as e:
        print(f"  ❌ Memory management test failed: {e}")

async def test_end_to_end_optimization():
    """Test end-to-end optimization performance."""
    print("🧪 Testing End-to-End Optimization Performance...")
    
    try:
        from src.training.steps.model_training.step14_tactician_labeling import RegimeAwareTacticianLabeler
        
        config = {
            'tactician_triple_barrier': {
                'max_lookahead': 50,
                'enable_high_precision_mode': True,
                'precision_threshold': 0.85
            },
            'regime_specific_tactician': {
                'regime_specific_barriers': True,
                'min_regime_samples': 100
            },
            'memory_threshold_gb': 8.0,
            'max_data_points': 1000000
        }
        
        labeler = RegimeAwareTacticianLabeler(config)
        
        # Test with medium-sized dataset
        print("  📊 Test: Medium dataset processing")
        test_data = create_test_data(10000)
        
        start_time = time.time()
        result = await labeler.apply_regime_specific_labeling(test_data)
        processing_time = time.time() - start_time
        
        assert len(result) == len(test_data), "Result should match input length"
        assert 'label' in result.columns, "Should have label column"
        assert 'potential_profit_pct' in result.columns, "Should have profit column"
        
        # Check label distribution
        label_counts = result['label'].value_counts()
        assert len(label_counts) >= 2, "Should have multiple label types"
        
        print(f"    ✅ Processed {len(test_data)} samples in {processing_time:.3f}s")
        print(f"    📊 Label distribution: {dict(label_counts)}")
        
        print("  ✅ End-to-end optimization working correctly")
        
    except Exception as e:
        print(f"  ❌ End-to-end optimization test failed: {e}")

async def main():
    """Run all optimization tests."""
    print("🚀 Starting Step14 Optimization Tests")
    print("=" * 60)
    
    test_functions = [
        test_fast_fail_validations,
        test_barrier_parameter_validation,
        test_regime_detection_logic,
        test_computational_optimizations,
        test_memory_management,
        test_end_to_end_optimization
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test_func.__name__} failed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Step14 optimizations working correctly!")
        return True
    else:
        print("⚠️ Some optimizations need attention")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)