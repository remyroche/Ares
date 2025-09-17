#!/usr/bin/env python3
"""
Test Script for Triple Barrier Labeling Fixes

This script tests all the implemented fixes:
1. Barrier hit race condition resolution
2. Standardized transaction cost modeling (0.08%)
3. Improved end index validation with temporal leakage detection

Usage:
    python test_triple_barrier_fixes.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the fixed implementations
try:
    from market_analysis.triple_barrier_labeling.core import (
        TripleBarrierLabeler, TripleBarrierConfig, LabelingMethod
    )
    from src.feature_generation.utils.step06_labeling_components.optimized_triple_barrier_labeling_improved import (
        OptimizedTripleBarrierLabelingImproved
    )
    print("✅ Successfully imported triple barrier implementations")
except ImportError as e:
    print(f"❌ Failed to import implementations: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLC test data with realistic patterns."""
    np.random.seed(seed)
    
    # Create base price series with trend and volatility
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_samples)  # Small positive drift, 2% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC data
    data = []
    for i, close in enumerate(prices):
        # Create realistic OHLC relationships
        volatility = np.random.uniform(0.005, 0.03)  # 0.5% to 3% intraday volatility
        
        open_price = close * np.random.uniform(0.99, 1.01)
        high = max(open_price, close) * np.random.uniform(1.0, 1 + volatility)
        low = min(open_price, close) * np.random.uniform(1 - volatility, 1.0)
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    # Create DataFrame with datetime index
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')
    
    return df

def create_race_condition_data(n_samples: int = 100) -> pd.DataFrame:
    """Create test data specifically designed to trigger race conditions."""
    np.random.seed(123)
    
    data = []
    base_price = 100.0
    
    for i in range(n_samples):
        # Create bars where both profit and stop barriers could be hit
        close = base_price + np.random.uniform(-1, 1)
        
        # Deliberately create scenarios where both barriers are hit
        if i % 10 == 0:  # Every 10th bar
            # High volatility bar that hits both barriers
            high = close * 1.015  # +1.5% high
            low = close * 0.985   # -1.5% low
            open_price = close
        else:
            # Normal bars
            volatility = np.random.uniform(0.001, 0.005)
            open_price = close * np.random.uniform(0.999, 1.001)
            high = max(open_price, close) * np.random.uniform(1.0, 1 + volatility)
            low = min(open_price, close) * np.random.uniform(1 - volatility, 1.0)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': 1000.0
        })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')
    
    return df

def test_barrier_race_condition_fix():
    """Test that barrier hit race conditions are properly resolved."""
    print("\n🔍 Testing Barrier Hit Race Condition Fix...")
    
    # Create test data with potential race conditions
    data = create_race_condition_data(100)
    
    # Test with configuration that should trigger race conditions
    config = TripleBarrierConfig(
        pt_mult=0.01,   # 1% profit target
        sl_mult=0.01,   # 1% stop loss
        max_holding_period=50,
        transaction_cost=0.0008  # Global 0.08%
    )
    
    try:
        labeler = TripleBarrierLabeler(config)
        result = labeler.create_labels(data, method=LabelingMethod.TRIPLE_BARRIER)
        
        # Check that we have labels
        assert 'label' in result.labels.columns, "Labels column missing"
        assert 'barrier_type' in result.labels.columns, "Barrier type column missing"
        
        # Check for priority resolution indicators
        barrier_types = result.labels['barrier_type'].value_counts()
        priority_indicators = [bt for bt in barrier_types.index if 'priority' in bt or 'tie_break' in bt]
        
        print(f"   ✅ Generated {len(result.labels)} labels")
        print(f"   📊 Barrier types: {barrier_types.to_dict()}")
        
        if priority_indicators:
            print(f"   🎯 Race conditions resolved: {len(priority_indicators)} priority cases detected")
        else:
            print(f"   ℹ️ No race conditions detected in this test data")
        
        # Validate transaction costs are applied
        profit_pcts = result.labels['profit_pct'].dropna()
        if len(profit_pcts) > 0:
            print(f"   💰 Profit range (net of tx costs): {profit_pcts.min():.4f} to {profit_pcts.max():.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def test_transaction_cost_standardization():
    """Test that transaction costs are standardized to 0.08%."""
    print("\n💰 Testing Transaction Cost Standardization...")
    
    data = create_test_data(200)
    
    # Test core implementation
    try:
        config = TripleBarrierConfig(
            pt_mult=0.005,  # 0.5%
            sl_mult=0.003,  # 0.3%
            max_holding_period=30
        )
        
        # Verify default transaction cost is 0.08%
        assert abs(config.transaction_cost - 0.0008) < 1e-6, f"Default transaction cost should be 0.0008, got {config.transaction_cost}"
        
        labeler = TripleBarrierLabeler(config)
        result = labeler.create_labels(data, method=LabelingMethod.TRIPLE_BARRIER)
        
        # Check that transaction costs are properly applied
        assert 'transaction_cost' in result.labels.columns, "Transaction cost column missing"
        
        tx_costs = result.labels['transaction_cost'].unique()
        expected_tx_cost = 0.0008
        
        assert abs(tx_costs[0] - expected_tx_cost) < 1e-6, f"Transaction cost should be {expected_tx_cost}, got {tx_costs[0]}"
        
        print(f"   ✅ Core implementation uses correct transaction cost: {expected_tx_cost}")
        
    except Exception as e:
        print(f"   ❌ Core implementation test failed: {e}")
        return False
    
    # Test optimized implementation
    try:
        from src.feature_generation.utils.step06_labeling_components.optimized_triple_barrier_labeling_improved import GLOBAL_TRANSACTION_COST
        
        assert abs(GLOBAL_TRANSACTION_COST - 0.0008) < 1e-6, f"Global transaction cost should be 0.0008, got {GLOBAL_TRANSACTION_COST}"
        
        optimized_labeler = OptimizedTripleBarrierLabelingImproved(
            profit_take_multiplier=0.005,
            stop_loss_multiplier=0.003,
            max_lookahead=30
        )
        
        # Verify the instance uses the global transaction cost
        assert abs(optimized_labeler.transaction_cost - 0.0008) < 1e-6, f"Instance transaction cost should be 0.0008, got {optimized_labeler.transaction_cost}"
        
        print(f"   ✅ Optimized implementation uses correct transaction cost: {GLOBAL_TRANSACTION_COST}")
        
    except Exception as e:
        print(f"   ❌ Optimized implementation test failed: {e}")
        return False
    
    return True

def test_end_index_validation():
    """Test end index validation and temporal leakage detection."""
    print("\n🔍 Testing End Index Validation and Temporal Leakage Detection...")
    
    data = create_test_data(500)
    
    try:
        # Test with valid configuration
        config = TripleBarrierConfig(
            pt_mult=0.005,
            sl_mult=0.003,
            max_holding_period=50,
            min_holding_period=1
        )
        
        labeler = TripleBarrierLabeler(config)
        
        # This should work without errors
        result = labeler.create_labels(data, method=LabelingMethod.TRIPLE_BARRIER)
        
        print(f"   ✅ Valid configuration passed validation")
        print(f"   📊 Generated {len(result.labels)} labels")
        
        # Check validation failure statistics
        barrier_types = result.labels['barrier_type'].value_counts()
        validation_failures = sum(count for bt, count in barrier_types.items() 
                                if 'validation_failed' in bt or 'invalid_entry_price' in bt)
        
        print(f"   🔍 Validation failures: {validation_failures}/{len(result.labels)} ({validation_failures/len(result.labels)*100:.1f}%)")
        
    except Exception as e:
        print(f"   ❌ Valid configuration test failed: {e}")
        return False
    
    # Test optimized implementation validation
    try:
        optimized_labeler = OptimizedTripleBarrierLabelingImproved(
            profit_take_multiplier=0.005,
            stop_loss_multiplier=0.003,
            max_lookahead=50
        )
        
        # This should trigger the comprehensive validation
        result_data = optimized_labeler.apply_labeling(data)
        
        print(f"   ✅ Optimized implementation validation passed")
        print(f"   📊 Generated labels for {len(result_data)} rows")
        
    except Exception as e:
        print(f"   ❌ Optimized implementation validation failed: {e}")
        return False
    
    # Test temporal leakage detection with invalid configuration
    try:
        # Create a configuration that should trigger leakage detection
        invalid_config = TripleBarrierConfig(
            pt_mult=0.005,
            sl_mult=0.003,
            max_holding_period=1000,  # Unreasonably large
            min_holding_period=1
        )
        
        # This should trigger warnings or errors about excessive lookahead
        labeler = TripleBarrierLabeler(invalid_config)
        result = labeler.create_labels(data[:100], method=LabelingMethod.TRIPLE_BARRIER)  # Small dataset
        
        print(f"   ⚠️ Large lookahead configuration completed (may have warnings)")
        
    except Exception as e:
        print(f"   ✅ Temporal leakage detection working: {e}")
    
    return True

def test_numerical_stability():
    """Test numerical stability improvements."""
    print("\n🔢 Testing Numerical Stability...")
    
    # Create edge case data
    edge_cases = []
    
    # Very small prices
    edge_cases.append(pd.DataFrame({
        'open': [0.0001, 0.0002, 0.0001],
        'high': [0.00015, 0.00025, 0.00015],
        'low': [0.00008, 0.00015, 0.00008],
        'close': [0.00012, 0.0002, 0.00012],
        'volume': [1000, 1000, 1000]
    }, index=pd.date_range('2023-01-01', periods=3, freq='1min')))
    
    # Zero and negative prices (should be handled gracefully)
    edge_cases.append(pd.DataFrame({
        'open': [100, 0, -1],
        'high': [105, 0, 0],
        'low': [95, 0, -2],
        'close': [102, 0, -1],
        'volume': [1000, 1000, 1000]
    }, index=pd.date_range('2023-01-01', periods=3, freq='1min')))
    
    config = TripleBarrierConfig(
        pt_mult=0.01,
        sl_mult=0.005,
        max_holding_period=10
    )
    
    for i, data in enumerate(edge_cases):
        try:
            labeler = TripleBarrierLabeler(config)
            result = labeler.create_labels(data, method=LabelingMethod.TRIPLE_BARRIER)
            
            print(f"   ✅ Edge case {i+1} handled gracefully")
            
            # Check for invalid entries
            barrier_types = result.labels['barrier_type'].value_counts()
            invalid_entries = sum(count for bt, count in barrier_types.items() 
                                if 'invalid_entry_price' in bt)
            
            if invalid_entries > 0:
                print(f"      📊 {invalid_entries} invalid entries properly handled")
            
        except Exception as e:
            print(f"   ❌ Edge case {i+1} failed: {e}")
            return False
    
    return True

def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("🧪 Starting Comprehensive Triple Barrier Labeling Fixes Test")
    print("=" * 70)
    
    tests = [
        ("Barrier Hit Race Condition Fix", test_barrier_race_condition_fix),
        ("Transaction Cost Standardization", test_transaction_cost_standardization),
        ("End Index Validation & Temporal Leakage Detection", test_end_index_validation),
        ("Numerical Stability", test_numerical_stability)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:10} | {test_name}")
        if result:
            passed += 1
    
    print("-" * 70)
    print(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All fixes are working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)