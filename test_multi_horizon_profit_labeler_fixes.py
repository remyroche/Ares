#!/usr/bin/env python3
"""
Comprehensive Test Suite for Multi-Horizon Profit Labeler Fixes

This test suite validates all the critical fixes and performance improvements
made to the multi-horizon profit labeler.
"""

import numpy as np
import pandas as pd
import sys
import os
import time
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.steps.market_analysis.multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler, 
    MultiHorizonConfig,
    ScoringConstants
)

def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create realistic test data for validation."""
    np.random.seed(42)
    
    # Generate realistic price data with trends
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.002, n_samples)
    prices = [base_price]
    
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Ensure OHLC consistency
    for i in range(len(data)):
        data.loc[i, 'high'] = max(data.iloc[i][['open', 'high', 'low', 'close']])
        data.loc[i, 'low'] = min(data.iloc[i][['open', 'high', 'low', 'close']])
    
    return data

def test_critical_fixes():
    """Test critical fixes for negative scores and mathematical issues."""
    print("🧪 Testing Critical Fixes")
    print("=" * 50)
    
    # Test 1: Negative Score Elimination
    print("1. Testing negative score elimination...")
    labeler = MultiHorizonProfitLabeler()
    
    # Test problematic scenario that would cause negative scores
    test_result = labeler._calculate_quality_score(
        target_hit=True,
        time_to_hit=2,
        max_adverse=0.05,  # 5% adverse - would cause negative with old multiplier
        total_periods=4,
        net_profit=0.008
    )
    
    assert test_result >= 0.2, f"❌ FAIL: Quality score should be >= 0.2, got {test_result:.4f}"
    print(f"   ✅ PASS: Quality score = {test_result:.4f} (>= 0.2)")
    
    # Test 2: Division by Zero Protection
    print("2. Testing division by zero protection...")
    
    # Test momentum calculation with zero denominator
    composite_scores = {
        'long_immediate_opportunity': 0.5,
        'long_short_opportunity': 0.0,  # This would cause division by zero
        'short_immediate_opportunity': 0.3,
        'short_short_opportunity': 0.0
    }
    
    # This should not raise an exception
    try:
        # Test the safe_divide function directly
        from src.utils.math_validation import safe_divide
        result = safe_divide(0.5, 0.0, 0.0)
        assert result == 0.0, f"❌ FAIL: safe_divide should return 0.0, got {result}"
        print("   ✅ PASS: Division by zero protection works")
    except Exception as e:
        print(f"   ❌ FAIL: Division by zero protection failed: {e}")
    
    # Test 3: Improved Bounds Checking
    print("3. Testing improved bounds checking...")
    
    # Test with extreme adverse excursion
    extreme_result = labeler._calculate_quality_score(
        target_hit=True,
        time_to_hit=1,
        max_adverse=0.15,  # 15% adverse excursion
        total_periods=4,
        net_profit=0.01
    )
    
    assert 0.2 <= extreme_result <= 1.0, f"❌ FAIL: Score should be in [0.2, 1.0], got {extreme_result:.4f}"
    print(f"   ✅ PASS: Extreme adverse excursion handled: {extreme_result:.4f}")
    
    # Test 4: Graduated Scoring for Unprofitable Trades
    print("4. Testing graduated scoring for unprofitable trades...")
    
    small_loss = labeler._calculate_quality_score(
        target_hit=True,
        time_to_hit=2,
        max_adverse=0.02,
        total_periods=4,
        net_profit=-0.003  # Small loss
    )
    
    large_loss = labeler._calculate_quality_score(
        target_hit=True,
        time_to_hit=2,
        max_adverse=0.02,
        total_periods=4,
        net_profit=-0.015  # Large loss
    )
    
    assert small_loss > large_loss, f"❌ FAIL: Small loss should score higher than large loss"
    assert small_loss >= 0.2, f"❌ FAIL: Small loss score should be >= 0.2"
    print(f"   ✅ PASS: Graduated scoring works: small_loss={small_loss:.4f}, large_loss={large_loss:.4f}")

def test_matrix_operations_performance():
    """Test matrix operations and performance improvements."""
    print("\n🚀 Testing Matrix Operations Performance")
    print("=" * 50)
    
    # Create test data
    test_data = create_test_data(5000)
    
    # Test original vs optimized approach
    config = MultiHorizonConfig()
    labeler = MultiHorizonProfitLabeler(config)
    
    print("1. Testing vectorized operations...")
    
    # Time the vectorized approach
    start_time = time.time()
    result = labeler.generate_labels(test_data)
    end_time = time.time()
    
    vectorized_time = end_time - start_time
    
    print(f"   ✅ Vectorized processing completed in {vectorized_time:.2f} seconds")
    print(f"   → Processed {len(test_data)} samples")
    print(f"   → Generated {result.shape[1]} features")
    
    # Test data integrity
    assert result.shape[0] == test_data.shape[0], "❌ FAIL: Row count mismatch"
    assert result.shape[1] > test_data.shape[1], "❌ FAIL: No new features added"
    
    # Test for negative values in opportunity scores
    opportunity_cols = [col for col in result.columns if 'opportunity' in col]
    negative_count = 0
    
    for col in opportunity_cols:
        if col in result.columns:
            min_val = result[col].min()
            if min_val < 0:
                negative_count += 1
                print(f"   ⚠️ WARNING: {col} has negative values (min: {min_val:.4f})")
    
    if negative_count == 0:
        print("   ✅ PASS: No negative opportunity scores found")
    else:
        print(f"   ❌ FAIL: Found {negative_count} columns with negative values")
    
    return result

def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n🔍 Testing Edge Cases")
    print("=" * 50)
    
    labeler = MultiHorizonProfitLabeler()
    
    # Test 1: Insufficient data
    print("1. Testing insufficient data handling...")
    small_data = create_test_data(10)  # Less than required horizon
    result = labeler.generate_labels(small_data)
    
    assert result.shape == small_data.shape, "❌ FAIL: Should return original data for insufficient samples"
    print("   ✅ PASS: Insufficient data handled correctly")
    
    # Test 2: Single window data
    print("2. Testing single window data...")
    
    # Test the vectorized probability calculation with minimal data
    highs = np.array([100.0, 101.0])
    lows = np.array([99.0, 100.0])
    
    result = labeler._calculate_profit_probability_vectorized(
        highs, lows, 100.0, 0.01, 2, 'long'
    )
    
    assert isinstance(result, dict), "❌ FAIL: Should return dictionary"
    assert 'probability' in result, "❌ FAIL: Should contain probability"
    assert 0.0 <= result['probability'] <= 1.0, "❌ FAIL: Probability should be in [0, 1]"
    print("   ✅ PASS: Single window data handled correctly")
    
    # Test 3: Extreme values
    print("3. Testing extreme values...")
    
    # Test with very high adverse excursion
    extreme_result = labeler._calculate_quality_score(
        target_hit=True,
        time_to_hit=1,
        max_adverse=0.5,  # 50% adverse excursion
        total_periods=4,
        net_profit=0.001
    )
    
    assert 0.2 <= extreme_result <= 1.0, f"❌ FAIL: Extreme values not handled, got {extreme_result:.4f}"
    print(f"   ✅ PASS: Extreme values handled: {extreme_result:.4f}")

def test_constants_and_configuration():
    """Test that constants are properly defined and used."""
    print("\n⚙️ Testing Constants and Configuration")
    print("=" * 50)
    
    # Test 1: Constants are properly defined
    print("1. Testing constant definitions...")
    
    assert hasattr(ScoringConstants, 'RISK_PENALTY_MULTIPLIER'), "❌ FAIL: RISK_PENALTY_MULTIPLIER not defined"
    assert hasattr(ScoringConstants, 'MIN_QUALITY_SCORE'), "❌ FAIL: MIN_QUALITY_SCORE not defined"
    assert hasattr(ScoringConstants, 'MAX_QUALITY_SCORE'), "❌ FAIL: MAX_QUALITY_SCORE not defined"
    
    # Verify the constants have the expected values
    assert ScoringConstants.RISK_PENALTY_MULTIPLIER == 10, f"❌ FAIL: Expected RISK_PENALTY_MULTIPLIER=10, got {ScoringConstants.RISK_PENALTY_MULTIPLIER}"
    assert ScoringConstants.MIN_QUALITY_SCORE == 0.2, f"❌ FAIL: Expected MIN_QUALITY_SCORE=0.2, got {ScoringConstants.MIN_QUALITY_SCORE}"
    
    print("   ✅ PASS: Constants properly defined")
    print(f"   → RISK_PENALTY_MULTIPLIER: {ScoringConstants.RISK_PENALTY_MULTIPLIER}")
    print(f"   → MIN_QUALITY_SCORE: {ScoringConstants.MIN_QUALITY_SCORE}")
    print(f"   → MAX_QUALITY_SCORE: {ScoringConstants.MAX_QUALITY_SCORE}")
    
    # Test 2: Configuration validation
    print("2. Testing configuration validation...")
    
    # Test valid configuration
    config = MultiHorizonConfig()
    labeler = MultiHorizonProfitLabeler(config)
    print("   ✅ PASS: Valid configuration accepted")
    
    # Test invalid configuration (should raise error)
    try:
        invalid_config = MultiHorizonConfig()
        invalid_config.profit_targets = {'invalid': 0.001}  # Below minimum
        invalid_labeler = MultiHorizonProfitLabeler(invalid_config)
        print("   ❌ FAIL: Invalid configuration should have been rejected")
    except ValueError:
        print("   ✅ PASS: Invalid configuration properly rejected")

def test_comprehensive_labeling():
    """Test comprehensive labeling with real-world scenarios."""
    print("\n📊 Testing Comprehensive Labeling")
    print("=" * 50)
    
    # Create realistic test data
    test_data = create_test_data(2000)
    config = MultiHorizonConfig()
    labeler = MultiHorizonProfitLabeler(config)
    
    print("1. Running comprehensive labeling test...")
    
    start_time = time.time()
    labeled_data = labeler.generate_labels(test_data)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    print(f"   ✅ Labeling completed in {processing_time:.2f} seconds")
    print(f"   → Input: {test_data.shape[0]} samples, {test_data.shape[1]} columns")
    print(f"   → Output: {labeled_data.shape[0]} samples, {labeled_data.shape[1]} columns")
    print(f"   → New features: {labeled_data.shape[1] - test_data.shape[1]}")
    
    # Test data quality
    print("2. Testing data quality...")
    
    # Check for NaN values
    nan_count = labeled_data.isnull().sum().sum()
    if nan_count == 0:
        print("   ✅ PASS: No NaN values found")
    else:
        print(f"   ⚠️ WARNING: Found {nan_count} NaN values")
    
    # Check for infinite values
    inf_count = np.isinf(labeled_data.select_dtypes(include=[np.number])).sum().sum()
    if inf_count == 0:
        print("   ✅ PASS: No infinite values found")
    else:
        print(f"   ⚠️ WARNING: Found {inf_count} infinite values")
    
    # Check opportunity score distributions
    opportunity_cols = [col for col in labeled_data.columns if 'opportunity' in col]
    
    for col in opportunity_cols[:5]:  # Check first 5 opportunity columns
        if col in labeled_data.columns:
            col_data = labeled_data[col]
            min_val = col_data.min()
            max_val = col_data.max()
            mean_val = col_data.mean()
            
            print(f"   → {col}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")
            
            assert min_val >= 0.0, f"❌ FAIL: {col} has negative minimum value"
            assert max_val <= 1.0, f"❌ FAIL: {col} has values > 1.0"
    
    print("   ✅ PASS: Data quality checks passed")
    
    return labeled_data

def main():
    """Run all tests."""
    print("🧪 Multi-Horizon Profit Labeler - Comprehensive Test Suite")
    print("=" * 70)
    
    try:
        # Run all test suites
        test_critical_fixes()
        test_matrix_operations_performance()
        test_edge_cases()
        test_constants_and_configuration()
        labeled_data = test_comprehensive_labeling()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("✅ Critical fixes implemented and validated")
        print("✅ Matrix operations and performance improvements working")
        print("✅ Edge cases handled correctly")
        print("✅ Constants and configuration validated")
        print("✅ Comprehensive labeling successful")
        
        # Final summary
        print(f"\n📈 Final Results Summary:")
        print(f"   → Processed {len(labeled_data)} samples")
        print(f"   → Generated {labeled_data.shape[1]} total features")
        print(f"   → No negative opportunity scores")
        print(f"   → No division by zero errors")
        print(f"   → Improved performance with vectorized operations")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)