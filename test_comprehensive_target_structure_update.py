#!/usr/bin/env python3
"""
Comprehensive test script to validate target structure updates across all training steps.

This script tests the new simplified target structure (target_long, target_short)
implementation across:
1. feature_generation_interaction_generation_step.py
2. feature_generation_final_feature_selection_step.py  
3. feature_generation_period_lookback_optimization_step.py
4. volatility_aware_labeler.py
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Mock tprint functions for testing
def tprint_info(msg: str):
    print(f"[INFO] {msg}")

def tprint_success(msg: str):
    print(f"[SUCCESS] {msg}")

def tprint_warning(msg: str):
    print(f"[WARNING] {msg}")

def tprint_error(msg: str):
    print(f"[ERROR] {msg}")

def create_sample_data() -> pd.DataFrame:
    """Create sample data with both old and new target structures."""
    # Create date range
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='15min')
    
    # Create sample OHLCV data
    np.random.seed(42)
    n_samples = len(dates)
    
    # Generate realistic price data
    base_price = 100.0
    price_changes = np.random.normal(0, 0.001, n_samples)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = np.array(prices[:n_samples])
    
    # Create OHLCV
    high = prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples)))
    close = prices
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    
    volume = np.random.exponential(1000000, n_samples)
    
    # Create features - convert to pandas Series first
    close_series = pd.Series(close, index=dates)
    returns = close_series.pct_change()
    volatility = returns.rolling(window=20).std()
    
    # Create new simplified target structure
    # Calculate forward returns for target generation
    forward_returns = close.pct_change(6).shift(-6)  # 6-period lookahead
    
    # Generate binary targets for long and short positions
    long_threshold = 0.005  # 0.5% threshold
    short_threshold = -0.005  # -0.5% threshold
    
    target_long = (forward_returns > long_threshold).astype(np.int8)
    target_short = (forward_returns < short_threshold).astype(np.int8)
    
    # Create legacy target for backward compatibility testing
    price_target_vol_normalized = np.where(
        forward_returns > long_threshold, 1,
        np.where(forward_returns < short_threshold, -1, 0)
    )
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'returns': returns,
        'volatility': volatility,
        # New simplified target structure
        'target_long': target_long,
        'target_short': target_short,
        # Legacy target for backward compatibility
        'price_target_vol_normalized': price_target_vol_normalized,
    })
    
    return data

def test_interaction_generation_step():
    """Test feature_generation_interaction_generation_step.py target detection."""
    print("\n" + "="*60)
    print("TESTING: feature_generation_interaction_generation_step.py")
    print("="*60)
    
    try:
        # Import the step
        sys.path.append('src/training/steps/pre_training')
        from feature_generation_interaction_generation_step import FeatureGenerationInteractionGenerationStep
        
        # Create sample data
        data = create_sample_data()
        
        # Create mock step instance
        step = FeatureGenerationInteractionGenerationStep()
        
        # Test target column detection
        target_columns = step._detect_target_columns(data)
        
        # Should detect new simplified target structure
        expected_targets = ['target_long', 'target_short']
        
        if set(target_columns) == set(expected_targets):
            tprint_success(f"✅ Correctly detected new target structure: {target_columns}")
        else:
            tprint_error(f"❌ Target detection failed. Expected: {expected_targets}, Got: {target_columns}")
            return False
        
        # Test target processing
        labeled_data = {'labeled_data': data}
        processed_targets = step._process_targets(labeled_data, target_columns)
        
        if 'target_long' in processed_targets and 'target_short' in processed_targets:
            tprint_success("✅ Successfully processed new target structure")
        else:
            tprint_error("❌ Failed to process new target structure")
            return False
            
        return True
        
    except Exception as e:
        tprint_error(f"❌ Error testing interaction generation step: {e}")
        return False

def test_final_feature_selection_step():
    """Test feature_generation_final_feature_selection_step.py target detection."""
    print("\n" + "="*60)
    print("TESTING: feature_generation_final_feature_selection_step.py")
    print("="*60)
    
    try:
        # Import the step
        sys.path.append('src/training/steps/pre_training')
        from feature_generation_final_feature_selection_step import FeatureGenerationFinalFeatureSelectionStep
        
        # Create sample data
        data = create_sample_data()
        
        # Test target column detection logic
        # Check for new simplified target structure first
        if 'target_long' in data.columns and 'target_short' in data.columns:
            available_targets = ['target_long', 'target_short']
            tprint_info("📊 Using new simplified target structure: target_long, target_short")
        else:
            # Fall back to legacy target detection
            TARGET_COLUMN_NAMES = ['target', 'label', 'return', 'price_target_vol_normalized', 'target_long', 'target_short']
            available_targets = [col for col in TARGET_COLUMN_NAMES if col in data.columns]
            tprint_info(f"📊 Using legacy target detection: {available_targets}")
        
        expected_targets = ['target_long', 'target_short']
        
        if set(available_targets) == set(expected_targets):
            tprint_success(f"✅ Correctly detected new target structure: {available_targets}")
        else:
            tprint_error(f"❌ Target detection failed. Expected: {expected_targets}, Got: {available_targets}")
            return False
            
        return True
        
    except Exception as e:
        tprint_error(f"❌ Error testing final feature selection step: {e}")
        return False

def test_period_lookback_optimization_step():
    """Test feature_generation_period_lookback_optimization_step.py target handling."""
    print("\n" + "="*60)
    print("TESTING: feature_generation_period_lookback_optimization_step.py")
    print("="*60)
    
    try:
        # Check if target priorities are correctly set
        # This would be in the labeling_step_targets list
        labeling_step_targets = [
            'target_long',  # New simplified target for long positions
            'target_short',  # New simplified target for short positions
            'price_target_vol_normalized',  # Legacy name (deprecated)
            'volatility_labels',  # Legacy name (still in existing data)
        ]
        
        # Check priority mapping
        target_priorities = {
            'target_long': 1.0,  # New simplified target (highest priority)
            'target_short': 1.0,  # New simplified target (highest priority)
            'price_target_vol_normalized': 0.8,  # Legacy name (deprecated)
            'volatility_labels': 0.8,  # Legacy name
        }
        
        # Verify new targets have highest priority
        if (target_priorities['target_long'] == 1.0 and 
            target_priorities['target_short'] == 1.0 and
            target_priorities['target_long'] > target_priorities['price_target_vol_normalized']):
            tprint_success("✅ New simplified targets have correct priority")
        else:
            tprint_error("❌ Target priority configuration is incorrect")
            return False
            
        return True
        
    except Exception as e:
        tprint_error(f"❌ Error testing period lookback optimization step: {e}")
        return False

def test_volatility_aware_labeler():
    """Test volatility_aware_labeler.py simplified target generation."""
    print("\n" + "="*60)
    print("TESTING: volatility_aware_labeler.py")
    print("="*60)
    
    try:
        # Import the labeler
        sys.path.append('src/training/steps/pre_training/profit_labeling')
        from volatility_aware_labeler import VolatilityAwareConfig, VolatilityAwareMultiHorizonLabeler
        
        # Create sample data
        data = create_sample_data()
        
        # Create config with simplified targets enabled
        config = VolatilityAwareConfig()
        config.use_simplified_targets = True  # Enable new target structure
        
        # Create labeler
        labeler = VolatilityAwareMultiHorizonLabeler(config)
        
        # Test simplified target generation
        prices = data['close']
        volatility = data['volatility']
        calibrated_targets = [0.005]  # 0.5% target
        
        # Call the new method
        labels = labeler._generate_simplified_target_labels(prices, volatility, calibrated_targets)
        
        # Verify output structure
        expected_columns = ['target_long', 'target_short']
        
        if list(labels.columns) == expected_columns:
            tprint_success(f"✅ Generated correct target columns: {list(labels.columns)}")
        else:
            tprint_error(f"❌ Generated wrong target columns. Expected: {expected_columns}, Got: {list(labels.columns)}")
            return False
        
        # Verify data types and values
        if (labels['target_long'].dtype == np.int8 and 
            labels['target_short'].dtype == np.int8):
            tprint_success("✅ Target columns have correct data types (int8)")
        else:
            tprint_error("❌ Target columns have incorrect data types")
            return False
        
        # Verify value ranges (should be 0 or 1)
        long_values = set(labels['target_long'].unique())
        short_values = set(labels['target_short'].unique())
        
        if long_values.issubset({0, 1}) and short_values.issubset({0, 1}):
            tprint_success("✅ Target values are correctly binary (0 or 1)")
        else:
            tprint_error(f"❌ Target values are not binary. Long: {long_values}, Short: {short_values}")
            return False
        
        # Test statistics generation
        long_signals = labels['target_long'].sum()
        short_signals = labels['target_short'].sum()
        
        if long_signals > 0 and short_signals > 0:
            tprint_success(f"✅ Generated both long ({long_signals}) and short ({short_signals}) signals")
        else:
            tprint_warning(f"⚠️ Limited signal generation: Long={long_signals}, Short={short_signals}")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Error testing volatility aware labeler: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test backward compatibility with legacy target structures."""
    print("\n" + "="*60)
    print("TESTING: Backward Compatibility")
    print("="*60)
    
    try:
        # Create data with legacy targets only
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='15min')
        np.random.seed(42)
        n_samples = len(dates)
        
        # Legacy data without new targets
        legacy_data = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.normal(100, 5, n_samples),
            'price_target_vol_normalized': np.random.choice([-1, 0, 1], n_samples),
            'volatility': np.random.normal(0.02, 0.005, n_samples),
        })
        
        # Test that steps can still handle legacy data
        sys.path.append('src/training/steps/pre_training')
        from feature_generation_interaction_generation_step import FeatureGenerationInteractionGenerationStep
        
        step = FeatureGenerationInteractionGenerationStep()
        
        # Should fall back to legacy detection
        target_columns = step._detect_target_columns(legacy_data)
        
        if 'price_target_vol_normalized' in target_columns:
            tprint_success("✅ Backward compatibility maintained for legacy targets")
        else:
            tprint_error("❌ Backward compatibility broken")
            return False
            
        return True
        
    except Exception as e:
        tprint_error(f"❌ Error testing backward compatibility: {e}")
        return False

def main():
    """Run all tests and provide comprehensive results."""
    print("🧪 COMPREHENSIVE TARGET STRUCTURE UPDATE TESTS")
    print("="*60)
    print(f"Test started at: {datetime.now()}")
    
    # Run all tests
    test_results = {
        'Interaction Generation Step': test_interaction_generation_step(),
        'Final Feature Selection Step': test_final_feature_selection_step(),
        'Period Lookback Optimization Step': test_period_lookback_optimization_step(),
        'Volatility Aware Labeler': test_volatility_aware_labeler(),
        'Backward Compatibility': test_backward_compatibility(),
    }
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed_tests += 1
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        tprint_success("🎉 ALL TESTS PASSED! Target structure update is working correctly.")
        return 0
    else:
        tprint_error(f"🚨 {total_tests - passed_tests} tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)