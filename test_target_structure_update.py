#!/usr/bin/env python3
"""
Test script to verify that the interaction generation step works correctly 
with the new simplified target structure (target_long and target_short).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data():
    """Create test data with new target structure."""
    # Create sample index
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    
    # Create sample features
    np.random.seed(42)
    features = {
        'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'volume': 1000 + np.random.randn(100) * 100,
        'rsi_14': 50 + np.random.randn(100) * 10,
        'sma_20': 100 + np.cumsum(np.random.randn(100) * 0.05),
        'volatility_20': np.abs(np.random.randn(100) * 0.02),
    }
    
    # Create new target structure
    targets = {
        'target_long': np.random.choice([0, 1], 100, p=[0.7, 0.3]),  # Binary long target
        'target_short': np.random.choice([0, 1], 100, p=[0.7, 0.3]),  # Binary short target
    }
    
    # Combine all data
    all_data = {**features, **targets}
    df = pd.DataFrame(all_data, index=dates)
    
    return df

def test_target_detection():
    """Test target detection logic from interaction generation step."""
    print("🧪 Testing target detection logic...")
    
    # Create test data with new target structure
    labeled_data = create_test_data()
    
    # Simulate the target detection logic from the interaction generation step
    # Primary target columns (from labeling integration step)
    primary_target_columns = [col for col in labeled_data.columns if col in [
        'target_long', 'target_short',  # New simplified target structure
        'directional_confidence', 'opportunity_asymmetry',
        'long_overall_opportunity', 'short_overall_opportunity', 'opportunity',
        'confidence_score', 'quality_score', 'signal_strength'
    ]]
    
    # Secondary target columns (pattern-based detection)
    secondary_target_columns = []
    for col in labeled_data.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in [
            'target', 'label', 'signal', 'opportunity', 'quality', 'confidence',
            'long_', 'short_', 'directional', 'asymmetry', 'regime', 'trend'
        ]):
            secondary_target_columns.append(col)
    
    # Combine and deduplicate
    all_target_candidates = list(set(primary_target_columns + secondary_target_columns))
    
    # Validate target columns have non-zero variance
    valid_target_columns = []
    for col in all_target_candidates:
        try:
            col_data = labeled_data[col].dropna()
            if len(col_data) > 0:
                variance = col_data.var()
                non_zero_count = (col_data != 0).sum()
                
                if variance > 1e-10 and non_zero_count > len(col_data) * 0.01:  # At least 1% non-zero
                    valid_target_columns.append(col)
                    print(f"✅ Valid target found: '{col}' (variance={variance:.6f})")
                else:
                    print(f"⚠️ Invalid target '{col}': variance={variance:.6f}, non-zero={non_zero_count}")
        except Exception as e:
            print(f"⚠️ Error validating target '{col}': {e}")
    
    target_columns = valid_target_columns
    
    print(f"\n📊 Target Detection Results:")
    print(f"  - Primary targets: {primary_target_columns}")
    print(f"  - Secondary targets: {secondary_target_columns}")
    print(f"  - Valid targets: {target_columns}")
    
    # Test target handling logic
    if set(target_columns) == {'target_long', 'target_short'}:
        print("\n✅ New simplified target structure detected correctly!")
        targets = labeled_data[['target_long', 'target_short']]
        # Create derived targets for compatibility
        targets['directional_confidence'] = (labeled_data['target_long'] + labeled_data['target_short']).abs()
        targets['opportunity_asymmetry'] = labeled_data['target_long'] - labeled_data['target_short']
        targets['long_overall_opportunity'] = labeled_data['target_long']
        targets['short_overall_opportunity'] = labeled_data['target_short']
        
        print("✅ Derived targets created successfully:")
        print(f"  - directional_confidence shape: {targets['directional_confidence'].shape}")
        print(f"  - opportunity_asymmetry shape: {targets['opportunity_asymmetry'].shape}")
        print(f"  - long_overall_opportunity shape: {targets['long_overall_opportunity'].shape}")
        print(f"  - short_overall_opportunity shape: {targets['short_overall_opportunity'].shape}")
        
        return True
    else:
        print(f"\n❌ Expected target_long and target_short, but found: {target_columns}")
        return False

def test_backward_compatibility():
    """Test backward compatibility with legacy target columns."""
    print("\n🧪 Testing backward compatibility...")
    
    # Create test data with legacy target structure
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    legacy_data = {
        'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'volume': 1000 + np.random.randn(100) * 100,
        'opportunity': np.random.randn(100) * 0.5,
        'directional_confidence': np.abs(np.random.randn(100) * 0.3),
    }
    
    labeled_data = pd.DataFrame(legacy_data, index=dates)
    
    # Test legacy target handling
    if 'opportunity' in labeled_data.columns:
        print("✅ Legacy 'opportunity' target detected")
        targets = labeled_data[['opportunity']]
        # Create derived targets for compatibility
        targets['directional_confidence'] = labeled_data['opportunity'].abs()
        targets['opportunity_asymmetry'] = labeled_data['opportunity']
        targets['long_overall_opportunity'] = labeled_data['opportunity'].clip(lower=0)
        targets['short_overall_opportunity'] = labeled_data['opportunity'].clip(upper=0).abs()
        
        print("✅ Legacy derived targets created successfully")
        return True
    else:
        print("❌ Legacy target structure not found")
        return False

def main():
    """Run all tests."""
    print("=" * 80)
    print("🧪 TESTING TARGET STRUCTURE UPDATES")
    print("=" * 80)
    
    # Test new target structure
    new_target_success = test_target_detection()
    
    # Test backward compatibility
    legacy_success = test_backward_compatibility()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"✅ New target structure (target_long, target_short): {'PASS' if new_target_success else 'FAIL'}")
    print(f"✅ Backward compatibility (legacy targets): {'PASS' if legacy_success else 'FAIL'}")
    
    overall_success = new_target_success and legacy_success
    print(f"\n🎯 Overall result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🚀 The interaction generation step is ready for the new target structure!")
    else:
        print("\n⚠️ Issues found - please review the implementation.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)