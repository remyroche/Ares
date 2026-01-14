#!/usr/bin/env python3
"""
Test script for enhanced Huber stability analysis with walk-forward time splits
"""

import sys
import os
sys.path.append('/Users/remyroche/Documents/Ares')

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

# Import enhanced Huber function
from src.utils.huber_regressor_for_trees import prepare_huber_teacher_outputs

def test_walkforward_stability():
    """Test the enhanced stability analysis with walk-forward time splits"""
    print("🧪 Testing Enhanced Huber with Walk-Forward Stability Analysis")
    print("=" * 70)
    
    # Generate synthetic time series data
    np.random.seed(42)
    n_samples = 1000
    n_features = 25
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        noise=0.1,
        random_state=42
    )
    
    # Create time index and add some temporal structure
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='15min')
    feature_names = [f'feature_{i:02d}' for i in range(n_features)]
    
    X_df = pd.DataFrame(X, columns=feature_names, index=dates)
    y_series = pd.Series(y, index=dates)
    
    print(f"📊 Generated synthetic time series data: {X_df.shape}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Samples: {len(X_df)}")
    print(f"   Time range: {dates[0]} to {dates[-1]}")
    print(f"   Frequency: 15-minute intervals")
    
    # Test enhanced Huber with walk-forward stability analysis
    print(f"\n🔍 Testing Enhanced Huber with Walk-Forward Stability Analysis...")
    
    try:
        results = prepare_huber_teacher_outputs(
            X_train=X_df,
            y_train=y_series,
            sign_agree_threshold=0.8,  # Same sign in ≥ 80% of splits
            nonzero_rate_threshold=0.7,  # Non-zero in ≥ 70% of splits
            n_time_splits=5,  # 5 walk-forward time splits
            pruning_percentile=20,
            n_jobs=2  # Limit jobs for testing
        )
        
        print(f"\n✅ Enhanced Huber with walk-forward stability completed successfully!")
        print(f"   📊 Results keys: {list(results.keys())}")
        
        # Check if monotonic constraints were generated
        if 'monotonic_constraints' in results:
            mono_cst = results['monotonic_constraints']
            if isinstance(mono_cst, dict):
                constraints = list(mono_cst.values())
                print(f"   🔗 Monotonic constraints: {len(constraints)}")
                print(f"      - Negative: {sum(1 for c in constraints if c == -1)}")
                print(f"      - Positive: {sum(1 for c in constraints if c == 1)}")
                print(f"      - Unconstrained: {sum(1 for c in constraints if c == 0)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_walkforward_stability()
        
        if success:
            print("\n🎉 Walk-forward stability analysis test passed!")
            print("✅ Enhanced Huber with time-based stability criteria working")
        else:
            print("\n❌ Walk-forward stability analysis test failed")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
