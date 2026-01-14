#!/usr/bin/env python3
"""
Test script for enhanced Huber stability analysis with sign consensus and nonzero rate
"""

import sys
import os
sys.path.append('/Users/remyroche/Documents/Ares')

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

# Import enhanced Huber function
from src.utils.huber_regressor_for_trees import prepare_huber_teacher_outputs

def test_stability_analysis():
    """Test the enhanced stability analysis with sign consensus and nonzero rate"""
    print("🧪 Testing Enhanced Huber Stability Analysis")
    print("=" * 60)
    
    # Generate synthetic regression data
    np.random.seed(42)
    X, y = make_regression(
        n_samples=500,
        n_features=30,
        n_informative=20,
        noise=0.1,
        random_state=42
    )
    
    # Convert to DataFrame with feature names
    feature_names = [f'feature_{i:02d}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    
    print(f"📊 Generated synthetic data: {X_df.shape}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Samples: {len(X_df)}")
    
    # Test enhanced Huber with stability analysis
    print("\n🔍 Testing Enhanced Huber with Stability Analysis...")
    
    try:
        results = prepare_huber_teacher_outputs(
            X_train=X_df,
            y_train=y_series,
            sign_agree_threshold=0.8,  # Same sign in ≥ 80% of splits
            nonzero_rate_threshold=0.7,  # Non-zero in ≥ 70% of splits
            pruning_percentile=20,
            n_jobs=2  # Limit jobs for testing
        )
        
        print(f"\n✅ Enhanced Huber completed successfully!")
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
        success = test_stability_analysis()
        
        if success:
            print("\n🎉 Stability analysis test passed!")
            print("✅ Enhanced Huber with stability criteria working")
        else:
            print("\n❌ Stability analysis test failed")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
