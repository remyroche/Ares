#!/usr/bin/env python3
"""
Basic test for specialist orthogonalization core functionality
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_basic_functionality():
    """Test basic orthogonalizer functionality"""
    try:
        from src.utils.ml_common.specialist_orthogonalizer import OptimizedSpecialistOrthogonalizer
        
        print("✅ Import successful")
        
        # Test initialization
        orthogonalizer = OptimizedSpecialistOrthogonalizer()
        print("✅ Initialization successful")
        
        # Test with sample data that has features for multiple specialists
        sample_data = pd.DataFrame({
            'risk_score': np.random.random(100),
            'risk_regime_0_prob': np.random.random(100),
            'liquidity_regime_1_prob': np.random.random(100),
            'liquidity_score': np.random.random(100),
            'momentum_persistence_5': np.random.random(100),
            'macro_trend_1': np.random.randn(100),
        })
        
        target_series = pd.Series(np.random.randint(0, 2, 100), index=sample_data.index)
        
        # Test orthogonal target generation
        orthogonal_targets, auc_weights = orthogonalizer.generate_auc_weighted_orthogonal_targets(
            specialist_df=sample_data,
            target_series=target_series
        )
        
        print("✅ Orthogonal target generation successful")
        print(f"  Generated {len(orthogonal_targets.columns)} orthogonal targets")
        print(f"  AUC weights: {auc_weights}")
        
        # Test feature extraction
        coverage = orthogonalizer.validate_specialist_coverage(sample_data)
        print("✅ Feature extraction successful")
        
        available_specialists = [s for s, has in coverage.items() if has]
        print(f"  Available specialists: {available_specialists}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Basic Orthogonalization Test")
    print("=" * 40)
    
    if test_basic_functionality():
        print("\n🎉 BASIC TEST PASSED!")
        print("✅ Core orthogonalization functionality is working!")
        exit_code = 0
    else:
        print("\n❌ BASIC TEST FAILED!")
        exit_code = 1
    
    sys.exit(exit_code)
