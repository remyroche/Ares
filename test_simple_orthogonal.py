#!/usr/bin/env python3
"""
Simple test for specialist orthogonalization - just test the core functionality
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_simple():
    """Test basic orthogonalizer functionality"""
    try:
        from src.utils.ml_common.specialist_orthogonalizer import OptimizedSpecialistOrthogonalizer
        
        print("✅ Import successful")
        
        # Test initialization
        orthogonalizer = OptimizedSpecialistOrthogonalizer()
        print("✅ Initialization successful")
        
        # Test with simple data
        sample_data = pd.DataFrame({
            'risk_score': np.random.random(100),
            'liquidity_regime_1_prob': np.random.random(100),
            'momentum_persistence_5': np.random.random(100),
            'macro_trend_1': np.random.randn(100),
        })
        
        target_series = pd.Series(np.random.randint(0, 2, 100), index=sample_data.index)
        
        # Test feature extraction
        coverage = orthogonalizer.validate_specialist_coverage(sample_data)
        print("✅ Feature extraction successful")
        
        available_specialists = [s for s, has in coverage.items() if has]
        print(f"  Available specialists: {available_specialists}")
        
        # Test AUC calculation for one specialist
        risk_features = orthogonalizer.extract_specialist_features(sample_data, 'risk')
        if not risk_features.empty:
            auc = orthogonalizer.calculate_specialist_auc(risk_features, target_series)
            print(f"✅ AUC calculation successful: {auc:.4f}")
        
        print("✅ Basic functionality test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Simple Orthogonalization Test")
    print("=" * 40)
    
    if test_simple():
        print("\n🎉 SIMPLE TEST PASSED!")
        print("✅ Core orthogonalization functionality is working!")
        exit_code = 0
    else:
        print("\n❌ SIMPLE TEST FAILED!")
        exit_code = 1
    
    sys.exit(exit_code)
