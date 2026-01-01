#!/usr/bin/env python3
"""
Test script for specialist orthogonalization implementation
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_imports():
    """Test that all imports work correctly"""
    try:
        from src.utils.ml_common.specialist_orthogonalizer import OptimizedSpecialistOrthogonalizer, SPECIALIST_CATEGORIES
        print("✅ SpecialistOrthogonalizer import successful")
        print(f"✅ Found {len(SPECIALIST_CATEGORIES)} specialist categories")
        return True, OptimizedSpecialistOrthogonalizer
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False, None

def test_initialization(OrthogonalizerClass):
    """Test orthogonalizer initialization"""
    try:
        orthogonalizer = OrthogonalizerClass(anchor_specialist='xgb_macro')
        print("✅ Orthogonalizer initialization successful")
        
        # Check specialist categories
        categories = orthogonalizer.specialist_categories
        print(f"✅ Found {len(categories)} specialist categories")
        
        # Check anchor specialist
        print(f"✅ Anchor specialist: {orthogonalizer.anchor_specialist}")
        
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def test_feature_extraction(OrthogonalizerClass):
    """Test feature extraction with sample data"""
    try:
        orthogonalizer = OrthogonalizerClass()
        
        # Create sample data with features for all specialists
        sample_data = pd.DataFrame({
            # XGB Macro features
            'macro_trend_1': np.random.randn(100),
            'xgb_macro_signal': np.random.randn(100),
            'regime_macro_prob': np.random.random(100),
            
            # Risk features
            'risk_score': np.random.random(100),
            'risk_regime_0_prob': np.random.random(100),
            'risk_pred_1': np.random.random(100),
            
            # Liquidity features
            'liquidity_regime_1_prob': np.random.random(100),
            'liquidity_score': np.random.random(100),
            
            # Path features
            'path_trend_r2': np.random.random(100),
            'path_quality_score': np.random.random(100),
            
            # Other specialists
            'momentum_persistence_5': np.random.random(100),
            'vol_force_breakout': np.random.random(100),
            'candlestick_doji': np.random.random(100),
            'spectral_fft_1': np.random.random(100),
            'mr_probability': np.random.random(100),
            'volatility_burst_signal': np.random.random(100),
            'smc_predicted': np.random.random(100),
        })
        
        # Test feature extraction for each specialist
        coverage = orthogonalizer.validate_specialist_coverage(sample_data)
        
        print("✅ Feature extraction test:")
        for specialist, has_features in coverage.items():
            status = "✅" if has_features else "❌"
            print(f"  {status} {specialist}: {'has features' if has_features else 'no features'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_orthogonal_targets(OrthogonalizerClass):
    """Test orthogonal target generation"""
    try:
        orthogonalizer = OrthogonalizerClass()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'risk_score': np.random.random(100),
            'liquidity_regime_1_prob': np.random.random(100),
            'momentum_persistence_5': np.random.random(100),
            'macro_trend_1': np.random.randn(100),
        })
        
        # Create sample target
        target_series = pd.Series(np.random.randint(0, 2, 100), index=sample_data.index)
        
        # Generate orthogonal targets
        orthogonal_targets, auc_weights = orthogonalizer.generate_auc_weighted_orthogonal_targets(
            specialist_df=sample_data,
            target_series=target_series
        )
        
        print("✅ Orthogonal target generation test:")
        print(f"  Generated {len(orthogonal_targets.columns)} orthogonal targets")
        print(f"  AUC weights: {auc_weights}")
        
        return True
        
    except Exception as e:
        print(f"❌ Orthogonal target test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Specialist Orthogonalization Implementation")
    print("=" * 60)
    
    # Test imports first
    import_success, OrthogonalizerClass = test_imports()
    if not import_success:
        print("❌ Cannot proceed without successful imports")
        return 1
    
    tests = [
        ("Initialization", lambda: test_initialization(OrthogonalizerClass)),
        ("Feature Extraction", lambda: test_feature_extraction(OrthogonalizerClass)),
        ("Orthogonal Targets", lambda: test_orthogonal_targets(OrthogonalizerClass)),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Specialist orthogonalization implementation is ready!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
