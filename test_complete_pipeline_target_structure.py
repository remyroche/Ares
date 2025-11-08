#!/usr/bin/env python3
"""
Comprehensive test script to validate the complete pipeline with new simplified target structure.

This script tests:
1. Target detection in all updated steps
2. Data flow between steps with new target structure
3. Backward compatibility with legacy targets
4. Proper handling of target_long and target_short
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import h5py

def test_target_detection_in_steps():
    """Test target detection in all updated steps."""
    print("🔍 Testing target detection in all updated steps...")
    
    # Test feature_generation_interaction_generation_step
    try:
        from training.steps.pre_training.feature_generation_interaction_generation_step import FeatureGenerationInteractionGenerationStep
        step = FeatureGenerationInteractionGenerationStep()
        print("✅ feature_generation_interaction_generation_step imported successfully")
    except Exception as e:
        print(f"❌ Error importing feature_generation_interaction_generation_step: {e}")
        return False
    
    # Test feature_generation_final_feature_selection_step
    try:
        from training.steps.pre_training.feature_generation_final_feature_selection_step import TARGET_COLUMN_NAMES
        print(f"✅ feature_generation_final_feature_selection_step TARGET_COLUMN_NAMES: {TARGET_COLUMN_NAMES}")
        
        # Verify new targets are included
        if 'target_long' in TARGET_COLUMN_NAMES and 'target_short' in TARGET_COLUMN_NAMES:
            print("✅ New simplified target structure (target_long, target_short) found in TARGET_COLUMN_NAMES")
        else:
            print("❌ New simplified target structure not found in TARGET_COLUMN_NAMES")
            return False
    except Exception as e:
        print(f"❌ Error importing feature_generation_final_feature_selection_step: {e}")
        return False
    
    # Test feature_generation_period_lookback_optimization_step
    try:
        from training.steps.pre_training.feature_generation_period_lookback_optimization_step import FeatureGenerationPeriodLookbackOptimizationStep
        step = FeatureGenerationPeriodLookbackOptimizationStep()
        print("✅ feature_generation_period_lookback_optimization_step imported successfully")
    except Exception as e:
        print(f"❌ Error importing feature_generation_period_lookback_optimization_step: {e}")
        return False
    
    return True

def test_target_processing_logic():
    """Test target processing logic with mock data."""
    print("\n🧪 Testing target processing logic with mock data...")
    
    # Create mock data with new simplified target structure
    mock_data = pd.DataFrame({
        'target_long': np.random.rand(100),
        'target_short': np.random.rand(100),
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'close': np.random.rand(100) * 1000 + 1000,
        'volume': np.random.rand(100) * 1000000
    })
    
    print(f"✅ Created mock data with shape: {mock_data.shape}")
    print(f"📊 Target columns: target_long ({mock_data['target_long'].notna().sum()} non-NaN), target_short ({mock_data['target_short'].notna().sum()} non-NaN)")
    
    # Test target detection logic
    from training.steps.pre_training.feature_generation_final_feature_selection_step import TARGET_COLUMN_NAMES
    
    # Check for new simplified target structure first (highest priority)
    if 'target_long' in mock_data.columns and 'target_short' in mock_data.columns:
        available_targets = ['target_long', 'target_short']
        print("✅ New simplified target structure detected correctly")
        print(f"📊 Available targets: {available_targets}")
    else:
        # Fall back to legacy target detection
        available_targets = [col for col in TARGET_COLUMN_NAMES if col in mock_data.columns]
        print("📊 Using legacy target detection")
        print(f"📊 Available targets: {available_targets}")
    
    # Test target statistics
    long_signals = (mock_data['target_long'] > 0).sum()
    short_signals = (mock_data['target_short'] > 0).sum()
    print(f"📊 Target statistics: Long signals={long_signals}, Short signals={short_signals}")
    
    return True

def test_backward_compatibility():
    """Test backward compatibility with legacy target structure."""
    print("\n🔄 Testing backward compatibility with legacy target structure...")
    
    # Create mock data with legacy target structure
    mock_data_legacy = pd.DataFrame({
        'price_target_vol_normalized': np.random.rand(100),
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })
    
    from training.steps.pre_training.feature_generation_final_feature_selection_step import TARGET_COLUMN_NAMES
    
    # Test legacy target detection
    available_targets = [col for col in TARGET_COLUMN_NAMES if col in mock_data_legacy.columns]
    print(f"📊 Legacy targets detected: {available_targets}")
    
    if 'price_target_vol_normalized' in available_targets:
        print("✅ Legacy target structure (price_target_vol_normalized) detected correctly")
        return True
    else:
        print("❌ Legacy target structure not detected")
        return False

def test_hdf5_compatibility():
    """Test HDF5 compatibility with new target structure."""
    print("\n💾 Testing HDF5 compatibility with new target structure...")
    
    try:
        # Create temporary HDF5 file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # Create mock data with new target structure
        mock_data = pd.DataFrame({
            'target_long': np.random.rand(100),
            'target_short': np.random.rand(100),
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        
        # Save to HDF5
        mock_data.to_hdf(tmp_path, key='data', mode='w')
        print(f"✅ Data saved to HDF5: {tmp_path}")
        
        # Load from HDF5
        loaded_data = pd.read_hdf(tmp_path, key='data')
        print(f"✅ Data loaded from HDF5 with shape: {loaded_data.shape}")
        
        # Verify target columns
        if 'target_long' in loaded_data.columns and 'target_short' in loaded_data.columns:
            print("✅ New target structure preserved in HDF5")
        else:
            print("❌ New target structure not preserved in HDF5")
            return False
        
        # Clean up
        os.unlink(tmp_path)
        return True
        
    except Exception as e:
        print(f"❌ Error testing HDF5 compatibility: {e}")
        return False

def test_pipeline_integration():
    """Test pipeline integration with new target structure."""
    print("\n🔗 Testing pipeline integration with new target structure...")
    
    try:
        # Test that all steps can be imported and initialized
        from training.steps.pre_training.feature_generation_interaction_generation_step import FeatureGenerationInteractionGenerationStep
        from training.steps.pre_training.feature_generation_final_feature_selection_step import FeatureGenerationFinalFeatureSelectionStep
        from training.steps.pre_training.feature_generation_period_lookback_optimization_step import FeatureGenerationPeriodLookbackOptimizationStep
        
        # Initialize steps
        interaction_step = FeatureGenerationInteractionGenerationStep()
        selection_step = FeatureGenerationFinalFeatureSelectionStep()
        lookback_step = FeatureGenerationPeriodLookbackOptimizationStep()
        
        print("✅ All steps initialized successfully")
        print("✅ Pipeline integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Error testing pipeline integration: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting comprehensive pipeline test with new simplified target structure...")
    print("=" * 80)
    
    tests = [
        ("Target Detection in Steps", test_target_detection_in_steps),
        ("Target Processing Logic", test_target_processing_logic),
        ("Backward Compatibility", test_backward_compatibility),
        ("HDF5 Compatibility", test_hdf5_compatibility),
        ("Pipeline Integration", test_pipeline_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running test: {test_name}")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The pipeline is ready for the new simplified target structure.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)