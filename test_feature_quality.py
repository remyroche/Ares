#!/usr/bin/env python3
"""
Test script for feature quality checks and transformations.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.steps.labeling.train_specialists_with_gmm_step import TrainSpecialistsWithGMMStep

def test_feature_quality_checks():
    """Test the feature quality checking functionality."""
    print("🧪 Testing feature quality checks...")
    
    # Create a test step instance
    step = TrainSpecialistsWithGMMStep()
    
    # Create test data with various feature types
    data = {
        'constant_feature': [1.0, 1.0, 1.0, 1.0, 1.0],
        'low_variance_feature': [1.0, 1.0001, 0.9999, 1.0002, 0.9998],
        'normal_feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'normal_feature_2': [10.0, 20.0, 15.0, 25.0, 30.0],
        'zero_feature': [0.0, 0.0, 0.0, 0.0, 0.0]
    }
    
    test_features = pd.DataFrame(data)
    print(f"📊 Created test features with shape: {test_features.shape}")
    print(f"   Columns: {list(test_features.columns)}")
    
    # Test feature quality checking
    cleaned_features, quality_report = step._check_feature_quality(test_features)
    
    print(f"\n🔍 Feature Quality Report:")
    print(f"   - Total features: {quality_report['total_features']}")
    print(f"   - Constant features removed: {len(quality_report['constant_features'])}")
    print(f"   - Low variance features: {len(quality_report['low_variance_features'])}")
    print(f"   - Features kept: {quality_report['final_feature_count']}")
    print(f"   - Removed features: {quality_report['removed_features']}")
    
    if quality_report['constant_features']:
        print(f"   - Constant features: {quality_report['constant_features']}")
    
    if quality_report['low_variance_features']:
        print(f"   - Low variance features:")
        for item in quality_report['low_variance_features']:
            print(f"     * {item['feature']}: variance={item['variance']:.8f}")
    
    print(f"\n✅ Cleaned features shape: {cleaned_features.shape}")
    print(f"   Remaining columns: {list(cleaned_features.columns)}")
    
    return True

def test_normalization():
    """Test the feature normalization functionality."""
    print("\n🧪 Testing feature normalization...")
    
    # Create a test step instance
    step = TrainSpecialistsWithGMMStep()
    
    # Create test data
    data = {
        'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature_2': [10.0, 20.0, 15.0, 25.0, 30.0],
        'feature_3': [-5.0, 0.0, 5.0, 10.0, 15.0]
    }
    
    test_features = pd.DataFrame(data)
    print(f"📊 Original features:\n{test_features}")
    
    # Test normalization
    normalized_features = step._normalize_features(test_features)
    print(f"\n✅ Normalized features:\n{normalized_features}")
    
    # Verify normalization (values should be in [0, 1] range)
    for col in normalized_features.columns:
        min_val = normalized_features[col].min()
        max_val = normalized_features[col].max()
        print(f"   - {col}: min={min_val:.4f}, max={max_val:.4f}")
        
        if min_val < 0 or max_val > 1:
            print(f"⚠️  Warning: {col} not properly normalized")
        else:
            print(f"   ✓ {col} properly normalized to [0, 1] range")
    
    return True

def test_standardization():
    """Test the feature standardization functionality."""
    print("\n🧪 Testing feature standardization...")
    
    # Create a test step instance
    step = TrainSpecialistsWithGMMStep()
    
    # Create test data
    data = {
        'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature_2': [10.0, 20.0, 15.0, 25.0, 30.0],
        'feature_3': [-5.0, 0.0, 5.0, 10.0, 15.0]
    }
    
    test_features = pd.DataFrame(data)
    print(f"📊 Original features:\n{test_features}")
    
    # Test standardization
    standardized_features = step._standardize_features(test_features)
    print(f"\n✅ Standardized features:\n{standardized_features}")
    
    # Verify standardization (mean should be ~0, std should be ~1)
    for col in standardized_features.columns:
        mean_val = standardized_features[col].mean()
        std_val = standardized_features[col].std()
        print(f"   - {col}: mean={mean_val:.4f}, std={std_val:.4f}")
        
        if abs(mean_val) < 0.01 and abs(std_val - 1.0) < 0.01:
            print(f"   ✓ {col} properly standardized")
        else:
            print(f"⚠️  Warning: {col} not properly standardized")
    
    return True

def test_gmm_feature_listing():
    """Test the GMM feature listing functionality."""
    print("\n🧪 Testing GMM feature listing...")
    
    # Create a test step instance
    step = TrainSpecialistsWithGMMStep()
    
    # Create test GMM features
    data = {
        'gmm_feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'gmm_feature_2': [0.1, 0.2, 0.15, 0.25, 0.3],
        'gmm_feature_3': [10.0, 20.0, 15.0, 25.0, 30.0],
        'constant_gmm_feature': [1.0, 1.0, 1.0, 1.0, 1.0]
    }
    
    test_features = pd.DataFrame(data)
    
    # Test GMM feature listing
    feature_info = step._list_gmm_features(test_features, "test_pipeline")
    
    print(f"\n📊 GMM Feature Information:")
    print(f"   - Source: {feature_info['source']}")
    print(f"   - Total features: {feature_info['total_features']}")
    print(f"   - Constant features: {feature_info['statistics']['constant_features']}")
    print(f"   - Low variance features: {feature_info['statistics']['low_variance_features']}")
    print(f"   - Mean variance: {feature_info['statistics']['mean_variance']:.6f}")
    print(f"   - Min variance: {feature_info['statistics']['min_variance']:.6f}")
    print(f"   - Max variance: {feature_info['statistics']['max_variance']:.6f}")
    
    print(f"\n🔍 Feature Details:")
    for feature in feature_info['feature_details']:
        print(f"   - {feature['feature_name']}:")
        print(f"     * mean: {feature['mean']:.4f}")
        print(f"     * variance: {feature['variance']:.6f}")
        print(f"     * std: {feature['std_deviation']:.4f}")
        print(f"     * is_constant: {feature['is_constant']}")
        print(f"     * is_low_variance: {feature['is_low_variance']}")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting feature quality and transformation tests...\n")
    
    try:
        # Run all tests
        test_feature_quality_checks()
        test_normalization()
        test_standardization()
        test_gmm_feature_listing()
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)