#!/usr/bin/env python3
"""
Test script for the enhanced multi-stage feature selection pipeline.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.training.steps.pre_training.feature_selection.core.config import FeatureSelectionConfig
from src.training.steps.pre_training.feature_selection.core.pipeline import MultiStageFeatureSelector
from src.training.steps.pre_training.feature_selection.core.enhanced_pipeline import EnhancedMultiStageFeatureSelector

def create_test_data(n_samples=1000, n_features=120):
    """Create test data for feature selection."""
    np.random.seed(42)
    
    # Create feature matrix
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i:03d}" for i in range(n_features)]
    )
    
    # Create target variable with some features being important
    # Use first 20 features as important
    important_features = X.iloc[:, :20]
    y = (
        0.1 * important_features.sum(axis=1) + 
        0.05 * np.random.randn(n_samples)
    )
    
    return X, y

def test_enhanced_pipeline():
    """Test the enhanced pipeline."""
    print("🧪 Testing Enhanced Multi-Stage Feature Selection Pipeline")
    print("=" * 60)
    
    # Create test data
    print("📊 Creating test data...")
    X, y = create_test_data(n_samples=1000, n_features=120)
    print(f"   📊 Data shape: {X.shape}")
    print(f"   📊 Target shape: {y.shape}")
    
    # Test 1: Enhanced pipeline with new configuration
    print("\n🚀 Test 1: Enhanced Pipeline")
    print("-" * 40)
    
    config = FeatureSelectionConfig()
    config.enable_new_pipeline = True
    config.target_features = 60
    config.stage1_mrmr_weight = 0.7
    config.stage1_spearman_weight = 0.3
    config.stage1_target_ratio = 0.5
    
    selector = MultiStageFeatureSelector(config)
    
    try:
        result = selector.select_features(X, y)
        
        if result.success:
            print(f"✅ Enhanced pipeline completed successfully!")
            print(f"   📊 Selected features: {len(result.selected_features)}")
            print(f"   📊 Execution time: {result.execution_time:.2f}s")
            print(f"   📊 Reduction ratio: {len(result.selected_features)/len(X.columns):.1%}")
            
            # Show some selected features
            print(f"   📊 First 10 selected features:")
            for i, feature in enumerate(result.selected_features[:10], 1):
                print(f"      {i:2d}. {feature}")
        else:
            print(f"❌ Enhanced pipeline failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Enhanced pipeline test failed: {e}")
    
    # Test 2: Direct enhanced selector
    print("\n🚀 Test 2: Direct Enhanced Selector")
    print("-" * 40)
    
    try:
        enhanced_selector = EnhancedMultiStageFeatureSelector(config)
        result = enhanced_selector.select_features(X, y)
        
        if result.success:
            print(f"✅ Direct enhanced selector completed successfully!")
            print(f"   📊 Selected features: {len(result.selected_features)}")
            print(f"   📊 Execution time: {result.execution_time:.2f}s")
            print(f"   📊 Reduction ratio: {len(result.selected_features)/len(X.columns):.1%}")
            
            # Show stage results
            if 'stage_1' in result.stage_results:
                stage1 = result.stage_results['stage_1']
                print(f"   📊 Stage 1 (mRMR+Spearman): {stage1.get('target_count', 0)} features")
                print(f"   📊 Stage 1 method: {stage1.get('method', 'unknown')}")
            
            if 'stage_2' in result.stage_results:
                stage2 = result.stage_results['stage_2']
                print(f"   📊 Stage 2 (Progressive refinement): {len(stage2.get('refinement_steps', []))} steps")
                print(f"   📊 Stage 2 method: {stage2.get('method', 'unknown')}")
        else:
            print(f"❌ Direct enhanced selector failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Direct enhanced selector test failed: {e}")
    
    # Test 3: Original pipeline (fallback)
    print("\n🚀 Test 3: Original Pipeline (Fallback)")
    print("-" * 40)
    
    config_original = FeatureSelectionConfig()
    config_original.enable_new_pipeline = False  # Disable new pipeline
    
    selector_original = MultiStageFeatureSelector(config_original)
    
    try:
        result = selector_original.select_features(X, y)
        
        if result.success:
            print(f"✅ Original pipeline completed successfully!")
            print(f"   📊 Selected features: {len(result.selected_features)}")
            print(f"   📊 Execution time: {result.execution_time:.2f}s")
            print(f"   📊 Reduction ratio: {len(result.selected_features)/len(X.columns):.1%}")
        else:
            print(f"❌ Original pipeline failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Original pipeline test failed: {e}")
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    test_enhanced_pipeline()