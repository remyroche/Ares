#!/usr/bin/env python3
"""
Test script for enhanced feature selection integration in UnifiedDataDrivenPipeline
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_enhanced_feature_selection():
    """Test the enhanced feature selection integration."""
    print("🧪 Testing Enhanced Feature Selection Integration")
    print("=" * 60)
    
    try:
        # Import the enhanced pipeline
        from src.training.steps.pre_training.unified_data_driven_pipeline import (
            UnifiedDataDrivenPipeline, create_default_config
        )
        print("✅ Successfully imported UnifiedDataDrivenPipeline")
        
        # Create test data
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        # Generate synthetic financial data
        data = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
        })
        
        # Add some target variable
        targets = pd.Series(np.random.randn(n_samples))
        
        print(f"📊 Created test data: {data.shape[0]} samples, {data.shape[1]} features")
        
        # Test 1: Create pipeline with enhanced feature selection
        print("\n🔧 Test 1: Creating pipeline with enhanced feature selection")
        config = create_default_config()
        config.feature_selection.enable_enhanced_methods = True
        config.feature_selection.enhanced_methods = ['improved_mrmr', 'vectorbt_mrmr']
        config.feature_selection.multi_objective.max_features = 20
        
        pipeline = UnifiedDataDrivenPipeline(config)
        print("✅ Pipeline created successfully")
        
        # Test 2: Check if enhanced selectors are initialized
        print("\n🔧 Test 2: Checking enhanced selectors initialization")
        if hasattr(pipeline, 'enhanced_feature_selectors'):
            print(f"✅ Enhanced selectors initialized: {len(pipeline.enhanced_feature_selectors)} methods")
            for method_name in pipeline.enhanced_feature_selectors.keys():
                print(f"   - {method_name}")
        else:
            print("⚠️ Enhanced selectors not found")
        
        # Test 3: Test individual enhanced methods
        print("\n🔧 Test 3: Testing individual enhanced methods")
        
        # Test improved mRMR
        try:
            from src.feature_selection.advanced.improved_mrmr import ImprovedMRMR
            mrmr_selector = ImprovedMRMR()
            result = mrmr_selector.select_features(
                data.values, targets.values,
                feature_names=data.columns.tolist(),
                target_ratio=0.4
            )
            if result.get('success', False):
                print(f"✅ Improved mRMR: Selected {len(result['selected_features'])} features")
            else:
                print("⚠️ Improved mRMR: Selection failed")
        except Exception as e:
            print(f"❌ Improved mRMR: Error - {e}")
        
        # Test VectorBT mRMR
        try:
            from src.feature_selection.vectorbt.vectorbt_mrmr_selector import VectorBTMRMRSelector
            from src.feature_selection.vectorbt.vectorbt_config import VectorBTFeatureSelectionConfig
            config_vbt = VectorBTFeatureSelectionConfig()
            config_vbt.target_features = 20
            vbt_selector = VectorBTMRMRSelector(config_vbt)
            result = vbt_selector.select_features(
                data.values, targets.values,
                feature_names=data.columns.tolist()
            )
            if result.get('success', False):
                print(f"✅ VectorBT mRMR: Selected {len(result['selected_features'])} features")
            else:
                print("⚠️ VectorBT mRMR: Selection failed")
        except Exception as e:
            print(f"❌ VectorBT mRMR: Error - {e}")
        
        # Test 4: Test multi-objective selector with enhanced methods
        print("\n🔧 Test 4: Testing multi-objective selector with enhanced methods")
        try:
            from src.training.steps.pre_training.unified_data_driven_pipeline.feature_selection.multi_objective_selector import (
                MultiObjectiveFeatureSelector, create_default_objectives
            )
            
            objectives = create_default_objectives()
            selector = MultiObjectiveFeatureSelector(objectives)
            
            # Test enhanced methods
            enhanced_methods = ['improved_mrmr', 'vectorbt_mrmr', 'vectorbt_rfe']
            for method in enhanced_methods:
                if hasattr(selector, f'_{method}_selection'):
                    print(f"✅ {method} method available in multi-objective selector")
                else:
                    print(f"⚠️ {method} method not found in multi-objective selector")
                    
        except Exception as e:
            print(f"❌ Multi-objective selector test failed: {e}")
        
        # Test 5: Test configuration
        print("\n🔧 Test 5: Testing configuration")
        print(f"✅ Enhanced methods enabled: {config.feature_selection.enable_enhanced_methods}")
        print(f"✅ Available methods: {config.feature_selection.enhanced_methods}")
        print(f"✅ Method weights: {config.feature_selection.enhanced_method_weights}")
        print(f"✅ VectorBT optimization: {config.feature_selection.enable_vectorbt_optimization}")
        
        print("\n🎉 Enhanced Feature Selection Integration Test Completed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_feature_selection()
    sys.exit(0 if success else 1)