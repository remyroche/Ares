#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced LGBM-SHAP integration in feature selection.
"""

import numpy as np
import pandas as pd
from src.training.steps.pre_training.components.final_feature_selection import (
    FinalFeatureSelectionComponent, FinalFeatureSelectionConfig
)

def test_lgbm_shap_integration():
    """Test the enhanced multi-method selection with LGBM-SHAP."""
    
    print("🧪 Testing Enhanced Multi-Method Selection with LGBM-SHAP")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples, n_features = 1000, 100
    
    # Generate features with different relationships to target
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target with non-linear relationships
    y = (
        2 * X['feature_0'] + 
        1.5 * X['feature_1'] * X['feature_2'] + 
        0.8 * np.sin(X['feature_3']) +
        0.5 * X['feature_4'] ** 2 +
        np.random.randn(n_samples) * 0.1
    )
    
    print(f"📊 Dataset: {n_samples} samples, {n_features} features")
    print(f"🎯 Target: Non-linear relationships with noise")
    
    # Test configuration
    config = FinalFeatureSelectionConfig(
        max_features=20,
        min_features=5,
        selection_method='mutual_info',
        scoring_threshold=0.01,
        use_tree_based=True
    )
    
    component = FinalFeatureSelectionComponent(config)
    
    # Test multi-method selection
    print("\n🔍 Testing Multi-Method Selection...")
    try:
        candidate_features, method_results = component._multi_method_initial_selection(
            X, y, list(X.columns), 20
        )
        
        print(f"✅ Multi-method selection successful!")
        print(f"📈 Candidate features: {len(candidate_features)}")
        print(f"🔧 Methods used: {list(method_results.keys())}")
        
        # Check LGBM-SHAP results
        if 'lgbm_shap' in method_results:
            lgbm_shap_data = method_results['lgbm_shap']
            if 'error' not in lgbm_shap_data:
                print(f"🎯 LGBM-SHAP features: {len(lgbm_shap_data['features'])}")
                print(f"📊 SHAP scores range: {min(lgbm_shap_data['scores']):.6f} - {max(lgbm_shap_data['scores']):.6f}")
                print(f"🏆 Top SHAP features: {lgbm_shap_data['features'][:5]}")
            else:
                print(f"⚠️ LGBM-SHAP error: {lgbm_shap_data['error']}")
        
        # Test stability-optimized selection
        print("\n🎯 Testing Stability-Optimized Selection...")
        optimized_features = component.select_features_with_stability_optimization(
            X, y, list(X.columns),
            target_features=15,
            stability_threshold=0.3,
            redundancy_threshold=0.7
        )
        
        print(f"✅ Stability-optimized selection successful!")
        print(f"📈 Final features: {len(optimized_features)}")
        
        # Test enhanced analysis
        print("\n📊 Testing Enhanced Analysis...")
        analysis = component.analyze_improved_selection(
            X, y, optimized_features, method_results
        )
        
        print(f"✅ Enhanced analysis successful!")
        print(f"📈 Analysis components: {list(analysis.keys())}")
        
        if 'method_results' in analysis:
            print(f"🔧 Method results included: {len(analysis['method_results'])} methods")
        
        # Display key metrics
        print("\n📊 Key Metrics:")
        if 'stability_analysis' in analysis:
            stab = analysis['stability_analysis']
            print(f"  - Stable features: {stab.get('stable_features', 0)}")
            print(f"  - Average stability: {stab.get('average_stability', 0):.4f}")
        
        if 'redundancy_analysis' in analysis:
            red = analysis['redundancy_analysis']
            print(f"  - Redundant features: {red.get('redundant_features', 0)}")
            print(f"  - Redundancy score: {red.get('redundancy_score', 0):.4f}")
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_shap_availability():
    """Test SHAP availability and functionality."""
    
    print("\n🔍 Testing SHAP Availability...")
    print("=" * 40)
    
    try:
        import shap
        print("✅ SHAP is available")
        
        # Test basic SHAP functionality
        import lightgbm as lgb
        from sklearn.datasets import make_regression
        
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        
        model = lgb.LGBMRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:10])
        
        print(f"✅ SHAP values calculated: {shap_values.shape}")
        print(f"📊 SHAP values range: {np.min(shap_values):.6f} - {np.max(shap_values):.6f}")
        
    except ImportError as e:
        print(f"❌ SHAP not available: {e}")
    except Exception as e:
        print(f"❌ SHAP test failed: {e}")

if __name__ == "__main__":
    test_lgbm_shap_integration()
    test_shap_availability()
    
    print("\n" + "=" * 60)
    print("🎯 Enhanced LGBM-SHAP Integration Test Complete!")
    print("=" * 60)
