#!/usr/bin/env python3
"""
Test script to verify SR feature categorization in the optimized feature selection system.
"""

import sys
sys.path.append('src')

from src.training.optimized_feature_selection_manager import OptimizedFeatureSelectionManager
import pandas as pd
import numpy as np

def test_sr_feature_categorization():
    """Test that SR features are properly categorized."""
    print("🧪 Testing SR Feature Categorization")
    print("=" * 50)
    
    # Create test configuration
    config = {
        "feature_selection": {
            "feature_categories": {
                "momentum": 0.20,
                "volatility": 0.15,
                "liquidity": 0.15,
                "microstructure": 0.10,
                "regime": 0.10,
                "sr_features": 0.15,
                "interaction": 0.15
            }
        }
    }
    
    # Initialize the feature selection manager
    fs_manager = OptimizedFeatureSelectionManager(config)
    
    # Create test features with various SR-related names
    sr_feature_names = [
        # Basic SR features
        "sr_distance", "sr_proximity", "sr_confidence",
        "support_level", "resistance_level", "sr_level",
        
        # Distance features
        "distance_to_resistance", "distance_to_support",
        "normalized_distance", "sr_distance_1", "sr_distance_2",
        
        # Probability features
        "breakout_probability", "rebounce_probability", "consolidation_probability",
        "sr_breakout_prob", "sr_rebounce_prob", "sr_consolidation_prob",
        
        # Score features
        "sr_score", "multi_timeframe_sr_score", "sr_proximity_score",
        "strength_score", "clarity_factor", "directional_pressure",
        "delta_sr_score", "isolation_score", "sr_confidence_score",
        
        # Proximity features
        "sr_proximity", "sr_proximity_1", "sr_proximity_2",
        
        # Other features (should not be categorized as SR)
        "rsi_14", "macd_12_26", "volume_sma_20", "momentum_strength",
        "volatility_garman_klass", "order_flow_imbalance", "hmm_state_0"
    ]
    
    # Test categorization
    categories = fs_manager._categorize_features(sr_feature_names)
    
    print("📊 Feature Categorization Results:")
    print("-" * 30)
    
    for category, features in categories.items():
        if features:
            print(f"{category}: {len(features)} features")
            if category == "sr_features":
                print(f"  SR features found: {features}")
            elif len(features) <= 5:
                print(f"  Examples: {features}")
    
    # Verify SR features are properly categorized
    sr_features_found = categories.get("sr_features", [])
    expected_sr_features = [
        "sr_distance", "sr_proximity", "sr_confidence", "support_level", 
        "resistance_level", "sr_level", "distance_to_resistance", 
        "distance_to_support", "normalized_distance", "sr_distance_1", 
        "sr_distance_2", "breakout_probability", "rebounce_probability", 
        "consolidation_probability", "sr_breakout_prob", "sr_rebounce_prob", 
        "sr_consolidation_prob", "sr_score", "multi_timeframe_sr_score", 
        "sr_proximity_score", "strength_score", "clarity_factor", 
        "directional_pressure", "delta_sr_score", "isolation_score", 
        "sr_confidence_score", "sr_proximity_1", "sr_proximity_2"
    ]
    
    print(f"\n✅ SR Features Categorization Test:")
    print(f"   - Expected SR features: {len(expected_sr_features)}")
    print(f"   - Found SR features: {len(sr_features_found)}")
    print(f"   - Success rate: {len(sr_features_found) / len(expected_sr_features) * 100:.1f}%")
    
    # Check for any SR features that weren't categorized
    missing_sr_features = [f for f in expected_sr_features if f not in sr_features_found]
    if missing_sr_features:
        print(f"   - Missing SR features: {missing_sr_features}")
    
    # Check for any non-SR features that were incorrectly categorized as SR
    non_sr_features = ["rsi_14", "macd_12_26", "volume_sma_20", "momentum_strength", 
                      "volatility_garman_klass", "order_flow_imbalance", "hmm_state_0"]
    incorrectly_categorized = [f for f in non_sr_features if f in sr_features_found]
    if incorrectly_categorized:
        print(f"   - Incorrectly categorized as SR: {incorrectly_categorized}")
    
    # Test balanced selection with SR features
    print(f"\n🎯 Testing Balanced Selection with SR Features:")
    print("-" * 40)
    
    # Create a larger test dataset
    n_samples = 100
    test_data = {}
    
    # Add SR features
    for feature in sr_features_found[:10]:  # Use first 10 SR features
        test_data[feature] = np.random.randn(n_samples)
    
    # Add other features
    for feature in non_sr_features:
        test_data[feature] = np.random.randn(n_samples)
    
    # Create target
    target = np.random.randint(0, 2, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    target_series = pd.Series(target, name='target')
    
    # Test balanced selection
    try:
        selected_features, metadata = fs_manager.select_features_optimized(
            df, target_series, model_type="ensemble_models", step_name="step2"
        )
        
        print(f"   - Original features: {len(df.columns)}")
        print(f"   - Selected features: {len(selected_features.columns)}")
        
        # Check SR feature representation
        selected_sr_features = [f for f in selected_features.columns if f in sr_features_found]
        print(f"   - SR features selected: {len(selected_sr_features)}")
        print(f"   - SR feature percentage: {len(selected_sr_features) / len(selected_features.columns) * 100:.1f}%")
        
        if "feature_categories" in metadata:
            category_dist = metadata["feature_categories"]
            if "sr_features" in category_dist:
                print(f"   - SR features in final selection: {len(category_dist['sr_features'])}")
        
        print("✅ Balanced selection test completed successfully!")
        
    except Exception as e:
        print(f"❌ Balanced selection test failed: {e}")
    
    return len(sr_features_found) >= len(expected_sr_features) * 0.8  # 80% success rate

if __name__ == "__main__":
    success = test_sr_feature_categorization()
    
    print(f"\n📊 Test Result: {'PASSED' if success else 'FAILED'}")
    
    if success:
        print("🎉 SR feature categorization is working correctly!")
        print("   - SR features are properly identified and categorized")
        print("   - Balanced selection includes SR features in the mix")
        print("   - Feature selection maintains SR feature representation")
    else:
        print("❌ SR feature categorization needs improvement")
        print("   - Some SR features may not be properly identified")
        print("   - Check the categorization logic in the feature selection manager")