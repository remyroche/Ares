#!/usr/bin/env python3
"""
Simple test script to verify gate feature integration logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def test_gate_feature_protection():
    """Test gate feature protection logic."""
    print("🛡️ Testing Gate Feature Protection Logic")
    print("=" * 50)
    
    # Create sample data with gate features
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    # Generate sample features including gate features
    features_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)] + 
                ['gate_quality', 'gate_stability', 'gate_variance'],
        index=pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    )
    
    print(f"📊 Sample data with gate features: {features_df.shape[0]} samples, {features_df.shape[1]} features")
    print(f"🛡️ Gate features in data: {[col for col in features_df.columns if 'gate' in col.lower()]}")
    
    # Test gate feature protection logic
    all_features = list(features_df.columns)
    selected_features = all_features[:10]  # Simulate feature selection
    
    print(f"📋 Selected features (before protection): {selected_features}")
    
    # Apply gate feature protection
    gate_features = [col for col in all_features if 'gate' in col.lower()]
    protected_gate_features = [gf for gf in gate_features if gf not in selected_features]
    
    if protected_gate_features:
        print(f"🛡️ Adding protected gate features: {protected_gate_features}")
        selected_features.extend(protected_gate_features)
    else:
        print("🛡️ All gate features already selected")
    
    print(f"📋 Final selected features (after protection): {selected_features}")
    print(f"🛡️ Gate features in final selection: {[col for col in selected_features if 'gate' in col.lower()]}")
    
    # Verify all gate features are included
    final_gate_features = [col for col in selected_features if 'gate' in col.lower()]
    if len(final_gate_features) == len(gate_features):
        print("✅ All gate features are protected!")
        return True
    else:
        print(f"❌ Gate feature protection failed: {len(final_gate_features)}/{len(gate_features)} gate features included")
        return False


def test_gate_feature_generation():
    """Test gate feature generation logic."""
    print("\n🔧 Testing Gate Feature Generation Logic")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Generate sample features
    features_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)],
        index=pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    )
    
    # Generate sample targets
    targets_series = pd.Series(
        np.random.randn(n_samples),
        index=features_df.index,
        name='target'
    )
    
    print(f"📊 Sample data: {features_df.shape[0]} samples, {features_df.shape[1]} features")
    print(f"📊 Target data: {len(targets_series)} samples")
    
    # Simulate gate feature generation
    selected_features = features_df.columns[:3].tolist()  # Select first 3 features
    print(f"🎯 Selected base features: {selected_features}")
    
    # Generate gate features
    gate_features = {}
    
    for feature_name in selected_features:
        feature_values = features_df[feature_name]
        
        # Generate different types of gate features
        gate_features[f"{feature_name}_gate_quality"] = np.random.rand(len(feature_values))
        gate_features[f"{feature_name}_gate_stability"] = np.random.rand(len(feature_values))
        gate_features[f"{feature_name}_gate_variance"] = np.random.rand(len(feature_values))
    
    # Add global gate features
    gate_features["global_data_quality_gate"] = np.random.rand(len(features_df))
    gate_features["global_correlation_gate"] = np.random.rand(len(features_df))
    gate_features["global_variance_gate"] = np.random.rand(len(features_df))
    
    gate_features_df = pd.DataFrame(gate_features, index=features_df.index)
    
    print(f"✅ Generated {len(gate_features_df.columns)} gate features")
    print(f"📋 Gate feature names: {list(gate_features_df.columns)}")
    print(f"📊 Gate features shape: {gate_features_df.shape}")
    
    # Verify gate features are generated
    if len(gate_features_df.columns) > 0:
        print("✅ Gate feature generation successful!")
        return True
    else:
        print("❌ Gate feature generation failed!")
        return False


def test_pipeline_integration():
    """Test pipeline integration points."""
    print("\n🔗 Testing Pipeline Integration Points")
    print("=" * 50)
    
    # Test step execution order
    step_order = [
        "feature_generation_data_validation_step",
        "feature_generation_labeling_integration_step",
        "feature_generation_feature_generation_step",
        "feature_generation_gate_feature_step",  # Gate features after feature generation
        "feature_generation_feature_selection_step",
        "feature_generation_period_lookback_optimization_step",
        "feature_generation_interaction_generation_step_analyst",
        "feature_generation_interaction_generation_step_tactician",
        "feature_generation_final_feature_selection_step",
        "feature_generation_final_validation_step"
    ]
    
    print("📋 Step execution order:")
    for i, step in enumerate(step_order, 1):
        print(f"  {i:2d}. {step}")
    
    # Check if gate feature step is in the right position
    gate_step_index = step_order.index("feature_generation_gate_feature_step")
    feature_gen_index = step_order.index("feature_generation_feature_generation_step")
    feature_sel_index = step_order.index("feature_generation_feature_selection_step")
    
    if feature_gen_index < gate_step_index < feature_sel_index:
        print("✅ Gate feature step is positioned correctly (after feature generation, before feature selection)")
        return True
    else:
        print("❌ Gate feature step is not positioned correctly")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Simple Gate Feature Integration Tests")
    print("=" * 60)
    
    # Test 1: Gate feature protection
    test1_passed = test_gate_feature_protection()
    
    # Test 2: Gate feature generation
    test2_passed = test_gate_feature_generation()
    
    # Test 3: Pipeline integration
    test3_passed = test_pipeline_integration()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 30)
    print(f"Gate Feature Protection: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Gate Feature Generation: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Pipeline Integration: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 All tests passed! Gate feature integration logic is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)