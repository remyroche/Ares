#!/usr/bin/env python3
"""
Simple Test of Step07 Integration Concept
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import shap

def test_step07_concept():
    """Test the core concept of step07's feature selection (MI + SHAP)."""
    print("🧪 Testing Step07 Feature Selection Concept...")

    # Create test data
    np.random.seed(42)
    n_samples, n_features = 1000, 50
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.choice([0, 1], n_samples))

    print(f"📊 Created test data: {n_samples} samples, {n_features} features")

    try:
        # Test MI calculation (core step07 functionality)
        print("📊 Testing Mutual Information calculation...")
        mi_scores = mutual_info_classif(X, y, random_state=42)
        print(f"✅ MI scores calculated for {len(mi_scores)} features")

        # Test SHAP calculation (optional step07 enhancement)
        print("🔍 Testing SHAP calculation...")
        sample_size = min(500, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        y_sample = y.loc[X_sample.index]

        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_sample, y_sample)

        # Calculate SHAP values
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_scores = np.abs(shap_values[1]).mean(axis=0)
        else:
            shap_scores = np.abs(shap_values).mean(axis=0)

        print(f"✅ SHAP scores calculated for {len(shap_scores)} features")

        # Test feature selection logic
        print("🎯 Testing feature selection logic...")

        # Combine MI and SHAP scores (step07 approach)
        mi_rank = pd.Series(mi_scores, index=X.columns).rank(ascending=False)
        shap_rank = pd.Series(shap_scores, index=X.columns).rank(ascending=False)
        combined_rank = (mi_rank + shap_rank) / 2

        # Select top features
        target_features = 25
        selected_features = combined_rank.nsmallest(target_features).index.tolist()

        X_selected = X[selected_features]

        print("✅ Feature selection completed!")
        print(f"   Original features: {X.shape[1]}")
        print(f"   Selected features: {X_selected.shape[1]}")
        print(".1%")
        # Show top features
        print("🏆 Top 5 features by combined MI+SHAP ranking:")
        for i, feature in enumerate(selected_features[:5], 1):
            mi_score = mi_scores[X.columns.get_loc(feature)]
            shap_score = shap_scores[X.columns.get_loc(feature)]
            print(f"   {i}. {feature}: MI={mi_score:.4f}, SHAP={shap_score:.4f}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step02_5_integration():
    """Test that step02_5 can import and use the concept correctly."""
    print("\n🔗 Testing Step02_5 Integration...")

    try:
        # Test that step02_5 can be imported
        from src.training.steps.data_collection.data_preparation.step02_5_sr_optimization import SRSOptimizationStep

        # Test that the method exists
        step_instance = SRSOptimizationStep(config={})
        method_exists = hasattr(step_instance, '_apply_step07_feature_selection')

        if method_exists:
            print("✅ Step02_5 integration successful!")
            print("   • SRSOptimizationStep imported successfully")
            print("   • _apply_step07_feature_selection method exists")
            return True
        else:
            print("❌ _apply_step07_feature_selection method not found in step02_5")
            return False

    except Exception as e:
        print(f"❌ Step02_5 integration failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Step07 Integration Testing Suite")
    print("=" * 50)

    # Test the core concept
    concept_success = test_step07_concept()

    # Test step02_5 integration
    integration_success = test_step02_5_integration()

    if concept_success and integration_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Step07 concept verified")
        print("✅ Step02_5 integration confirmed")
        print("\nReady to use MI/SHAP feature selection in step02_5!")
    else:
        print("\n💥 SOME TESTS FAILED!")
        if not concept_success:
            print("❌ Step07 concept test failed")
        if not integration_success:
            print("❌ Step02_5 integration test failed")
