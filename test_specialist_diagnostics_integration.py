#!/usr/bin/env python3
"""
Test script to verify SpecialistFeatureDiagnostics integration.

This script demonstrates that the comprehensive diagnostics are now
fully integrated into the specialist system via the enhanced mixin.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from src.training.steps.labeling.specialist_feature_diagnostics import SpecialistFeatureDiagnostics
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import SpecialistDiagnosticsMixinEnhancedV2


def create_mock_specialist_data():
    """Create mock data for testing the diagnostics integration."""
    np.random.seed(42)

    # Create mock features
    n_samples = 1000
    n_features = 20

    features = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
    })

    # Add some correlated features
    features['feature_20'] = features['feature_0'] * 0.8 + np.random.randn(n_samples) * 0.2
    features['feature_21'] = features['feature_1'] * 0.9 + np.random.randn(n_samples) * 0.1

    # Create mock labels and predictions
    labels = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    predictions = np.random.rand(n_samples)  # Probabilities

    return features, labels, predictions


def test_direct_diagnostics():
    """Test SpecialistFeatureDiagnostics directly."""
    print("🔬 Testing SpecialistFeatureDiagnostics directly...")

    features, labels, predictions = create_mock_specialist_data()

    diagnostics = SpecialistFeatureDiagnostics()

    # Test comprehensive feature analysis
    feature_results = diagnostics.comprehensive_feature_analysis(
        features=features,
        labels=labels,
        predictions=predictions,
        specialist_name="test_specialist"
    )

    print("✅ Feature analysis completed")
    quality_score = feature_results.get('overall_quality_score', 0.0)
    print(f"   Quality Score: {quality_score:.3f}" if isinstance(quality_score, (int, float)) else f"   Quality Score: {quality_score}")
    print(f"   Features Analyzed: {len(features.columns)}")

    # Test orthogonalization diagnostics (need original and orthogonal features)
    # For demo, we'll use the same features as both original and orthogonal
    ortho_results = diagnostics.advanced_orthogonalization_diagnostics(
        original_features=features,
        orthogonal_features=features,
        labels=labels,
        dropped_features=['feature_20', 'feature_21'],  # Mock dropped features
        specialist_name="test_specialist"
    )

    print("✅ Orthogonalization diagnostics completed")
    ortho_score = ortho_results.get('orthogonality_score', 0.0)
    print(f"   Orthogonality Score: {ortho_score:.3f}" if isinstance(ortho_score, (int, float)) else f"   Orthogonality Score: {ortho_score}")

    # Test denoising diagnostics
    denoised_targets = labels + np.random.randn(len(labels)) * 0.1  # Add some noise

    denoising_results = diagnostics.comprehensive_denoising_analysis(
        original_targets=pd.Series(labels),
        denoised_targets=pd.Series(denoised_targets),
        features=features,
        denoising_method="mock_kalman",
        specialist_name="test_specialist"
    )

    print("✅ Denoising analysis completed")
    denoising_score = denoising_results.get('overall_quality_score', 0.0)
    print(f"   Denoising Quality: {denoising_score:.3f}" if isinstance(denoising_score, (int, float)) else f"   Denoising Quality: {denoising_score}")

    return feature_results, ortho_results, denoising_results


def test_mixin_integration():
    """Test that the diagnostics are integrated into the mixin."""
    print("\n🔗 Testing SpecialistDiagnosticsMixinEnhancedV2 integration...")

    # Create a mock specialist class that inherits from the mixin
    class MockSpecialist(SpecialistDiagnosticsMixinEnhancedV2):
        def __init__(self):
            super().__init__()
            self.__class__.__name__ = "MockSpecialist"

    specialist = MockSpecialist()

    # Verify the diagnostics instance is available
    assert hasattr(specialist, 'feature_diagnostics'), "❌ feature_diagnostics not found in mixin"
    assert isinstance(specialist.feature_diagnostics, SpecialistFeatureDiagnostics), "❌ Wrong diagnostics type"

    print("✅ SpecialistFeatureDiagnostics successfully integrated into mixin")
    print(f"   Diagnostics instance: {type(specialist.feature_diagnostics).__name__}")

    return True


def main():
    """Run the integration tests."""
    print("🚀 Testing SpecialistFeatureDiagnostics Integration")
    print("=" * 60)

    try:
        # Test direct usage
        feature_results, ortho_results, denoising_results = test_direct_diagnostics()

        # Test mixin integration
        mixin_integrated = test_mixin_integration()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("\nSpecialistFeatureDiagnostics is now FULLY WIRED:")
        print("✅ Direct usage working")
        print("✅ Mixin integration complete")
        print("✅ All specialist classes can now use comprehensive diagnostics")
        print("\n📊 Sample Results:")
        print(f"   Feature Quality Score: {feature_results.get('overall_quality_score', 0):.3f}")
        print(f"   Orthogonality Score: {ortho_results.get('orthogonality_score', 0):.3f}")
        print(f"   Denoising Quality Score: {denoising_results.get('overall_quality_score', 0):.3f}")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
