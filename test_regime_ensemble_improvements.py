"""
Test script for regime ensemble improvements.

This script tests the new features:
1. Feature Contract Validation
2. Standardized Artifact Format
3. Meta-Features Generation
4. Circular Reference Handling
"""

import numpy as np
import sys
from typing import Dict, Any

# Add src to path
sys.path.insert(0, '/Users/remyroche/Documents/Ares')

from src.training.steps.market_analysis.components.regime_artifact_schema import (
    RegimeLabelsArtifact, FeatureContract, BaseModelContract,
    RegimeModelsArtifact, RegimeArtifactExtractor
)
from src.training.steps.market_analysis.components.ensemble_meta_features import (
    EnsembleMetaFeaturesGenerator
)
from src.utils.tprint import tprint

def test_feature_contract():
    """Test feature contract validation."""
    tprint("=" * 80, color="cyan")
    tprint("TEST 1: Feature Contract Validation", color="cyan", bold=True)
    tprint("=" * 80, color="cyan")
    
    # Create a feature contract
    contract = FeatureContract(
        feature_names=['feature1', 'feature2', 'feature3'],
        feature_count=3,
        feature_types={'feature1': 'base_prediction', 'feature2': 'uncertainty', 'feature3': 'confidence'},
        expected_shape=(None, 3)
    )
    
    tprint("✅ Created feature contract", color="green")
    tprint(f"   Features: {contract.feature_names}", color="blue")
    tprint(f"   Count: {contract.feature_count}", color="blue")
    
    # Test validation with correct features
    X_correct = np.random.rand(100, 3)
    try:
        contract.validate_features(X_correct, ['feature1', 'feature2', 'feature3'])
        tprint("✅ Validation passed for correct features", color="green")
    except ValueError as e:
        tprint(f"❌ Validation failed: {e}", color="red")
        return False
    
    # Test validation with incorrect feature count
    X_incorrect = np.random.rand(100, 5)
    try:
        contract.validate_features(X_incorrect)
        tprint("❌ Validation should have failed for incorrect feature count", color="red")
        return False
    except ValueError as e:
        tprint(f"✅ Validation correctly failed: {e}", color="green")
    
    tprint("✅ TEST 1 PASSED", color="green", bold=True)
    return True


def test_regime_labels_artifact():
    """Test regime labels artifact."""
    tprint("\n" + "=" * 80, color="cyan")
    tprint("TEST 2: Regime Labels Artifact", color="cyan", bold=True)
    tprint("=" * 80, color="cyan")
    
    # Create regime labels
    regime_labels = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    
    artifact = RegimeLabelsArtifact(
        cluster_assignments=regime_labels,
        n_regimes=3,
        regime_distribution={0: 4, 1: 3, 2: 3},
        clustering_method='gmm',
        clustering_params={'n_components': 3}
    )
    
    tprint("✅ Created regime labels artifact", color="green")
    tprint(f"   N regimes: {artifact.n_regimes}", color="blue")
    tprint(f"   Distribution: {artifact.regime_distribution}", color="blue")
    
    # Validate
    if artifact.validate():
        tprint("✅ Artifact validation passed", color="green")
    else:
        tprint("❌ Artifact validation failed", color="red")
        return False
    
    # Test serialization
    artifact_dict = artifact.to_dict()
    artifact_reconstructed = RegimeLabelsArtifact.from_dict(artifact_dict)
    
    if np.array_equal(artifact_reconstructed.cluster_assignments, regime_labels):
        tprint("✅ Serialization/deserialization passed", color="green")
    else:
        tprint("❌ Serialization/deserialization failed", color="red")
        return False
    
    tprint("✅ TEST 2 PASSED", color="green", bold=True)
    return True


def test_base_model_contract():
    """Test base model contract and ensemble detection."""
    tprint("\n" + "=" * 80, color="cyan")
    tprint("TEST 3: Base Model Contract & Circular Reference Handling", color="cyan", bold=True)
    tprint("=" * 80, color="cyan")
    
    # Create contracts for different model types
    base_model_contract = BaseModelContract(
        model_name='catboost_regime',
        model_type='classifier',
        output_type='probabilities',
        n_classes=5,
        feature_contract=FeatureContract(
            feature_names=['f1', 'f2'],
            feature_count=2,
            feature_types={'f1': 'base', 'f2': 'base'},
            expected_shape=(None, 2)
        )
    )
    
    ensemble_contract = BaseModelContract(
        model_name='stacker_lgbm_calibrated',
        model_type='ensemble',
        output_type='probabilities',
        n_classes=5,
        feature_contract=FeatureContract(
            feature_names=['meta1', 'meta2'],
            feature_count=2,
            feature_types={'meta1': 'meta', 'meta2': 'meta'},
            expected_shape=(None, 2)
        )
    )
    
    # Test ensemble detection
    if base_model_contract.is_base_model():
        tprint("✅ Correctly identified base model", color="green")
    else:
        tprint("❌ Failed to identify base model", color="red")
        return False
    
    if ensemble_contract.is_ensemble_model():
        tprint("✅ Correctly identified ensemble model", color="green")
    else:
        tprint("❌ Failed to identify ensemble model", color="red")
        return False
    
    # Test with name-based detection
    name_based_ensemble = BaseModelContract(
        model_name='voting_ensemble_model',
        model_type='classifier',  # Type says classifier
        output_type='probabilities',
        n_classes=5,
        feature_contract=base_model_contract.feature_contract
    )
    
    if name_based_ensemble.is_ensemble_model():
        tprint("✅ Correctly identified ensemble by name", color="green")
    else:
        tprint("❌ Failed to identify ensemble by name", color="red")
        return False
    
    tprint("✅ TEST 3 PASSED", color="green", bold=True)
    return True


def test_meta_features_generator():
    """Test meta-features generation."""
    tprint("\n" + "=" * 80, color="cyan")
    tprint("TEST 4: Meta-Features Generation", color="cyan", bold=True)
    tprint("=" * 80, color="cyan")
    
    # Create mock base models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 3, 100)
    
    # Train simple models
    rf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    lr = LogisticRegression(random_state=42, max_iter=100)
    
    rf.fit(X, y)
    lr.fit(X, y)
    
    base_models = {
        'random_forest': rf,
        'logistic_regression': lr
    }
    
    tprint("✅ Created and trained 2 base models", color="green")
    
    # Generate meta-features
    generator = EnsembleMetaFeaturesGenerator(component_name="TEST")
    
    meta_features, feature_names = generator.generate_meta_features(
        base_models=base_models,
        X=X,
        y=y,
        include_uncertainty=True,
        include_confidence=True,
        include_disagreement=True
    )
    
    tprint(f"✅ Generated meta-features: shape {meta_features.shape}", color="green")
    tprint(f"   Total features: {len(feature_names)}", color="blue")
    
    # Count feature types
    base_pred_count = sum(1 for name in feature_names if 'prob' in name and 'class' in name)
    uncertainty_count = sum(1 for name in feature_names if 'uncertainty' in name)
    confidence_count = sum(1 for name in feature_names if 'confidence' in name)
    disagreement_count = sum(1 for name in feature_names if 'disagreement' in name)
    
    tprint(f"   Base predictions: {base_pred_count}", color="blue")
    tprint(f"   Uncertainty features: {uncertainty_count}", color="blue")
    tprint(f"   Confidence features: {confidence_count}", color="blue")
    tprint(f"   Disagreement features: {disagreement_count}", color="blue")
    
    # Validate
    if meta_features.shape[1] == len(feature_names):
        tprint("✅ Feature count matches feature names", color="green")
    else:
        tprint(f"❌ Feature count mismatch: {meta_features.shape[1]} != {len(feature_names)}", color="red")
        return False
    
    if uncertainty_count > 0 and confidence_count > 0 and disagreement_count > 0:
        tprint("✅ All feature types generated", color="green")
    else:
        tprint("❌ Some feature types missing", color="red")
        return False
    
    tprint("✅ TEST 4 PASSED", color="green", bold=True)
    return True


def main():
    """Run all tests."""
    tprint("\n" + "=" * 80, color="magenta")
    tprint("REGIME ENSEMBLE IMPROVEMENTS - TEST SUITE", color="magenta", bold=True)
    tprint("=" * 80, color="magenta")
    tprint()
    
    results = []
    
    # Run tests
    results.append(("Feature Contract Validation", test_feature_contract()))
    results.append(("Regime Labels Artifact", test_regime_labels_artifact()))
    results.append(("Base Model Contract", test_base_model_contract()))
    results.append(("Meta-Features Generation", test_meta_features_generator()))
    
    # Summary
    tprint("\n" + "=" * 80, color="magenta")
    tprint("TEST SUMMARY", color="magenta", bold=True)
    tprint("=" * 80, color="magenta")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        color = "green" if result else "red"
        tprint(f"{test_name}: {status}", color=color)
    
    tprint()
    tprint(f"Total: {passed}/{total} tests passed", color="cyan", bold=True)
    
    if passed == total:
        tprint("✅ ALL TESTS PASSED!", color="green", bold=True)
        return 0
    else:
        tprint(f"❌ {total - passed} TEST(S) FAILED", color="red", bold=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

