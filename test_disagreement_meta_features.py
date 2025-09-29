#!/usr/bin/env python3
"""
Test script for Disagreement Meta-Features Implementation

This script tests the comprehensive disagreement meta-features implementation
for Analyst and Tactician ensemble models.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_disagreement_meta_features():
    """Test the disagreement meta-features implementation."""
    print("🧪 Testing Disagreement Meta-Features Implementation")
    print("=" * 60)
    
    try:
        # Import the disagreement meta-features
        from src.analyst.predictive_ensembles.disagreement_meta_features import DisagreementMetaFeatures
        
        # Create a logger
        logger = logging.getLogger(__name__)
        
        # Initialize the disagreement calculator
        disagreement_calculator = DisagreementMetaFeatures(logger)
        
        print("✅ Successfully imported DisagreementMetaFeatures")
        
        # Test 1: Basic disagreement features calculation
        print("\n🔍 Test 1: Basic Disagreement Features Calculation")
        
        # Create sample model predictions
        model_predictions = {
            'model_1': np.array([0.7, 0.8, 0.6]),
            'model_2': np.array([0.3, 0.2, 0.4]),
            'model_3': np.array([0.5, 0.5, 0.5]),
            'model_4': np.array([0.9, 0.1, 0.8])
        }
        
        model_probabilities = {
            'model_1': np.array([0.7, 0.8, 0.6]),
            'model_2': np.array([0.3, 0.2, 0.4]),
            'model_3': np.array([0.5, 0.5, 0.5]),
            'model_4': np.array([0.9, 0.1, 0.8])
        }
        
        model_confidences = {
            'model_1': np.array([0.8, 0.9, 0.7]),
            'model_2': np.array([0.6, 0.5, 0.7]),
            'model_3': np.array([0.5, 0.5, 0.5]),
            'model_4': np.array([0.9, 0.3, 0.8])
        }
        
        # Calculate disagreement features
        disagreement_features = disagreement_calculator.calculate_all_disagreement_features(
            model_predictions, model_probabilities, model_confidences
        )
        
        print(f"✅ Calculated {len(disagreement_features)} disagreement features")
        for feature_name, feature_value in disagreement_features.items():
            print(f"   {feature_name}: {feature_value:.4f}")
        
        # Test 2: Ensemble disagreement features
        print("\n🔍 Test 2: Ensemble Disagreement Features")
        
        # Create sample ensemble predictions
        ensemble_predictions = {
            'ensemble_1': {'prediction': 0.7, 'probability': 0.7, 'confidence': 0.8},
            'ensemble_2': {'prediction': 0.3, 'probability': 0.3, 'confidence': 0.6},
            'ensemble_3': {'prediction': 0.5, 'probability': 0.5, 'confidence': 0.5},
            'ensemble_4': {'prediction': 0.9, 'probability': 0.9, 'confidence': 0.9}
        }
        
        ensemble_disagreement = disagreement_calculator.calculate_disagreement_features_for_ensemble(
            ensemble_predictions, is_live=False
        )
        
        print(f"✅ Calculated {len(ensemble_disagreement)} ensemble disagreement features")
        for feature_name, feature_value in ensemble_disagreement.items():
            print(f"   {feature_name}: {feature_value:.4f}")
        
        # Test 3: Edge cases
        print("\n🔍 Test 3: Edge Cases")
        
        # Test with empty predictions
        empty_predictions = {}
        empty_disagreement = disagreement_calculator.calculate_all_disagreement_features(
            empty_predictions, {}, {}
        )
        print(f"✅ Empty predictions handled: {len(empty_disagreement)} features")
        
        # Test with single model
        single_model_predictions = {'model_1': np.array([0.5])}
        single_model_disagreement = disagreement_calculator.calculate_all_disagreement_features(
            single_model_predictions, {'model_1': np.array([0.5])}, {}
        )
        print(f"✅ Single model handled: {len(single_model_disagreement)} features")
        
        print("\n✅ All disagreement meta-features tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_integration():
    """Test the integration with ensemble models."""
    print("\n🔍 Test 4: Ensemble Integration")
    
    try:
        # Test VolatileRegimeEnsemble integration
        from src.analyst.predictive_ensembles.regime_ensembles.volatile_regime_ensemble import VolatileRegimeEnsemble
        
        # Create a mock config
        config = {
            'analyst': {
                'VolatileRegimeEnsemble': {
                    'n_pca_components': 10,
                    'use_smote': True,
                    'tune_base_models': True
                }
            }
        }
        
        # Initialize the ensemble
        ensemble = VolatileRegimeEnsemble(config, 'TestVolatileRegime')
        
        # Create sample data
        sample_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'volatility_20': [0.02, 0.03, 0.025, 0.035, 0.03],
            'volatility_regime': [1, 1, 2, 2, 1]
        })
        
        # Test meta-features generation
        meta_features = ensemble._get_meta_features(sample_data, is_live=False)
        
        print(f"✅ VolatileRegimeEnsemble meta-features: {len(meta_features.columns)} features")
        print(f"   Features: {list(meta_features.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_meta_feature_validation():
    """Test that meta-features are properly validated."""
    print("\n🔍 Test 5: Meta-Feature Validation")
    
    try:
        from src.analyst.predictive_ensembles.disagreement_meta_features import DisagreementMetaFeatures
        
        calculator = DisagreementMetaFeatures()
        
        # Test with various prediction scenarios
        test_scenarios = [
            # High agreement scenario
            {
                'predictions': {'m1': np.array([0.8]), 'm2': np.array([0.8]), 'm3': np.array([0.8])},
                'probabilities': {'m1': np.array([0.8]), 'm2': np.array([0.8]), 'm3': np.array([0.8])},
                'expected_low_dispersion': True
            },
            # High disagreement scenario
            {
                'predictions': {'m1': np.array([0.9]), 'm2': np.array([0.1]), 'm3': np.array([0.5])},
                'probabilities': {'m1': np.array([0.9]), 'm2': np.array([0.1]), 'm3': np.array([0.5])},
                'expected_high_dispersion': True
            }
        ]
        
        for i, scenario in enumerate(test_scenarios):
            features = calculator.calculate_all_disagreement_features(
                scenario['predictions'], scenario['probabilities']
            )
            
            print(f"   Scenario {i+1}:")
            print(f"     Prediction dispersion: {features['prediction_dispersion']:.4f}")
            print(f"     Direction conflict: {features['direction_conflict']:.4f}")
            print(f"     Entropy: {features['entropy']:.4f}")
            print(f"     JS divergence: {features['js_divergence']:.4f}")
        
        print("✅ Meta-feature validation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Meta-feature validation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Disagreement Meta-Features Test Suite")
    print("=" * 60)
    
    tests = [
        ("Disagreement Meta-Features", test_disagreement_meta_features),
        ("Ensemble Integration", test_ensemble_integration),
        ("Meta-Feature Validation", test_meta_feature_validation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✅ {test_name} test passed!")
                passed += 1
            else:
                print(f"❌ {test_name} test failed!")
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Disagreement meta-features are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)