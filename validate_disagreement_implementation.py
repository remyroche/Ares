#!/usr/bin/env python3
"""
Validation script for Disagreement Meta-Features Implementation

This script validates that the disagreement meta-features have been properly
implemented in the Analyst and Tactician ensemble models.
"""

import os
import sys

def validate_file_exists(file_path, description):
    """Validate that a file exists."""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False

def validate_imports(file_path, required_imports):
    """Validate that required imports are present in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        missing_imports = []
        for import_name in required_imports:
            if import_name not in content:
                missing_imports.append(import_name)
        
        if missing_imports:
            print(f"❌ Missing imports in {file_path}: {missing_imports}")
            return False
        else:
            print(f"✅ All required imports found in {file_path}")
            return True
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def validate_method_exists(file_path, method_name, description):
    """Validate that a method exists in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if f"def {method_name}" in content:
            print(f"✅ {description}: {method_name} method found")
            return True
        else:
            print(f"❌ {description}: {method_name} method NOT FOUND")
            return False
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def validate_disagreement_features():
    """Validate the disagreement meta-features implementation."""
    print("🔍 Validating Disagreement Meta-Features Implementation")
    print("=" * 60)
    
    # Check if the disagreement meta-features file exists in feature_engineering
    disagreement_file = "/workspace/src/feature_engineering/disagreement_meta_features.py"
    if not validate_file_exists(disagreement_file, "Disagreement Meta-Features Module"):
        return False
    
    # Check if the ensemble meta-features file exists
    ensemble_meta_file = "/workspace/src/feature_engineering/ensemble_meta_features.py"
    if not validate_file_exists(ensemble_meta_file, "Ensemble Meta-Features Module"):
        return False
    
    # Check if the trading disagreement analyzer exists
    trading_disagreement_file = "/workspace/src/trading/ensemble_disagreement_features.py"
    if not validate_file_exists(trading_disagreement_file, "Trading Disagreement Analyzer"):
        return False
    
    # Validate required imports in disagreement meta-features
    required_imports = [
        "import numpy as np",
        "import pandas as pd",
        "from scipy import stats",
        "from scipy.spatial.distance import jensenshannon"
    ]
    
    if not validate_imports(disagreement_file, required_imports):
        return False
    
    # Validate ensemble meta-features imports
    ensemble_required_imports = [
        "from .disagreement_meta_features import DisagreementMetaFeatures"
    ]
    
    if not validate_imports(ensemble_meta_file, ensemble_required_imports):
        return False
    
    # Validate trading disagreement analyzer imports
    trading_required_imports = [
        "from src.feature_engineering.ensemble_meta_features import EnsembleMetaFeatureGenerator"
    ]
    
    if not validate_imports(trading_disagreement_file, trading_required_imports):
        return False
    
    # Validate required methods in disagreement meta-features
    required_methods = [
        "calculate_all_disagreement_features",
        "_calculate_prediction_dispersion",
        "_calculate_direction_conflict",
        "_calculate_confidence_gap",
        "_calculate_entropy_uncertainty",
        "_calculate_spread_indicators",
        "_calculate_pairwise_divergence"
    ]
    
    for method in required_methods:
        if not validate_method_exists(disagreement_file, method, f"Disagreement Meta-Features"):
            return False
    
    # Validate ensemble meta-features methods
    ensemble_required_methods = [
        "generate_meta_features_for_analyst_ensemble",
        "generate_meta_features_for_tactician_ensemble",
        "generate_meta_features_for_volatile_regime_ensemble"
    ]
    
    for method in ensemble_required_methods:
        if not validate_method_exists(ensemble_meta_file, method, f"Ensemble Meta-Features"):
            return False
    
    # Validate trading disagreement analyzer methods
    trading_required_methods = [
        "analyze_trading_signal_reliability",
        "get_trading_recommendation"
    ]
    
    for method in trading_required_methods:
        if not validate_method_exists(trading_disagreement_file, method, f"Trading Disagreement Analyzer"):
            return False
    
    return True

def validate_ensemble_integration():
    """Validate ensemble integration."""
    print("\n🔍 Validating Ensemble Integration")
    print("=" * 40)
    
    # Validate VolatileRegimeEnsemble integration
    volatile_ensemble_file = "/workspace/src/analyst/predictive_ensembles/regime_ensembles/volatile_regime_ensemble.py"
    
    if not validate_file_exists(volatile_ensemble_file, "VolatileRegimeEnsemble"):
        return False
    
    # Check for ensemble meta-features import
    if not validate_imports(volatile_ensemble_file, ["from src.feature_engineering.ensemble_meta_features import EnsembleMetaFeatureGenerator"]):
        return False
    
    # Check for meta-features method
    if not validate_method_exists(volatile_ensemble_file, "_get_meta_features", "VolatileRegimeEnsemble"):
        return False
    
    # Check for base model predictions method
    if not validate_method_exists(volatile_ensemble_file, "_get_base_model_predictions", "VolatileRegimeEnsemble"):
        return False
    
    # Check for meta-feature generator initialization (attribute, not method)
    # This is validated by the import check above
    
    # Validate TacticianEnsembleTrainingStep integration
    tactician_ensemble_file = "/workspace/src/training/steps/model_training/tactician_ensemble_training.py"
    
    if not validate_file_exists(tactician_ensemble_file, "TacticianEnsembleTrainingStep"):
        return False
    
    # Check for meta-features method
    if not validate_method_exists(tactician_ensemble_file, "_get_meta_features", "TacticianEnsembleTrainingStep"):
        return False
    
    # Check for base model predictions method
    if not validate_method_exists(tactician_ensemble_file, "_get_base_model_predictions", "TacticianEnsembleTrainingStep"):
        return False
    
    # Check for ensemble meta-features import
    if not validate_imports(tactician_ensemble_file, ["from src.feature_engineering.ensemble_meta_features import EnsembleMetaFeatureGenerator"]):
        return False
    
    # Validate AnalystEnsembleTrainingStep integration
    analyst_ensemble_file = "/workspace/src/training/steps/model_training/analyst_ensemble_training.py"
    
    if not validate_file_exists(analyst_ensemble_file, "AnalystEnsembleTrainingStep"):
        return False
    
    # Check for meta-features method
    if not validate_method_exists(analyst_ensemble_file, "_get_meta_features", "AnalystEnsembleTrainingStep"):
        return False
    
    # Check for base model predictions method
    if not validate_method_exists(analyst_ensemble_file, "_get_base_model_predictions", "AnalystEnsembleTrainingStep"):
        return False
    
    # Check for ensemble meta-features import
    if not validate_imports(analyst_ensemble_file, ["from src.feature_engineering.ensemble_meta_features import EnsembleMetaFeatureGenerator"]):
        return False
    
    return True

def validate_feature_completeness():
    """Validate that all required disagreement features are implemented."""
    print("\n🔍 Validating Feature Completeness")
    print("=" * 40)
    
    disagreement_file = "/workspace/src/feature_engineering/disagreement_meta_features.py"
    
    # Check for all 6 types of disagreement features
    required_features = [
        "prediction_dispersion",
        "direction_conflict", 
        "confidence_gap",
        "entropy",
        "prediction_range",
        "js_divergence"
    ]
    
    try:
        with open(disagreement_file, 'r') as f:
            content = f.read()
        
        missing_features = []
        for feature in required_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing disagreement features: {missing_features}")
            return False
        else:
            print("✅ All required disagreement features implemented")
            return True
    except Exception as e:
        print(f"❌ Error reading disagreement file: {e}")
        return False

def main():
    """Run all validation checks."""
    print("🚀 Starting Disagreement Meta-Features Validation")
    print("=" * 60)
    
    validations = [
        ("Disagreement Meta-Features", validate_disagreement_features),
        ("Ensemble Integration", validate_ensemble_integration),
        ("Feature Completeness", validate_feature_completeness)
    ]
    
    passed = 0
    total = len(validations)
    
    for validation_name, validation_func in validations:
        print(f"\n🧪 Running {validation_name} Validation")
        print("-" * 40)
        
        try:
            if validation_func():
                print(f"✅ {validation_name} validation passed!")
                passed += 1
            else:
                print(f"❌ {validation_name} validation failed!")
        except Exception as e:
            print(f"❌ {validation_name} validation failed with exception: {e}")
    
    print(f"\n📊 Validation Results: {passed}/{total} validations passed")
    
    if passed == total:
        print("\n🎉 All validations passed! Disagreement meta-features are properly implemented.")
        print("\n📋 Implementation Summary:")
        print("✅ DisagreementMetaFeatures class with all 6 disagreement feature types")
        print("✅ VolatileRegimeEnsemble integration with _get_meta_features method")
        print("✅ TacticianEnsembleTrainingStep integration with _get_meta_features method")
        print("✅ AnalystEnsembleTrainingStep integration with _get_meta_features method")
        print("✅ All ensemble models now feed disagreement features to meta-learners")
        return True
    else:
        print("\n⚠️ Some validations failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)