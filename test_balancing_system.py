#!/usr/bin/env python3
"""
Simple test to verify the balancing system structure and imports.
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that our balancing system modules can be imported."""
    print("Testing imports...")

    try:
        from src.training.steps.pre_training.label_balancing import (
            ComprehensiveBalancingSystem,
            BalancingConfig,
            WeightingConfig,
            RegimeConfig,
            ValidationFairnessConfig,
            BalancingTechnique,
            WeightingScheme
        )
        print("✅ Label balancing imports successful")
    except ImportError as e:
        print(f"❌ Label balancing import failed: {e}")
        return False

    try:
        from src.training.steps.model_training.tactician_balanced_training import (
            BalancedTacticianTrainingStep,
            BalancedTrainingConfig
        )
        print("✅ Balanced training imports successful")
    except ImportError as e:
        print(f"❌ Balanced training import failed: {e}")
        return False

    return True

def test_class_structure():
    """Test that our classes have the expected structure."""
    print("\nTesting class structure...")

    try:
        from src.training.steps.pre_training.label_balancing import BalancingTechnique, WeightingScheme

        # Check balancing techniques
        expected_techniques = [
            'UNDER_SAMPLING', 'OVER_SAMPLING', 'SMOTE', 'ADASYN',
            'MIXUP', 'STRATIFIED_BATCHING', 'HYBRID'
        ]

        for technique in expected_techniques:
            if hasattr(BalancingTechnique, technique):
                print(f"✅ BalancingTechnique.{technique} exists")
            else:
                print(f"❌ BalancingTechnique.{technique} missing")
                return False

        # Check weighting schemes
        expected_schemes = [
            'VOLATILITY', 'CONFIDENCE', 'EVENT_OVERLAP', 'TIME_DECAY',
            'REGIME_AWARE', 'INFORMATION_CONTENT'
        ]

        for scheme in expected_schemes:
            if hasattr(WeightingScheme, scheme):
                print(f"✅ WeightingScheme.{scheme} exists")
            else:
                print(f"❌ WeightingScheme.{scheme} missing")
                return False

        print("✅ All expected techniques and schemes available")
        return True

    except ImportError as e:
        print(f"❌ Class structure test failed: {e}")
        return False

def test_configuration_defaults():
    """Test that default configurations are available."""
    print("\nTesting default configurations...")

    try:
        from src.training.steps.pre_training.label_balancing import (
            DEFAULT_BALANCING_CONFIG,
            DEFAULT_WEIGHTING_CONFIG,
            DEFAULT_REGIME_CONFIG,
            DEFAULT_FAIRNESS_CONFIG
        )

        configs = [
            ('DEFAULT_BALANCING_CONFIG', DEFAULT_BALANCING_CONFIG),
            ('DEFAULT_WEIGHTING_CONFIG', DEFAULT_WEIGHTING_CONFIG),
            ('DEFAULT_REGIME_CONFIG', DEFAULT_REGIME_CONFIG),
            ('DEFAULT_FAIRNESS_CONFIG', DEFAULT_FAIRNESS_CONFIG)
        ]

        for name, config in configs:
            if config is not None:
                print(f"✅ {name} available")
            else:
                print(f"❌ {name} is None")
                return False

        print("✅ All default configurations available")
        return True

    except ImportError as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("BALANCING SYSTEM STRUCTURE TEST")
    print("=" * 60)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False

    # Test class structure
    if not test_class_structure():
        all_passed = False

    # Test configurations
    if not test_configuration_defaults():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Balancing system structure is correct")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️ Check the error messages above")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)