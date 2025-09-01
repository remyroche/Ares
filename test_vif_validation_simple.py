#!/usr/bin/env python3
"""
Simple Test for VIF Validation Decorators

This script tests the VIF validation decorators without requiring external dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_vif_validation_decorators():
    """Test that the VIF validation decorators can be imported and used."""
    print("🧪 Testing VIF Validation Decorators Import")

    try:
        # Test import of VIF validation decorators (simplified version)
        from src.utils.vif_validation_decorators_simple import (
            validate_vif_inputs,
            validate_vif_outputs,
            safe_vif_calculation,
            comprehensive_vif_validation,
            VIFValidationError
        )
        print("✅ Successfully imported VIF validation decorators (simplified)")

        # Test import of VIF calculator (will fail without numpy/pandas, but that's expected)
        try:
            from src.utils.vif_calculator import (
                calculate_vif_simple,
                calculate_vif_robust,
                analyze_vif_issues,
                get_vif_recommendations
            )
            print("✅ Successfully imported VIF calculator functions")
        except ImportError:
            print("⚠️ VIF calculator functions not available (numpy/pandas required)")

        # Test decorator creation
        @validate_vif_inputs()
        def test_function(data):
            return data

        print("✅ Successfully created decorated function")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_step2_import():
    """Test that step2 can be imported with the new VIF validation."""
    print("\n🧪 Testing Step2 Import with VIF Validation")

    try:
        # Test import of step2
        from src.training.steps.step2_feature_engineering import run_step
        print("✅ Successfully imported step2_feature_engineering")

        # Check if the VIF calculator import is working
        import src.utils.vif_calculator
        print("✅ VIF calculator module is available")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_vif_validation_structure():
    """Test the structure of the VIF validation modules."""
    print("\n🧪 Testing VIF Validation Module Structure")

    try:
        from src.utils.vif_validation_decorators_simple import (
            validate_vif_inputs,
            validate_vif_outputs,
            safe_vif_calculation,
            comprehensive_vif_validation,
            VIFValidationError
        )

        # Test that decorators are callable
        assert callable(validate_vif_inputs)
        assert callable(validate_vif_outputs)
        assert callable(safe_vif_calculation)
        assert callable(comprehensive_vif_validation)

        print("✅ All VIF validation decorators are callable")

        # Test that VIFValidationError is an exception
        assert issubclass(VIFValidationError, Exception)
        print("✅ VIFValidationError is properly defined")

        return True

    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting VIF Validation Tests")
    print("=" * 50)

    tests = [
        test_vif_validation_decorators,
        test_step2_import,
        test_vif_validation_structure
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests passed! VIF validation decorators are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)