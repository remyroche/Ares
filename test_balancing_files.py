#!/usr/bin/env python3
"""
Simple test to verify the balancing system files exist and have correct structure.
"""

import os
import ast
import sys

def check_file_exists(filepath):
    """Check if file exists."""
    if os.path.exists(filepath):
        print(f"✅ {filepath} exists")
        return True
    else:
        print(f"❌ {filepath} missing")
        return False

def check_class_in_file(filepath, class_name):
    """Check if a class is defined in a file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Simple check for class definition
        if f"class {class_name}" in content:
            print(f"✅ {class_name} class found in {filepath}")
            return True
        else:
            print(f"❌ {class_name} class not found in {filepath}")
            return False
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return False

def check_function_in_file(filepath, function_name):
    """Check if a function is defined in a file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Simple check for function definition
        if f"def {function_name}" in content:
            print(f"✅ {function_name} function found in {filepath}")
            return True
        else:
            print(f"❌ {function_name} function not found in {filepath}")
            return False
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("BALANCING SYSTEM FILE STRUCTURE TEST")
    print("=" * 60)

    all_passed = True

    # Check main files exist
    files_to_check = [
        "src/training/steps/pre_training/label_balancing.py",
        "src/training/steps/model_training/tactician_balanced_training.py",
        "src/training/steps/pre_training/README_BALANCING_SYSTEM.md",
        "examples/balanced_training_example.py"
    ]

    for filepath in files_to_check:
        if not check_file_exists(filepath):
            all_passed = False

    # Check key classes exist
    classes_to_check = [
        ("src/training/steps/pre_training/label_balancing.py", "LabelBalancer"),
        ("src/training/steps/pre_training/label_balancing.py", "SampleWeighter"),
        ("src/training/steps/pre_training/label_balancing.py", "RegimeAwareBalancer"),
        ("src/training/steps/pre_training/label_balancing.py", "ValidationFairnessChecker"),
        ("src/training/steps/pre_training/label_balancing.py", "ComprehensiveBalancingSystem"),
        ("src/training/steps/model_training/tactician_balanced_training.py", "BalancedTacticianTrainingStep"),
    ]

    for filepath, class_name in classes_to_check:
        if not check_class_in_file(filepath, class_name):
            all_passed = False

    # Check key functions exist
    functions_to_check = [
        ("src/training/steps/pre_training/label_balancing.py", "balance_dataset"),
        ("src/training/steps/pre_training/label_balancing.py", "compute_weights"),
        ("src/training/steps/model_training/tactician_balanced_training.py", "balance_and_weight"),
    ]

    for filepath, function_name in functions_to_check:
        if not check_function_in_file(filepath, function_name):
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Balancing system files and structure are correct")
        print("\n📋 SUMMARY:")
        print("✅ Main balancing module created")
        print("✅ Balanced training integration created")
        print("✅ Documentation created")
        print("✅ Example script created")
        print("✅ All key classes and functions defined")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️ Check the error messages above")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)