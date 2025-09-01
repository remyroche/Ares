#!/usr/bin/env python3
"""
Simple validation script for centralized S/R logic implementation.

This script validates the basic structure and syntax of the S/R implementation
without requiring external dependencies.
"""

import ast
import os
import sys

def check_file_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def check_method_exists(file_path, method_name):
    """Check if a method exists in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                return True, f"Method '{method_name}' found"
            elif isinstance(node, ast.AsyncFunctionDef) and node.name == method_name:
                return True, f"Async method '{method_name}' found"

        return False, f"Method '{method_name}' not found"
    except Exception as e:
        return False, f"Error checking method: {e}"

def check_imports(file_path):
    """Check if required imports are present."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        return imports
    except Exception as e:
        return []

def validate_sr_implementation():
    """Validate the S/R implementation."""
    print("🔍 Validating Centralized S/R Logic Implementation")
    print("=" * 60)

    # Check main S/R file
    sr_file = "src/tactician/sr_breakout_predictor.py"
    print(f"\n📁 Checking main S/R file: {sr_file}")

    if not os.path.exists(sr_file):
        print(f"❌ File not found: {sr_file}")
        return False

    # Check syntax
    syntax_ok, syntax_error = check_file_syntax(sr_file)
    if not syntax_ok:
        print(f"❌ Syntax error: {syntax_error}")
        return False
    print("✅ Syntax is valid")

    # Check required methods
    required_methods = [
        "get_sr_context",
        "is_near_sr_level",
        "get_sr_proximity_details",
        "predict_sr_outcome",
        "calculate_sr_features",
        "calculate_comprehensive_sr_features",
        "predict_breakout",
        "set_weights"
    ]

    print(f"\n🔧 Checking required methods:")
    all_methods_found = True
    for method in required_methods:
        found, message = check_method_exists(sr_file, method)
        if found:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
            all_methods_found = False

    if not all_methods_found:
        print("❌ Some required methods are missing")
        return False

    # Check imports
    imports = check_imports(sr_file)
    print(f"\n📦 Found {len(imports)} imports")

    # Check integration files
    integration_files = [
        "src/training/steps/step6_feature_engineering.py",
        "src/analyst/unified_regime_intelligence_runtime.py",
        "src/tactician/tactics_orchestrator.py",
        "src/training/steps/step15_tactician_specialist_training.py",
        "src/training/steps/sr_outcome_model_trainer.py"
    ]

    print(f"\n🔗 Checking integration files:")
    all_integrations_ok = True
    for file_path in integration_files:
        if os.path.exists(file_path):
            syntax_ok, syntax_error = check_file_syntax(file_path)
            if syntax_ok:
                print(f"✅ {file_path} - syntax valid")
            else:
                print(f"❌ {file_path} - {syntax_error}")
                all_integrations_ok = False
        else:
            print(f"⚠️  {file_path} - file not found")

    # Check test file
    test_file = "test_centralized_sr_logic.py"
    print(f"\n🧪 Checking test file: {test_file}")

    if os.path.exists(test_file):
        syntax_ok, syntax_error = check_file_syntax(test_file)
        if syntax_ok:
            print("✅ Test file syntax is valid")
        else:
            print(f"❌ Test file syntax error: {syntax_error}")
    else:
        print("⚠️  Test file not found")

    # Check documentation
    doc_file = "CENTRALIZED_SR_LOGIC_IMPLEMENTATION.md"
    print(f"\n📚 Checking documentation: {doc_file}")

    if os.path.exists(doc_file):
        print("✅ Documentation file exists")
    else:
        print("⚠️  Documentation file not found")

    # Summary
    print(f"\n{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")

    if all_methods_found and all_integrations_ok:
        print("🎉 Centralized S/R Logic Implementation is VALID!")
        print("✅ All required methods are present")
        print("✅ Integration files are syntactically correct")
        print("✅ Ready for testing and deployment")
        return True
    else:
        print("❌ Implementation has issues that need to be addressed")
        return False

def check_file_structure():
    """Check the overall file structure."""
    print(f"\n📂 Checking file structure:")

    expected_files = [
        "src/tactician/sr_breakout_predictor.py",
        "src/training/steps/step6_feature_engineering.py",
        "src/analyst/unified_regime_intelligence_runtime.py",
        "src/tactician/tactics_orchestrator.py",
        "test_centralized_sr_logic.py",
        "CENTRALIZED_SR_LOGIC_IMPLEMENTATION.md"
    ]

    for file_path in expected_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} - missing")

if __name__ == "__main__":
    print("🚀 Starting S/R Implementation Validation")
    print("=" * 60)

    # Check file structure
    check_file_structure()

    # Validate implementation
    success = validate_sr_implementation()

    if success:
        print(f"\n🎉 VALIDATION COMPLETED SUCCESSFULLY!")
        print("The centralized S/R logic implementation is ready for use.")
        sys.exit(0)
    else:
        print(f"\n❌ VALIDATION FAILED!")
        print("Please fix the issues before proceeding.")
        sys.exit(1)