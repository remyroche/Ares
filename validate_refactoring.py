#!/usr/bin/env python3
"""
Simple validation script for refactored feature generators.

This script validates that the refactored files have correct syntax
and contain the expected centralized utility usage.
"""

import ast
import os
import re
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate that a Python file has correct syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(content)
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def validate_centralized_imports(file_path):
    """Validate that centralized imports are present."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for centralized utility imports
        required_imports = [
            "get_vectorbt_rolling_optimizer",
            "create_vectorbt_scaler",
            "get_global_feature_bank"
        ]
        
        missing_imports = []
        for import_name in required_imports:
            if import_name not in content:
                missing_imports.append(import_name)
        
        if missing_imports:
            return False, f"Missing imports: {missing_imports}"
        else:
            return True, "All required imports present"
            
    except Exception as e:
        return False, f"Error reading file: {e}"

def validate_optimization_methods(file_path):
    """Validate that optimization methods are present."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for optimization methods
        required_methods = [
            "_optimized_rolling_operation",
            "_fallback_rolling_operation", 
            "_normalize_feature",
            "_fallback_normalize"
        ]
        
        missing_methods = []
        for method_name in required_methods:
            if f"def {method_name}" not in content:
                missing_methods.append(method_name)
        
        if missing_methods:
            return False, f"Missing methods: {missing_methods}"
        else:
            return True, "All optimization methods present"
            
    except Exception as e:
        return False, f"Error reading file: {e}"

def validate_rolling_operations_replaced(file_path):
    """Validate that rolling operations have been replaced."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for old rolling patterns
        old_patterns = [
            r'\.rolling\(window=\w+\)\.mean\(\)',
            r'\.rolling\(window=\w+\)\.std\(\)',
            r'\.rolling\(window=\w+\)\.var\(\)',
            r'\.rolling\(window=\w+\)\.min\(\)',
            r'\.rolling\(window=\w+\)\.max\(\)',
            r'\.rolling\(window=\w+\)\.sum\(\)'
        ]
        
        old_patterns_found = []
        for pattern in old_patterns:
            if re.search(pattern, content):
                old_patterns_found.append(pattern)
        
        if old_patterns_found:
            return False, f"Old rolling patterns found: {old_patterns_found}"
        else:
            return True, "No old rolling patterns found"
            
    except Exception as e:
        return False, f"Error reading file: {e}"

def validate_file(file_path):
    """Validate a single file."""
    print(f"\nValidating {file_path}...")
    
    results = {}
    
    # Test syntax
    syntax_ok, syntax_msg = validate_python_syntax(file_path)
    results['syntax'] = (syntax_ok, syntax_msg)
    print(f"  Syntax: {'✅' if syntax_ok else '❌'} {syntax_msg}")
    
    # Test centralized imports
    imports_ok, imports_msg = validate_centralized_imports(file_path)
    results['imports'] = (imports_ok, imports_msg)
    print(f"  Imports: {'✅' if imports_ok else '❌'} {imports_msg}")
    
    # Test optimization methods
    methods_ok, methods_msg = validate_optimization_methods(file_path)
    results['methods'] = (methods_ok, methods_msg)
    print(f"  Methods: {'✅' if methods_ok else '❌'} {methods_msg}")
    
    # Test rolling operations
    rolling_ok, rolling_msg = validate_rolling_operations_replaced(file_path)
    results['rolling'] = (rolling_ok, rolling_msg)
    print(f"  Rolling: {'✅' if rolling_ok else '❌'} {rolling_msg}")
    
    return results

def main():
    """Main validation function."""
    print("🔍 Validating refactored feature generators...")
    
    # Target files to validate
    target_files = [
        "src/feature_generation/categories/momentum.py",
        "src/feature_generation/categories/trend.py",
        "src/feature_generation/categories/oscillator.py",
        "src/feature_generation/categories/legacy.py",
        "src/feature_generation/categories/volatility.py",
        "src/feature_generation/categories/volume.py"
    ]
    
    all_results = {}
    total_tests = 0
    passed_tests = 0
    
    for file_path in target_files:
        if os.path.exists(file_path):
            results = validate_file(file_path)
            all_results[file_path] = results
            
            # Count tests
            for test_name, (passed, _) in results.items():
                total_tests += 1
                if passed:
                    passed_tests += 1
        else:
            print(f"\n❌ File not found: {file_path}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    for file_path, results in all_results.items():
        print(f"\n{file_path}:")
        for test_name, (passed, msg) in results.items():
            status = "✅" if passed else "❌"
            print(f"  {test_name}: {status} {msg}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All validations passed! Refactoring successful!")
        return True
    else:
        print("❌ Some validations failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)