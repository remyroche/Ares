#!/usr/bin/env python3
"""
Basic validation script for RL_NAS_Optimizer
Tests the basic structure without external dependencies
"""

import sys
import os
import ast
import inspect

def validate_python_syntax(filepath):
    """Validate Python syntax of the file."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source)
        print("✅ Python syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error parsing file: {e}")
        return False

def check_class_definitions(filepath):
    """Check if required classes are defined."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        required_classes = [
            'RL_NAS_Optimizer',
            'ArchitectureConfig', 
            'OptimizationConfig',
            'OptimizationResult'
        ]
        
        required_enums = [
            'OptimizationObjective',
            'ArchitectureType'
        ]
        
        found_classes = []
        found_enums = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                found_classes.append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in required_enums:
                        found_enums.append(target.id)
        
        print(f"✅ Found classes: {[c for c in found_classes if c in required_classes]}")
        print(f"✅ Found enums: {[e for e in found_enums if e in required_enums]}")
        
        missing_classes = [c for c in required_classes if c not in found_classes]
        if missing_classes:
            print(f"❌ Missing classes: {missing_classes}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking class definitions: {e}")
        return False

def check_method_definitions(filepath):
    """Check if required methods are defined in RL_NAS_Optimizer."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        required_methods = [
            '__init__',
            'optimize',
            'save_result',
            'load_result',
            'get_optimization_summary'
        ]
        
        found_methods = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'RL_NAS_Optimizer':
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        found_methods.append(item.name)
        
        print(f"✅ Found methods: {[m for m in found_methods if m in required_methods]}")
        
        missing_methods = [m for m in required_methods if m not in found_methods]
        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking method definitions: {e}")
        return False

def check_imports(filepath):
    """Check if required imports are present."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        required_imports = [
            'logging',
            'time',
            'json',
            'pathlib'
        ]
        
        found_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    found_imports.append(node.module)
        
        print(f"✅ Found imports: {[i for i in found_imports if i in required_imports]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking imports: {e}")
        return False

def check_file_structure(filepath):
    """Check overall file structure."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        print(f"✅ File has {len(lines)} lines")
        
        # Check for docstrings
        has_module_docstring = lines[0].strip().startswith('"""') or lines[0].strip().startswith("'''")
        if has_module_docstring:
            print("✅ Module has docstring")
        else:
            print("⚠️ Module docstring not found")
        
        # Check for main block
        has_main_block = any('if __name__ == "__main__":' in line for line in lines)
        if has_main_block:
            print("✅ Has main block for testing")
        else:
            print("⚠️ No main block found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking file structure: {e}")
        return False

def main():
    """Main validation function."""
    filepath = '/workspace/rl_nas.py'
    
    print("🔍 Validating RL_NAS_Optimizer implementation...")
    print("=" * 50)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    print(f"✅ File exists: {filepath}")
    
    # Run all validations
    validations = [
        ("Python Syntax", validate_python_syntax),
        ("Class Definitions", check_class_definitions),
        ("Method Definitions", check_method_definitions),
        ("Imports", check_imports),
        ("File Structure", check_file_structure)
    ]
    
    all_passed = True
    
    for name, validation_func in validations:
        print(f"\n🔍 Checking {name}...")
        if not validation_func(filepath):
            all_passed = False
        print()
    
    print("=" * 50)
    if all_passed:
        print("🎉 All validations passed! RL_NAS_Optimizer is properly structured.")
    else:
        print("❌ Some validations failed. Please check the implementation.")
    
    return all_passed

if __name__ == "__main__":
    main()