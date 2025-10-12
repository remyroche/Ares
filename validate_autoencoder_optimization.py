#!/usr/bin/env python3
"""
Simple validation script for VectorBT autoencoder optimization.

This script validates the syntax and structure of the optimized autoencoder files.
"""

import ast
import os
import sys

def validate_python_file(file_path):
    """Validate Python file syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(content)
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_imports(file_path):
    """Check if required imports are present."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_imports = [
            'VectorBTRollingOptimizer',
            'UnifiedVectorizationManager',
            'VectorBTFeatureGenerator',
            'VectorBTOptimizationMixin'
        ]
        
        missing_imports = []
        for import_name in required_imports:
            if import_name not in content:
                missing_imports.append(import_name)
        
        return missing_imports
    except Exception as e:
        return [f"Error reading file: {e}"]

def check_class_inheritance(file_path):
    """Check if classes inherit from VectorBT classes."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for class definitions
        tree = ast.parse(content)
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'bases': [base.id if hasattr(base, 'id') else str(base) for base in node.bases]
                })
        
        return classes
    except Exception as e:
        return [f"Error parsing classes: {e}"]

def main():
    """Main validation function."""
    print("🔍 Validating VectorBT autoencoder optimization...")
    
    # Files to validate
    files_to_check = [
        '/workspace/src/feature_generation/utils/unified_vectorization_manager.py',
        '/workspace/src/feature_generation/categories/autoencoder.py'
    ]
    
    all_passed = True
    
    for file_path in files_to_check:
        print(f"\n📁 Checking {os.path.basename(file_path)}...")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            all_passed = False
            continue
        
        # Check syntax
        is_valid, message = validate_python_file(file_path)
        if is_valid:
            print(f"✅ Syntax: {message}")
        else:
            print(f"❌ Syntax: {message}")
            all_passed = False
        
        # Check imports
        missing_imports = check_imports(file_path)
        if not missing_imports:
            print("✅ All required imports present")
        else:
            print(f"⚠️ Missing imports: {missing_imports}")
        
        # Check class inheritance
        classes = check_class_inheritance(file_path)
        if isinstance(classes, list) and len(classes) > 0:
            print(f"📋 Found {len(classes)} classes:")
            for cls in classes:
                print(f"   - {cls['name']}: {cls['bases']}")
        else:
            print(f"⚠️ Could not parse classes: {classes}")
    
    # Summary
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All files passed validation!")
    else:
        print("⚠️ Some files failed validation. Check the output above.")
    print(f"{'='*50}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)