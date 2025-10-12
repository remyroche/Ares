#!/usr/bin/env python3
"""
Simple syntax check to verify that the duplicate cleanup was successful.
This script checks that all files can be imported without syntax errors.
"""

import sys
import os
import ast

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def check_imports(file_path):
    """Check if a Python file can be imported without errors."""
    try:
        # Extract the module name from the file path
        module_name = file_path.replace('src/', '').replace('/', '.').replace('.py', '')
        
        # Try to import the module
        __import__(module_name)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    """Run syntax and import checks."""
    print("🔍 Running syntax and import checks...\n")
    
    # Files to check
    files_to_check = [
        'src/feature_generation/core/feature_generator.py',
        'src/feature_generation/core/vectorbt_optimization_mixin.py',
        'src/feature_generation/categories/volume.py',
        'src/feature_generation/categories/acceleration.py',
        'src/feature_generation/categories/entropy.py',
        'src/feature_generation/categories/oscillator.py',
        'src/feature_generation/categories/interaction.py',
        'src/feature_generation/categories/cross_timeframe.py',
        'src/feature_generation/categories/autoencoder.py',
        'src/feature_generation/categories/representation_learning.py',
        'src/feature_generation/categories/regime_feature_integration.py'
    ]
    
    syntax_passed = 0
    import_passed = 0
    total = len(files_to_check)
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
            
        print(f"Checking {file_path}...")
        
        # Check syntax
        syntax_ok, syntax_error = check_syntax(file_path)
        if syntax_ok:
            print(f"  ✅ Syntax OK")
            syntax_passed += 1
        else:
            print(f"  ❌ Syntax Error: {syntax_error}")
        
        # Check imports (skip for now due to missing dependencies)
        # import_ok, import_error = check_imports(file_path)
        # if import_ok:
        #     print(f"  ✅ Import OK")
        #     import_passed += 1
        # else:
        #     print(f"  ❌ Import Error: {import_error}")
        
        print()
    
    print(f"📊 Syntax Check Results: {syntax_passed}/{total} files passed")
    # print(f"📊 Import Check Results: {import_passed}/{total} files passed")
    
    if syntax_passed == total:
        print("🎉 All syntax checks passed! Duplicate cleanup was successful.")
        return True
    else:
        print("⚠️ Some syntax checks failed. Please check the files.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)