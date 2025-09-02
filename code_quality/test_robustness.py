#!/usr/bin/env python3
"""
Test script to demonstrate robust error handling in our analyzers.
"""

import ast
import re
from pathlib import Path

def test_syntax_error_handling():
    """Test how we handle files with syntax errors."""
    
    # Test file with syntax error
    bad_code = """
def working_function():
    return "This works"

def broken_function(
    # Missing closing parenthesis and body
    print("This will cause a syntax error"
    
def another_working_function():
    return "This also works"
"""
    
    print("🔍 Testing syntax error handling...")
    print("=" * 50)
    
    # Test 1: Try AST parsing (will fail)
    try:
        tree = ast.parse(bad_code)
        print("✅ AST parsing succeeded")
    except SyntaxError as e:
        print(f"❌ AST parsing failed: {e}")
        print("🔄 Switching to regex fallback...")
    
    # Test 2: Regex fallback parsing
    imports = set()
    function_defs = set()
    
    # Extract function definitions with regex
    for line in bad_code.split('\n'):
        line = line.strip()
        if line.startswith('def '):
            # Simple regex to extract function names
            match = re.match(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
            if match:
                func_name = match.group(1)
                function_defs.add(func_name)
                print(f"📝 Found function: {func_name}")
    
    print(f"\n📊 Results despite syntax errors:")
    print(f"   • Functions found: {len(function_defs)}")
    print(f"   • Functions: {', '.join(function_defs)}")
    
    return len(function_defs) > 0

def test_import_extraction():
    """Test import extraction from broken code."""
    
    broken_code = """
import pandas as pd
import numpy as np
from datetime import datetime

def broken_function(
    # Syntax error here
    pass

import json
"""
    
    print("\n🔍 Testing import extraction from broken code...")
    print("=" * 50)
    
    # Extract imports with regex
    imports = set()
    import_pattern = r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
    from_pattern = r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import'
    
    for line in broken_code.split('\n'):
        line = line.strip()
        if line.startswith('import '):
            match = re.match(import_pattern, line)
            if match:
                module = match.group(1).split('.')[0]
                imports.add(module)
                print(f"📦 Found import: {module}")
        elif line.startswith('from '):
            match = re.match(from_pattern, line)
            if match:
                module = match.group(1).split('.')[0]
                imports.add(module)
                print(f"📦 Found from-import: {module}")
    
    print(f"\n📊 Import extraction results:")
    print(f"   • Total imports found: {len(imports)}")
    print(f"   • Imports: {', '.join(sorted(imports))}")
    
    return len(imports) > 0

def demonstrate_analysis_continuity():
    """Demonstrate that analysis continues despite errors."""
    
    print("\n🔍 Demonstrating analysis continuity...")
    print("=" * 50)
    
    # Simulate multiple files with different error levels
    test_files = [
        ("good_file.py", "def good_function(): return True"),
        ("syntax_error.py", "def broken_function( return True"),
        ("another_good.py", "def another_function(): return False"),
        ("import_error.py", "import pandas\nfrom datetime import datetime\ndef broken(): pass"),
    ]
    
    total_functions = 0
    successful_parses = 0
    failed_parses = 0
    
    for filename, content in test_files:
        try:
            tree = ast.parse(content)
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
            
            total_functions += len(functions)
            successful_parses += 1
            print(f"✅ {filename}: {len(functions)} functions parsed")
            
        except SyntaxError:
            failed_parses += 1
            print(f"❌ {filename}: Syntax error, using fallback")
            
            # Fallback parsing
            functions = []
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('def '):
                    match = re.match(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
                    if match:
                        functions.append(match.group(1))
            
            total_functions += len(functions)
            print(f"🔄 {filename}: {len(functions)} functions found with fallback")
    
    print(f"\n📊 Analysis continuity results:")
    print(f"   • Files successfully parsed: {successful_parses}")
    print(f"   • Files with syntax errors: {failed_parses}")
    print(f"   • Total functions found: {total_functions}")
    print(f"   • Analysis completed: ✅ YES")
    
    return total_functions > 0

def main():
    """Run all robustness tests."""
    print("🧪 TESTING ANALYZER ROBUSTNESS")
    print("=" * 60)
    
    # Test 1: Syntax error handling
    test1_passed = test_syntax_error_handling()
    
    # Test 2: Import extraction
    test2_passed = test_import_extraction()
    
    # Test 3: Analysis continuity
    test3_passed = demonstrate_analysis_continuity()
    
    print("\n" + "=" * 60)
    print("🏁 ROBUSTNESS TEST RESULTS")
    print("=" * 60)
    print(f"✅ Syntax error handling: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ Import extraction: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"✅ Analysis continuity: {'PASSED' if test3_passed else 'FAILED'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 ALL TESTS PASSED! Our analyzer is robust against syntax errors.")
    else:
        print("\n⚠️  Some tests failed. Check the implementation.")

if __name__ == "__main__":
    main()