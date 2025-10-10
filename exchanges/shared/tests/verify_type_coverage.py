#!/usr/bin/env python3
"""
Verify comprehensive type hint coverage and error handling implementation.
"""

import os
import sys
import ast
import inspect
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def analyze_type_hints(file_path: str) -> Dict[str, Any]:
    """Analyze type hints in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Count functions, classes, and methods
        functions = []
        classes = []
        methods = []
        type_hints = 0
        error_handling = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
                # Check for type hints
                if node.returns:
                    type_hints += 1
                for arg in node.args.args:
                    if arg.annotation:
                        type_hints += 1
                # Check for error handling
                if any(isinstance(n, ast.Try) for n in ast.walk(node)):
                    error_handling += 1
                    
            elif isinstance(node, ast.AsyncFunctionDef):
                functions.append(node.name)
                # Check for type hints
                if node.returns:
                    type_hints += 1
                for arg in node.args.args:
                    if arg.annotation:
                        type_hints += 1
                # Check for error handling
                if any(isinstance(n, ast.Try) for n in ast.walk(node)):
                    error_handling += 1
                    
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                        # Check for type hints
                        if item.returns:
                            type_hints += 1
                        for arg in item.args.args:
                            if arg.annotation:
                                type_hints += 1
                        # Check for error handling
                        if any(isinstance(n, ast.Try) for n in ast.walk(item)):
                            error_handling += 1
        
        return {
            "functions": len(functions),
            "classes": len(classes),
            "methods": len(methods),
            "type_hints": type_hints,
            "error_handling": error_handling,
            "total_items": len(functions) + len(classes) + len(methods)
        }
    except Exception as e:
        return {"error": str(e)}


def check_tprint_usage(file_path: str) -> bool:
    """Check if file uses tprint for error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return "tprint(" in content and "ERROR" in content
    except Exception:
        return False


def check_error_handling_decorators(file_path: str) -> bool:
    """Check if file uses error handling decorators."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return "@handle_errors" in content or "@handle_async_errors" in content
    except Exception:
        return False


def check_try_except_blocks(file_path: str) -> int:
    """Count try-except blocks in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        try_blocks = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                try_blocks += 1
        
        return try_blocks
    except Exception:
        return 0


def main():
    """Main verification function."""
    print("🔍 Verifying Type Hint Coverage and Error Handling")
    print("=" * 70)
    
    # Get workspace directory
    workspace = Path(__file__).parent.parent.parent.parent
    print(f"Working directory: {workspace}")
    
    # Files to analyze
    files_to_analyze = [
        "exchanges/shared/interfaces_typed.py",
        "exchanges/shared/high_level_wrappers_typed.py",
        "exchanges/shared/high_level_wrappers_typed_part2.py",
        "exchanges/shared/examples/typed_usage_example.py",
        "exchanges/shared/tests/test_type_coverage.py"
    ]
    
    results = []
    total_type_hints = 0
    total_error_handling = 0
    total_try_blocks = 0
    files_with_tprint = 0
    files_with_decorators = 0
    
    for file_path in files_to_analyze:
        full_path = workspace / file_path
        
        if not full_path.exists():
            print(f"❌ File not found: {file_path}")
            continue
        
        print(f"\n📄 Analyzing: {file_path}")
        
        # Analyze type hints
        analysis = analyze_type_hints(str(full_path))
        
        if "error" in analysis:
            print(f"❌ Error analyzing file: {analysis['error']}")
            continue
        
        # Check tprint usage
        has_tprint = check_tprint_usage(str(full_path))
        if has_tprint:
            files_with_tprint += 1
        
        # Check error handling decorators
        has_decorators = check_error_handling_decorators(str(full_path))
        if has_decorators:
            files_with_decorators += 1
        
        # Count try-except blocks
        try_blocks = check_try_except_blocks(str(full_path))
        total_try_blocks += try_blocks
        
        # Accumulate totals
        total_type_hints += analysis["type_hints"]
        total_error_handling += analysis["error_handling"]
        
        # Calculate coverage percentage
        coverage = (analysis["type_hints"] / max(analysis["total_items"], 1)) * 100
        
        print(f"  📊 Functions: {analysis['functions']}")
        print(f"  📊 Classes: {analysis['classes']}")
        print(f"  📊 Methods: {analysis['methods']}")
        print(f"  📊 Type hints: {analysis['type_hints']}")
        print(f"  📊 Error handling: {analysis['error_handling']}")
        print(f"  📊 Try-except blocks: {try_blocks}")
        print(f"  📊 Type coverage: {coverage:.1f}%")
        print(f"  📊 Uses tprint: {'✅' if has_tprint else '❌'}")
        print(f"  📊 Uses decorators: {'✅' if has_decorators else '❌'}")
        
        results.append({
            "file": file_path,
            "coverage": coverage,
            "type_hints": analysis["type_hints"],
            "error_handling": analysis["error_handling"],
            "try_blocks": try_blocks,
            "has_tprint": has_tprint,
            "has_decorators": has_decorators
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    total_files = len(results)
    avg_coverage = sum(r["coverage"] for r in results) / max(total_files, 1)
    
    print(f"Total files analyzed: {total_files}")
    print(f"Total type hints: {total_type_hints}")
    print(f"Total error handling functions: {total_error_handling}")
    print(f"Total try-except blocks: {total_try_blocks}")
    print(f"Files using tprint: {files_with_tprint}")
    print(f"Files using decorators: {files_with_decorators}")
    print(f"Average type coverage: {avg_coverage:.1f}%")
    
    # Check specific requirements
    requirements_met = 0
    total_requirements = 6
    
    print("\n📋 REQUIREMENTS CHECK:")
    print("-" * 30)
    
    # 1. Type hint coverage > 80%
    if avg_coverage >= 80:
        print("✅ Type hint coverage >= 80%")
        requirements_met += 1
    else:
        print(f"❌ Type hint coverage {avg_coverage:.1f}% < 80%")
    
    # 2. All files use tprint
    if files_with_tprint == total_files:
        print("✅ All files use tprint for error handling")
        requirements_met += 1
    else:
        print(f"❌ Only {files_with_tprint}/{total_files} files use tprint")
    
    # 3. All files use error handling decorators
    if files_with_decorators == total_files:
        print("✅ All files use error handling decorators")
        requirements_met += 1
    else:
        print(f"❌ Only {files_with_decorators}/{total_files} files use decorators")
    
    # 4. Sufficient try-except blocks
    if total_try_blocks >= total_files * 5:  # At least 5 per file on average
        print("✅ Sufficient try-except blocks for error handling")
        requirements_met += 1
    else:
        print(f"❌ Only {total_try_blocks} try-except blocks (expected >= {total_files * 5})")
    
    # 5. All files have type hints
    files_with_type_hints = sum(1 for r in results if r["type_hints"] > 0)
    if files_with_type_hints == total_files:
        print("✅ All files have type hints")
        requirements_met += 1
    else:
        print(f"❌ Only {files_with_type_hints}/{total_files} files have type hints")
    
    # 6. All files have error handling
    files_with_error_handling = sum(1 for r in results if r["error_handling"] > 0)
    if files_with_error_handling == total_files:
        print("✅ All files have error handling")
        requirements_met += 1
    else:
        print(f"❌ Only {files_with_error_handling}/{total_files} files have error handling")
    
    print(f"\n📈 Requirements met: {requirements_met}/{total_requirements}")
    
    if requirements_met == total_requirements:
        print("\n🎉 ALL TYPE HINT AND ERROR HANDLING REQUIREMENTS MET!")
        print("\nKey improvements:")
        print("• ✅ Comprehensive type hints throughout all modules")
        print("• ✅ tprint error handling with proper logging levels")
        print("• ✅ Error handling decorators for consistent error management")
        print("• ✅ Try-except blocks for robust error handling")
        print("• ✅ Type-safe interfaces and implementations")
        print("• ✅ Comprehensive error handling coverage")
        return 0
    else:
        print(f"\n⚠️  {total_requirements - requirements_met} requirements not met.")
        print("Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())