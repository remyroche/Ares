#!/usr/bin/env python3
"""
Simple syntax validation script for Python files.
Uses Python's built-in ast module to check for syntax errors.
"""

import os
import ast
import sys
from pathlib import Path
from typing import List, Dict, Any

def find_python_files(directory: str, exclude_patterns: List[str] = None) -> List[str]:
    """Find all Python files in a directory."""
    if exclude_patterns is None:
        exclude_patterns = ["__pycache__", "*.pyc", ".git", "venv", "env", "node_modules", ".pytest_cache"]
    
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_patterns]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    
    return python_files

def validate_python_file(file_path: str) -> Dict[str, Any]:
    """Validate a single Python file for syntax errors."""
    result = {
        "file_path": file_path,
        "valid": True,
        "errors": [],
        "ast_parseable": True,
        "compilable": True
    }
    
    try:
        # Try to read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse with AST
        try:
            ast.parse(content)
        except SyntaxError as e:
            result["ast_parseable"] = False
            result["valid"] = False
            result["errors"].append({
                "type": "syntax_error",
                "line": e.lineno,
                "column": e.offset or 0,
                "message": str(e)
            })
        except Exception as e:
            result["ast_parseable"] = False
            result["valid"] = False
            result["errors"].append({
                "type": "parse_error",
                "line": 0,
                "column": 0,
                "message": str(e)
            })
        
        # Try to compile
        try:
            compile(content, file_path, 'exec')
        except SyntaxError as e:
            result["compilable"] = False
            result["valid"] = False
            if not result["errors"]:  # Only add if not already added
                result["errors"].append({
                    "type": "compilation_error",
                    "line": e.lineno,
                    "column": e.offset or 0,
                    "message": str(e)
                })
        except Exception as e:
            result["compilable"] = False
            result["valid"] = False
            if not result["errors"]:  # Only add if not already added
                result["errors"].append({
                    "type": "compilation_error",
                    "line": 0,
                    "column": 0,
                    "message": str(e)
                })
            
    except Exception as e:
        result["valid"] = False
        result["errors"].append({
            "type": "file_error",
            "line": 0,
            "column": 0,
            "message": f"Could not read file: {str(e)}"
        })
    
    return result

def main():
    """Main function to validate all Python files in the workspace."""
    workspace_dir = "."
    
    print("🔍 Performing syntax validation on Python files...")
    print(f"📁 Scanning directory: {os.path.abspath(workspace_dir)}")
    print()
    
    # Find all Python files
    python_files = find_python_files(workspace_dir)
    print(f"📄 Found {len(python_files)} Python files")
    print()
    
    if not python_files:
        print("❌ No Python files found in the workspace")
        return
    
    # Validate each file
    results = []
    valid_files = 0
    invalid_files = 0
    total_errors = 0
    
    for file_path in python_files:
        result = validate_python_file(file_path)
        results.append(result)
        
        if result["valid"]:
            valid_files += 1
            print(f"✅ {file_path}")
        else:
            invalid_files += 1
            total_errors += len(result["errors"])
            print(f"❌ {file_path}")
            for error in result["errors"]:
                print(f"   🔴 Line {error['line']}: {error['message']}")
    
    print()
    print("📊 SYNTAX VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total files: {len(python_files)}")
    print(f"Valid files: {valid_files}")
    print(f"Invalid files: {invalid_files}")
    print(f"Total errors: {total_errors}")
    
    if invalid_files > 0:
        print()
        print("🚨 FILES WITH SYNTAX ISSUES:")
        print("=" * 50)
        for result in results:
            if not result["valid"]:
                print(f"\n📁 {result['file_path']}")
                for error in result["errors"]:
                    print(f"   🔴 {error['type'].upper()}: Line {error['line']}, Column {error['column']}")
                    print(f"      {error['message']}")
    else:
        print()
        print("🎉 All Python files have valid syntax!")
    
    return invalid_files

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)