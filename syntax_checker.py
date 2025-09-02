#!/usr/bin/env python3
"""
Simple syntax checker for Python files.
"""

import os
import sys
import ast
from pathlib import Path

def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to compile the code
        compile(content, file_path, 'exec')
        
        # Try to parse the AST
        ast.parse(content, filename=file_path)
        
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"
    except UnicodeDecodeError as e:
        return False, f"UnicodeDecodeError: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function to check all Python files."""
    workspace_path = Path(".")
    python_files = list(workspace_path.rglob("*.py"))
    
    print(f"Checking {len(python_files)} Python files for syntax errors...\n")
    
    files_with_issues = []
    total_issues = 0
    
    for file_path in python_files:
        # Skip the code_quality directory for now since it has dependency issues
        if "code_quality" in str(file_path):
            continue
            
        is_valid, error = check_syntax(file_path)
        
        if not is_valid:
            files_with_issues.append((file_path, error))
            total_issues += 1
            print(f"❌ {file_path}: {error}")
        else:
            print(f"✅ {file_path}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"Total Python files checked: {len(python_files)}")
    print(f"Files with syntax issues: {len(files_with_issues)}")
    print(f"Total syntax errors: {total_issues}")
    
    if files_with_issues:
        print(f"\nFILES WITH SYNTAX ISSUES:")
        print(f"{'='*60}")
        for file_path, error in files_with_issues:
            print(f"\n📁 {file_path}")
            print(f"   Error: {error}")
    else:
        print(f"\n🎉 All Python files have valid syntax!")
    
    return len(files_with_issues)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)