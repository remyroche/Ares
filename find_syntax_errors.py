#!/usr/bin/env python3
"""Find all Python files with syntax errors."""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict


def check_file_syntax(file_path):
    """Check if a Python file has syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse with ast
        ast.parse(content)
        return None
    except SyntaxError as e:
        return {
            'file': str(file_path),
            'line': e.lineno,
            'offset': e.offset,
            'msg': e.msg,
            'text': e.text
        }
    except Exception as e:
        return {
            'file': str(file_path),
            'line': 0,
            'offset': 0,
            'msg': f"Error reading file: {str(e)}",
            'text': None
        }


def find_all_syntax_errors():
    """Find all Python files with syntax errors in the workspace."""
    workspace_path = Path("/workspace")
    src_path = workspace_path / "src"
    
    syntax_errors = []
    error_types = defaultdict(int)
    files_checked = 0
    
    # Find all Python files in src/
    for py_file in src_path.rglob("*.py"):
        files_checked += 1
        error = check_file_syntax(py_file)
        
        if error:
            syntax_errors.append(error)
            error_types[error['msg']] += 1
    
    print(f"\nChecked {files_checked} Python files")
    print(f"Found {len(syntax_errors)} files with syntax errors")
    
    if syntax_errors:
        print("\nError type distribution:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")
        
        print("\nFirst 20 files with syntax errors:")
        for i, error in enumerate(syntax_errors[:20], 1):
            print(f"\n{i}. {error['file']}")
            print(f"   Line {error['line']}: {error['msg']}")
            if error['text']:
                print(f"   Code: {error['text'].strip()}")
    
    # Save to file
    import json
    with open("/workspace/actual_syntax_errors.json", "w") as f:
        json.dump({
            'total_files': files_checked,
            'files_with_errors': len(syntax_errors),
            'error_types': dict(error_types),
            'errors': syntax_errors
        }, f, indent=2)
    
    print(f"\nDetailed report saved to: /workspace/actual_syntax_errors.json")
    
    return syntax_errors


if __name__ == "__main__":
    find_all_syntax_errors()