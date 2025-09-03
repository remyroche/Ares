#!/usr/bin/env python3
"""Get detailed list of files with syntax errors."""

import ast
import json
from pathlib import Path

def check_file_syntax(file_path):
    """Check if a file has syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
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
            'error': str(e)
        }

def main():
    src_dir = Path('/workspace/src')
    errors = []
    
    for py_file in src_dir.rglob('*.py'):
        error = check_file_syntax(py_file)
        if error:
            errors.append(error)
    
    # Sort by error message for grouping
    errors.sort(key=lambda x: x.get('msg', x.get('error', '')))
    
    print(f"Found {len(errors)} files with syntax errors\n")
    
    # Group by error type
    error_groups = {}
    for error in errors:
        msg = error.get('msg', error.get('error', 'Unknown'))
        if msg not in error_groups:
            error_groups[msg] = []
        error_groups[msg].append(error)
    
    # Show first few files from each error group
    for msg, files in error_groups.items():
        print(f"\n{msg} ({len(files)} files):")
        for error in files[:5]:  # Show first 5 of each type
            print(f"  - {error['file']} (line {error.get('line', 'unknown')})")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
    
    # Save full list
    with open('/workspace/syntax_errors_detailed.json', 'w') as f:
        json.dump(errors, f, indent=2)
    
    print(f"\nFull list saved to /workspace/syntax_errors_detailed.json")

if __name__ == "__main__":
    main()