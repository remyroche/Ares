#!/usr/bin/env python3
"""
Script to fix common syntax errors in Python files.
"""

import ast
import re
from pathlib import Path
import json

def check_syntax(file_path):
    """Check if a file has valid Python syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, e

def fix_common_issues(content, error):
    """Try to fix common syntax issues."""
    fixed = False
    
    if "unexpected indent" in str(error):
        # Try to fix indentation issues
        lines = content.split('\n')
        if error.lineno and 0 < error.lineno <= len(lines):
            line_idx = error.lineno - 1
            line = lines[line_idx]
            
            # Check if line has only whitespace followed by code
            if line.strip() and line[0] in ' \t':
                # Remove leading whitespace if it seems wrong
                stripped = line.lstrip()
                # Try to determine correct indentation from previous lines
                prev_indent = 0
                for i in range(line_idx - 1, -1, -1):
                    if lines[i].strip():
                        prev_indent = len(lines[i]) - len(lines[i].lstrip())
                        break
                
                # Apply same indentation
                lines[line_idx] = ' ' * prev_indent + stripped
                content = '\n'.join(lines)
                fixed = True
    
    elif "unmatched" in str(error) and ")" in str(error):
        # Try to find and remove unmatched closing parenthesis
        lines = content.split('\n')
        if error.lineno and 0 < error.lineno <= len(lines):
            line_idx = error.lineno - 1
            line = lines[line_idx]
            
            # Count parentheses
            open_count = line.count('(')
            close_count = line.count(')')
            
            if close_count > open_count:
                # Remove the last closing parenthesis
                last_close = line.rfind(')')
                if last_close != -1:
                    lines[line_idx] = line[:last_close] + line[last_close+1:]
                    content = '\n'.join(lines)
                    fixed = True
    
    return content, fixed

def process_file(file_path):
    """Process a single file and try to fix syntax errors."""
    print(f"Processing: {file_path}")
    
    valid, error = check_syntax(file_path)
    if valid:
        print(f"  ✓ No syntax errors")
        return True
    
    print(f"  ✗ Syntax error: {error}")
    
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to fix
    fixed_content, was_fixed = fix_common_issues(content, error)
    
    if was_fixed:
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        # Check again
        valid, new_error = check_syntax(file_path)
        if valid:
            print(f"  ✓ Fixed successfully!")
            return True
        else:
            print(f"  ✗ Still has errors: {new_error}")
            return False
    else:
        print(f"  ✗ Could not auto-fix")
        return False

def main():
    # Load error data
    with open('/workspace/syntax_errors.json', 'r') as f:
        error_data = json.load(f)
    
    # Process files with indentation errors first
    indentation_files = error_data['categories'].get('indentation', [])
    
    fixed_count = 0
    failed_count = 0
    
    print("Fixing indentation errors...")
    for file_path in indentation_files[:10]:  # Process first 10 files
        if Path(file_path).exists():
            if process_file(file_path):
                fixed_count += 1
            else:
                failed_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nSummary:")
    print(f"Fixed: {fixed_count}")
    print(f"Failed: {failed_count}")

if __name__ == "__main__":
    main()