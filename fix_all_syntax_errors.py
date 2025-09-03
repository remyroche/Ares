#!/usr/bin/env python3
"""
Script to fix all syntax errors in Python files.
"""

import ast
import os
import re
from pathlib import Path
import subprocess
import sys


def check_syntax(file_path):
    """Check if a file has valid Python syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, e
    except Exception as e:
        return False, e


def fix_common_syntax_errors(content):
    """Fix common syntax errors found in the codebase."""
    
    # Fix parenthesized keyword arguments (old Python 2 syntax)
    # Pattern: func(param=(value))
    content = re.sub(r'(\w+)\s*=\s*\(([^)]+)\)', r'\1=\2', content)
    
    # Fix re.search with incorrect keyword argument syntax
    content = re.sub(r're\.search\(pattern=([^,\)]+)\)', r're.search(\1', content)
    content = re.sub(r're\.search\(([^,]+),\s*line=([^,\)]+)\)', r're.search(\1, \2', content)
    
    # Fix malformed regex patterns with unclosed quotes
    content = re.sub(r"r'([^']*)'([^']*)'", r"r'\1\2'", content)
    
    # Fix tuple append with incorrect syntax
    content = re.sub(r'\.append\(\((\w+)=([^)]+)\)\)', r'.append((\1, \2))', content)
    
    # Fix incorrect parameter syntax in function definitions
    # Pattern: def func(param=(default), param2):
    content = re.sub(r'def\s+(\w+)\s*\(([^)]*)\)\s*:', fix_function_params, content)
    
    # Fix broken import statements
    content = re.sub(r"from\s+(\w+)'?\s+import", r'from \1 import', content)
    content = re.sub(r"import\s+(\w+)'?\s*$", r'import \1', content, flags=re.MULTILINE)
    
    # Fix line continuation issues
    content = re.sub(r'\\s*\n\s*', ' ', content)
    
    return content


def fix_function_params(match):
    """Fix function parameter definitions."""
    func_name = match.group(1)
    params = match.group(2)
    
    # Fix parameters with defaults that come after parameters without defaults
    param_list = []
    for param in params.split(','):
        param = param.strip()
        if '=' in param and param.count('=') == 1:
            name, default = param.split('=')
            param_list.append(f"{name.strip()}={default.strip()}")
        else:
            param_list.append(param)
    
    # Ensure parameters with defaults come after those without
    no_default = [p for p in param_list if '=' not in p]
    with_default = [p for p in param_list if '=' in p]
    
    fixed_params = ', '.join(no_default + with_default)
    return f"def {func_name}({fixed_params}):"


def fix_file(file_path):
    """Fix syntax errors in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply fixes
        fixed_content = fix_common_syntax_errors(content)
        
        # Only write if content changed
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            # Verify the fix
            is_valid, error = check_syntax(file_path)
            if is_valid:
                print(f"✅ Fixed: {file_path}")
                return True
            else:
                # Revert if still has errors
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"❌ Could not fix: {file_path} - {error}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False


def main():
    """Fix syntax errors in all Python files."""
    # Get list of files with syntax errors from ruff
    try:
        cmd = ["ruff", "check", ".", "--output-format=concise"]
        env = os.environ.copy()
        env["PATH"] = f"{os.path.expanduser('~')}/.local/bin:" + env.get("PATH", "")
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        # Extract files with syntax errors
        files_with_errors = set()
        for line in result.stdout.splitlines():
            if "invalid-syntax" in line:
                file_path = line.split(':')[0]
                if file_path.endswith('.py'):
                    files_with_errors.add(file_path)
        
        print(f"Found {len(files_with_errors)} files with syntax errors")
        
        # Fix each file
        fixed_count = 0
        for file_path in sorted(files_with_errors):
            if os.path.exists(file_path):
                if fix_file(file_path):
                    fixed_count += 1
        
        print(f"\n✅ Fixed {fixed_count}/{len(files_with_errors)} files")
        
        # Run ruff again to check remaining issues
        print("\nRunning ruff again to check remaining syntax errors...")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        remaining_syntax_errors = sum(1 for line in result.stdout.splitlines() if "invalid-syntax" in line)
        print(f"Remaining syntax errors: {remaining_syntax_errors}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()