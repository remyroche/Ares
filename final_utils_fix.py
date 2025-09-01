#!/usr/bin/env python3
"""
Final script to fix remaining indentation and syntax issues in src/utils/ files
"""

import os
import re
import glob

def fix_indentation_and_syntax(content):
    """Fix indentation and syntax issues."""

    # Fix malformed try/except blocks
    content = re.sub(r'try:\s*\n\s*pass\s*\n', r'try:\n    pass\n', content)

    # Fix import statements that are not properly indented
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Fix import statements that should be inside try blocks
        if (line.strip().startswith('import ') or line.strip().startswith('from ')) and i > 0:
            prev_line = lines[i-1].strip()
            if prev_line == 'try:':
                # This import should be indented inside the try block
                line = '    ' + line

        # Fix function definitions that are not properly indented
        if line.strip().startswith('def ') and not line.startswith('    '):
            # Check if this should be indented (inside a class or function)
            if i > 0 and (lines[i-1].strip().endswith(':') or lines[i-1].strip().startswith('class ')):
                line = '    ' + line

        # Fix class definitions that are not properly indented
        if line.strip().startswith('class ') and not line.startswith('    '):
            # Check if this should be indented
            if i > 0 and lines[i-1].strip().endswith(':'):
                line = '    ' + line

        # Fix variable assignments that are not properly indented
        if '=' in line and not line.startswith('    ') and i > 0:
            prev_line = lines[i-1].strip()
            if prev_line.endswith(':') or prev_line.startswith('def ') or prev_line.startswith('class '):
                line = '    ' + line

        fixed_lines.append(line)
        i += 1

    content = '\n'.join(fixed_lines)

    # Fix specific syntax patterns
    content = re.sub(r'(\w+): dict\[str = Any\]', r'\1: dict[str, Any]', content)
    content = re.sub(r'(\w+): dict\[str , Any\]', r'\1: dict[str, Any]', content)
    content = re.sub(r'(\w+): (\w+) = (\w+)', r'\1: \2 = \3', content)
    content = re.sub(r'(\w+), (\w+)', r'\1 = \2', content)

    # Fix function parameter issues
    content = re.sub(r'def (\w+)\(self = (\w+):', r'def \1(self, \2):', content)
    content = re.sub(r'def (\w+)\((\w+): (\w+)\):', r'def \1(\2: \3):', content)

    # Fix import statement issues
    content = re.sub(r'from src\.utils\.error_handler import (\w+)\s*\n\s*(\w+)', r'from src.utils.error_handler import \1, \2', content)

    return content

def fix_file(filepath):
    """Fix a single file."""
    try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        content = fix_indentation_and_syntax(content)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
        else:
            return False

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Main function to fix remaining issues."""
    utils_dir = "src/utils"
    py_files = glob.glob(os.path.join(utils_dir, "*.py"))

    fixed_count = 0
    total_count = len(py_files)

    for filepath in py_files:
        if fix_file(filepath):
            fixed_count += 1

    print(f"\nFixed {fixed_count} out of {total_count} files")

if __name__ == "__main__":
    main()
