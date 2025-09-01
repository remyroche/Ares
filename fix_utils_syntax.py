#!/usr/bin/env python3
"""
Script to fix common syntax issues in src/utils/ files
"""

import os
import re
import glob

def fix_common_issues(content):
    """Fix common syntax issues in Python files."""

    # Fix malformed import statements
    content = re.sub(r'from typing import Any, import (\w+)', r'from typing import Any\nimport \1', content)
    content = re.sub(r'from (\w+) import (\w+), import (\w+)', r'from \1 import \2\nimport \3', content)

    # Fix malformed try/except blocks
    content = re.sub(r'try:\s*\n\s*pass\s*\nexcept Exception as e:\s*\n\s*pass\s*\n\s*pass\s*\nexcept Exception as e:\s*\n\s*pass\s*\n\s*pass\s*\nexcept Exception as e:\s*\n\s*pass\s*\n', '', content)

    # Fix incorrect assignment operators
    content = re.sub(r'(\w+) = (\w+), (\w+)', r'\1, \2, \3', content)
    content = re.sub(r'return (\w+) = (\w+)', r'return \1, \2', content)

    # Fix incorrect function signatures
    content = re.sub(r'def (\w+)\(\s*self = (\w+):', r'def \1(self, \2):', content)
    content = re.sub(r'(\w+): (\w+) = (\w+): (\w+)', r'\1: \2, \3: \4', content)

    # Fix incorrect parameter assignments
    content = re.sub(r'(\w+) = (\w+): (\w+)', r'\1: \2, \3', content)

    # Fix incorrect string formatting
    content = re.sub(r'(\w+) = (\w+)', r'\1, \2', content)

    # Fix incorrect parentheses
    content = re.sub(r'(\w+) = (\w+)\)', r'\1, \2)', content)

    # Fix incorrect commas
    content = re.sub(r',\s*,', ',', content)

    # Fix indentation issues - remove leading spaces before imports
    lines = content.split('\n')
    fixed_lines = []
    for line in lines:
        # Remove leading spaces before import statements
        if re.match(r'^\s+import\s+', line) or re.match(r'^\s+from\s+', line):
            line = line.lstrip()
        # Remove leading spaces before function definitions
        elif re.match(r'^\s+def\s+', line):
            line = line.lstrip()
        # Remove leading spaces before class definitions
        elif re.match(r'^\s+class\s+', line):
            line = line.lstrip()
        fixed_lines.append(line)

    content = '\n'.join(fixed_lines)

    # Fix specific import patterns
    content = re.sub(r'from src\.utils\.warning_symbols import warning, as , _warn_symbol, import h5py',
                    r'from src.utils.warning_symbols import warning, _warn_symbol\nimport h5py', content)

    content = re.sub(r'from src\.utils\.error_handler import \(from src\.utils\.warning_symbols, import \(handle_errors\)',
                    r'from src.utils.error_handler import handle_errors', content)

    # Fix specific function call patterns
    content = re.sub(r'(\w+)\((\w+) = (\w+), (\w+)\)', r'\1(\2, \3, \4)', content)

    # Fix specific assignment patterns
    content = re.sub(r'(\w+) = (\w+) = (\w+)', r'\1 = \2, \3', content)

    # Fix specific return patterns
    content = re.sub(r'return (\w+) = (\w+) = (\w+)', r'return \1, \2, \3', content)

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
        content = fix_common_issues(content)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
        else:
            print(f"No changes needed: {filepath}")
            return False

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Main function to fix all utils files."""
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
