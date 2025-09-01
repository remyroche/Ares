#!/usr/bin/env python3
"""
Comprehensive script to fix remaining syntax issues in the 4 problematic files.
This script addresses the complex structural issues that couldn't be fixed with simple regex replacements.
"""

import re
import os
from pathlib import Path

def fix_state_manager():
    """Fix src/utils/state_manager.py"""
    file_path = "src/utils/state_manager.py"

    with open(file_path, 'r') as f:
        content = f.read()

    # Fix variable assignments
    content = re.sub(r'(\w+), (\w+)', r'\1 = \2', content)

    # Fix function parameter syntax
    content = re.sub(r'(\w+): (\w+), (\w+)', r'\1: \2 = \3', content)

    # Fix decorator parameters
    content = re.sub(r'default_return, (\w+)', r'default_return=\1', content)

    # Fix indentation in try blocks
    lines = content.split('\n')
    fixed_lines = []
    in_try_block = False
    try_indent = 0

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('try:'):
            in_try_block = True
            try_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
        elif in_try_block and stripped.startswith('except'):
            in_try_block = False
            fixed_lines.append(line)
        elif in_try_block and stripped and not stripped.startswith('#'):
            # Fix indentation for try block content
            if not line.startswith(' ' * (try_indent + 4)):
                line = ' ' * (try_indent + 4) + stripped
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    content = '\n'.join(fixed_lines)

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✅ Fixed {file_path}")

def fix_model_manager():
    """Fix src/utils/model_manager.py"""
    file_path = "src/utils/model_manager.py"

    with open(file_path, 'r') as f:
        content = f.read()

    # Fix variable assignments
    content = re.sub(r'(\w+), (\w+)', r'\1 = \2', content)

    # Fix function parameter syntax
    content = re.sub(r'(\w+): (\w+), (\w+)', r'\1: \2 = \3', content)

    # Fix getattr calls
    content = re.sub(r'getattr\((\w+) = (\w+)', r'getattr(\1, \2', content)

    # Fix indentation issues
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        stripped = line.strip()

        # Fix excessive indentation
        if line.count(' ') > 40:
            line = ' ' * 8 + stripped

        fixed_lines.append(line)

    content = '\n'.join(fixed_lines)

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✅ Fixed {file_path}")

def fix_config_loader():
    """Fix src/utils/config_loader.py"""
    file_path = "src/utils/config_loader.py"

    with open(file_path, 'r') as f:
        content = f.read()

    # Fix function parameter syntax
    content = re.sub(r'(\w+): (\w+), (\w+)', r'\1: \2 = \3', content)

    # Fix decorator parameters
    content = re.sub(r'default_return, (\w+)', r'default_return=\1', content)

    # Fix function indentation inside classes
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = 0

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('class '):
            in_class = True
            class_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
        elif in_class and stripped.startswith('def ') and not line.startswith(' ' * (class_indent + 4)):
            # Fix function indentation inside class
            line = ' ' * (class_indent + 4) + stripped
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    content = '\n'.join(fixed_lines)

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✅ Fixed {file_path}")

def fix_async_utils():
    """Fix src/utils/async_utils.py"""
    file_path = "src/utils/async_utils.py"

    with open(file_path, 'r') as f:
        content = f.read()

    # Fix variable assignments
    content = re.sub(r'(\w+), (\w+)', r'\1 = \2', content)

    # Fix decorator parameters
    content = re.sub(r'default_return, (\w+)', r'default_return=\1', content)

    # Fix function indentation inside classes
    lines = content.split('\n')
    fixed_lines = []
    in_class = False
    class_indent = 0

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('class '):
            in_class = True
            class_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
        elif in_class and stripped.startswith('def ') and not line.startswith(' ' * (class_indent + 4)):
            # Fix function indentation inside class
            line = ' ' * (class_indent + 4) + stripped
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    content = '\n'.join(fixed_lines)

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✅ Fixed {file_path}")

def main():
    """Main function to fix all remaining files"""
    print("🔧 Fixing remaining syntax issues...")

    # Fix each file
    fix_state_manager()
    fix_model_manager()
    fix_config_loader()
    fix_async_utils()

    print("✅ All files processed!")

if __name__ == "__main__":
    main()
