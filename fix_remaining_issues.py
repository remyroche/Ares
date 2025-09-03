#!/usr/bin/env python3
"""
Script to fix remaining specific syntax issues in src/utils/ files
"""

import glob
import os
import re


def fix_specific_issues(content):
    """Fix specific syntax issues that remain."""

    # Fix function parameter syntax errors
    content = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\|\s*None\s*,\s*None,", r"\\1: \\2 | None=None,", content)
    content = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)", r"\\1: \\2=\\3", content)
    content = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*False\)", r"\\1: \\2, \\3, bold: bool=False)", content)

    # Fix unmatched parentheses
    content = re.sub(r"handle_file_operations\'\)", r"handle_file_operations", content)
    content = re.sub(r"missing\'\)", r"missing", content)

    # Fix specific import issues
    content = re.sub(r"from\s+src\.utils\.error_handler\s+import\s+\(handle_file_operations\'\)",
                    r"from src.utils.error_handler import handle_file_operations", content)

    # Fix indentation issues by ensuring proper structure
    lines = content.split("\n")
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Fix specific indentation issues
        if "def colorize(" in line and "False)" in line:
            # Fix the colorize function signature
            line = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*False\)", r"\\1: \\2, \\3, bold: bool=False)", line)

        # Fix if statement indentation
        if line.strip().startswith("if ") and i + 1 < len(lines):
            next_line = lines[i + 1]
            if not next_line.strip():
                # Ensure at least a pass after if
                fixed_lines.append(line)
                fixed_lines.append("    pass")
                i += 2
                continue

        fixed_lines.append(line)
        i += 1

    return "\n".join(fixed_lines)

def fix_file(filepath):
    """Fix a single file."""
    try:
        with open(filepath, encoding="utf-8") as f:
            content=f.read()

        original_content=content
        content = fix_specific_issues(content)

        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
        return False

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Main function to fix remaining issues."""
    utils_dir="src/utils"
    py_files = glob.glob(os.path.join(utils_dir, "*.py"))

    fixed_count=0
    total_count = len(py_files)

    for filepath in py_files:
        if fix_file(filepath):
            fixed_count += 1

    print(f"\nFixed {fixed_count} out of {total_count} files")

if __name__== "__main__":
    main()
