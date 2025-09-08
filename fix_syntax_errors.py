#!/usr/bin/env python3
"""
Comprehensive syntax error fixer for the Ares project.
Fixes common try-except indentation issues across the codebase.
"""

import os
import re
import sys
from pathlib import Path

def fix_try_except_indentation(content):
    """Fix try-except blocks with improper indentation."""

    # Pattern to find try blocks with improperly indented import statements
    pattern = r'(try:\s*\n(?:\s+.*?\n)*?)(\n\s*)(import\s+.*?)(?=\n\s*(?:\n\s*)?except|\n\s*(?:\n\s*)?finally|\n\s*(?:\n\s*)?\Z)'

    def replace_func(match):
        try_block = match.group(1)
        spacing = match.group(2)
        import_stmt = match.group(3)

        # Get the indentation level of the try block
        try_lines = try_block.strip().split('\n')
        if try_lines:
            try_indent = len(try_lines[0]) - len(try_lines[0].lstrip())
            # Indent the import statement to match the try block content
            indented_import = ' ' * (try_indent + 4) + import_stmt.lstrip()
            return try_block + spacing + indented_import

        return match.group(0)

    return re.sub(pattern, replace_func, content, flags=re.MULTILINE | re.DOTALL)

def fix_malformed_imports(content):
    """Fix malformed import blocks where imports are mixed with from statements."""

    # Pattern to find malformed from-import blocks
    pattern = r'from\s+.*?\s+import\s*\(\s*\n(?:\s+.*?\n)*?\s*(import\s+.*?)(?:\n\s*(?:import\s+.*?))*\n(?:\s+.*?\n)*?\s*\)'

    def replace_func(match):
        full_match = match.group(0)
        lines = full_match.split('\n')

        # Separate regular imports from from-imports
        regular_imports = []
        from_imports = []

        for line in lines:
            line = line.strip()
            if line.startswith('import ') and not line.startswith('from '):
                regular_imports.append(line)
            elif line.startswith('from ') or (line.strip() and not line.startswith('import ') and line != '(' and line != ')'):
                from_imports.append(line)

        # Reconstruct the block
        result = []
        for imp in regular_imports:
            result.append(' ' * 4 + imp)
        result.append('')
        for imp in from_imports:
            if imp:
                result.append(' ' * 4 + imp)

        return '\n'.join(result)

    return re.sub(pattern, replace_func, content, flags=re.MULTILINE | re.DOTALL)

def fix_file(filepath):
    """Fix syntax errors in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply fixes
        content = fix_try_except_indentation(content)
        content = fix_malformed_imports(content)

        # Write back if changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True

        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Main function to fix syntax errors across the codebase."""
    if len(sys.argv) != 2:
        print("Usage: python fix_syntax_errors.py <directory>")
        sys.exit(1)

    directory = Path(sys.argv[1])
    if not directory.exists():
        print(f"Directory {directory} does not exist")
        sys.exit(1)

    fixed_count = 0

    # Find all Python files
    for py_file in directory.rglob('*.py'):
        if fix_file(py_file):
            fixed_count += 1

    print(f"\nFixed {fixed_count} files with syntax errors")

    # Run syntax check on all files
    print("\nRunning syntax check...")
    syntax_errors = []
    for py_file in directory.rglob('*.py'):
        try:
            compile(open(py_file, 'r').read(), str(py_file), 'exec')
        except SyntaxError as e:
            syntax_errors.append(f"{py_file}: {e}")

    if syntax_errors:
        print(f"\nRemaining syntax errors ({len(syntax_errors)}):")
        for error in syntax_errors[:10]:  # Show first 10
            print(f"  {error}")
        if len(syntax_errors) > 10:
            print(f"  ... and {len(syntax_errors) - 10} more")
    else:
        print("\nNo syntax errors remaining!")

if __name__ == "__main__":
    main()