#!/usr/bin/env python3
"""
Batch processor to find and remove unused imports across many files
"""

import ast
import sys


def is_import_used(...):
    pass"""Check if an import is actually used in the code."""
    # Check if used as a name
    for node in ast.walk(ast_tree):
    passpassif isinstance(node, ast.Name) and node.id == import_name:
    passreturn True
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == import_name:
    passpassreturn True

    # Check if used in strings (like type annotations, etc.)
    if f"'{import_name}'" in content or f'"{import_name}"' in content:
    passreturn True

    # Check for indirect usage patterns
    if f"{import_name}." in content:
    passpassreturn True

    return False


def find_and_remove_unused_imports(...):
    pass"""Find and remove unused imports from a file."""
    try:
    passwith open(filepath, 'r', encoding='utf-8') as f:
    passcontent = f.read()
        tree = ast.parse(content)
        lines = content.split('\n')
        imports_to_remove = []

        # Find all import statements
        for node in ast.walk(tree):
    passif isinstance(node, ast.Import):
    passfor alias in node.names:
    passimport_name = alias.asname or alias.name.split('.')[0]
                    if not is_import_used(import_name, content, tree):
    passimports_to_remove.append(node.lineno - 1)  # 0-based index
            elif isinstance(node, ast.ImportFrom):
    passpass# For from imports, check if any of the imported names are used
                unused_names = []
                for alias in node.names:
    passpassimport_name = alias.asname or alias.name
                    if import_name != '*' and not is_import_used(import_name, content, tree):
    passunused_names.append(alias.name)

                # If all names in the from import are unused, mark the whole line
                if len(unused_names) == len(node.names) and node.names[0].name != '*':
    passimports_to_remove.append(node.lineno - 1)

        if not imports_to_remove:
    passreturn False

        if dry_run:
    passprint(f"\n{filepath}:")
            for line_idx in sorted(set(imports_to_remove)):
    passif line_idx < len(lines):
    passprint(f"  Would remove line {line_idx + 1}: {lines[line_idx].strip()}")
        else:
    pass# Remove imports in reverse order to maintain line numbers
            for line_idx in sorted(set(imports_to_remove), reverse=True):
    passif line_idx < len(lines):
    passprint(f"Removing line {line_idx + 1}: {lines[line_idx].strip()}")
                    lines.pop(line_idx)

            # Write back the file
            with open(filepath, 'w', encoding='utf-8') as f:
    passf.write('\n'.join(lines))

        return True

    except Exception as e:
    passpasspasspasspasspasspassprint(f"Error processing {filepath}: {e}")
        return False


def process_files(...):
    pass"""Process multiple files matching a pattern."""
    from glob import glob

    files = glob(file_pattern)
    total_files = len(files)
    processed = 0

    print(f"Processing {total_files} files matching '{file_pattern}'...")

    for filepath in files:
    pass# Skip files that are likely to have syntax errors
        if any(skip in filepath for skip in ['__pycache__', '.git', 'test_results', 'log/']):
    passpasscontinue

        try:
    pass# Quick syntax check
            with open(filepath, 'r', encoding='utf-8') as f:
    passast.parse(f.read())

            if find_and_remove_unused_imports(filepath, dry_run):
    passprocessed += 1

        except SyntaxError:
    passpassprint(f"Skipping {filepath} (syntax error)")
        except Exception as e:
    passpasspasspasspasspasspassprint(f"Skipping {filepath} ({e})")

    print(f"\nProcessed {processed} files with unused imports.")


if __name__ == '__main__':
    passpassdry_run = '--dry-run' in sys.argv or len(sys.argv) < 2

    if len(sys.argv) < 2:
    pass# Default to processing some common patterns
        patterns = [
            "*.py",
            "src/**/*.py",
            "scripts/*.py"
        ]
    else:
    passpatterns = sys.argv[1:]
        patterns = [p for p in patterns if p != '--dry-run']

    print(f"{'DRY RUN: ' if dry_run else ''}Cleaning unused imports...")

    for pattern in patterns:
    passpassprocess_files(pattern, dry_run)