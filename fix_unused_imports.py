#!/usr/bin/env python3
"""
Simple script to remove unused imports from specific files
"""

import ast
import sys
from pathlib import Path


def find_unused_imports(filepath):
    """Find unused imports in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)
        imports = []
        used_names = set()

        # Collect all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'name': alias.name,
                        'asname': alias.asname,
                        'lineno': node.lineno,
                        'used_name': alias.asname or alias.name.split('.')[0]
                    })
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': node.module,
                        'name': alias.name,
                        'asname': alias.asname,
                        'lineno': node.lineno,
                        'used_name': alias.asname or alias.name
                    })

        # Collect all used names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)

        # Check for string usage (some imports used in strings)
        for line in content.split('\n'):
            for imp in imports:
                if f"'{imp['used_name']}'" in line or f'"{imp["used_name"]}"' in line:
                    used_names.add(imp['used_name'])

        # Find unused imports
        unused = []
        for imp in imports:
            if imp['used_name'] not in used_names and imp['name'] != '*':
                unused.append(imp)

        return unused, content.split('\n')

    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return [], []


def remove_unused_imports(filepath, dry_run=True):
    """Remove unused imports from a file."""
    unused_imports, lines = find_unused_imports(filepath)

    if not unused_imports:
        return False

    lines_to_remove = set()
    for imp in unused_imports:
        lines_to_remove.add(imp['lineno'] - 1)  # Convert to 0-based indexing

    if dry_run:
        print(f"\n{filepath}:")
        for line_idx in sorted(lines_to_remove):
            if line_idx < len(lines):
                print(f"  Would remove line {line_idx + 1}: {lines[line_idx].strip()}")
        return bool(lines_to_remove)

    # Actually remove the lines
    for line_idx in sorted(lines_to_remove, reverse=True):
        if line_idx < len(lines):
            print(f"Removing line {line_idx + 1}: {lines[line_idx].strip()}")
            lines.pop(line_idx)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return True


# List of files to fix (valid files only)
files_to_fix = [
    './test_multi_output_final_integration.py',
    './test_steps_1_7_compatibility.py',
    './test_sr_training_integration.py',
    './fix_metadata_and_naming.py',
    './complete_remaining_steps_integration.py',
    './test_sr_optimization_integration.py',
    './test_tactician_multi_outcome_predictions_updated.py',
    './test_enhanced_dynamic_feature_selection.py',
    './targeted_fix.py'
]

if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv

    print(f"{'DRY RUN: ' if dry_run else ''}Fixing unused imports...")

    for filepath in files_to_fix:
        if Path(filepath).exists():
            remove_unused_imports(filepath, dry_run=dry_run)
        else:
            print(f"File not found: {filepath}")