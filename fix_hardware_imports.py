#!/usr/bin/env python3
"""
Script to fix hardware import paths in ml_common module.

The issue is that files in src/utils/ml_common/ are trying to import from '..hardware'
but should import from 'src.utils.hardware' instead.
"""

import os
import re
import glob

def fix_hardware_imports():
    """Fix all hardware import paths in the ml_common module."""

    # Pattern to match the incorrect imports
    pattern = r'from \.\.hardware\.([a-zA-Z_][a-zA-Z0-9_]*)\s+import\s+(.+)'

    # Replacement template
    replacement = r'from src.utils.hardware.\1 import \2'

    # Find all Python files in ml_common directory
    ml_common_dir = '/Users/remyroche/Documents/Ares/src/utils/ml_common'
    python_files = glob.glob(os.path.join(ml_common_dir, '**', '*.py'), recursive=True)

    fixed_files = []

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all matches
            matches = re.findall(pattern, content)
            if matches:
                print(f"Fixing {len(matches)} import(s) in {file_path}")

                # Apply replacements
                new_content = re.sub(pattern, replacement, content)

                # Write back if changed
                if new_content != content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    fixed_files.append(file_path)
                    print(f"  ✅ Fixed imports in {os.path.basename(file_path)}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return fixed_files

if __name__ == "__main__":
    print("🔧 Fixing hardware import paths in ml_common module...")
    fixed_files = fix_hardware_imports()
    print(f"\n✅ Fixed imports in {len(fixed_files)} files:")
    for file in fixed_files:
        print(f"  - {os.path.basename(file)}")
