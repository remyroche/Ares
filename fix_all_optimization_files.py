#!/usr/bin/env python3
"""
Fix indentation in all optimization files
"""

import os
from pathlib import Path

def fix_file(filepath):
    """Fix the indentation issues in a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the problematic patterns
    # Pattern 1: Fix try-except blocks with pass statements
    content = content.replace(
        '        try:\n    pass  # TODO: Add proper exception handling\nexcept Exception as e:\n    pass  # TODO: Add proper exception handling',
        '        try:'
    )
    
    # Pattern 2: Fix try-except blocks with pass statements (different indentation)
    content = content.replace(
        '        try:\n            pass  # TODO: Add proper exception handling\n        except Exception as e:\n            pass  # TODO: Add proper exception handling',
        '        try:'
    )
    
    # Pattern 3: Fix try-except blocks with pass statements (no indentation)
    content = content.replace(
        'try:\n    pass  # TODO: Add proper exception handling\nexcept Exception as e:\n    pass  # TODO: Add proper exception handling',
        'try:'
    )
    
    # Pattern 4: Fix try-except blocks with pass statements (different spacing)
    content = content.replace(
        '        try:\n            pass  # TODO: Add proper exception handling\n        except Exception as e:\n            pass  # TODO: Add proper exception handling',
        '        try:'
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Fix all optimization files."""
    optimization_dir = Path("src/training/optimization")
    
    if not optimization_dir.exists():
        print("Optimization directory not found")
        return
    
    files_fixed = 0
    total_files = 0
    
    for py_file in optimization_dir.glob("*.py"):
        total_files += 1
        print(f"Processing {py_file}...")
        
        try:
            fix_file(py_file)
            files_fixed += 1
            print(f"Fixed {py_file}")
        except Exception as e:
            print(f"Failed to fix {py_file}: {e}")
    
    print(f"\nSummary: Fixed {files_fixed}/{total_files} files")

if __name__ == "__main__":
    main()