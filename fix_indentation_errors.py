#!/usr/bin/env python3
"""
Fix indentation errors in optimization files
"""

import os
import re
from pathlib import Path

def fix_indentation_errors(filepath):
    """Fix indentation errors in a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the specific pattern: try: followed by unindented pass statements
    # Pattern: try:\n    pass  # TODO: Add proper exception handling\nexcept Exception as e:\n    pass  # TODO: Add proper exception handling
    content = re.sub(
        r'try:\n\s*pass  # TODO: Add proper exception handling\nexcept Exception as e:\n\s*pass  # TODO: Add proper exception handling',
        'try:',
        content
    )
    
    # Fix the specific pattern: try: followed by unindented pass statements (different spacing)
    content = re.sub(
        r'try:\n\s+pass  # TODO: Add proper exception handling\nexcept Exception as e:\n\s+pass  # TODO: Add proper exception handling',
        'try:',
        content
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
            fix_indentation_errors(py_file)
            files_fixed += 1
            print(f"Fixed {py_file}")
        except Exception as e:
            print(f"Failed to fix {py_file}: {e}")
    
    print(f"\nSummary: Fixed {files_fixed}/{total_files} files")

if __name__ == "__main__":
    main()