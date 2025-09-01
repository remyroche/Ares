#!/usr/bin/env python3
"""
Remove problematic pass statements from optimization files
"""

import os
import re
from pathlib import Path

def remove_problematic_passes(filepath):
    """Remove problematic pass statements from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove the problematic pass statements
        # Pattern: lines that are just "pass  # TODO: Add proper exception handling"
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped == 'pass  # TODO: Add proper exception handling':
                # Skip this line entirely
                continue
            else:
                fixed_lines.append(line)
        
        # Write the fixed content back
        fixed_content = '\n'.join(fixed_lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        return True
        
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Main function to fix all optimization files."""
    optimization_dir = Path("src/training/optimization")
    
    if not optimization_dir.exists():
        print("Optimization directory not found")
        return
    
    files_fixed = 0
    total_files = 0
    
    for py_file in optimization_dir.glob("*.py"):
        total_files += 1
        print(f"Processing {py_file}...")
        
        if remove_problematic_passes(py_file):
            files_fixed += 1
            print(f"Fixed {py_file}")
        else:
            print(f"Failed to fix {py_file}")
    
    print(f"\nSummary: Fixed {files_fixed}/{total_files} files")

if __name__ == "__main__":
    main()