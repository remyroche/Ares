#!/usr/bin/env python3
"""
Fix empty try blocks in optimization files
"""

import os
import re
from pathlib import Path

def fix_empty_try_blocks(filepath):
    """Fix empty try blocks in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this line has a try: statement
            if line.strip() == 'try:':
                # Look ahead to see if the next line is an except
                if i + 1 < len(lines) and lines[i + 1].strip().startswith('except'):
                    # This is an empty try block, remove both try and except
                    i += 2
                    continue
                else:
                    # Keep the try and continue
                    fixed_lines.append(line)
                    i += 1
            else:
                fixed_lines.append(line)
                i += 1
        
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
        
        if fix_empty_try_blocks(py_file):
            files_fixed += 1
            print(f"Fixed {py_file}")
        else:
            print(f"Failed to fix {py_file}")
    
    print(f"\nSummary: Fixed {files_fixed}/{total_files} files")

if __name__ == "__main__":
    main()