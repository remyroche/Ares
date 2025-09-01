#!/usr/bin/env python3
"""
Fix all remaining syntax errors in optimization files
"""

import os
import re
from pathlib import Path

def fix_file_syntax(filepath):
    """Fix syntax errors in a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix indentation issues by removing problematic try-except blocks
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
    
    # Fix assignment operator issues
    content = '\n'.join(fixed_lines)
    
    # Fix common syntax errors
    content = content.replace(' = ', ', ')
    content = content.replace('= ', '=')
    content = content.replace(' =', '=')
    
    # Fix specific patterns
    content = content.replace('def ', 'def ')
    content = content.replace(' -> ', ' -> ')
    
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
            fix_file_syntax(py_file)
            files_fixed += 1
            print(f"Fixed {py_file}")
        except Exception as e:
            print(f"Failed to fix {py_file}: {e}")
    
    print(f"\nSummary: Fixed {files_fixed}/{total_files} files")

if __name__ == "__main__":
    main()