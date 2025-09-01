#!/usr/bin/env python3
"""
Fix indentation errors in optimization files - Version 2
"""

import os
import re
from pathlib import Path

def fix_indentation_errors(filepath):
    """Fix indentation errors in a file."""
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
                fixed_lines.append(line)
                i += 1
                
                # Skip the problematic pass statements and except blocks
                while i < len(lines):
                    current_line = lines[i].strip()
                    
                    # Skip pass statements in try block
                    if current_line == 'pass  # TODO: Add proper exception handling':
                        i += 1
                        continue
                    
                    # Skip except statements with pass
                    if current_line.startswith('except') and i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line == 'pass  # TODO: Add proper exception handling':
                            # Skip both the except and the pass
                            i += 2
                            continue
                        else:
                            # Keep the except but skip the pass
                            fixed_lines.append(lines[i])
                            i += 1
                            if i < len(lines) and lines[i].strip() == 'pass  # TODO: Add proper exception handling':
                                i += 1
                            break
                    
                    # If we reach here, it's not a problematic pattern, keep the line
                    fixed_lines.append(lines[i])
                    i += 1
                    break
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
        
        if fix_indentation_errors(py_file):
            files_fixed += 1
            print(f"Fixed {py_file}")
        else:
            print(f"Failed to fix {py_file}")
    
    print(f"\nSummary: Fixed {files_fixed}/{total_files} files")

if __name__ == "__main__":
    main()