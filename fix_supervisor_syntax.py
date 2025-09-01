#!/usr/bin/env python3
"""
Comprehensive syntax fixer for supervisor directory files.
This script addresses common syntax errors found in the supervisor files.
"""

import os
import re
from pathlib import Path

def fix_common_syntax_errors(content):
    """Fix common syntax errors in Python code."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Fix 1: Remove duplicate function definitions
        if line.strip().startswith('def ') and i + 1 < len(lines):
            next_line = lines[i + 1]
            if next_line.strip().startswith('def ') and 'def ' in next_line:
                # Skip duplicate def lines
                while i < len(lines) and lines[i].strip().startswith('def '):
                    i += 1
                continue
        
        # Fix 2: Fix indentation for class methods
        if line.strip().startswith('def ') and not line.startswith('    '):
            # This should be indented if it's inside a class
            if any('class ' in prev_line for prev_line in fixed_lines[-10:] if prev_line.strip()):
                line = '    ' + line
        
        # Fix 3: Fix missing try/except blocks
        if 'try:' in line and i + 1 < len(lines):
            next_line = lines[i + 1]
            if not next_line.strip().startswith('    ') and not next_line.strip().startswith('except'):
                # Add proper indentation
                fixed_lines.append(line)
                i += 1
                continue
        
        # Fix 4: Fix unindented code blocks
        if line.strip() and not line.startswith('    ') and not line.startswith('\t'):
            # Check if previous line ends with ':'
            if fixed_lines and fixed_lines[-1].strip().endswith(':'):
                line = '    ' + line
        
        # Fix 5: Fix missing except blocks
        if 'try:' in line and i + 1 < len(lines):
            # Look ahead to see if there's an except block
            has_except = False
            for j in range(i + 1, min(i + 10, len(lines))):
                if 'except' in lines[j]:
                    has_except = True
                    break
                elif lines[j].strip() and not lines[j].strip().startswith('    '):
                    break
            
            if not has_except:
                fixed_lines.append(line)
                fixed_lines.append('    pass  # TODO: Add proper exception handling')
                i += 1
                continue
        
        # Fix 6: Fix invalid import statements
        if line.strip().startswith('from ') and 'import (' in line:
            # Fix multi-line imports
            import_lines = [line]
            j = i + 1
            while j < len(lines) and ')' not in lines[j]:
                import_lines.append(lines[j])
                j += 1
            if j < len(lines):
                import_lines.append(lines[j])
            
            # Reconstruct the import
            fixed_import = 'from ' + line.split('from ')[1].split(' import')[0] + ' import ('
            for imp_line in import_lines[1:]:
                if ')' in imp_line:
                    fixed_import += imp_line.strip()
                else:
                    fixed_import += imp_line.strip() + ', '
            
            fixed_lines.append(fixed_import)
            i = j
            continue
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_file(filepath):
    """Fix syntax errors in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply fixes
        fixed_content = fix_common_syntax_errors(content)
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        return True
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Main function to fix all supervisor files."""
    supervisor_dir = Path('src/supervisor')
    
    if not supervisor_dir.exists():
        print("Supervisor directory not found!")
        return
    
    python_files = list(supervisor_dir.glob('*.py'))
    print(f"Found {len(python_files)} Python files to fix")
    
    fixed_count = 0
    for filepath in python_files:
        print(f"Fixing {filepath}...")
        if fix_file(filepath):
            fixed_count += 1
            print(f"✅ Fixed {filepath}")
        else:
            print(f"❌ Failed to fix {filepath}")
    
    print(f"\nFixed {fixed_count} out of {len(python_files)} files")

if __name__ == '__main__':
    main()