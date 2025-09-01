#!/usr/bin/env python3
"""
Improved syntax fixer for supervisor directory files.
Addresses specific syntax errors like unmatched parentheses and missing except blocks.
"""

import os
import re
from pathlib import Path

def fix_unmatched_parentheses(content):
    """Fix unmatched parentheses in import statements."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Fix unmatched parentheses in imports
        if 'from ' in line and 'import (' in line:
            # Count opening and closing parentheses
            open_count = line.count('(')
            close_count = line.count(')')
            
            if open_count > close_count:
                # Add missing closing parentheses
                missing = open_count - close_count
                line += ')' * missing
        
        # Fix multi-line imports with unmatched parentheses
        if line.strip().startswith('from ') and '(' in line and ')' not in line:
            # Look for the closing parenthesis in subsequent lines
            j = i + 1
            while j < len(lines) and ')' not in lines[j]:
                j += 1
            
            if j < len(lines):
                # Found the closing parenthesis, reconstruct the import
                import_parts = []
                for k in range(i, j + 1):
                    import_parts.append(lines[k].strip())
                
                # Join the import parts properly
                fixed_import = ' '.join(import_parts)
                fixed_lines.append(fixed_import)
                i = j
                continue
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_missing_except_blocks(content):
    """Fix missing except blocks after try statements."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for try statements without except blocks
        if 'try:' in line:
            # Look ahead for except block
            has_except = False
            j = i + 1
            while j < len(lines) and j < i + 20:  # Look up to 20 lines ahead
                if 'except' in lines[j]:
                    has_except = True
                    break
                elif lines[j].strip() and not lines[j].strip().startswith('    '):
                    # Found non-indented line, probably end of try block
                    break
                j += 1
            
            if not has_except:
                # Add a basic except block
                fixed_lines.append(line)
                # Add indented content if there's any
                j = i + 1
                while j < len(lines) and lines[j].strip().startswith('    '):
                    fixed_lines.append(lines[j])
                    j += 1
                
                # Add except block
                fixed_lines.append('    except Exception as e:')
                fixed_lines.append('        pass  # TODO: Add proper exception handling')
                i = j - 1  # Skip the lines we've already processed
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
        
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_file_v2(filepath):
    """Fix syntax errors in a single file with improved logic."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply fixes in order
        content = fix_unmatched_parentheses(content)
        content = fix_missing_except_blocks(content)
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
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
        if fix_file_v2(filepath):
            fixed_count += 1
            print(f"✅ Fixed {filepath}")
        else:
            print(f"❌ Failed to fix {filepath}")
    
    print(f"\nFixed {fixed_count} out of {len(python_files)} files")

if __name__ == '__main__':
    main()