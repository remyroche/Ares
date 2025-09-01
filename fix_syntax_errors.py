#!/usr/bin/env python3
"""
Script to fix common syntax errors in Python files.
"""

import os
import re
import ast

def fix_common_syntax_errors(filepath):
    """Fix common syntax errors in a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Fix 1: Add missing pass statements after empty blocks
            if line.strip().endswith(':') and i + 1 < len(lines):
                next_line = lines[i + 1]
                if not next_line.strip() or next_line.strip().startswith('#'):
                    # Add pass statement
                    fixed_lines.append(line)
                    fixed_lines.append('    pass')
                    i += 1
                    continue
            
            # Fix 2: Fix indentation issues
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                # Check if this should be indented
                if i > 0 and lines[i-1].strip().endswith(':'):
                    # This line should be indented
                    line = '    ' + line
            
            # Fix 3: Fix missing except/finally blocks
            if line.strip().startswith('try:') and i + 1 < len(lines):
                # Look for the next non-empty line
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                
                if j < len(lines) and not lines[j].strip().startswith(('except', 'finally', '#')):
                    # Add except block
                    fixed_lines.append(line)
                    fixed_lines.append('    pass')
                    fixed_lines.append('except Exception as e:')
                    fixed_lines.append('    pass')
                    i += 1
                    continue
            
            # Fix 4: Fix parameter order issues
            if 'def ' in line and '=' in line:
                # Check for parameters without defaults after parameters with defaults
                if re.search(r'[^=,]+=[^,]+,[^=]+[^,]*\)', line):
                    # This is a complex fix that would require parsing
                    # For now, just add a comment
                    line = line + '  # FIXME: Parameter order issue'
            
            # Fix 5: Fix invalid decimal literals
            if re.search(r'\b\d+\.\d+\.\d+\b', line):
                # Fix invalid decimal literals like 1.2.3
                line = re.sub(r'\b(\d+\.\d+)\.(\d+)\b', r'\1_2', line)
            
            # Fix 6: Fix unmatched parentheses
            open_parens = line.count('(')
            close_parens = line.count(')')
            if open_parens > close_parens:
                # Add missing closing parentheses
                line = line + ')' * (open_parens - close_parens)
            
            fixed_lines.append(line)
            i += 1
        
        fixed_content = '\n'.join(fixed_lines)
        
        # Test if the fixed content is valid Python
        try:
            ast.parse(fixed_content)
            if fixed_content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"Fixed syntax errors in {filepath}")
                return True
        except SyntaxError:
            print(f"Could not fix all syntax errors in {filepath}")
            return False
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def process_directory(directory):
    """Process all Python files in a directory."""
    fixed_count = 0
    total_count = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'test_results', 'test_models', 'log']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                total_count += 1
                
                try:
                    # Quick syntax check
                    with open(filepath, 'r', encoding='utf-8') as f:
                        ast.parse(f.read())
                    # File is already valid
                except SyntaxError:
                    # File has syntax errors, try to fix them
                    if fix_common_syntax_errors(filepath):
                        fixed_count += 1
    
    print(f"\nFixed {fixed_count} out of {total_count} Python files")

if __name__ == "__main__":
    # Process the current directory
    process_directory('.')
