#!/usr/bin/env python3
"""
Fix indentation errors by adding pass statements where needed.
"""

import re
from pathlib import Path

def fix_empty_blocks(file_path: Path):
    """Add pass statements to empty if/else/for/while blocks."""
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    fixes_count = 0
    
    while i < len(lines):
        line = lines[i]
        fixed_lines.append(line)
        
        # Check if this is a block start (if/else/elif/for/while/try/except/finally/def/class)
        if re.match(r'^\s*(if|elif|else|for|while|try|except|finally|def|class|with)\s*.*:\s*$', line):
            # Get indentation level
            indent = len(line) - len(line.lstrip())
            
            # Check next line
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                next_indent = len(next_line) - len(next_line.lstrip()) if next_line.strip() else 0
                
                # If next line is not indented more (or is another block at same level), add pass
                if next_line.strip() and next_indent <= indent:
                    # Add pass with proper indentation
                    fixed_lines.append(' ' * (indent + 4) + 'pass\n')
                    fixes_count += 1
                    print(f"Added 'pass' after line {i+1}: {line.strip()}")
                elif not next_line.strip() and i + 2 < len(lines):
                    # Check line after blank line
                    next_next_line = lines[i + 2]
                    next_next_indent = len(next_next_line) - len(next_next_line.lstrip()) if next_next_line.strip() else 0
                    if next_next_line.strip() and next_next_indent <= indent:
                        # Add pass with proper indentation
                        fixed_lines.append(' ' * (indent + 4) + 'pass\n')
                        fixes_count += 1
                        print(f"Added 'pass' after line {i+1}: {line.strip()}")
        
        i += 1
    
    if fixes_count > 0:
        with open(file_path, 'w') as f:
            f.writelines(fixed_lines)
        print(f"\n✅ Fixed {fixes_count} empty blocks")
    else:
        print(f"ℹ️  No empty blocks found")
    
    return fixes_count

if __name__ == '__main__':
    file_path = Path('src/training/steps/pre_training/feature_generation_interaction_generation_step.py')
    fix_empty_blocks(file_path)

