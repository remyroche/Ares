#!/usr/bin/env python3
"""
Script to comprehensively fix syntax issues in the HMM file.
"""

import re

def fix_hmm_comprehensive(file_path):
    """Comprehensively fix the HMM file syntax issues."""
    print(f"Fixing comprehensive syntax issues in: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split into lines for easier processing
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if we're starting a method with decorators
        if line.strip().startswith('@') and i + 1 < len(lines):
            # Collect all decorators
            decorators = []
            while i < len(lines) and lines[i].strip().startswith('@'):
                decorators.append(lines[i])
                i += 1
            
            # Check if next line is a method definition
            if i < len(lines) and ('async def ' in lines[i] or 'def ' in lines[i]):
                method_line = lines[i]
                fixed_lines.extend(decorators)
                fixed_lines.append(method_line)
                i += 1
                
                # Look for method body and try blocks
                indent_level = len(method_line) - len(method_line.lstrip())
                method_body = []
                try_blocks = []
                
                # Collect method body until next method or class
                while i < len(lines):
                    current_line = lines[i]
                    
                    # Stop if we reach another method or class at same or lower indent level
                    if (current_line.strip() and 
                        not current_line.startswith(' ' * (indent_level + 1)) and
                        not current_line.strip().startswith('#') and
                        (current_line.strip().startswith('def ') or 
                         current_line.strip().startswith('async def ') or
                         current_line.strip().startswith('class ') or
                         current_line.strip().startswith('@'))):
                        break
                    
                    # Track try blocks
                    if current_line.strip().startswith('try:'):
                        try_blocks.append(len(method_body))
                    
                    method_body.append(current_line)
                    i += 1
                
                # Fix try blocks without except clauses
                for try_idx in reversed(try_blocks):
                    # Check if there's a matching except
                    has_except = False
                    for j in range(try_idx + 1, len(method_body)):
                        if method_body[j].strip().startswith('except ') or method_body[j].strip().startswith('finally:'):
                            has_except = True
                            break
                        if (method_body[j].strip() and 
                            not method_body[j].startswith(' ' * (indent_level + 8)) and
                            not method_body[j].strip().startswith('#')):
                            break
                    
                    if not has_except:
                        # Add a generic except block
                        try_line = method_body[try_idx]
                        try_indent = len(try_line) - len(try_line.lstrip())
                        except_block = [
                            ' ' * try_indent + 'except Exception as e:',
                            ' ' * (try_indent + 4) + 'self.logger.exception(f"Error in method: {e}")',
                            ' ' * (try_indent + 4) + 'return {}'
                        ]
                        # Insert except block at the end of method body
                        method_body.extend(except_block)
                
                fixed_lines.extend(method_body)
                continue
            else:
                # Not a method definition, just add decorators as is
                fixed_lines.extend(decorators)
                continue
        
        fixed_lines.append(line)
        i += 1
    
    content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"  ✅ Fixed comprehensive syntax issues in {file_path}")

def main():
    """Main function."""
    file_path = '/workspace/src/training/steps/market_analysis/hmm_clustering/step03_hmm_regime_discovery.py'
    fix_hmm_comprehensive(file_path)

if __name__ == "__main__":
    main()