#!/usr/bin/env python3
"""
Fix syntax errors in cluster_quality_assessor.py
"""

import re

def fix_syntax_errors():
    with open('src/training/steps/market_analysis/clusters/cluster_quality_assessor.py', 'r') as f:
        content = f.read()
    
    # Fix the with statement indentation issues
    # Pattern: with tprint_timer(...): should be at the same level as the code inside
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix with statement indentation
        if 'with tprint_timer(' in line:
            # Count leading spaces
            leading_spaces = len(line) - len(line.lstrip())
            # Should be at the same level as the try block (8 spaces)
            fixed_line = '        ' + line.lstrip()
            fixed_lines.append(fixed_line)
        # Fix lines that follow with tprint_timer (should be indented more)
        elif i > 0 and 'with tprint_timer(' in lines[i-1]:
            # These should be indented 4 more spaces than the with statement
            fixed_line = '            ' + line.lstrip()
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    
    # Write back
    with open('src/training/steps/market_analysis/clusters/cluster_quality_assessor.py', 'w') as f:
        f.write('\n'.join(fixed_lines))
    
    print("Fixed syntax errors")

if __name__ == "__main__":
    fix_syntax_errors()
