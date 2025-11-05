#!/usr/bin/env python3
"""
Fix syntax errors in cluster_quality_assessor.py properly
"""

def fix_syntax_errors():
    with open('src/training/steps/market_analysis/clusters/cluster_quality_assessor.py', 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for try block followed by with tprint_timer
        if line.strip().startswith('try:') and i + 1 < len(lines):
            next_line = lines[i + 1]
            if 'with tprint_timer(' in next_line:
                # Add the try line
                fixed_lines.append(line)
                # Fix the with line (should be indented 4 spaces more than try)
                fixed_lines.append('    ' + next_line.lstrip())
                i += 2
                # Add the content inside with block (should be indented 8 spaces more than try)
                while i < len(lines) and not lines[i].strip().startswith('except') and not lines[i].strip().startswith('finally'):
                    if lines[i].strip():  # Non-empty line
                        fixed_lines.append('        ' + lines[i].lstrip())
                    else:
                        fixed_lines.append(lines[i])
                    i += 1
                # Add the except/finally line
                if i < len(lines):
                    fixed_lines.append(lines[i])
                    i += 1
                continue
        
        fixed_lines.append(line)
        i += 1
    
    # Write back
    with open('src/training/steps/market_analysis/clusters/cluster_quality_assessor.py', 'w') as f:
        f.writelines(fixed_lines)
    
    print("Fixed syntax errors properly")

if __name__ == "__main__":
    fix_syntax_errors()
