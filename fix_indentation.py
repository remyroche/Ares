#!/usr/bin/env python3
"""
Script to fix indentation issues in the probabilistic_bayesian_optimizer.py file.
"""

import re

def fix_indentation_issues(file_path):
    """Fix indentation issues in a single file."""
    print(f"Fixing indentation in: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Fix improperly indented except statements
    pattern = r'^except (.*?):\n'
    matches = re.finditer(pattern, content, flags=re.MULTILINE)
    
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        if line.strip().startswith('except ') and not line.startswith('    ') and not line.startswith('\t'):
            # This is an improperly indented except statement
            # Look for the matching try statement above
            for j in range(i - 1, -1, -1):
                if lines[j].strip().startswith('try:'):
                    try_indent = len(lines[j]) - len(lines[j].lstrip())
                    # Add the same indentation to the except statement
                    fixed_line = ' ' * try_indent + line.strip()
                    fixed_lines.append(fixed_line)
                    break
            else:
                # Fallback: add 8 spaces (double indent)
                fixed_lines.append('        ' + line.strip())
        else:
            fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  ✅ Fixed indentation issues in {file_path}")
        return True
    else:
        print(f"  ℹ️  No indentation issues found in {file_path}")
        return False

def main():
    """Main function to fix indentation issues."""
    file_path = '/workspace/src/training/probabilistic_bayesian_optimizer.py'
    fix_indentation_issues(file_path)

if __name__ == "__main__":
    main()