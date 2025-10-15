#!/usr/bin/env python3
"""
Script to fix unclosed triple-quoted docstrings in volume.py
"""

import re

def fix_docstrings_in_file(file_path):
    """Fix all unclosed triple-quoted docstrings in a file"""
    with open(file_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this line starts a docstring
        if '"""' in line and not line.strip().endswith('"""'):
            # Find the start of the docstring
            docstring_start = line.find('"""')
            prefix = line[:docstring_start]
            suffix = line[docstring_start + 3:]

            # Add the line with proper closing if it's a single-line docstring
            if suffix.strip():  # Multi-line docstring
                fixed_lines.append(line)
                i += 1
                # Look for the closing triple quotes in subsequent lines
                while i < len(lines) and '"""' not in lines[i]:
                    fixed_lines.append(lines[i])
                    i += 1

                if i < len(lines):
                    # Close the docstring on the line that contains the closing quotes
                    closing_line = lines[i]
                    if '"""' in closing_line:
                        # Make sure it's properly closed
                        if not closing_line.strip().endswith('"""'):
                            closing_line = closing_line.rstrip() + '"""'
                        fixed_lines.append(closing_line)
                    i += 1
            else:
                # Single-line docstring, ensure it's properly closed
                if not line.strip().endswith('"""'):
                    line = line.rstrip() + '"""'
                fixed_lines.append(line)
                i += 1
        else:
            fixed_lines.append(line)
            i += 1

    # Write back the fixed content
    fixed_content = '\n'.join(fixed_lines)
    with open(file_path, 'w') as f:
        f.write(fixed_content)

    print(f"Fixed docstrings in {file_path}")

if __name__ == "__main__":
    fix_docstrings_in_file("src/feature_generation/categories/volume.py")

