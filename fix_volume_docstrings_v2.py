#!/usr/bin/env python3
"""
Improved script to fix unclosed triple-quoted docstrings in volume.py
"""

import re

def fix_docstrings_in_file(file_path):
    """Fix all unclosed triple-quoted docstrings in a file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip('\n\r')

        # Check if this line contains triple quotes
        if '"""' in line:
            # Count quotes in this line
            quote_count = line.count('"""')

            if quote_count == 1:
                # Single triple quote - could be start or end
                if line.strip().startswith('"""'):
                    # This is a docstring start, need to find where it ends
                    # Add this line
                    fixed_lines.append(line)

                    # Look for the closing triple quotes in subsequent lines
                    j = i + 1
                    found_closing = False
                    while j < len(lines):
                        next_line = lines[j].rstrip('\n\r')
                        if '"""' in next_line:
                            # Found closing quotes
                            if next_line.count('"""') >= 1:
                                # Close the docstring properly
                                if not next_line.strip().endswith('"""'):
                                    next_line = next_line.rstrip() + '"""'
                                fixed_lines.append(next_line)
                                found_closing = True
                                i = j
                                break
                        else:
                            fixed_lines.append(next_line)
                        j += 1

                    if not found_closing:
                        # No closing found, add closing quotes at the end
                        fixed_lines.append('"""')
                        i = j - 1
                else:
                    # This might be a closing quote in the middle of a line
                    # Add it as is for now
                    fixed_lines.append(line)
            elif quote_count >= 2:
                # Multiple quotes in one line - should be properly closed
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

        i += 1

    # Write back the fixed content
    with open(file_path, 'w') as f:
        for line in fixed_lines:
            f.write(line + '\n')

    print(f"Fixed docstrings in {file_path}")

if __name__ == "__main__":
    fix_docstrings_in_file("src/feature_generation/categories/volume.py")

