#!/usr/bin/env python3
"""
Final script to fix unclosed triple-quoted docstrings in volume.py
This script processes the file line by line and ensures all docstrings are properly closed.
"""

def fix_docstrings_in_file(file_path):
    """Fix all unclosed triple-quoted docstrings in a file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    i = 0
    in_docstring = False
    docstring_start_line = -1

    while i < len(lines):
        line = lines[i]

        # Check if this line contains triple quotes
        if '"""' in line:
            quote_count = line.count('"""')

            if not in_docstring:
                # Starting a new docstring
                if quote_count == 1 and line.strip().startswith('"""'):
                    # Single triple quote at start - start of docstring
                    in_docstring = True
                    docstring_start_line = i
                    fixed_lines.append(line)
                elif quote_count >= 2:
                    # Multiple quotes - could be single line docstring or malformed
                    fixed_lines.append(line)
                else:
                    # Single quote not at start - might be in middle of line
                    fixed_lines.append(line)
            else:
                # We're in a docstring, looking for the end
                if quote_count >= 1:
                    # Found the end of the docstring
                    in_docstring = False
                    # Ensure it's properly closed
                    if not line.strip().endswith('"""'):
                        line = line.rstrip() + '"""'
                    fixed_lines.append(line)
                else:
                    # Still in docstring, no closing quotes found
                    fixed_lines.append(line)
        else:
            # Regular line
            if in_docstring:
                # We're in a docstring but no quotes in this line
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        i += 1

    # If we're still in a docstring at the end, close it
    if in_docstring:
        fixed_lines.append('"""')

    # Write back the fixed content
    with open(file_path, 'w') as f:
        for line in fixed_lines:
            f.write(line)

    print(f"Fixed docstrings in {file_path}")

    # Verify the fix
    import subprocess
    try:
        result = subprocess.run(['python3', '-m', 'py_compile', file_path],
                              capture_output=True, text=True, cwd='/Users/remyroche/Documents/Ares')
        if result.returncode == 0:
            print("✅ File compiles successfully!")
        else:
            print(f"❌ Still has syntax errors: {result.stderr}")
    except Exception as e:
        print(f"❌ Error testing compilation: {e}")

if __name__ == "__main__":
    fix_docstrings_in_file("src/feature_generation/categories/volume.py")
