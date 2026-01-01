
import os

file_path = 'src/training/steps/labeling/label_based_layer_1.py'

with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # Only process lines from 337 (0-indexed) approx line 338 onwards
    if i < 337:
        new_lines.append(line)
        continue
    
    # Check for 8 spaces indent
    if line.startswith('        '):
        # explicit check to avoid dedenting multiline strings incorrectly if they start with spaces
        # but here we assume code structure
        new_lines.append(line[4:])
    else:
        # Lines with 4 spaces or less, or empty lines
        new_lines.append(line)

with open(file_path, 'w') as f:
    f.writelines(new_lines)

print(f"Fixed indentation for {file_path}")
