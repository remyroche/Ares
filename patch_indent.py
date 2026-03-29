import re

with open("extreme_price_movements/training.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.strip() == 'if best_m is not None:':
        # Let's check the next line
        next_line = lines[i+1]
        if next_line.startswith('            best_m["models_by_h"]'):
            lines[i+1] = '    ' + next_line

with open("extreme_price_movements/training.py", "w") as f:
    f.writelines(lines)
