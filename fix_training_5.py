with open("extreme_price_movements/training.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "if _pre_h is not None and variant is None:" in line:
        new_lines.append(line)
        new_lines.append("            pass\n")
    else:
        new_lines.append(line)

with open("extreme_price_movements/training.py", "w") as f:
    f.writelines(new_lines)
