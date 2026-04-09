with open("extreme_price_movements/features.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "    )\n" == line and "Transform cache can be enabled" in lines[i+1]:
        lines[i] = ''
        break

with open("extreme_price_movements/features.py", "w") as f:
    f.writelines(lines)
