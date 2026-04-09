with open("extreme_price_movements/features.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "tprint(" in line and "tprint(f\"Features:" in lines[i+1]:
        # we have a hanging tprint( without a closing parenthesis. Let's delete line i
        lines[i] = ''
        break

with open("extreme_price_movements/features.py", "w") as f:
    f.writelines(lines)
