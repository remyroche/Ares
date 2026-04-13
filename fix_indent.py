with open("extreme_price_movements/pipeline_steps.py", "r") as f:
    lines = f.readlines()

for i in range(len(lines)):
    if "tprint(\"Resolving unique symbols and timestamps for feature injection...\")" in lines[i]:
        print(f"Line {i} starts with {len(lines[i]) - len(lines[i].lstrip())} spaces")
        # remove the 4 spaces we added
        for j in range(i, len(lines)):
            if lines[j].startswith("    "):
                lines[j] = lines[j][4:]

with open("extreme_price_movements/pipeline_steps.py", "w") as f:
    f.writelines(lines)
