with open("extreme_price_movements/pipeline_steps.py", "r") as f:
    lines = f.readlines()

in_inject = False
for i in range(len(lines)):
    if "def inject_features_into_datasets(" in lines[i]:
        in_inject = True
    elif in_inject and "tprint(\"Resolving unique symbols and timestamps for feature injection...\")" in lines[i]:
        # Add indent back
        for j in range(i, len(lines)):
            lines[j] = "    " + lines[j]
        break

with open("extreme_price_movements/pipeline_steps.py", "w") as f:
    f.writelines(lines)
