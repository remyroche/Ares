with open("extreme_price_movements/pipeline_steps.py", "r") as f:
    lines = f.readlines()

in_inject = False
for i in range(len(lines)):
    if "def inject_features_into_datasets(" in lines[i]:
        in_inject = True
    elif in_inject and "return datasets" in lines[i]:
        # Indent everything after this return line
        for j in range(i + 1, len(lines)):
            lines[j] = "    " + lines[j]
        break

with open("extreme_price_movements/pipeline_steps.py", "w") as f:
    f.writelines(lines)
