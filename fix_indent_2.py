with open("extreme_price_movements/pipeline_steps.py", "r") as f:
    lines = f.readlines()

# The code from 5742 to end should be part of inject_features_into_datasets
for i in range(len(lines)):
    if "tprint(\"Resolving unique symbols and timestamps for feature injection...\")" in lines[i]:
        # Add the indent back
        for j in range(i, len(lines)):
            lines[j] = "    " + lines[j]
        break

with open("extreme_price_movements/pipeline_steps.py", "w") as f:
    f.writelines(lines)
