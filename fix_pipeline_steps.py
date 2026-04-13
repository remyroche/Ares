with open("extreme_price_movements/pipeline_steps.py", "r") as f:
    lines = f.readlines()

import re

# find where "tprint("Resolving unique symbols and timestamps for feature injection...")" is.
# This entire section from there to the end of the file is outside a function because of indentation
start_idx = -1
for i, line in enumerate(lines):
    if "tprint(\"Resolving unique symbols and timestamps for feature injection...\")" in line:
        start_idx = i
        break

if start_idx != -1:
    # indent from start_idx to end
    for i in range(start_idx, len(lines)):
        lines[i] = "    " + lines[i]

with open("extreme_price_movements/pipeline_steps.py", "w") as f:
    f.writelines(lines)
