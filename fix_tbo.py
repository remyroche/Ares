import re
with open("extreme_price_movements/labeling.py", "r") as f:
    content = f.read()

content = content.replace("return outcomes, returns, quality, exit_idxs, conflict_j, conflict_j", "return outcomes, returns, quality, exit_idxs, conflict_j")

with open("extreme_price_movements/labeling.py", "w") as f:
    f.write(content)
