import re

with open("extreme_price_movements/policy_optimiser.py", "r") as f:
    content = f.read()

# remove line 746
lines = content.split('\n')
del lines[745] # 0-indexed

content = '\n'.join(lines)

with open("extreme_price_movements/policy_optimiser.py", "w") as f:
    f.write(content)
