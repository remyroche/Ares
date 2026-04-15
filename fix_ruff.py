import re

with open("extreme_price_movements/policy_optimiser.py", "r") as f:
    content = f.read()

content = content.replace("from typing import Tuple\n", "")

with open("extreme_price_movements/policy_optimiser.py", "w") as f:
    f.write(content)
