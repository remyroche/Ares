with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

# Let's see what the current RV portability hardening looks like
import re

matches = re.findall(r"(if scale_rv is not None:.*?# 4\) Heavy-tailed)", content, flags=re.DOTALL)
if matches:
    print(matches[0])
else:
    print("Not found")
