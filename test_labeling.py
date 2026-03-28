with open("extreme_price_movements/labeling.py", "r") as f:
    content = f.read()

import re
matches = re.findall(r".{0,100}mfe.{0,100}", content, re.IGNORECASE)
for m in matches:
    print(m)
