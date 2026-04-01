import re
with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

import subprocess
out = subprocess.getoutput("git diff HEAD extreme_price_movements/lgbm_based_mask_generation.py | grep -i reject")
print(out)
