with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

# Replace _roll_std and _roll_mean calls everywhere since they're just local to _compute_features_impl
import re

content = re.sub(r'_roll_std\("[^"]*", (.+?), (\d+)\)', r'ff.numba_rolling_std(\1, \2)', content)
content = re.sub(r'_roll_mean\("[^"]*", (.+?), (\d+)\)', r'ff.numba_rolling_mean(\1, \2)', content)

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
