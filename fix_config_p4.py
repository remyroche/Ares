import re

with open("extreme_price_movements/config.py", "r") as f:
    content = f.read()

# I checked the config and it doesn't contain any lr_* fields, they are already using ret1h etc.
# And ret24h is present.
# Just to be sure, I'll add ret24h to MODEL_FEATURES if it's missing, since it's the canonical name.

idx = content.find("MODEL_FEATURES = [")
if "ret24h" not in content[idx:content.find("]", idx)]:
    idx2 = content.find("]", idx)
    content = content[:idx2] + '    "ret24h",\n' + content[idx2:]

with open("extreme_price_movements/config.py", "w") as f:
    f.write(content)
