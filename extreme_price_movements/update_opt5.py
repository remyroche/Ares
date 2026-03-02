import re

with open("extreme_price_movements/optimise_tpsl_ratio.py", "r") as f:
    content = f.read()

idx1 = content.find("rH_prefix_max = np.maximum.accumulate")
if idx1 != -1:
    print(content[idx1:idx1+1000])

idx2 = content.find("def build_event_cache_15m")
if idx2 != -1:
    print(content[idx2:idx2+1000])
