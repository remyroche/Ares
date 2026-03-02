import re

with open("extreme_price_movements/optimise_tpsl_ratio.py", "r") as f:
    content = f.read()

idx1 = content.find("def build_event_cache_15m")
if idx1 != -1:
    print(content[idx1:idx1+2000])
