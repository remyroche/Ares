import re

with open("extreme_price_movements/optimise_tpsl_ratio.py", "r") as f:
    content = f.read()

# Debug: try to find how build_event_cache actually looks
idx1 = content.find("def build_event_cache(")
if idx1 != -1:
    print(content[idx1:idx1+1000])
