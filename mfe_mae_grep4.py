import re
with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "r") as f:
    content = f.read()

for v in ["mfe", "mae", "MFE", "MAE"]:
    print(f"--- {v} ---")
    matches = re.findall(r".{0,100}" + v + r".{0,100}", content, re.IGNORECASE)
    for m in matches[:5]:
        print(m.strip())
