with open("extreme_price_movements/labeling.py", "r") as f:
    content = f.read()

for v in ["median_mfe_mae", "p10_mfe_mae", "p90_mae", "p50_mfe", "pct_MFE_before_MAE"]:
    print(f"--- {v} ---")
    if v in content:
        print("Found!")
    else:
        print("Not found.")
