import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# Add fold_pnl_raws to the output dict of `assess_rules` so we can do std dev
old_dict = """                    "ridge_pnl_raw": ridge_pnl_raw,
                    "ridge_trade_sortino": ridge_trade_sortino,"""

new_dict = """                    "ridge_pnl_raw": ridge_pnl_raw,
                    "fold_pnl_raws": pnl_info.get("fold_pnl_raws", {}) if 'pnl_info' in locals() else {},
                    "ridge_trade_sortino": ridge_trade_sortino,"""

source = source.replace(old_dict, new_dict)

with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
    f.write(source)

print("Added fold_pnl_raws to results.")
