import re

with open("extreme_price_movements/mask_optimiser.py", "r") as f:
    code = f.read()

# Make sure df_short (Tier 0) has tier = 0 before the loop
start_str = '    df_short = df2.head(shortlist_max).copy()'
new_str = '    df_short = df2.head(shortlist_max).copy()\n    df_short["tier"] = 0\n    df_short["conditioner_mode"] = "none"'

code = code.replace(start_str, new_str)

with open("extreme_price_movements/mask_optimiser.py", "w") as f:
    f.write(code)
