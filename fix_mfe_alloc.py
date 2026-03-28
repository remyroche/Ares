import re

with open("extreme_price_movements/labeling.py", "r") as f:
    c = f.read()

target_loop = """        trailing_active = False
        exit_found = False
        stall_checked = False
        extreme = entry_p

        # Scan only within the time window
        for j in range(j_start, min(j_end, n)):"""

replace_loop = """        trailing_active = False
        exit_found = False
        stall_checked = False
        extreme = entry_p

        mfe_val = 0.0
        mae_val = 0.0
        t_mfe = 0.0
        t_mae = 0.0

        # Scan only within the time window
        for j in range(j_start, min(j_end, n)):"""

c = c.replace(target_loop, replace_loop)

with open("extreme_price_movements/labeling.py", "w") as f:
    f.write(c)
