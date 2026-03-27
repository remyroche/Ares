import re

with open("extreme_price_movements/labeling.py", "r") as f:
    c = f.read()

target = """        trailing_active = False
        exit_found = False

        mfe_val = 0.0
        mae_val = 0.0
        t_mfe = 0.0
        t_mae = 0.0

        for j in range(j_start, min(j_end, n)):"""

# Wait, `mfe_val = 0.0` might be defined *after* `for i in range` but we need to check exactly where it is. Let's inspect `_numba_triple_barrier_fast`.
