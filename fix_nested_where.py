import re

with open("extreme_price_movements/policy_optimiser.py", "r") as f:
    content = f.read()

# Fix the nested np.where intrabar priority logic on line 1137+
old_nested = """        bar_exit = np.where(
            np.isnan(bar_exit),
            np.where(
                simple_tp_hit,
                tp_dist_a,
                np.where(
                    trail_hit,
                    trail_floor_ret,
                    np.where(sl_hit, -sl_eff, np.nan),
                ),
            ),
            bar_exit,
        )"""

new_nested = """        bar_exit = np.where(
            np.isnan(bar_exit),
            np.where(
                sl_hit,
                -sl_eff,
                np.where(
                    trail_hit,
                    trail_floor_ret,
                    np.where(simple_tp_hit, tp_dist_a, np.nan),
                ),
            ),
            bar_exit,
        )"""

content = content.replace(old_nested, new_nested)

with open("extreme_price_movements/policy_optimiser.py", "w") as f:
    f.write(content)
