import re

with open("extreme_price_movements/policy_optimiser.py", "r") as f:
    content = f.read()

new_logic = """        bar_exit = np.where(
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

content = re.sub(
    r"        bar_exit = np.where\(\n"
    r"            np.isnan\(bar_exit\),\n"
    r"            np.where\(\n"
    r"                simple_tp_hit,\n"
    r"                tp_dist_a,\n"
    r"                np.where\(\n"
    r"                    trail_hit,\n"
    r"                    trail_floor_ret,\n"
    r"                    np.where\(sl_hit, -sl_eff, np.nan\),\n"
    r"                \),\n"
    r"            \),\n"
    r"            bar_exit,\n"
    r"        \)",
    new_logic,
    content
)

with open("extreme_price_movements/policy_optimiser.py", "w") as f:
    f.write(content)
