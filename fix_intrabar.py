import re

with open("extreme_price_movements/policy_optimiser.py", "r") as f:
    content = f.read()

# Fix the intrabar priority logic
# Current:
# bar_exit = np.where(simple_tp_hit, tp_dist_a, bar_exit)
# bar_exit = np.where(open_trail & np.isnan(bar_exit), trail_floor_ret, bar_exit)
# bar_exit = np.where(open_sl & np.isnan(bar_exit), -sl_eff, bar_exit)
#
# Replacement (SL and trailing hit first before TP to be pessimistic):
new_logic = """        bar_exit = np.where(open_sl, -sl_eff, bar_exit)
        bar_exit = np.where(open_trail & np.isnan(bar_exit), trail_floor_ret, bar_exit)
        bar_exit = np.where(simple_tp_hit & np.isnan(bar_exit), tp_dist_a, bar_exit)"""

content = re.sub(
    r"        bar_exit = np.where\(simple_tp_hit, tp_dist_a, bar_exit\)\n"
    r"        bar_exit = np.where\(open_trail & np.isnan\(bar_exit\), trail_floor_ret, bar_exit\)\n"
    r"        bar_exit = np.where\(open_sl & np.isnan\(bar_exit\), -sl_eff, bar_exit\)",
    new_logic,
    content
)

with open("extreme_price_movements/policy_optimiser.py", "w") as f:
    f.write(content)
