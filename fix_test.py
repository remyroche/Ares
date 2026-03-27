import re

with open("extreme_price_movements/tests/test_labeling.py", "r") as f:
    c = f.read()

target1 = """        labels, rets = compute_triple_barrier_labels(panel, tp_df, sl_df, 5)"""
replace1 = """        labels, rets, _, _, _, _ = compute_triple_barrier_labels(panel, tp_df, sl_df, 5)"""
c = c.replace(target1, replace1)

target2 = """        labels, rets = compute_triple_barrier_labels(panel, 0.05, 0.05, 5)"""
replace2 = """        labels, rets, _, _, _, _ = compute_triple_barrier_labels(panel, 0.05, 0.05, 5)"""
c = c.replace(target2, replace2)

with open("extreme_price_movements/tests/test_labeling.py", "w") as f:
    f.write(c)
