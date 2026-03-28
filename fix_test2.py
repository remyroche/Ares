import re
with open("extreme_price_movements/tests/test_labeling.py", "r") as f:
    c = f.read()

target = """        labels_s, rets_s = compute_triple_barrier_labels(panel, 0.05, 0.05, 5, side="short")"""
replace = """        labels_s, rets_s, _, _, _, _ = compute_triple_barrier_labels(panel, 0.05, 0.05, 5, side="short")"""
c = c.replace(target, replace)

with open("extreme_price_movements/tests/test_labeling.py", "w") as f:
    f.write(c)
