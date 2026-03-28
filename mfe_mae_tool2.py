with open("extreme_price_movements/labeling.py", "r") as f:
    c = f.read()

# Let's inspect the `compute_triple_barrier_labels` returns
if "return outcomes, quality, returns, exit_idxs, conflict_j" in c:
    print("Found return signature 1")
if "return outcomes, returns, quality, exit_idxs, conflict_j" in c:
    print("Found return signature 2")
