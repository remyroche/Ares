with open("extreme_price_movements/labeling.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "def compute_triple_barrier_labels(" in line:
        start = i
        break

print("".join(lines[start:start+30]))
