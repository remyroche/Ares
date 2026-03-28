with open("extreme_price_movements/labeling.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "def _numba_triple_barrier_fast(" in line:
        start = i
        break

print("".join(lines[start:start+120]))
