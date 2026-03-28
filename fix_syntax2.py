with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "def _refine_ambiguous_labels_with_intrabar(" in line:
        start = i
        break

print("".join(lines[start:start+30]))
