with open("extreme_price_movements/features.py", "r") as f:
    lines = f.readlines()

# Find PORTABILITY HARDENING (IN-PLACE) and its end
start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if "PORTABILITY HARDENING (IN-PLACE)" in line:
        start_idx = i - 1
        break

for i in range(start_idx + 3, len(lines)):
    if "Final check for Inf/NaN" in lines[i]:
        end_idx = i - 1
        break

if start_idx != -1 and end_idx != -1:
    block = lines[start_idx:end_idx]
    del lines[start_idx:end_idx]

    # Insert before CausalFeatureTransformer
    insert_idx = -1
    for i, line in enumerate(lines):
        if "transformer = CausalFeatureTransformer(" in line:
            insert_idx = i - 6 # before Transform cache can be enabled
            break

    if insert_idx != -1:
        lines = lines[:insert_idx] + block + lines[insert_idx:]
        with open("extreme_price_movements/features.py", "w") as f:
            f.writelines(lines)
        print(f"Moved hardening block to {insert_idx}")
