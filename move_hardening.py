with open("extreme_price_movements/features.py", "r") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "PORTABILITY HARDENING (IN-PLACE)" in line:
        start_idx = i - 1
        break

if start_idx != -1:
    # Find the end of the block, marked by "return feats" for the compute_regime_features function
    for i in range(start_idx + 3, len(lines)):
        if "return feats" in lines[i]:
            end_idx = i
            break

    if end_idx != -1:
        hardening_block = lines[start_idx:end_idx]

        # Remove it from the original location
        del lines[start_idx:end_idx]

        # Find where to insert it: before "Final check for Inf/NaN (numpy arrays now)" in _compute_features_impl
        insert_idx = -1
        for i, line in enumerate(lines):
            if "Final check for Inf/NaN (numpy arrays now)" in line:
                insert_idx = i - 1
                break

        if insert_idx != -1:
            lines = lines[:insert_idx] + hardening_block + lines[insert_idx:]

        with open("extreme_price_movements/features.py", "w") as f:
            f.writelines(lines)
            print(f"Moved block from {start_idx}-{end_idx} to {insert_idx}")
