import sys
import re

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# Look for `tprint("Top 15 Final Diverse Rules:")`
# That part is in `run_side_pipeline`, wait no, `run_mining_stage`. Let's search for `Top 15 Final Diverse Rules:` in the file.
print("Searching for 'Top 15 Final Diverse Rules:'...")
lines = content.split('\n')
for i, line in enumerate(lines):
    if "Top 15 Final Diverse Rules:" in line:
        print(f"Found at line {i+1}: {line}")
