#!/usr/bin/env python3
"""Investigate per_cell_rows construction in compare_tbm_parameters.py"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    lines = f.readlines()

# Find all lines with per_cell_rows
print("Lines containing 'per_cell_rows':")
for i, line in enumerate(lines, 1):
    if 'per_cell_rows' in line:
        print(f"{i}: {line.rstrip()[:100]}")
