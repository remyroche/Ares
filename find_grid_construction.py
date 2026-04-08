#!/usr/bin/env python3
"""Find where geometry grid rows are populated"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    lines = f.readlines()

# Look for where _grid_df or grid rows are constructed
print("Searching for grid DataFrame construction:")
for i, line in enumerate(lines):
    if '_grid_df' in line or 'grid_rows' in line or 'per_cell_grids' in line:
        print(f"{i+1}: {line.rstrip()[:100]}")
        if i > 9500:  # Focus on later part of file where export happens
            # Show more context
            for j in range(max(0, i-2), min(len(lines), i+3)):
                print(f"  {j+1}: {lines[j].rstrip()[:90]}")
            print()
