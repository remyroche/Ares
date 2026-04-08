#!/usr/bin/env python3
"""Find where geometry grid is populated - search for DataFrame construction"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    content = f.read()

# Search for grid DataFrame or rows construction
patterns = [
    '_grid_df',
    'grid_rows',
    'per_cell_grids',
    'TBM_GEOMETRY_GRID',
    'geometry_grid',
    'to_csv.*grid',
]

lines = content.split('\n')

print("Lines with grid DataFrame construction:\n")
for i, line in enumerate(lines):
    for pattern in patterns:
        if pattern in line:
            print(f"{i+1}: {line.rstrip()[:100]}")
            break
