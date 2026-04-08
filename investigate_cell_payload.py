#!/usr/bin/env python3
"""Investigate _cell_payload construction"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    lines = f.readlines()

# Show lines around 9725 where per_cell_rows is populated
start = 9700
end = 9750
print(f"Lines {start}-{end} (per_cell_rows append context):")
for i in range(start-1, min(end, len(lines))):
    print(f"{i+1}: {lines[i].rstrip()[:120]}")
