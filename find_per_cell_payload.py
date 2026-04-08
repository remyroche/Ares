#!/usr/bin/env python3
"""Find where per_cell_payload is constructed with placeholder metrics"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    lines = f.readlines()

# Search for where per_cell_rows or _cell_payload is constructed
for i, line in enumerate(lines):
    if '_cell_payload' in line or 'per_cell_rows' in line:
        print(f"{i+1}: {line.rstrip()[:100]}")
        # Show context
        for j in range(max(0, i-3), min(len(lines), i+4)):
            if j != i:
                print(f"  {j+1}: {lines[j].rstrip()[:80]}")
        print()
        if i > 9500:  # Focus on export section
            break
