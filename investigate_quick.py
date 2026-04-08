#!/usr/bin/env python3
"""Investigate _cell_payload construction - lines around 9725"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    lines = f.readlines()

# Lines around 9725
for i in range(9700, min(9750, len(lines))):
    line = lines[i]
    if 'cell' in line.lower() or 'payload' in line.lower() or 'append' in line.lower():
        print(f"{i+1}: {line.rstrip()[:100]}")
