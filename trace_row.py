#!/usr/bin/env python3
"""Trace _row origin and understand why metrics are missing"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    lines = f.readlines()

# Find where _row is defined (around line 9693)
print("Lines around 9680-9725 (_row definition and _cell_payload construction):\n")

for i in range(9679, min(9730, len(lines))):
    line = lines[i]
    print(f"{i+1}: {line.rstrip()[:100]}")
