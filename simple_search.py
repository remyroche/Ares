#!/usr/bin/env python3
"""Simple search for simple patterns"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    lines = f.readlines()

print("Lines with 'simple' or 'SIMPLE':")
for i, line in enumerate(lines):
    if 'simple' in line.lower():
        print(f"{i+1}: {line.rstrip()[:80]}")
