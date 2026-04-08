#!/usr/bin/env python3
"""Search for hardcoded metric values in the geometry grid creation"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    content = f.read()

# Search for the exact values we see in the CSV
patterns = ['0.55,', '0.52,', '0.05,', '0.3,', '0.6,', '0.15,', '0.2,', '0.5,']

lines = content.split('\n')

print("Lines containing hardcoded metric values:\n")
for i, line in enumerate(lines):
    for pattern in patterns:
        if pattern in line and ('cell' in line.lower() or 'row' in line.lower() or 'dict' in line.lower()):
            print(f"{i+1}: {line.rstrip()[:120]}")
            break
