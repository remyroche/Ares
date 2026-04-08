#!/usr/bin/env python3
"""Find where simple_generated configs and placeholder values are created"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    content = f.read()
    lines = content.split('\n')

print("Searching for simple_tight, simple_wide, simple_generated, 0.55 patterns:\n")

for i, line in enumerate(lines):
    if any(x in line for x in ['simple_tight', 'simple_wide', 'simple_generated', '"0.55"', "'0.55'", '= 0.55', ': 0.55']):
        print(f"{i+1}: {line.rstrip()[:100]}")
