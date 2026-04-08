#!/usr/bin/env python3
"""Find where compare_tbm_parameters.py generates simple configs"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    content = f.read()

# Search for the exact placeholder values seen in CSV
patterns_to_find = [
    'probe_status',
    'simple_generated',
    'simple_tight',
    'simple_wide',
    'SIMPLE_',
]

lines = content.split('\n')
print("Lines containing simple/fallback patterns:\n")

for i, line in enumerate(lines):
    found_any = False
    for pattern in patterns_to_find:
        if pattern in line:
            found_any = True
            break
    if found_any:
        print(f"{i+1}: {line.rstrip()[:100]}")
        # Show surrounding context
        if i > 0:
            print(f"{i}: {lines[i-1].rstrip()[:80]}")
        if i < len(lines) - 1:
            print(f"{i+2}: {lines[i+1].rstrip()[:80]}")
        print()
