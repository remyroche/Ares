#!/usr/bin/env python3
"""Find where 0.55 and default metrics are set"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    content = f.read()

# Find where 0.55 appears
lines = content.split('\n')
for i, line in enumerate(lines):
    if '0.55' in line and ('auc' in line.lower() or 'cell' in line.lower() or 'score' in line.lower()):
        print(f"{i+1}: {line.rstrip()[:100]}")
        # Show context
        if i > 0:
            print(f"{i}: {lines[i-1].rstrip()[:100]}")
        if i < len(lines) - 1:
            print(f"{i+2}: {lines[i+1].rstrip()[:100]}")
        print()
