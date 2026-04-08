#!/usr/bin/env python3
"""Find all simple mode configuration code"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    content = f.read()

# Search for simple config patterns
patterns = [
    'simple_generated',
    'simple_tight',
    'simple_wide',
    'SIMPLE_',
    '_generate_simple',
    'simple_mode',
    'simple_fallback',
    'use_simple',
]

lines = content.split('\n')

print("Lines containing simple mode patterns:\n")
for i, line in enumerate(lines):
    for pattern in patterns:
        if pattern in line:
            print(f"{i+1}: {line.rstrip()[:100]}")
            break
