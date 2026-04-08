#!/usr/bin/env python3
"""Quick grep for simple config patterns"""

import re

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    content = f.read()

# Find lines with simple_tight or simple_wide
matches = []
for i, line in enumerate(content.split('\n'), 1):
    if 'simple_tight' in line or 'simple_wide' in line:
        matches.append((i, line.strip()))

print(f"Found {len(matches)} lines with simple_tight/simple_wide")
for i, (line_num, line) in enumerate(matches[:20]):
    print(f"{line_num}: {line[:80]}")
