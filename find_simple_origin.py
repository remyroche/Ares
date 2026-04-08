#!/usr/bin/env python3
"""Find where SIMPLE_ configs and simple_generated status are created"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    content = f.read()

# Find SIMPLE_ patterns
import re
matches = list(re.finditer(r'SIMPLE_|simple_generated|simple_tight|simple_wide', content))

print(f"Found {len(matches)} matches for simple patterns")
print("\nFirst 20 matches with context:")
lines = content.split('\n')
for m in matches[:20]:
    start = max(0, m.start() - 200)
    end = min(len(content), m.end() + 100)
    context = content[start:end].replace('\n', ' ')
    print(f"  ...{context}...")
    print()
