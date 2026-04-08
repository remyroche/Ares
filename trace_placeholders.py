#!/usr/bin/env python3
"""Trace the source of placeholder metrics (0.55, 0.3, 0.5, 0.15, 0.2)"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    content = f.read()

# Search for hardcoded values
search_terms = ['0.55', '0.52', '0.05', '1.0', '0.15', '0.2', '0.3', '0.6', '0.5,']
lines = content.split('\n')

print("Searching for hardcoded placeholder values:\n")

for i, line in enumerate(lines):
    for term in search_terms:
        if term in line and ('cell' in line.lower() or 'auc' in line.lower() or 'default' in line.lower() or 'placeholder' in line.lower()):
            if '=' in line or ':' in line:
                print(f"{i+1}: {line.rstrip()[:120]}")
                break

print("\n" + "="*80)
print("\nSearching for _row initialization and cell metric defaults:\n")

for i, line in enumerate(lines):
    if '_row' in line and ('get' in line or 'cell' in line.lower()):
        if any(x in line for x in ['cell_auc', 'cell_bind', 'cell_score', 'cell_timeout', 'cell_ece', 'cell_brier']):
            print(f"{i+1}: {line.rstrip()[:120]}")
