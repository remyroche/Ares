#!/usr/bin/env python3
"""Find geometry grid row construction with column assignments"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    content = f.read()

# Look for where rows are constructed with cell_auc, cell_bind etc
lines = content.split('\n')

print("Searching for cell metric column assignments in row construction:\n")

for i, line in enumerate(lines):
    # Look for patterns like "cell_auc": value or cell_auc=value
    if ('cell_auc' in line or 'cell_bind' in line or 'cell_score' in line) and ('=' in line or ':' in line):
        print(f"{i+1}: {line.rstrip()[:100]}")
        # Show surrounding context
        for j in range(max(0, i-2), min(len(lines), i+3)):
            if j != i:
                print(f"  {j+1}: {lines[j].rstrip()[:80]}")
        print()
        if i > 100:  # Stop after finding enough
            break
