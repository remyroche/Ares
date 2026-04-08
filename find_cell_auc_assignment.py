#!/usr/bin/env python3
"""Find where geometry grid rows are created with cell metrics"""

with open('/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py') as f:
    content = f.read()

# Find cell_auc assignment patterns
lines = content.split('\n')

print("Lines with cell_auc assignment or dictionary creation:")
for i, line in enumerate(lines):
    if '"cell_auc"' in line or "'cell_auc'" in line or 'cell_auc:' in line or 'cell_auc =' in line:
        print(f"{i+1}: {line.rstrip()[:100]}")
        # Show context
        if i > 0:
            print(f"{i}: {lines[i-1].rstrip()[:80]}")
        if i < len(lines) - 1:
            print(f"{i+2}: {lines[i+1].rstrip()[:80]}")
        print()
        if i > 50:  # Stop after finding some
            break
