import sys

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    lines = f.readlines()

for i in range(5015, 5025):
    print(f"{i+1}: {lines[i]}", end='')

print("---")
# Also check at the end of run_lgbm_mask_generation_triad
for i in range(5330, 5360):
    print(f"{i+1}: {lines[i]}", end='')
