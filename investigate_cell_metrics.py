#!/usr/bin/env python3
"""Investigate the cell metrics in tbm_parameter_comparison.csv"""

import pandas as pd
from pathlib import Path

comparison_csv = Path("/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/reports/tbm_parameter_comparison.csv")
df = pd.read_csv(comparison_csv)

print("All columns with 'cell' in name:")
cell_cols = [c for c in df.columns if 'cell' in c.lower()]
for col in cell_cols:
    print(f"  {col}")

print("\n\nChecking if min=median (suggests single value, not distribution):")
for col in cell_cols:
    if 'min' in col or 'median' in col:
        continue
    min_col = f"min_{col}"
    med_col = f"median_{col}"
    if min_col in df.columns and med_col in df.columns:
        diff_count = (df[min_col] != df[med_col]).sum()
        print(f"  {col}: {diff_count}/{len(df)} rows have min != median")

print("\n\nSample row with all cell metrics:")
if len(df) > 0:
    row = df.iloc[0]
    for col in cell_cols[:15]:  # First 15
        print(f"  {col}: {row.get(col, 'N/A')}")
