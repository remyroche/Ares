#!/usr/bin/env python3
"""Check tbm_geometry_grid.csv for true per-cell metrics"""

import pandas as pd
from pathlib import Path

grid_csv = Path("/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/reports/tbm_geometry_grid.csv")

df = pd.read_csv(grid_csv)

print(f"Geometry grid: {len(df)} rows")
print(f"Columns: {list(df.columns)[:20]}...")  # First 20 columns

print("\n\nUnique cell_keys:")
print(df['cell_key'].value_counts().head(10))

print("\n\nSample per-cell metrics (first 3 rows):")
metric_cols = ['cell_key', 'config_id', 'cell_auc', 'cell_bind', 'cell_ece', 'cell_brier', 'k_tp', 'sl_as_tp_pct']
available_cols = [c for c in metric_cols if c in df.columns]
print(df[available_cols].head(3).to_string())
