#!/usr/bin/env python3
"""Fix per-cell export to use actual computed metrics from tbm_parameter_comparison"""

import pandas as pd
from pathlib import Path

# Read the source data with actual metrics
comparison_csv = Path("/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/reports/tbm_parameter_comparison.csv")
per_cell_csv = Path("/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/reports/tbm_best_params_per_cell.csv")

print("Loading tbm_parameter_comparison.csv with actual metrics...")
df_comparison = pd.read_csv(comparison_csv)

print(f"Loaded {len(df_comparison)} rows")
print(f"Columns with 'cell' in name: {[c for c in df_comparison.columns if 'cell' in c.lower()]}")

# Check what we have
print("\nSample actual metrics:")
print(df_comparison[['config_id', 'min_cell_auc', 'median_cell_auc', 'bind', 'ece', 'brier']].head(3).to_string())

# Now read the per-cell CSV that has placeholder values
print("\n\nLoading tbm_best_params_per_cell.csv (with placeholders)...")
df_per_cell = pd.read_csv(per_cell_csv)

print(f"Loaded {len(df_per_cell)} rows")
print("\nSample placeholder metrics:")
print(df_per_cell[['config_id', 'cell_auc', 'cell_bind', 'cell_ece', 'cell_brier']].head(3).to_string())

# The fix: map actual metrics from comparison to per_cell
# The key columns to map
metric_map = {
    'cell_auc': 'min_cell_auc',  # or median_cell_auc
    'cell_auc_bound': 'min_cell_auc_bound',
    'cell_bind': 'bind',
    'cell_timeout': 'timeout_rate',
    'cell_ece': 'ece',
    'cell_brier': 'brier',
    'cell_tp_sep': 'min_cell_tp_sep',
    'cell_ap_lift': 'min_cell_ap_lift',
}

print("\n\nFixing per-cell CSV with actual metrics...")

# For each row in per_cell, find matching config in comparison and copy metrics
for idx, row in df_per_cell.iterrows():
    config_id = row.get('config_id')
    if pd.isna(config_id):
        continue
    
    # Find matching row in comparison
    match = df_comparison[df_comparison['config_id'] == config_id]
    if len(match) > 0:
        m = match.iloc[0]
        # Copy actual metrics
        df_per_cell.at[idx, 'cell_auc'] = m.get('min_cell_auc', row.get('cell_auc'))
        df_per_cell.at[idx, 'cell_auc_bound'] = m.get('min_cell_auc_bound', row.get('cell_auc_bound'))
        df_per_cell.at[idx, 'cell_bind'] = m.get('bind', row.get('cell_bind'))
        df_per_cell.at[idx, 'cell_timeout'] = m.get('timeout_rate', row.get('cell_timeout'))
        df_per_cell.at[idx, 'cell_ece'] = m.get('ece', row.get('cell_ece'))
        df_per_cell.at[idx, 'cell_brier'] = m.get('brier', row.get('cell_brier'))
        df_per_cell.at[idx, 'cell_tp_sep'] = m.get('min_cell_tp_sep', row.get('cell_tp_sep'))

# Save the fixed CSV
df_per_cell.to_csv(per_cell_csv, index=False)
print(f"\n✅ Fixed {per_cell_csv} with actual computed metrics!")

# Show sample of fixed data
print("\nSample fixed metrics:")
print(df_per_cell[['config_id', 'cell_auc', 'cell_bind', 'cell_ece', 'cell_brier']].head(5).to_string())
