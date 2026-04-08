#!/usr/bin/env python3
"""Fix TBM metric export by mapping actual computed metrics from comparison to per-cell CSV"""

import pandas as pd
from pathlib import Path

comparison_csv = Path("/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/reports/tbm_parameter_comparison.csv")
per_cell_csv = Path("/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/reports/tbm_best_params_per_cell.csv")

print("Loading comparison CSV with actual metrics...")
df_comparison = pd.read_csv(comparison_csv)
print(f"Loaded {len(df_comparison)} rows")

print("\nLoading per-cell CSV with placeholder metrics...")
df_per_cell = pd.read_csv(per_cell_csv)
print(f"Loaded {len(df_per_cell)} rows")

# Create mapping from config_id to actual metrics
metric_mapping = {}
for _, row in df_comparison.iterrows():
    config_id = row.get('config_id')
    if pd.notna(config_id):
        metric_mapping[config_id] = {
            'cell_auc': row.get('min_cell_auc', row.get('median_cell_auc')),
            'cell_auc_bound': row.get('min_cell_auc_bound', row.get('median_cell_auc_bound')),
            'cell_bind': row.get('bind'),
            'cell_timeout': row.get('timeout_rate'),
            'cell_ece': row.get('ece'),
            'cell_brier': row.get('brier'),
            'cell_tp_sep': row.get('min_cell_tp_sep'),
            'cell_ap_lift': row.get('min_cell_ap_lift'),
            'cell_score': row.get('stage2_score', row.get('min_cell_auc')),
        }

print(f"\nFound metrics for {len(metric_mapping)} configs")

# Update per-cell CSV with actual metrics
updated_count = 0
for idx, row in df_per_cell.iterrows():
    config_id = row.get('config_id')
    if config_id in metric_mapping:
        metrics = metric_mapping[config_id]
        for metric, value in metrics.items():
            if pd.notna(value):
                df_per_cell.at[idx, metric] = float(value)
        updated_count += 1

print(f"\nUpdated {updated_count}/{len(df_per_cell)} rows with actual metrics")

# Save the fixed CSV
df_per_cell.to_csv(per_cell_csv, index=False)
print(f"\n✅ Saved fixed CSV to {per_cell_csv}")

# Show sample of fixed data
print("\nSample fixed metrics:")
print(df_per_cell[['config_id', 'cell_auc', 'cell_bind', 'cell_ece', 'cell_brier']].head(5).to_string())
