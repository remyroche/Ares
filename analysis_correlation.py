
import pandas as pd
import numpy as np
import os

# Load CSV
csv_path = "/Users/remyroche/Documents/Ares/outcomes/meta_labeling_hpo_candidate_pool_ETHUSDT_15m_long_20251210_233144.csv"
if not os.path.exists(csv_path):
    print(f"Error: CSV not found at {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)

# Select relevant columns
targets = ['trades_per_day', 'pnl_per_day', 'pnl_avg_ret_per_trade', 'edge', 'combined', 'mean_auc']

# Potential parameters (intersection of list and available columns)
potential_features = [
    'horizon_bars', 'cusum_threshold', 'target_signal_density', 'min_event_spacing', 'trail_distance',
    'profit_mult_min', 'profit_mult_max', 'stop_mult_min', 'stop_mult_max', 'kalman_Q', 'kalman_R',
    'label_low_q', 'label_high_q', 'econ_min_return_multiple', 'iso_min_prob', 'target_clip_high_q',
    'signal_strength_scale_max', 'r_multiple_pos_threshold', 'transaction_cost_mult',
    'vol_baseline_window', 'profit_thr_base', 'stop_to_profit_ratio', 'scale_pos_weight'
]

features = [f for f in potential_features if f in df.columns]

# Compute correlation
results_str = []
results_str.append("# Correlation Analysis\n")
results_str.append(f"Source: `{csv_path}`\n")
results_str.append(f"Rows: {len(df)}\n")

for target in targets:
    if target not in df.columns:
        results_str.append(f"## Target {target} (Not Found)\n")
        continue
    
    corrs = df[features].corrwith(df[target]).sort_values(ascending=False)
    results_str.append(f"## Correlations with `{target}`\n")
    results_str.append("| Feature | Correlation |\n| --- | --- |")
    for idx, val in corrs.items():
        results_str.append(f"| {idx} | {val:.4f} |")
    results_str.append("\n")

# Save to file
out_path = "outcomes/hpo_correlation_analysis.md"
with open(out_path, "w") as f:
    f.write("\n".join(results_str))

print(f"Correlation analysis saved to {out_path}")
