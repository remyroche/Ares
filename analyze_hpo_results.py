import json
import pandas as pd
from pathlib import Path
import glob
import sys

def find_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=lambda p: Path(p).stat().st_mtime)

def analyze_layer2(outcomes_dir):
    print("\n--- Layer 2 Analysis ---")
    
    # 1. Unified Report for basic metrics
    report_json = find_latest_file(f"{outcomes_dir}/label_based_unified_report_*.json")
    if report_json:
        with open(report_json, 'r') as f:
            data = json.load(f)
            l2 = data.get('layer2', {})
            print(f"Coverage: {l2.get('coverage')}")
            print(f"Pos Rate: {l2.get('pos_rate')}")
            # AUC is usually not in L2 metrics directly unless computed from OOF
    else:
        print("No unified report found.")

    # 2. Winning ML models from selected geometries
    geo_json = find_latest_file(f"{outcomes_dir}/layer2_selected_geometries.json")
    if geo_json:
        with open(geo_json, 'r') as f:
            geometries = json.load(f)
            print(f"Selected Geometries Count: {len(geometries)}")
            
            # Count families
            families = {}
            for geo in geometries:
                fam = geo.get('barrier_family', 'unknown')
                families[fam] = families.get(fam, 0) + 1
            print("Winning Barrier Families:", json.dumps(families, indent=2))
    else:
        print("No layer2_selected_geometries.json found.")

def analyze_layer3(outcomes_dir):
    print("\n--- Layer 3 Analysis ---")
    report_json = find_latest_file(f"{outcomes_dir}/label_based_unified_report_*.json")
    if report_json:
        with open(report_json, 'r') as f:
            data = json.load(f)
            l3 = data.get('layer3', {})
            print(f"AUC: {l3.get('auc')}")
            print(f"Log Loss: {l3.get('log_loss')}")
            print(f"ECE: {l3.get('ece')}")
            print(f"N Eval: {l3.get('n_eval')}")
            print(f"Prob Mean: {l3.get('prob_mean')}")
    else:
        print("No unified report found.")

def analyze_layer4(outcomes_dir):
    print("\n--- Layer 4 Analysis ---")
    report_json = find_latest_file(f"{outcomes_dir}/label_based_unified_report_*.json")
    if report_json:
        with open(report_json, 'r') as f:
            data = json.load(f)
            l5 = data.get('layer5', {}) # Layer 5 metrics often contain Layer 4 gate metrics
            
            # Coverage / Trades Accepted
            n_events = data.get('layer3', {}).get('n_eval', 0)
            n_gate = l5.get('n_prob_ge_pmin', 0)
            coverage = n_gate / n_events if n_events > 0 else 0
            print(f"Coverage (Events Accepted): {coverage:.2%} ({n_gate}/{n_events})")
            
            # Trades/Day
            # Need date range or total days. L2 report might have it.
            # Assuming ~15m bars, 96/day.
            
            print(f"Avg Trade PnL: {l5.get('avg_trade_pnl')}")
            print(f"Win Rate: {l5.get('win_rate')}")
            
            # Compare to base win rate if available (from L2 pos_rate)
            l2_pos_rate = data.get('layer2', {}).get('pos_rate', 0)
            print(f"Base Win Rate (L2): {l2_pos_rate}")
            if l2_pos_rate:
                print(f"Win Rate Delta: {l5.get('win_rate', 0) - l2_pos_rate}")

    else:
        print("No unified report found.")

def analyze_layer5(outcomes_dir):
    print("\n--- Layer 5 Analysis ---")
    # PnL/day at different thresholds
    sweep_csv = find_latest_file(f"{outcomes_dir}/layer5_pmin_sweep_*.csv")
    if sweep_csv:
        df = pd.read_csv(sweep_csv)
        print("PnL/Day and Metrics at different p_min thresholds:")
        # Select relevant columns
        cols = ['p_min', 'n_trades', 'avg_trade_pnl', 'total_pnl', 'win_rate', 'profit_factor']
        if 'n_days' not in df.columns:
            # Estimate days from n_trades / trades_per_day if available? 
            # Or just total PnL. User asked for PnL/day.
            # I'll just print total PnL for now and assume user knows duration or I can find it.
             print(df[cols].to_markdown(index=False))
             
        # Start/End date to calc PnL/Day?
        # Maybe check unified report context
    else:
        print("No layer5_pmin_sweep_*.csv found.")

if __name__ == "__main__":
    outcomes_dir = "outcomes"
    if len(sys.argv) > 1:
        outcomes_dir = sys.argv[1]
    
    print(f"Analyzing outcomes in: {outcomes_dir}")
    analyze_layer2(outcomes_dir)
    analyze_layer3(outcomes_dir)
    analyze_layer4(outcomes_dir)
    analyze_layer5(outcomes_dir)
