"""
Diagnostic: Check if committee label=1 correlates with return>0

This investigates why committee labels don't correlate with returns
despite TP thresholds being above transaction costs.
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np

# Try to load the committee matrices from a recent HPO run
def analyze_label_return_correlation():
    """Check correlation between labels and returns in committee matrices."""
    
    print("="*60)
    print("Committee Label-Return Correlation Analysis")
    print("="*60)
    
    # Check outcomes directory for relevant data
    outcomes_dir = "outcomes"
    
    # Look for Layer 2 history which might contain diagnostic data
    import json
    import glob
    
    history_files = sorted(glob.glob(f"{outcomes_dir}/hpo_layer2_history_*.json"))
    if not history_files:
        print("No Layer 2 history files found")
        return
    
    latest_history = history_files[-1]
    print(f"Loading: {latest_history}")
    
    with open(latest_history, 'r') as f:
        history = json.load(f)
    
    # Check what diagnostic data is available
    if "trials" in history:
        print(f"Found {len(history['trials'])} trials")
        
        # Look at first trial for structure
        if history["trials"]:
            trial = history["trials"][0]
            print(f"Trial keys: {list(trial.keys())[:20]}")
    
    # The real test: simulate what happens when label=1 but return is negative
    # This requires understanding the triple barrier logic
    print("\n" + "="*60)
    print("Simulating Triple Barrier Logic")
    print("="*60)
    
    # Create simple test case
    np.random.seed(42)
    n_events = 1000
    
    # Simulate triple barrier outcomes
    # profit_threshold = 0.5% (above 0.3% fees)
    # stop_threshold = 0.3%
    
    profit_threshold = 0.005  # 0.5%
    stop_threshold = 0.003    # 0.3%
    transaction_cost = 0.003  # 0.3%
    
    # Generate random returns (simulating MFE/MAE)
    mfe = np.abs(np.random.normal(0, 0.008, n_events))  # Max favorable excursion
    mae = np.abs(np.random.normal(0, 0.005, n_events))  # Max adverse excursion
    
    # Determine exit reason
    hit_profit = mfe >= profit_threshold
    hit_stop = mae >= stop_threshold
    
    # First barrier hit wins
    # Assume profit hits first 50% of time when both hit
    profit_first = np.random.random(n_events) < 0.5
    
    labels = np.zeros(n_events, dtype=int)
    returns = np.zeros(n_events, dtype=float)
    
    for i in range(n_events):
        if hit_profit[i] and hit_stop[i]:
            # Both barriers hit - which came first?
            if profit_first[i]:
                labels[i] = 1
                returns[i] = profit_threshold  # Exit exactly at TP
            else:
                labels[i] = -1
                returns[i] = -stop_threshold  # Exit exactly at SL
        elif hit_profit[i]:
            labels[i] = 1
            returns[i] = profit_threshold
        elif hit_stop[i]:
            labels[i] = -1
            returns[i] = -stop_threshold
        else:
            labels[i] = 0  # Timeout
            returns[i] = np.random.normal(0, 0.002)  # Small random return
    
    # Apply transaction cost
    returns_net = returns - transaction_cost
    
    print(f"\nSimulated {n_events} events:")
    print(f"  Profit hits (label=1): {np.sum(labels == 1)}")
    print(f"  Stop hits (label=-1): {np.sum(labels == -1)}")
    print(f"  Timeouts (label=0): {np.sum(labels == 0)}")
    
    # Key check: When label=1, is return_net > 0?
    label_1_mask = labels == 1
    label_1_returns_net = returns_net[label_1_mask]
    
    print(f"\nWhen label=1 (profit hit):")
    print(f"  Mean net return: {np.mean(label_1_returns_net):.4f}")
    print(f"  % with positive net return: {100*np.mean(label_1_returns_net > 0):.1f}%")
    print(f"  Min net return: {np.min(label_1_returns_net):.4f}")
    print(f"  Max net return: {np.max(label_1_returns_net):.4f}")
    
    # When label=-1, what's the return?
    label_neg1_mask = labels == -1
    label_neg1_returns_net = returns_net[label_neg1_mask]
    
    print(f"\nWhen label=-1 (stop hit):")
    print(f"  Mean net return: {np.mean(label_neg1_returns_net):.4f}")
    print(f"  % with positive net return: {100*np.mean(label_neg1_returns_net > 0):.1f}%")
    
    # Now the critical question: correlation with committee voting system
    print("\n" + "="*60)
    print("Committee Voting Simulation")
    print("="*60)
    
    # 6 experts with different configs
    n_experts = 6
    
    # Simulate expert labels (they should mostly agree for real data)
    # Add some noise to simulate different TP/SL configs
    expert_labels = np.zeros((n_events, n_experts), dtype=int)
    expert_returns = np.zeros((n_events, n_experts), dtype=float)
    
    for exp in range(n_experts):
        # Each expert has slightly different thresholds
        exp_profit_thr = profit_threshold * (0.8 + 0.4 * np.random.random())
        exp_stop_thr = stop_threshold * (0.8 + 0.4 * np.random.random())
        
        exp_hit_profit = mfe >= exp_profit_thr
        exp_hit_stop = mae >= exp_stop_thr
        
        for i in range(n_events):
            if exp_hit_profit[i] and exp_hit_stop[i]:
                if profit_first[i]:
                    expert_labels[i, exp] = 1
                    expert_returns[i, exp] = exp_profit_thr - transaction_cost
                else:
                    expert_labels[i, exp] = -1
                    expert_returns[i, exp] = -exp_stop_thr - transaction_cost
            elif exp_hit_profit[i]:
                expert_labels[i, exp] = 1
                expert_returns[i, exp] = exp_profit_thr - transaction_cost
            elif exp_hit_stop[i]:
                expert_labels[i, exp] = -1
                expert_returns[i, exp] = -exp_stop_thr - transaction_cost
            else:
                expert_labels[i, exp] = 0
                expert_returns[i, exp] = 0
    
    # Committee voting (average of labels)
    fired = expert_labels != 0
    n_fired = np.sum(fired, axis=1)
    
    # Weighted sum of labels
    label_sum = np.sum(expert_labels, axis=1).astype(float)
    committee_score = label_sum / np.maximum(n_fired, 1)
    
    # Committee label: score > 0 => 1
    committee_label = (committee_score > 0).astype(int)
    
    # Average return across experts (for comparison)
    expert_returns_masked = np.where(fired, expert_returns, np.nan)
    avg_return = np.nanmean(expert_returns_masked, axis=1)
    
    print(f"\nCommittee voting results:")
    print(f"  Events with all experts fired: {np.sum(n_fired == n_experts)}")
    print(f"  Events with no expert fired: {np.sum(n_fired == 0)}")
    
    # Key correlation check
    valid_mask = np.isfinite(avg_return) & (n_fired > 0)
    
    print(f"\nCorrelation: committee_label vs avg_return > 0")
    committee_label_valid = committee_label[valid_mask]
    avg_return_positive = (avg_return[valid_mask] > 0).astype(int)
    
    agreement = np.mean(committee_label_valid == avg_return_positive)
    print(f"  Agreement rate: {100*agreement:.1f}%")
    
    # When committee says 1, is avg return positive?
    comm_1_mask = committee_label_valid == 1
    if np.sum(comm_1_mask) > 0:
        print(f"\n  When committee_label=1:")
        print(f"    N events: {np.sum(comm_1_mask)}")
        print(f"    % with avg_return > 0: {100*np.mean(avg_return[valid_mask][comm_1_mask] > 0):.1f}%")
        print(f"    Mean avg_return: {np.mean(avg_return[valid_mask][comm_1_mask]):.4f}")
    
    comm_0_mask = committee_label_valid == 0
    if np.sum(comm_0_mask) > 0:
        print(f"\n  When committee_label=0:")
        print(f"    N events: {np.sum(comm_0_mask)}")
        print(f"    % with avg_return > 0: {100*np.mean(avg_return[valid_mask][comm_0_mask] > 0):.1f}%")
        print(f"    Mean avg_return: {np.mean(avg_return[valid_mask][comm_0_mask]):.4f}")

if __name__ == "__main__":
    analyze_label_return_correlation()
