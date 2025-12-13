
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import sys
import argparse

def load_hpo_history(history_files):
    all_trials = []
    for file_path in history_files:
        try:
            with open(file_path, "r") as f:
                history = json.load(f)
                # history is a list of dicts: {'params': {...}, 'score': ..., 'trial_number': ...}
                for trial in history:
                    # Flatten params
                    flat_trial = trial.get('params', {}).copy()
                    flat_trial['score'] = trial.get('score', np.nan)
                    # Add metadata if needed (e.g. layer)
                    flat_trial['source_file'] = Path(file_path).name
                    all_trials.append(flat_trial)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return pd.DataFrame(all_trials)

def analyze_correlations(df, output_path=None):
    if df.empty:
        print("No data to analyze.")
        return

    # Filter numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation with score
    if 'score' not in numeric_df.columns:
        print("Score column missing.")
        return

    correlations = numeric_df.corr(method='spearman')['score'].sort_values(ascending=False)
    
    print("\n=== Parameter-Score Correlations (Spearman) ===")
    print(correlations)
    
    # Save to CSV
    if output_path:
        correlations.to_csv(output_path)
        print(f"\nCorrelations saved to {output_path}")

    return correlations

def main():
    parser = argparse.ArgumentParser(description="Analyze HPO History Correlations")
    parser.add_argument("--outcomes_dir", type=str, default="outcomes", help="Directory containing HPO history JSONs")
    parser.add_argument("--layer", type=str, default="2", choices=["2", "3", "all"], help="Layer to analyze (2, 3, or all)")
    args = parser.parse_args()
    
    layer_pattern = args.layer if args.layer != "all" else "*"
    search_pattern = f"{args.outcomes_dir}/hpo_layer{layer_pattern}_history_*.json"
    
    files = glob(search_pattern)
    if not files:
        print(f"No files found matching {search_pattern}")
        return
        
    print(f"Found {len(files)} history files.")
    
    df = load_hpo_history(files)
    print(f"Loaded {len(df)} trials.")
    
    if args.layer == "all":
        # Analyze separately per layer? Or mixed?
        # Mixed might be messy because params differ.
        pass
        
    analyze_correlations(df, output_path=f"{args.outcomes_dir}/hpo_correlations_layer{args.layer}.csv")

if __name__ == "__main__":
    main()
