
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_probs():
    # Find latest bundle
    bundles = sorted(Path("outcomes").glob("layer2_oof_bundle.joblib"), key=lambda p: p.stat().st_mtime)
    if not bundles:
        print("No bundle found")
        return

    latest_bundle = bundles[-1]
    print(f"Loading {latest_bundle}")
    data = joblib.load(latest_bundle)
    
    # 'l2_score' contains the probability scores
    # Check structure
    key = 'l2_score'
    if key in data:
        scores = data[key]
        if isinstance(scores, pd.DataFrame):
            # If it's a dataframe, it might have one column or be the scores directly
            scores = scores.iloc[:, 0] if scores.shape[1] > 0 else pd.Series()
        elif not isinstance(scores, pd.Series):
             scores = pd.Series(scores)
             
        scores = pd.to_numeric(scores, errors='coerce').dropna()
        
        n_total = len(scores)
        print(f"\nTotal OOF Events: {n_total}")
        
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
            n_pass = (scores > thr).sum()
            pct = (n_pass / n_total) * 100
            print(f"Coverage > {thr:.1f}: {n_pass} events ({pct:.2f}%)")
            
        print("\nStatistics:")
        print(scores.describe())
        
        print("\nTop 20 Most Frequent Scores:")
        print(scores.value_counts().head(20))
        
        cluster_mask = (scores > 0.48) & (scores < 0.52)
        print(f"\nCluster (0.48-0.52) count: {cluster_mask.sum()} ({(cluster_mask.sum()/n_total)*100:.2f}%)")
    else:
        print("Key 'oof_labels' not found in bundle. Keys:", data.keys())

if __name__ == "__main__":
    analyze_probs()
