
import pandas as pd
import numpy as np
import os
from tabulate import tabulate

ARTIFACT_DIR = "data/artifacts/20260204_220000/labels"

def analyze_file(filename):
    path = os.path.join(ARTIFACT_DIR, filename)
    if not os.path.exists(path):
        print(f"Skipping {filename}: Not found")
        return

    print(f"\n{'='*50}")
    print(f"Analyzing: {filename}")
    print(f"{'='*50}")

    try:
        df = pd.read_parquet(path)
        print(f"Shape: {df.shape}")
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Summary Statistics
        if not numeric_cols.empty:
            stats = df[numeric_cols].describe().T
            stats["nan_count"] = df[numeric_cols].isna().sum()
            print("\nStatistics:")
            print(tabulate(stats, headers="keys", tablefmt="simple", floatfmt=".4f"))
        else:
            print("\nNo numeric columns found.")
            
        # Class Balance (if applicable)
        for target in ["y", "__y__", "target", "label"]:
            if target in df.columns:
                print(f"\nTarget distribution ({target}):")
                counts = df[target].value_counts()
                props = df[target].value_counts(normalize=True)
                dist = pd.concat([counts, props], axis=1, keys=["Count", "Proportion"])
                print(tabulate(dist, headers="keys", tablefmt="simple", floatfmt=".4f"))
        
        # Check specific columns
        if "__w__" in df.columns:
            print("\nWeights (__w__) distributions:")
            print(df["__w__"].describe())

    except Exception as e:
        print(f"Error reading {filename}: {e}")

def main():
    files = ["spike_anatomy.parquet", "exh_up.parquet", "exh_down.parquet", "exhaustion_history.parquet"]
    for f in files:
        analyze_file(f)

if __name__ == "__main__":
    main()
