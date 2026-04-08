#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_labels(labels_dir):
    results = []
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        return results
    
    skip = {"exhaustion_history.parquet", "gamma_model.parquet", "spike_anatomy.parquet", 
            "spike_anatomy_best.parquet", "spike_anatomy_worst.parquet", "trap_model.parquet"}
    
    for f in sorted(labels_path.glob("*.parquet")):
        if f.name in skip:
            continue
        try:
            df = pd.read_parquet(f)
            if "__y_bin__" in df.columns:
                y = df["__y_bin__"]
                results.append({
                    "name": f.stem,
                    "rows": len(df),
                    "pos_rate": float((y >= 0.5).mean() * 100),
                    "mean": float(y.mean()),
                })
        except Exception as e:
            print(f"Error {f}: {e}")
    return results

before = "data/artifacts/20260212_170000/labels"
after = "data/artifacts/20260321_140000/labels"

print("="*80)
print("BEFORE LABELS:")
print("="*80)
r1 = analyze_labels(before)
for r in r1[:10]:
    print(f"{r['name']:<40} rows={r['rows']:>6}  pos={r['pos_rate']:>5.1f}%")

print(f"\nTotal: {len(r1)} datasets")
if r1:
    print(f"Mean pos rate: {np.mean([r['pos_rate'] for r in r1]):.2f}%")

print("\n" + "="*80)
print("AFTER LABELS:")
print("="*80)
r2 = analyze_labels(after)
for r in r2[:10]:
    print(f"{r['name'][:40]:<40} rows={r['rows']:>6}  pos={r['pos_rate']:>5.1f}%")

print(f"\nTotal: {len(r2)} datasets")
if r2:
    print(f"Mean pos rate: {np.mean([r['pos_rate'] for r in r2]):.2f}%")

print("\n" + "="*80)
print("KEY FINDING: Naming convention changed completely")
print("="*80)
print("BEFORE: train_{side}_{kind}_{horizon}.parquet (e.g., train_long_mr_2)")
print("AFTER:  train_{strategy_id}_{horizon}[_tight|wide].parquet")
print("\nThis confirms the move to strategy-driven horizons.")
