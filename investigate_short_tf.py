"""
Investigate SHORT_TF Negative Correlation

This script analyzes why SHORT_TF has negative score-return correlation
while other strategies have positive correlation.

Diagnostic steps:
1. Load latest backtest results
2. Analyze SHORT_TF vs SHORT_MR predictions
3. Check feature importances
4. Verify label generation
5. Identify root cause
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

# Find latest backtest results
data_dir = Path("/Users/remyroche/Documents/Ares/data")
artifacts_dirs = sorted(data_dir.glob("artifacts/202*"))

if not artifacts_dirs:
    print("ERROR: No artifacts directories found")
    exit(1)

latest_dir = artifacts_dirs[-1]
print(f"Using latest artifacts: {latest_dir}")

# Load backtest results
backtest_file = latest_dir / "backtest_results.csv"
if not backtest_file.exists():
    print(f"ERROR: {backtest_file} not found")
    exit(1)

df = pd.read_csv(backtest_file)
print(f"\nLoaded {len(df)} trades")
print(f"Columns: {df.columns.tolist()}")

# Separate by strategy
strategies = {
    "LONG_MR": df[(df["side"] == "long") & (df["dom"] == "mr")],
    "LONG_TF": df[(df["side"] == "long") & (df["dom"] == "tf")],
    "SHORT_MR": df[(df["side"] == "short") & (df["dom"] == "mr")],
    "SHORT_TF": df[(df["side"] == "short") & (df["dom"] == "tf")],
}

print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

for name, strat_df in strategies.items():
    if len(strat_df) == 0:
        continue
    
    scores = strat_df["score"].values
    returns = strat_df["ret"].values
    
    # Remove NaN
    mask = ~(np.isnan(scores) | np.isnan(returns))
    scores = scores[mask]
    returns = returns[mask]
    
    if len(scores) < 10:
        print(f"\n{name}: Too few trades ({len(scores)})")
        continue
    
    corr, pval = spearmanr(scores, returns)
    abs_corr, abs_pval = spearmanr(np.abs(scores), returns)
    
    print(f"\n{name} (n={len(scores)}):")
    print(f"  Score-Return Correlation: {corr:+.4f} (p={pval:.4f})")
    print(f"  |Score|-Return Correlation: {abs_corr:+.4f} (p={abs_pval:.4f})")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  Return range: [{returns.min():.3f}, {returns.max():.3f}]")
    print(f"  Mean score: {scores.mean():.3f}")
    print(f"  Mean return: {returns.mean():.3f}")

print("\n" + "="*80)
print("SHORT_TF vs SHORT_MR COMPARISON")
print("="*80)

short_tf = strategies["SHORT_TF"]
short_mr = strategies["SHORT_MR"]

if len(short_tf) > 0 and len(short_mr) > 0:
    print(f"\nSHORT_TF:")
    print(f"  n={len(short_tf)}")
    print(f"  WR={(short_tf['ret'] > 0).mean():.2%}")
    print(f"  Mean score: {short_tf['score'].mean():.3f}")
    print(f"  Mean return: {short_tf['ret'].mean():.3f}")
    print(f"  Score std: {short_tf['score'].std():.3f}")
    
    print(f"\nSHORT_MR:")
    print(f"  n={len(short_mr)}")
    print(f"  WR={(short_mr['ret'] > 0).mean():.2%}")
    print(f"  Mean score: {short_mr['score'].mean():.3f}")
    print(f"  Mean return: {short_mr['ret'].mean():.3f}")
    print(f"  Score std: {short_mr['score'].std():.3f}")
    
    # Check if SHORT_TF scores are inverted
    print(f"\n" + "="*80)
    print("INVERSION TEST")
    print("="*80)
    
    # Test if inverting SHORT_TF score improves correlation
    tf_scores = short_tf["score"].values
    tf_returns = short_tf["ret"].values
    mask = ~(np.isnan(tf_scores) | np.isnan(tf_returns))
    tf_scores = tf_scores[mask]
    tf_returns = tf_returns[mask]
    
    original_corr, _ = spearmanr(tf_scores, tf_returns)
    inverted_corr, _ = spearmanr(-tf_scores, tf_returns)
    
    print(f"\nSHORT_TF:")
    print(f"  Original correlation: {original_corr:+.4f}")
    print(f"  Inverted correlation: {inverted_corr:+.4f}")
    
    if abs(inverted_corr) > abs(original_corr):
        print(f"\n  ⚠️  INVERSION DETECTED!")
        print(f"  Inverting score would improve correlation by {abs(inverted_corr) - abs(original_corr):.4f}")
    
    # Analyze by score quartiles
    print(f"\n" + "="*80)
    print("QUARTILE ANALYSIS - SHORT_TF")
    print("="*80)
    
    quartiles = pd.qcut(tf_scores, q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"], duplicates='drop')
    
    for q in quartiles.categories:
        q_returns = tf_returns[quartiles == q]
        if len(q_returns) > 0:
            wr = (q_returns > 0).mean()
            mean_ret = q_returns.mean()
            print(f"  {q}: n={len(q_returns):3d}, WR={wr:.2%}, Mean Ret={mean_ret:+.4f}")
    
    # Check if high confidence predicts worse returns
    median_score = np.median(tf_scores)
    high_conf = tf_returns[tf_scores >= median_score]
    low_conf = tf_returns[tf_scores < median_score]
    
    print(f"\n  High confidence (score >= median): WR={(high_conf > 0).mean():.2%}, Mean={high_conf.mean():+.4f}")
    print(f"  Low confidence (score < median):  WR={(low_conf > 0).mean():.2%}, Mean={low_conf.mean():+.4f}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
