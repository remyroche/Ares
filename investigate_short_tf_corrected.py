"""
Enhanced SHORT_TF Investigation - Accounting for Short Direction

This script properly accounts for the fact that:
- For LONGS: positive score should predict positive returns
- For SHORTS: MORE NEGATIVE score should predict positive returns
  (i.e., we should use |score| for shorts)
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

# Load latest backtest results
data_dir = Path("/Users/remyroche/Documents/Ares/data")
artifacts_dirs = sorted(data_dir.glob("artifacts/202*"))
latest_dir = artifacts_dirs[-1]
backtest_file = latest_dir / "backtest_results.csv"

df = pd.read_csv(backtest_file)
print(f"Loaded {len(df)} trades from {latest_dir.name}\n")

# Separate by strategy
strategies = {
    "LONG_MR": df[(df["side"] == "long") & (df["dom"] == "mr")],
    "LONG_TF": df[(df["side"] == "long") & (df["dom"] == "tf")],
    "SHORT_MR": df[(df["side"] == "short") & (df["dom"] == "mr")],
    "SHORT_TF": df[(df["side"] == "short") & (df["dom"] == "tf")],
}

print("="*80)
print("CORRECTED CORRELATION ANALYSIS")
print("="*80)
print("\nFor LONGS: Use raw score (positive score → positive return)")
print("For SHORTS: Use |score| (more negative score → positive return)\n")

for name, strat_df in strategies.items():
    if len(strat_df) == 0:
        continue
    
    scores = strat_df["score"].values
    returns = strat_df["ret"].values
    side = strat_df["side"].iloc[0]
    
    # Remove NaN
    mask = ~(np.isnan(scores) | np.isnan(returns))
    scores = scores[mask]
    returns = returns[mask]
    
    if len(scores) < 10:
        continue
    
    # For shorts, use |score| for proper correlation
    if side == "short":
        confidence = np.abs(scores)
        corr_label = "|Score|-Return"
    else:
        confidence = scores
        corr_label = "Score-Return"
    
    corr, pval = spearmanr(confidence, returns)
    
    print(f"{name} (n={len(scores)}):")
    print(f"  {corr_label} Correlation: {corr:+.4f} (p={pval:.4f})")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"  Return range: [{returns.min():.3f}, {returns.max():.3f}]")
    
    # Quartile analysis by confidence
    quartiles = pd.qcut(confidence, q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"], duplicates='drop')
    
    print(f"  Quartile Analysis:")
    for q in quartiles.categories:
        q_returns = returns[quartiles == q]
        if len(q_returns) > 0:
            wr = (q_returns > 0).mean()
            mean_ret = q_returns.mean()
            print(f"    {q}: n={len(q_returns):3d}, WR={wr:.1%}, Mean={mean_ret:+.4f}")
    
    # High vs low confidence
    median_conf = np.median(confidence)
    high_conf = returns[confidence >= median_conf]
    low_conf = returns[confidence < median_conf]
    
    print(f"  High confidence: WR={(high_conf > 0).mean():.1%}, Mean={high_conf.mean():+.4f}")
    print(f"  Low confidence:  WR={(low_conf > 0).mean():.1%}, Mean={low_conf.mean():+.4f}")
    print()

print("="*80)
print("KEY INSIGHTS")
print("="*80)

# Compare SHORT_TF vs SHORT_MR
short_tf = strategies["SHORT_TF"]
short_mr = strategies["SHORT_MR"]

if len(short_tf) > 0 and len(short_mr) > 0:
    tf_conf = np.abs(short_tf["score"].values)
    tf_ret = short_tf["ret"].values
    mask = ~(np.isnan(tf_conf) | np.isnan(tf_ret))
    tf_conf = tf_conf[mask]
    tf_ret = tf_ret[mask]
    
    mr_conf = np.abs(short_mr["score"].values)
    mr_ret = short_mr["ret"].values
    mask = ~(np.isnan(mr_conf) | np.isnan(mr_ret))
    mr_conf = mr_conf[mask]
    mr_ret = mr_ret[mask]
    
    tf_corr, tf_p = spearmanr(tf_conf, tf_ret)
    mr_corr, mr_p = spearmanr(mr_conf, mr_ret)
    
    print(f"\nSHORT_TF: Correlation={tf_corr:+.4f} (p={tf_p:.4f})")
    print(f"SHORT_MR: Correlation={mr_corr:+.4f} (p={mr_p:.4f})")
    
    if tf_corr < 0 and mr_corr > 0:
        print(f"\n⚠️  SHORT_TF has NEGATIVE correlation while SHORT_MR is POSITIVE!")
        print(f"   This confirms SHORT_TF is broken: higher confidence → worse returns")
    elif abs(tf_corr) < 0.05:
        print(f"\n⚠️  SHORT_TF has NEAR-ZERO correlation (not significant)")
        print(f"   Model is not learning useful patterns")

# Compare LONG_MR vs LONG_TF
long_mr = strategies["LONG_MR"]
long_tf = strategies["LONG_TF"]

if len(long_mr) > 0 and len(long_tf) > 0:
    mr_scores = long_mr["score"].values
    mr_ret = long_mr["ret"].values
    mask = ~(np.isnan(mr_scores) | np.isnan(mr_ret))
    mr_scores = mr_scores[mask]
    mr_ret = mr_ret[mask]
    
    tf_scores = long_tf["score"].values
    tf_ret = long_tf["ret"].values
    mask = ~(np.isnan(tf_scores) | np.isnan(tf_ret))
    tf_scores = tf_scores[mask]
    tf_ret = tf_ret[mask]
    
    mr_corr, mr_p = spearmanr(mr_scores, mr_ret)
    tf_corr, tf_p = spearmanr(tf_scores, tf_ret)
    
    print(f"\nLONG_MR: Correlation={mr_corr:+.4f} (p={mr_p:.4f})")
    print(f"LONG_TF: Correlation={tf_corr:+.4f} (p={tf_p:.4f})")
    
    if abs(mr_corr) < 0.05 and abs(tf_corr) < 0.05:
        print(f"\n⚠️  Both LONG strategies have near-zero correlation")
        print(f"   Meta models need retraining with specialist features")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
