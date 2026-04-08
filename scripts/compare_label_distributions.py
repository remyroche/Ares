#!/usr/bin/env python3
"""
Compare __y_bin__ distributions and geometry selection parameters before/after changes.
This script analyzes label datasets from two artifact directories.

Usage:
    python scripts/compare_label_distributions.py --before data/artifacts/20260212_170000 --after data/artifacts/20260321_140000
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


def load_label_dataset(path: str) -> pd.DataFrame:
    """Load a label parquet file."""
    return pd.read_parquet(path)


def analyze_y_bin_distribution(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Analyze the __y_bin__ distribution in a label dataset."""
    result = {"name": name, "total_rows": len(df)}
    
    if "__y_bin__" in df.columns:
        y_bin = df["__y_bin__"]
        result["y_bin_positive"] = int((y_bin >= 0.5).sum())
        result["y_bin_negative"] = int((y_bin < 0.5).sum())
        result["y_bin_nan"] = int(y_bin.isna().sum())
        result["y_bin_mean"] = float(y_bin.mean())
        result["y_bin_std"] = float(y_bin.std())
        result["positive_rate"] = float((y_bin >= 0.5).mean() * 100)
    else:
        result["y_bin_positive"] = None
        result["y_bin_negative"] = None
        result["y_bin_nan"] = None
        result["y_bin_mean"] = None
        result["y_bin_std"] = None
        result["positive_rate"] = None
    
    # Check for __y_outcome__ (raw triple-barrier labels: 0=SL, 1=TO, 2=TP)
    if "__y_outcome__" in df.columns:
        y_outcome = df["__y_outcome__"]
        result["outcome_sl"] = int((y_outcome == 0).sum())
        result["outcome_to"] = int((y_outcome == 1).sum())
        result["outcome_tp"] = int((y_outcome == 2).sum())
        total_outcomes = result["outcome_sl"] + result["outcome_to"] + result["outcome_tp"]
        if total_outcomes > 0:
            result["outcome_sl_rate"] = float(result["outcome_sl"] / total_outcomes * 100)
            result["outcome_to_rate"] = float(result["outcome_to"] / total_outcomes * 100)
            result["outcome_tp_rate"] = float(result["outcome_tp"] / total_outcomes * 100)
        else:
            result["outcome_sl_rate"] = None
            result["outcome_to_rate"] = None
            result["outcome_tp_rate"] = None
    else:
        result["outcome_sl"] = None
        result["outcome_to"] = None
        result["outcome_tp"] = None
        result["outcome_sl_rate"] = None
        result["outcome_to_rate"] = None
        result["outcome_tp_rate"] = None
    
    # Check for __y_ret__ (realized returns)
    if "__y_ret__" in df.columns:
        y_ret = df["__y_ret__"]
        valid_ret = y_ret.dropna()
        result["y_ret_mean"] = float(valid_ret.mean()) if len(valid_ret) > 0 else None
        result["y_ret_std"] = float(valid_ret.std()) if len(valid_ret) > 0 else None
        result["y_ret_median"] = float(valid_ret.median()) if len(valid_ret) > 0 else None
        result["y_ret_positive_share"] = float((valid_ret > 0).mean() * 100) if len(valid_ret) > 0 else None
    else:
        result["y_ret_mean"] = None
        result["y_ret_std"] = None
        result["y_ret_median"] = None
        result["y_ret_positive_share"] = None
    
    # Check for __w__ (sample weights)
    if "__w__" in df.columns:
        w = df["__w__"]
        valid_w = w.dropna()
        result["w_mean"] = float(valid_w.mean()) if len(valid_w) > 0 else None
        result["w_std"] = float(valid_w.std()) if len(valid_w) > 0 else None
        result["w_min"] = float(valid_w.min()) if len(valid_w) > 0 else None
        result["w_max"] = float(valid_w.max()) if len(valid_w) > 0 else None
    else:
        result["w_mean"] = None
        result["w_std"] = None
        result["w_min"] = None
        result["w_max"] = None
    
    return result


def analyze_all_labels(labels_dir: str) -> List[Dict[str, Any]]:
    """Analyze all label files in a directory."""
    results = []
    labels_path = Path(labels_dir)
    
    if not labels_path.exists():
        print(f"ERROR: Labels directory not found: {labels_dir}")
        return results
    
    skip_files = {
        "exhaustion_history.parquet", "gamma_model.parquet", 
        "spike_anatomy.parquet", "spike_anatomy_best.parquet", 
        "spike_anatomy_worst.parquet", "spike_oof_best.parquet", 
        "spike_oof_worst.parquet", "trap_model.parquet",
        "exh_up.parquet", "exh_down.parquet"
    }
    
    for parquet_file in sorted(labels_path.glob("*.parquet")):
        if parquet_file.name in skip_files:
            continue
        
        try:
            df = load_label_dataset(str(parquet_file))
            result = analyze_y_bin_distribution(df, parquet_file.stem)
            results.append(result)
        except Exception as e:
            print(f"ERROR loading {parquet_file}: {e}")
    
    return results


def compare_distributions(results1: List[Dict], results2: List[Dict]) -> None:
    """Compare two sets of label distributions."""
    print("\n" + "="*100)
    print("LABEL DISTRIBUTION COMPARISON")
    print("="*100)
    
    # Create lookup by name
    r1_by_name = {r["name"]: r for r in results1}
    r2_by_name = {r["name"]: r for r in results2}
    
    common_names = sorted(set(r1_by_name.keys()) & set(r2_by_name.keys()))
    
    if not common_names:
        print("\nNo common datasets found between the two artifact directories.")
        print(f"Before datasets: {list(r1_by_name.keys())[:5]}...")
        print(f"After datasets: {list(r2_by_name.keys())[:5]}...")
        return
    
    print(f"\n{'Dataset':<35} {'Before Pos%':>12} {'After Pos%':>12} {'Delta':>10}")
    print("-"*75)
    
    for name in common_names:
        r1 = r1_by_name[name]
        r2 = r2_by_name[name]
        
        pr1 = r1.get("positive_rate")
        pr2 = r2.get("positive_rate")
        if pr1 is not None and pr2 is not None:
            delta = pr2 - pr1
            delta_str = f"{delta:+.2f}%"
            if abs(delta) > 5:
                delta_str += " ***"
            elif abs(delta) > 2:
                delta_str += " **"
            elif abs(delta) > 1:
                delta_str += " *"
            print(f"{name:<35} {pr1:>11.2f}% {pr2:>11.2f}% {delta_str:>10}")
        else:
            print(f"{name:<35} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
    
    # Summary statistics
    print("\n" + "-"*75)
    print("SUMMARY STATISTICS")
    print("-"*75)
    
    pr1_values = [r.get("positive_rate") for r in results1 if r.get("positive_rate") is not None]
    pr2_values = [r.get("positive_rate") for r in results2 if r.get("positive_rate") is not None]
    
    if pr1_values and pr2_values:
        print(f"Mean positive rate (before): {np.mean(pr1_values):.2f}%")
        print(f"Mean positive rate (after):  {np.mean(pr2_values):.2f}%")
        print(f"Mean delta: {np.mean(pr2_values) - np.mean(pr1_values):+.2f}%")
    
    # Outcome distribution comparison
    print("\n" + "-"*75)
    print("OUTCOME DISTRIBUTION (TP/SL/TO)")
    print("-"*75)
    
    print(f"\n{'Dataset':<25} {'SL%':>6} {'TO%':>6} {'TP%':>6} || {'SL%':>6} {'TO%':>6} {'TP%':>6}")
    print(f"{'':25} {'--- BEFORE ---':>22} || {'--- AFTER ----':>22}")
    print("-"*75)
    
    for name in common_names:
        r1 = r1_by_name[name]
        r2 = r2_by_name[name]
        
        sl1 = r1.get("outcome_sl_rate")
        to1 = r1.get("outcome_to_rate")
        tp1 = r1.get("outcome_tp_rate")
        sl2 = r2.get("outcome_sl_rate")
        to2 = r2.get("outcome_to_rate")
        tp2 = r2.get("outcome_tp_rate")
        
        if all(v is not None for v in [sl1, to1, tp1, sl2, to2, tp2]):
            print(f"{name:<25} {sl1:>5.1f}% {to1:>5.1f}% {tp1:>5.1f}% || {sl2:>5.1f}% {to2:>5.1f}% {tp2:>5.1f}%")
    
    # Return comparison
    print("\n" + "-"*75)
    print("REALIZED RETURN STATISTICS")
    print("-"*75)
    
    print(f"\n{'Dataset':<25} {'Mean':>8} {'Std':>8} {'Pos%':>8} || {'Mean':>8} {'Std':>8} {'Pos%':>8}")
    print(f"{'':25} {'--- BEFORE ---':>26} || {'--- AFTER ----':>26}")
    print("-"*85)
    
    for name in common_names:
        r1 = r1_by_name[name]
        r2 = r2_by_name[name]
        
        m1, s1, p1 = r1.get("y_ret_mean"), r1.get("y_ret_std"), r1.get("y_ret_positive_share")
        m2, s2, p2 = r2.get("y_ret_mean"), r2.get("y_ret_std"), r2.get("y_ret_positive_share")
        
        if all(v is not None for v in [m1, s1, p1, m2, s2, p2]):
            m1_s = f"{m1*100:.3f}%" if abs(m1) < 1 else f"{m1:.4f}"
            m2_s = f"{m2*100:.3f}%" if abs(m2) < 1 else f"{m2:.4f}"
            s1_s = f"{s1*100:.3f}%" if abs(s1) < 1 else f"{s1:.4f}"
            s2_s = f"{s2*100:.3f}%" if abs(s2) < 1 else f"{s2:.4f}"
            print(f"{name:<25} {m1_s:>8} {s1_s:>8} {p1:>7.1f}% || {m2_s:>8} {s2_s:>8} {p2:>7.1f}%")


def load_bucket_params(artifact_dir: str) -> Optional[Dict[str, Any]]:
    """Load bucket_params.json from an artifact directory."""
    bucket_path = Path(artifact_dir) / "models" / "bucket_params.json"
    if bucket_path.exists():
        with open(bucket_path, "r") as f:
            return json.load(f)
    return None


def compare_geometries(artifact_dir1: str, artifact_dir2: str) -> None:
    """Compare geometry selection between two artifact directories."""
    
    print("\n" + "="*100)
    print("GEOMETRY SELECTION COMPARISON (bucket_params.json)")
    print("="*100)
    
    bucket1 = load_bucket_params(str(artifact_dir1))
    bucket2 = load_bucket_params(str(artifact_dir2))
    
    if bucket1 is None:
        print(f"WARNING: bucket_params.json not found in {artifact_dir1}")
    if bucket2 is None:
        print(f"WARNING: bucket_params.json not found in {artifact_dir2}")
        return
    
    buckets1 = bucket1.get("buckets", {})
    buckets2 = bucket2.get("buckets", {})
    
    common_buckets = sorted(set(buckets1.keys()) & set(buckets2.keys()))
    
    for bucket_name in common_buckets:
        b1 = buckets1[bucket_name]
        b2 = buckets2[bucket_name]
        
        print(f"\n{'='*60}")
        print(f"BUCKET: {bucket_name}")
        print(f"{'='*60}")
        
        # TP/SL comparison
        tp_sl1 = b1.get("tp_sl", {})
        tp_sl2 = b2.get("tp_sl", {})
        
        print(f"\n{'Parameter':<25} {'Before':>12} {'After':>12} {'Changed':>10}")
        print("-"*60)
        
        for key in ["sl_mult", "tp_mult", "atr_scale_hi", "atr_scale_lo"]:
            v1 = tp_sl1.get(key)
            v2 = tp_sl2.get(key)
            if v1 is not None or v2 is not None:
                changed = "YES" if v1 != v2 else ""
                print(f"tp_sl.{key:<18} {str(v1):>12} {str(v2):>12} {changed:>10}")
        
        # Position sizing comparison
        ps1 = b1.get("position_sizing", {})
        ps2 = b2.get("position_sizing", {})
        
        for key in ["c0", "k", "s_min", "s_max"]:
            v1 = ps1.get(key)
            v2 = ps2.get(key)
            if v1 is not None or v2 is not None:
                changed = "YES" if v1 != v2 else ""
                print(f"sizing.{key:<18} {str(v1):>12} {str(v2):>12} {changed:>10}")
        
        # Profit exit comparison
        pe1 = b1.get("profit_exit", {})
        pe2 = b2.get("profit_exit", {})
        
        for key in ["act_n", "be_act_n", "d_min", "d_max"]:
            v1 = pe1.get(key)
            v2 = pe2.get(key)
            if v1 is not None or v2 is not None:
                changed = "YES" if v1 != v2 else ""
                print(f"profit_exit.{key:<14} {str(v1):>12} {str(v2):>12} {changed:>10}")
        
        # Evaluation comparison
        eval1 = b1.get("evaluation", {})
        eval2 = b2.get("evaluation", {})
        
        print(f"\n{'Evaluation Metric':<25} {'Before':>12} {'After':>12} {'Delta':>12}")
        print("-"*65)
        
        for key in ["holdout_pnl_net", "holdout_trades", "holdout_win_rate"]:
            v1 = eval1.get(key)
            v2 = eval2.get(key)
            if v1 is not None and v2 is not None:
                if isinstance(v1, float):
                    delta = v2 - v1
                    print(f"{key:<25} {v1:>12.4f} {v2:>12.4f} {delta:>+12.4f}")
                else:
                    changed = "YES" if v1 != v2 else ""
                    print(f"{key:<25} {str(v1):>12} {str(v2):>12} {changed:>12}")"
