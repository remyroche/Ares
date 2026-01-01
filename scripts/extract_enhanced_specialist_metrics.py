#!/usr/bin/env python3
"""
Extract Enhanced Specialists Performance Metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def extract_enhanced_specialist_metrics():
    """Extract metrics from enhanced specialists outputs."""
    
    # Look for recent diagnostic outputs
    outcomes_dir = Path("outcomes")
    
    # Find the most recent diagnostic files
    diagnostic_files = list(outcomes_dir.glob("*specialist*diagnostics*.csv"))
    
    if not diagnostic_files:
        print("❌ No diagnostic files found")
        return
    
    # Get the most recent file
    latest_file = max(diagnostic_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Analyzing: {latest_file}")
    
    # Load the data
    df = pd.read_csv(latest_file)
    
    # Extract specialist metrics
    specialist_metrics = {}
    
    # Filter for model_reliability rows (these contain the specialist metrics)
    model_rows = df[df['row_type'] == 'model_reliability']
    
    for _, row in model_rows.iterrows():
        specialist_name = row['feature']
        
        # Extract metrics
        mi_mean_avg = row.get('mi_mean_avg', 0)
        r2_mean = row.get('r2_mean', 0)
        n_features = row.get('n_features', 0)
        
        specialist_metrics[specialist_name] = {
            'mi_score': mi_mean_avg,
            'r2_score': r2_mean,
            'n_features': n_features,
            'status': '✅ Success' if mi_mean_avg > 0.01 else '⚠️ Low MI'
        }
    
    # Create comprehensive table
    print("\n🎯 Enhanced Specialists Performance Metrics")
    print("=" * 80)
    
    # Table header
    print(f"{'Specialist':<30} {'MI Score':<12} {'R² Score':<12} {'Features':<10} {'Status':<15}")
    print("-" * 80)
    
    # Sort by MI score
    sorted_specialists = sorted(specialist_metrics.items(), 
                               key=lambda x: x[1]['mi_score'], 
                               reverse=True)
    
    for specialist, metrics in sorted_specialists:
        mi_score = metrics['mi_score']
        r2_score = metrics['r2_score']
        n_features = metrics['n_features']
        status = metrics['status']
        
        # Format scores
        mi_str = f"{mi_score:.4f}" if mi_score > 0 else "N/A"
        r2_str = f"{r2_score:.4f}" if r2_score > 0 else "N/A"
        features_str = f"{n_features}" if n_features > 0 else "N/A"
        
        print(f"{specialist:<30} {mi_str:<12} {r2_str:<12} {features_str:<10} {status:<15}")
    
    # Summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"   Total specialists: {len(specialist_metrics)}")
    
    mi_scores = [m['mi_score'] for m in specialist_metrics.values() if m['mi_score'] > 0]
    if mi_scores:
        print(f"   Average MI: {np.mean(mi_scores):.4f}")
        print(f"   Best MI: {np.max(mi_scores):.4f}")
        print(f"   MI > 0.02 (target): {sum(1 for s in mi_scores if s > 0.02)}/{len(mi_scores)}")
    
    # Cross-specialist redundancy analysis
    print(f"\n🔍 Cross-Specialist Redundancy Analysis:")
    
    # Filter for pairwise MI
    pairwise_rows = df[df['row_type'] == 'model_pairwise']
    
    if not pairwise_rows.empty:
        print(f"   High redundancy pairs (MI > 0.5):")
        high_redundancy = pairwise_rows[pairwise_rows['mi_proxy'] > 0.5]
        
        for _, row in high_redundancy.iterrows():
            specialist_i = row['model_i']
            specialist_j = row['model_j']
            mi_score = row['mi_proxy']
            print(f"     {specialist_i} | {specialist_j}: MI = {mi_score:.3f}")
        
        print(f"\n   Moderate redundancy pairs (MI 0.2-0.5):")
        moderate_redundancy = pairwise_rows[
            (pairwise_rows['mi_proxy'] > 0.2) & 
            (pairwise_rows['mi_proxy'] <= 0.5)
        ]
        
        for _, row in moderate_redundancy.iterrows():
            specialist_i = row['model_i']
            specialist_j = row['model_j']
            mi_score = row['mi_proxy']
            print(f"     {specialist_i} | {specialist_j}: MI = {mi_score:.3f}")
    
    # Recommendations
    print(f"\n🎯 Recommendations:")
    
    # Top performers
    top_performers = [(s, m) for s, m in sorted_specialists if m['mi_score'] > 0.02]
    if top_performers:
        print(f"   ✅ Top performers (MI > 0.02):")
        for specialist, metrics in top_performers[:3]:
            print(f"     - {specialist}: MI = {metrics['mi_score']:.4f}")
    
    # Underperformers
    underperformers = [(s, m) for s, m in sorted_specialists if m['mi_score'] < 0.02]
    if underperformers:
        print(f"   ⚠️ Underperformers (MI < 0.02):")
        for specialist, metrics in underperformers:
            print(f"     - {specialist}: MI = {metrics['mi_score']:.4f}")
    
    # Redundancy warnings
    if not high_redundancy.empty:
        print(f"   🔴 High redundancy detected - consider orthogonalization")
    
    return specialist_metrics

if __name__ == "__main__":
    extract_enhanced_specialist_metrics()
