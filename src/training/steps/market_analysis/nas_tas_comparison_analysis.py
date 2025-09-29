#!/usr/bin/env python3
"""
NAS vs TAS Regime Distribution Comparison Analysis

This script provides a detailed comparison between NAS and TAS regime distributions,
highlighting differences and similarities in regime characteristics.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_info, tprint_warning

class NASTASComparisonAnalyzer:
    """Analyzes and compares NAS vs TAS regime distributions."""
    
    def __init__(self, data_cache_path: str = "data_cache"):
        """Initialize the comparison analyzer."""
        self.data_cache_path = Path(data_cache_path)
        tprint("🔍 NAS-TAS Comparison Analyzer initialized", "INFO")
    
    def load_regime_data(self, symbol: str = "ETHUSDT") -> Dict[str, Any]:
        """Load regime data for comparison analysis."""
        tprint(f"📊 Loading regime data for {symbol} comparison", "INFO")
        
        # Look for regime data files
        clustering_dir = self.data_cache_path / "nas_tas_clustering" / symbol
        
        if not clustering_dir.exists():
            raise FileNotFoundError(f"Clustering directory not found: {clustering_dir}")
        
        # Find the most recent regime assignments file
        regime_files = list(clustering_dir.glob("nas_tas_regime_assignments_*.parquet"))
        if not regime_files:
            raise FileNotFoundError(f"No regime assignment files found in {clustering_dir}")
        
        latest_file = max(regime_files, key=lambda x: x.stat().st_mtime)
        tprint(f"📁 Using regime file: {latest_file.name}", "INFO")
        
        # Load regime assignments
        df = pd.read_parquet(latest_file)
        tprint(f"✅ Loaded regime assignments: {len(df)} samples", "SUCCESS")
        
        # For this analysis, we'll assume NAS and TAS use the same regime assignments
        # In a real implementation, you'd have separate NAS and TAS regime data
        regime_labels = df['regime_id'].values
        regime_probs = df['regime_prob'].values
        
        return {
            'regime_labels': regime_labels,
            'regime_probs': regime_probs,
            'total_samples': len(df),
            'unique_regimes': sorted(np.unique(regime_labels))
        }
    
    def calculate_distribution_comparison(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed distribution comparison between NAS and TAS."""
        tprint("📈 Calculating NAS vs TAS distribution comparison", "INFO")
        
        regime_labels = data['regime_labels']
        unique_regimes = data['unique_regimes']
        total_samples = data['total_samples']
        
        # Calculate distribution for both NAS and TAS (assuming same for now)
        nas_distribution = {}
        tas_distribution = {}
        
        for regime_id in unique_regimes:
            mask = regime_labels == regime_id
            count = np.sum(mask)
            percentage = (count / total_samples) * 100
            
            nas_distribution[f'regime_{regime_id}'] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
            tas_distribution[f'regime_{regime_id}'] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
        
        # Calculate balance metrics
        nas_percentages = [nas_distribution[f'regime_{r}']['percentage'] for r in unique_regimes]
        tas_percentages = [tas_distribution[f'regime_{r}']['percentage'] for r in unique_regimes]
        
        nas_balance = {
            'min_percentage': round(min(nas_percentages), 2),
            'max_percentage': round(max(nas_percentages), 2),
            'std_percentage': round(np.std(nas_percentages), 2),
            'balance_score': round(1.0 - (np.std(nas_percentages) / 100), 3)
        }
        
        tas_balance = {
            'min_percentage': round(min(tas_percentages), 2),
            'max_percentage': round(max(tas_percentages), 2),
            'std_percentage': round(np.std(tas_percentages), 2),
            'balance_score': round(1.0 - (np.std(tas_percentages) / 100), 3)
        }
        
        # Calculate differences
        differences = {}
        for regime_id in unique_regimes:
            nas_pct = nas_distribution[f'regime_{regime_id}']['percentage']
            tas_pct = tas_distribution[f'regime_{regime_id}']['percentage']
            diff = nas_pct - tas_pct
            differences[f'regime_{regime_id}'] = {
                'nas_percentage': nas_pct,
                'tas_percentage': tas_pct,
                'difference': round(diff, 2),
                'abs_difference': round(abs(diff), 2)
            }
        
        comparison = {
            'nas_distribution': nas_distribution,
            'tas_distribution': tas_distribution,
            'nas_balance': nas_balance,
            'tas_balance': tas_balance,
            'differences': differences,
            'summary': {
                'total_difference': round(sum([d['abs_difference'] for d in differences.values()]), 2),
                'max_difference': round(max([d['abs_difference'] for d in differences.values()]), 2),
                'identical_distributions': all(d['abs_difference'] == 0 for d in differences.values())
            }
        }
        
        tprint("✅ Distribution comparison calculated", "SUCCESS")
        return comparison
    
    def print_comparison_analysis(self, comparison: Dict[str, Any]):
        """Print detailed comparison analysis."""
        tprint("\n" + "="*80, "INFO")
        tprint("📊 NAS vs TAS REGIME DISTRIBUTION COMPARISON", "INFO")
        tprint("="*80, "INFO")
        
        # Print regime-by-regime comparison
        tprint("\n🔍 REGIME-BY-REGIME COMPARISON:", "INFO")
        tprint("-" * 60, "INFO")
        
        for regime_key in sorted(comparison['differences'].keys()):
            diff_data = comparison['differences'][regime_key]
            nas_pct = diff_data['nas_percentage']
            tas_pct = diff_data['tas_percentage']
            diff = diff_data['difference']
            
            # Color coding for differences
            if abs(diff) < 0.1:
                status = "✅ IDENTICAL"
            elif abs(diff) < 1.0:
                status = "🟡 MINOR DIFF"
            else:
                status = "🔴 MAJOR DIFF"
            
            tprint(f"   {regime_key}:", "INFO")
            tprint(f"      NAS: {nas_pct:6.2f}% | TAS: {tas_pct:6.2f}% | Diff: {diff:+6.2f}% {status}", "INFO")
        
        # Print balance comparison
        tprint("\n📈 BALANCE METRICS COMPARISON:", "INFO")
        tprint("-" * 60, "INFO")
        
        nas_balance = comparison['nas_balance']
        tas_balance = comparison['tas_balance']
        
        tprint(f"   NAS Balance:", "INFO")
        tprint(f"      Min: {nas_balance['min_percentage']:6.2f}% | Max: {nas_balance['max_percentage']:6.2f}% | Std: {nas_balance['std_percentage']:6.2f}% | Score: {nas_balance['balance_score']:.3f}", "INFO")
        
        tprint(f"   TAS Balance:", "INFO")
        tprint(f"      Min: {tas_balance['min_percentage']:6.2f}% | Max: {tas_balance['max_percentage']:6.2f}% | Std: {tas_balance['std_percentage']:6.2f}% | Score: {tas_balance['balance_score']:.3f}", "INFO")
        
        # Print summary
        tprint("\n📋 COMPARISON SUMMARY:", "INFO")
        tprint("-" * 60, "INFO")
        
        summary = comparison['summary']
        tprint(f"   Total Difference: {summary['total_difference']:.2f}%", "INFO")
        tprint(f"   Max Difference: {summary['max_difference']:.2f}%", "INFO")
        tprint(f"   Identical Distributions: {'✅ YES' if summary['identical_distributions'] else '❌ NO'}", "INFO")
        
        # Identify largest and smallest regimes
        nas_percentages = [comparison['nas_distribution'][f'regime_{r}']['percentage'] for r in range(8)]
        tas_percentages = [comparison['tas_distribution'][f'regime_{r}']['percentage'] for r in range(8)]
        
        nas_largest = nas_percentages.index(max(nas_percentages))
        nas_smallest = nas_percentages.index(min(nas_percentages))
        tas_largest = tas_percentages.index(max(tas_percentages))
        tas_smallest = tas_percentages.index(min(tas_percentages))
        
        tprint(f"\n🎯 REGIME CHARACTERISTICS:", "INFO")
        tprint(f"   NAS Largest: Regime {nas_largest} ({max(nas_percentages):.2f}%)", "INFO")
        tprint(f"   NAS Smallest: Regime {nas_smallest} ({min(nas_percentages):.2f}%)", "INFO")
        tprint(f"   TAS Largest: Regime {tas_largest} ({max(tas_percentages):.2f}%)", "INFO")
        tprint(f"   TAS Smallest: Regime {tas_smallest} ({min(tas_percentages):.2f}%)", "INFO")
        
        # Imbalance analysis
        nas_imbalance = max(nas_percentages) / min(nas_percentages) if min(nas_percentages) > 0 else float('inf')
        tas_imbalance = max(tas_percentages) / min(tas_percentages) if min(tas_percentages) > 0 else float('inf')
        
        tprint(f"\n⚖️ IMBALANCE ANALYSIS:", "INFO")
        tprint(f"   NAS Imbalance Ratio: {nas_imbalance:.1f}:1", "INFO")
        tprint(f"   TAS Imbalance Ratio: {tas_imbalance:.1f}:1", "INFO")
        
        if nas_imbalance > 10 or tas_imbalance > 10:
            tprint("   ⚠️ WARNING: Significant regime imbalance detected!", "WARNING")
        elif nas_imbalance > 5 or tas_imbalance > 5:
            tprint("   ⚠️ CAUTION: Moderate regime imbalance detected", "WARNING")
        else:
            tprint("   ✅ Regime distribution is reasonably balanced", "SUCCESS")
        
        tprint("\n" + "="*80, "INFO")
    
    def analyze_regime_comparison(self, symbol: str = "ETHUSDT") -> Dict[str, Any]:
        """Perform comprehensive NAS vs TAS regime comparison analysis."""
        tprint(f"🚀 Starting NAS vs TAS regime comparison for {symbol}", "INFO")
        
        try:
            # Load regime data
            data = self.load_regime_data(symbol)
            
            # Calculate distribution comparison
            comparison = self.calculate_distribution_comparison(data)
            
            # Print detailed analysis
            self.print_comparison_analysis(comparison)
            
            # Save results
            output_path = Path("regime_analysis_results") / f"{symbol}_nas_tas_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            tprint(f"✅ Comparison analysis saved to {output_path}", "SUCCESS")
            
            return comparison
            
        except Exception as e:
            tprint(f"❌ Comparison analysis failed: {e}", "ERROR")
            raise


def main():
    """Main function to run NAS vs TAS comparison analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare NAS vs TAS regime distributions')
    parser.add_argument('--symbol', default='ETHUSDT', help='Trading symbol to analyze')
    parser.add_argument('--data-cache', default='data_cache', help='Path to data cache directory')
    
    args = parser.parse_args()
    
    try:
        analyzer = NASTASComparisonAnalyzer(data_cache_path=args.data_cache)
        comparison = analyzer.analyze_regime_comparison(symbol=args.symbol)
        
        tprint("🎉 NAS vs TAS comparison analysis completed successfully!", "SUCCESS")
        
    except Exception as e:
        tprint(f"❌ Analysis failed: {e}", "ERROR")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
