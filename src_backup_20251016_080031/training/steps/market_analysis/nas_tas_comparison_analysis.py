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
        
        # Look for regime data files in multiple possible locations
        possible_dirs = [
            self.data_cache_path / "nas_tas_clustering" / symbol,
            self.data_cache_path / "regime_analysis" / symbol,
            self.data_cache_path / "clustering" / symbol,
            Path("data_cache") / "nas_tas_clustering" / symbol,
            Path("data_cache") / "regime_analysis" / symbol,
            Path("data_cache") / "clustering" / symbol
        ]
        
        clustering_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                clustering_dir = dir_path
                break
        
        if clustering_dir is None:
            raise FileNotFoundError(
                f"No clustering directory found for {symbol}. "
                f"Expected directories: {[str(d) for d in possible_dirs]}. "
                f"Please ensure regime analysis has been completed first."
            )
        
        # Look for separate NAS and TAS regime files
        nas_files = list(clustering_dir.glob("*nas*regime*assignments*.parquet"))
        tas_files = list(clustering_dir.glob("*tas*regime*assignments*.parquet"))
        combined_files = list(clustering_dir.glob("*regime*assignments*.parquet"))
        
        if nas_files and tas_files:
            # Load separate NAS and TAS data
            tprint("📁 Found separate NAS and TAS regime files", "INFO")
            nas_file = max(nas_files, key=lambda x: x.stat().st_mtime)
            tas_file = max(tas_files, key=lambda x: x.stat().st_mtime)
            
            nas_df = pd.read_parquet(nas_file)
            tas_df = pd.read_parquet(tas_file)
            
            tprint(f"✅ Loaded NAS regime assignments: {len(nas_df)} samples from {nas_file.name}", "SUCCESS")
            tprint(f"✅ Loaded TAS regime assignments: {len(tas_df)} samples from {tas_file.name}", "SUCCESS")
            
            return {
                'nas_regime_labels': nas_df['regime_id'].values,
                'nas_regime_probs': nas_df['regime_prob'].values if 'regime_prob' in nas_df.columns else None,
                'tas_regime_labels': tas_df['regime_id'].values,
                'tas_regime_probs': tas_df['regime_prob'].values if 'regime_prob' in tas_df.columns else None,
                'nas_total_samples': len(nas_df),
                'tas_total_samples': len(tas_df),
                'nas_unique_regimes': sorted(np.unique(nas_df['regime_id'])),
                'tas_unique_regimes': sorted(np.unique(tas_df['regime_id'])),
                'has_separate_data': True
            }
        
        elif combined_files:
            # Found combined data but no separate NAS/TAS files
            latest_file = max(combined_files, key=lambda x: x.stat().st_mtime)
            raise FileNotFoundError(
                f"Found combined regime file '{latest_file.name}' but no separate NAS and TAS files. "
                f"Expected separate files with patterns: '*nas*regime*assignments*.parquet' and '*tas*regime*assignments*.parquet'. "
                f"Please run separate NAS and TAS regime analysis to generate comparison data."
            )
        
        else:
            raise FileNotFoundError(
                f"No regime assignment files found in {clustering_dir}. "
                f"Expected files with patterns: '*nas*regime*assignments*.parquet' and '*tas*regime*assignments*.parquet'. "
                f"Please run NAS and TAS regime analysis first."
            )
    
    
    def calculate_distribution_comparison(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed distribution comparison between NAS and TAS."""
        tprint("📈 Calculating NAS vs TAS distribution comparison", "INFO")
        
        # Extract NAS and TAS data
        nas_labels = data['nas_regime_labels']
        tas_labels = data['tas_regime_labels']
        nas_total = data['nas_total_samples']
        tas_total = data['tas_total_samples']
        nas_unique_regimes = data['nas_unique_regimes']
        tas_unique_regimes = data['tas_unique_regimes']
        
        # Get all unique regimes from both approaches
        all_unique_regimes = sorted(set(nas_unique_regimes + tas_unique_regimes))
        
        # Calculate NAS distribution
        nas_distribution = {}
        for regime_id in all_unique_regimes:
            if regime_id in nas_unique_regimes:
                mask = nas_labels == regime_id
                count = np.sum(mask)
                percentage = (count / nas_total) * 100
            else:
                count = 0
                percentage = 0.0
            
            nas_distribution[f'regime_{regime_id}'] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
        
        # Calculate TAS distribution
        tas_distribution = {}
        for regime_id in all_unique_regimes:
            if regime_id in tas_unique_regimes:
                mask = tas_labels == regime_id
                count = np.sum(mask)
                percentage = (count / tas_total) * 100
            else:
                count = 0
                percentage = 0.0
            
            tas_distribution[f'regime_{regime_id}'] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
        
        # Calculate balance metrics for NAS
        nas_percentages = [nas_distribution[f'regime_{r}']['percentage'] for r in all_unique_regimes]
        nas_balance = {
            'min_percentage': round(min(nas_percentages), 2),
            'max_percentage': round(max(nas_percentages), 2),
            'std_percentage': round(np.std(nas_percentages), 2),
            'balance_score': round(1.0 - (np.std(nas_percentages) / 100), 3)
        }
        
        # Calculate balance metrics for TAS
        tas_percentages = [tas_distribution[f'regime_{r}']['percentage'] for r in all_unique_regimes]
        tas_balance = {
            'min_percentage': round(min(tas_percentages), 2),
            'max_percentage': round(max(tas_percentages), 2),
            'std_percentage': round(np.std(tas_percentages), 2),
            'balance_score': round(1.0 - (np.std(tas_percentages) / 100), 3)
        }
        
        # Calculate differences
        differences = {}
        for regime_id in all_unique_regimes:
            nas_pct = nas_distribution[f'regime_{regime_id}']['percentage']
            tas_pct = tas_distribution[f'regime_{regime_id}']['percentage']
            diff = nas_pct - tas_pct
            differences[f'regime_{regime_id}'] = {
                'nas_percentage': nas_pct,
                'tas_percentage': tas_pct,
                'difference': round(diff, 2),
                'abs_difference': round(abs(diff), 2)
            }
        
        # Calculate summary statistics
        abs_differences = [d['abs_difference'] for d in differences.values()]
        total_difference = round(sum(abs_differences), 2)
        max_difference = round(max(abs_differences), 2)
        identical_distributions = all(diff == 0 for diff in abs_differences)
        
        comparison = {
            'nas_distribution': nas_distribution,
            'tas_distribution': tas_distribution,
            'nas_balance': nas_balance,
            'tas_balance': tas_balance,
            'differences': differences,
            'summary': {
                'total_difference': total_difference,
                'max_difference': max_difference,
                'identical_distributions': identical_distributions,
                'nas_total_samples': nas_total,
                'tas_total_samples': tas_total,
                'nas_unique_regimes_count': len(nas_unique_regimes),
                'tas_unique_regimes_count': len(tas_unique_regimes)
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
        tprint(f"   NAS Samples: {summary['nas_total_samples']:,}", "INFO")
        tprint(f"   TAS Samples: {summary['tas_total_samples']:,}", "INFO")
        tprint(f"   NAS Regimes: {summary['nas_unique_regimes_count']}", "INFO")
        tprint(f"   TAS Regimes: {summary['tas_unique_regimes_count']}", "INFO")
        
        # Identify largest and smallest regimes
        all_regimes = sorted([int(k.split('_')[1]) for k in comparison['nas_distribution'].keys()])
        nas_percentages = [comparison['nas_distribution'][f'regime_{r}']['percentage'] for r in all_regimes]
        tas_percentages = [comparison['tas_distribution'][f'regime_{r}']['percentage'] for r in all_regimes]
        
        if nas_percentages and tas_percentages:
            nas_largest_idx = nas_percentages.index(max(nas_percentages))
            nas_smallest_idx = nas_percentages.index(min(nas_percentages))
            tas_largest_idx = tas_percentages.index(max(tas_percentages))
            tas_smallest_idx = tas_percentages.index(min(tas_percentages))
            
            tprint(f"\n🎯 REGIME CHARACTERISTICS:", "INFO")
            tprint(f"   NAS Largest: Regime {all_regimes[nas_largest_idx]} ({max(nas_percentages):.2f}%)", "INFO")
            tprint(f"   NAS Smallest: Regime {all_regimes[nas_smallest_idx]} ({min(nas_percentages):.2f}%)", "INFO")
            tprint(f"   TAS Largest: Regime {all_regimes[tas_largest_idx]} ({max(tas_percentages):.2f}%)", "INFO")
            tprint(f"   TAS Smallest: Regime {all_regimes[tas_smallest_idx]} ({min(tas_percentages):.2f}%)", "INFO")
            
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
        
        # Show most significant differences
        significant_diffs = [(k, v) for k, v in comparison['differences'].items() if v['abs_difference'] > 1.0]
        if significant_diffs:
            tprint(f"\n🔍 SIGNIFICANT DIFFERENCES (>1%):", "INFO")
            for regime_key, diff_data in sorted(significant_diffs, key=lambda x: x[1]['abs_difference'], reverse=True):
                regime_id = regime_key.split('_')[1]
                tprint(f"   Regime {regime_id}: NAS {diff_data['nas_percentage']:.2f}% vs TAS {diff_data['tas_percentage']:.2f}% (Δ{diff_data['difference']:+.2f}%)", "INFO")
        
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
            
        except FileNotFoundError as e:
            tprint(f"❌ Missing required data files: {e}", "ERROR")
            tprint("💡 To fix this issue:", "INFO")
            tprint("   1. Run separate NAS regime analysis to generate NAS regime assignments", "INFO")
            tprint("   2. Run separate TAS regime analysis to generate TAS regime assignments", "INFO")
            tprint("   3. Ensure files follow naming patterns: '*nas*regime*assignments*.parquet' and '*tas*regime*assignments*.parquet'", "INFO")
            raise
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
