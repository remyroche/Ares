#!/usr/bin/env python3
"""
Regime Detection Report Analyzer
Analyzes regime detection performance reports and generates insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def analyze_regime_reports():
    """Analyze the latest regime detection reports."""
    
    print("=" * 80)
    print("REGIME DETECTION PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()
    
    # Load reports
    base_path = Path("outcomes")
    
    # 1. Load ensemble metrics
    print("📊 Loading ensemble training metrics...")
    metrics_df = pd.read_csv(base_path / "regime_ensemble_training_metrics_ETHUSDT_20251111_024656.csv")
    
    # 2. Load model performance
    print("📊 Loading model performance by regime...")
    perf_df = pd.read_csv(base_path / "regime_performance_by_model_ETHUSDT_20251111_022208.csv")
    
    # 3. Load temporal analysis
    print("📊 Loading temporal regime analysis...")
    temporal_df = pd.read_csv(base_path / "temporal_regime_analysis_ETHUSDT_20251111_024656.csv")
    
    print()
    print("=" * 80)
    print("CRITICAL ISSUES DETECTED")
    print("=" * 80)
    print()
    
    # Issue 1: Ensemble failure
    ensemble_acc = metrics_df[metrics_df['Metric Name'] == 'Accuracy']['Value'].values[0]
    if float(ensemble_acc) < 0.01:
        print("🚨 CRITICAL: Ensemble Complete Failure")
        print(f"   Accuracy: {float(ensemble_acc):.4f} (Expected: >0.60)")
        print("   Status: CATASTROPHIC - System unusable")
        print()
    
    # Issue 2: Class imbalance
    print("⚠️  SEVERE CLASS IMBALANCE")
    print()
    regime_support = perf_df.groupby('Regime')['Support'].first().sort_values()
    for regime, support in regime_support.items():
        pct = support / regime_support.sum() * 100
        f1 = perf_df[perf_df['Regime'] == regime]['F1-Score'].mean()
        status = "❌" if f1 < 0.05 else "⚠️" if f1 < 0.30 else "✅"
        print(f"   {status} Regime {regime}: {support:3.0f} samples ({pct:5.1f}%) - F1: {f1:.4f}")
    print()
    
    # Issue 3: Model performance
    print("⚠️  POOR MODEL PERFORMANCE")
    print()
    for model in perf_df['Model Name'].unique():
        model_data = perf_df[perf_df['Model Name'] == model]
        avg_f1 = model_data['F1-Score'].mean()
        avg_prec = model_data['Precision'].mean()
        avg_recall = model_data['Recall'].mean()
        print(f"   {model:15s}: F1={avg_f1:.4f}, Precision={avg_prec:.4f}, Recall={avg_recall:.4f}")
    print()
    
    # Issue 4: Economic validity
    print("⚠️  QUESTIONABLE REGIME ECONOMICS")
    print()
    for idx, row in temporal_df.iterrows():
        regime = int(row['regime'])
        max_dd = row['max_drawdown']
        sharpe = row['sharpe_ratio']
        avg_ret = row['avg_return']
        
        if max_dd < -0.80:
            print(f"   ⚠️ Regime {regime}: Max DD = {max_dd:.2%} (EXTREME), Sharpe = {sharpe:.2f}")
        elif sharpe < -5:
            print(f"   ⚠️ Regime {regime}: Sharpe = {sharpe:.2f} (EXTREME), Avg Return = {avg_ret:.4%}")
    print()
    
    # Generate visualizations
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Per-model F1 scores
    ax1 = plt.subplot(2, 3, 1)
    model_f1 = perf_df.groupby('Model Name')['F1-Score'].mean().sort_values()
    model_f1.plot(kind='barh', ax=ax1, color='steelblue')
    ax1.axvline(x=0.60, color='green', linestyle='--', label='Target (0.60)')
    ax1.set_xlabel('Average F1 Score')
    ax1.set_title('Model Performance Comparison')
    ax1.legend()
    
    # 2. Per-regime F1 scores
    ax2 = plt.subplot(2, 3, 2)
    regime_f1 = perf_df.groupby('Regime')['F1-Score'].mean()
    colors = ['red' if f1 < 0.05 else 'orange' if f1 < 0.30 else 'green' for f1 in regime_f1]
    regime_f1.plot(kind='bar', ax=ax2, color=colors)
    ax2.axhline(y=0.30, color='orange', linestyle='--', label='Minimum (0.30)')
    ax2.axhline(y=0.60, color='green', linestyle='--', label='Target (0.60)')
    ax2.set_xlabel('Regime')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Per-Regime Detection Performance')
    ax2.legend()
    
    # 3. Class distribution
    ax3 = plt.subplot(2, 3, 3)
    regime_counts = perf_df.groupby('Regime')['Support'].first()
    colors_pie = ['red' if c < 50 else 'orange' if c < 100 else 'green' for c in regime_counts]
    ax3.pie(regime_counts, labels=[f'R{i}\n({c:.0f})' for i, c in enumerate(regime_counts)], 
            autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax3.set_title('Class Distribution (Imbalance)')
    
    # 4. Precision-Recall by regime
    ax4 = plt.subplot(2, 3, 4)
    regime_metrics = perf_df.groupby('Regime')[['Precision', 'Recall']].mean()
    x = np.arange(len(regime_metrics))
    width = 0.35
    ax4.bar(x - width/2, regime_metrics['Precision'], width, label='Precision', color='steelblue')
    ax4.bar(x + width/2, regime_metrics['Recall'], width, label='Recall', color='coral')
    ax4.set_xlabel('Regime')
    ax4.set_ylabel('Score')
    ax4.set_title('Precision vs Recall by Regime')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'R{i}' for i in regime_metrics.index])
    ax4.legend()
    
    # 5. Economic metrics
    ax5 = plt.subplot(2, 3, 5)
    temporal_df['regime'] = temporal_df['regime'].astype(int)
    colors_sharpe = ['red' if s < 0 else 'green' for s in temporal_df['sharpe_ratio']]
    ax5.bar(temporal_df['regime'], temporal_df['sharpe_ratio'], color=colors_sharpe)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.set_xlabel('Regime')
    ax5.set_ylabel('Sharpe Ratio')
    ax5.set_title('Regime Economic Performance (Sharpe Ratio)')
    
    # 6. Regime persistence
    ax6 = plt.subplot(2, 3, 6)
    ax6.bar(temporal_df['regime'], temporal_df['persistence_mean_duration'], color='steelblue')
    ax6.set_xlabel('Regime')
    ax6.set_ylabel('Mean Duration (periods)')
    ax6.set_title('Regime Persistence')
    
    plt.tight_layout()
    
    # Save figure
    output_path = base_path / f"regime_analysis_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    print()
    
    # Detailed heatmap
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    # Create confusion-style heatmap
    pivot_df = perf_df.pivot_table(
        index='Model Name',
        columns='Regime', 
        values='F1-Score',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.30, vmin=0, vmax=0.50, ax=ax, cbar_kws={'label': 'F1 Score'})
    ax.set_title('F1 Score Heatmap: Model vs Regime')
    ax.set_xlabel('Regime')
    ax.set_ylabel('Model')
    
    heatmap_path = base_path / f"regime_f1_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"✅ Heatmap saved to: {heatmap_path}")
    print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()
    
    print("📊 Overall Performance:")
    print(f"   Ensemble Accuracy: {float(ensemble_acc):.4f}")
    print(f"   Average Model F1: {perf_df['F1-Score'].mean():.4f}")
    print(f"   Best Model F1: {perf_df.groupby('Model Name')['F1-Score'].mean().max():.4f}")
    print(f"   Worst Regime F1: {perf_df.groupby('Regime')['F1-Score'].mean().min():.4f}")
    print()
    
    print("📊 Class Balance:")
    print(f"   Most common regime: {regime_counts.idxmax()} ({regime_counts.max():.0f} samples)")
    print(f"   Least common regime: {regime_counts.idxmin()} ({regime_counts.min():.0f} samples)")
    print(f"   Imbalance ratio: {regime_counts.max() / regime_counts.min():.1f}x")
    print()
    
    print("📊 Economic Validity:")
    n_extreme_dd = (temporal_df['max_drawdown'] < -0.80).sum()
    n_extreme_sharpe = (temporal_df['sharpe_ratio'].abs() > 10).sum()
    print(f"   Regimes with extreme drawdown (<-80%): {n_extreme_dd}")
    print(f"   Regimes with extreme Sharpe (|Sharpe| >10): {n_extreme_sharpe}")
    print()
    
    # Recommendations
    print("=" * 80)
    print("TOP 5 PRIORITY ACTIONS")
    print("=" * 80)
    print()
    print("1. 🚨 DEBUG ENSEMBLE TRAINING")
    print("   - Ensemble accuracy is 0.00 (catastrophic)")
    print("   - Check calibration, feature alignment, data leakage")
    print("   - Add extensive logging to regime_ensemble_training.py")
    print()
    print("2. 🚨 FIX CLASS IMBALANCE")
    print(f"   - Remove/merge regimes with <50 samples (Regimes: {[r for r, s in regime_counts.items() if s < 50]})")
    print("   - Apply adaptive SMOTE to remaining regimes")
    print("   - Use sample weights in model training")
    print()
    print("3. ⚠️  VALIDATE REGIME DEFINITIONS")
    print(f"   - {n_extreme_dd} regimes have complete/near-complete drawdowns")
    print("   - These may be data artifacts or invalid market states")
    print("   - Implement regime validation and filtering")
    print()
    print("4. ⚠️  IMPROVE FEATURE ENGINEERING")
    print("   - Current features lack discriminative power")
    print("   - Add regime-specific interaction features")
    print("   - Implement feature importance analysis")
    print()
    print("5. ⚠️  ENHANCE MODEL ARCHITECTURE")
    print("   - Try hierarchical classification (coarse -> fine)")
    print("   - Experiment with LSTM/CNN for temporal patterns")
    print("   - Use focal loss for minority classes")
    print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("📄 See REGIME_DETECTION_IMPROVEMENT_PLAN.md for detailed improvement strategy")
    print()

if __name__ == "__main__":
    analyze_regime_reports()
