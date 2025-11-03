"""
Multi-Timeframe Quality Score Validation

Tests the quality score calculation across different timeframes (1h, 4h, 24h)
to ensure fixes work consistently across all time horizons.
"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json

from src.tactician.sr_levels.ml_quality.sr_quality_data_collector import SRQualityDataCollector

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


async def collect_timeframe_data(timeframe: str, symbol: str = 'ETHUSDT'):
    """Collect quality score data for a specific timeframe."""
    print(f"\n{'='*80}")
    print(f"📊 COLLECTING DATA FOR {timeframe} TIMEFRAME")
    print(f"{'='*80}")
    
    collector = SRQualityDataCollector()
    
    # Adjust parameters based on timeframe
    params = get_timeframe_parameters(timeframe)
    
    print(f"\nParameters:")
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: binance")
    print(f"   Timeframe: {timeframe}")
    print(f"   Date range: {params['start_date']} to {params['end_date']}")
    print(f"   Forward days: {params['forward_days']}")
    print(f"   Sample frequency: {params['sample_freq_days']} days")
    
    try:
        training_data = await collector.collect_training_data(
            symbol=symbol,
            exchange='binance',
            start_date=params['start_date'],
            end_date=params['end_date'],
            timeframe=timeframe,
            forward_days=params['forward_days'],
            sample_freq_days=params['sample_freq_days']
        )
        
        # Save data
        output_dir = Path('data_cache/sr_ml_training/multi_timeframe')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f'sr_quality_{timeframe}_{symbol}.parquet'
        training_data.to_parquet(output_path, index=False)
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'timeframe': timeframe,
            'symbol': symbol,
            'samples': len(training_data),
            'date_range': {
                'start': str(training_data['date'].min()),
                'end': str(training_data['date'].max())
            },
            'parameters': params,
            'quality_stats': {
                'mean': float(training_data['quality_score'].mean()),
                'std': float(training_data['quality_score'].std()),
                'min': float(training_data['quality_score'].min()),
                'max': float(training_data['quality_score'].max())
            },
            'component_stats': {
                'bounce_strength': {
                    'mean': float(training_data['bounce_strength'].mean()),
                    'std': float(training_data['bounce_strength'].std()),
                    'at_max': float((training_data['bounce_strength'] >= 0.95).sum() / len(training_data) * 100)
                },
                'hold_strength': {
                    'mean': float(training_data['hold_strength'].mean()),
                    'std': float(training_data['hold_strength'].std())
                },
                'trade_profit': {
                    'mean': float(training_data['trade_profit'].mean()),
                    'std': float(training_data['trade_profit'].std()),
                    'positive_pct': float((training_data['trade_profit'] > 0).sum() / len(training_data) * 100)
                }
            }
        }
        
        metadata_path = output_dir / f'sr_quality_{timeframe}_{symbol}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Data collected successfully!")
        print(f"   Samples: {len(training_data):,}")
        print(f"   Saved to: {output_path}")
        
        return training_data, metadata
        
    except Exception as e:
        print(f"\n❌ Error collecting data for {timeframe}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def get_timeframe_parameters(timeframe: str) -> dict:
    """Get appropriate parameters for each timeframe."""
    
    # Base parameters
    base = {
        '1h': {
            'start_date': '2025-05-01',
            'end_date': '2025-09-30',
            'forward_days': 7,      # 1 week forward for 1h
            'sample_freq_days': 7   # Weekly samples
        },
        '4h': {
            'start_date': '2025-01-01',
            'end_date': '2025-09-30',
            'forward_days': 14,     # 2 weeks forward for 4h
            'sample_freq_days': 7   # Weekly samples
        },
        '24h': {
            'start_date': '2024-01-01',
            'end_date': '2025-09-30',
            'forward_days': 30,     # 1 month forward for daily
            'sample_freq_days': 14  # Bi-weekly samples
        }
    }
    
    return base.get(timeframe, base['1h'])


def analyze_timeframe_results(results_dict):
    """Analyze and compare results across timeframes."""
    print(f"\n{'='*80}")
    print(f"📊 MULTI-TIMEFRAME QUALITY SCORE ANALYSIS")
    print(f"{'='*80}")
    
    # Summary table
    print(f"\n{'Timeframe':<12} {'Samples':<10} {'Bounce Mean':<12} {'Trade Mean':<12} {'Quality Mean':<12}")
    print(f"{'-'*70}")
    
    for tf, (data, metadata) in results_dict.items():
        if data is not None:
            bounce_mean = data['bounce_strength'].mean()
            trade_mean = data['trade_profit'].mean()
            quality_mean = data['quality_score'].mean()
            
            print(f"{tf:<12} {len(data):<10} {bounce_mean:<12.4f} {trade_mean:<12.4f} {quality_mean:<12.4f}")
    
    # Component analysis
    print(f"\n{'='*80}")
    print(f"🔍 COMPONENT ANALYSIS BY TIMEFRAME")
    print(f"{'='*80}")
    
    for tf, (data, metadata) in results_dict.items():
        if data is not None:
            print(f"\n{tf} TIMEFRAME:")
            print(f"   Samples: {len(data):,}")
            
            # Bounce strength
            bounce = data['bounce_strength']
            print(f"\n   Bounce Strength:")
            print(f"      Mean: {bounce.mean():.4f}")
            print(f"      Median: {bounce.median():.4f}")
            print(f"      Std: {bounce.std():.4f}")
            print(f"      At max (≥0.95): {(bounce >= 0.95).sum()} ({(bounce >= 0.95).sum()/len(bounce)*100:.1f}%)")
            if bounce.mean() > 0.8:
                print(f"      ⚠️  WARNING: Still saturated!")
            else:
                print(f"      ✅ Good spread")
            
            # Trade profit
            profit = data['trade_profit']
            print(f"\n   Trade Profit:")
            print(f"      Mean: {profit.mean():.4f}")
            print(f"      Median: {profit.median():.4f}")
            print(f"      Std: {profit.std():.4f}")
            print(f"      Winning trades: {(profit > 0).sum()} ({(profit > 0).sum()/len(profit)*100:.1f}%)")
            if profit.mean() < 0:
                print(f"      ⚠️  WARNING: Negative expectancy!")
            else:
                print(f"      ✅ Positive expectancy")
            
            # Hold strength
            hold = data['hold_strength']
            print(f"\n   Hold Strength:")
            print(f"      Mean: {hold.mean():.4f}")
            print(f"      Median: {hold.median():.4f}")
            print(f"      Std: {hold.std():.4f}")
            
            # Quality score
            quality = data['quality_score']
            print(f"\n   Quality Score:")
            print(f"      Mean: {quality.mean():.4f}")
            print(f"      Median: {quality.median():.4f}")
            print(f"      Std: {quality.std():.4f}")
            print(f"      Range: [{quality.min():.4f}, {quality.max():.4f}]")
            
            # Feature correlations
            feature_cols = [c for c in data.columns if c.startswith('feature_')]
            if feature_cols:
                correlations = data[feature_cols].corrwith(data['quality_score']).abs().sort_values(ascending=False)
                strong_features = (correlations > 0.3).sum()
                print(f"\n   Feature Correlations:")
                print(f"      Top correlation: {correlations.iloc[0]:.4f} ({correlations.index[0].replace('feature_', '')})")
                print(f"      Strong (>0.3): {strong_features}")
                print(f"      Top 5:")
                for i, (feat, corr) in enumerate(correlations.head(5).items(), 1):
                    print(f"         {i}. {feat.replace('feature_', '')}: {corr:.4f}")


def visualize_multi_timeframe_comparison(results_dict, output_dir='analysis_output/multi_timeframe'):
    """Create comprehensive visualizations comparing timeframes."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter valid results
    valid_results = {tf: (data, meta) for tf, (data, meta) in results_dict.items() if data is not None}
    
    if not valid_results:
        print("\n❌ No valid results to visualize")
        return
    
    # Create comparison figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Quality Score Analysis Across Timeframes', fontsize=16, fontweight='bold', y=0.98)
    
    colors = {'1h': 'steelblue', '4h': 'coral', '24h': 'green'}
    
    # Row 1: Component distributions
    components = ['bounce_strength', 'hold_strength', 'trade_profit']
    for i, comp in enumerate(components):
        ax = fig.add_subplot(gs[0, i])
        
        for tf, (data, _) in valid_results.items():
            data[comp].plot(kind='kde', ax=ax, label=tf, color=colors.get(tf, 'gray'), linewidth=2)
        
        ax.set_xlabel(comp.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'{comp.replace("_", " ").title()} by Timeframe', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Row 2: Quality score comparison
    ax_quality = fig.add_subplot(gs[1, :])
    
    tf_labels = []
    quality_data = []
    
    for tf, (data, _) in valid_results.items():
        tf_labels.append(tf)
        quality_data.append(data['quality_score'].values)
    
    bp = ax_quality.boxplot(quality_data, labels=tf_labels, patch_artist=True)
    for patch, tf in zip(bp['boxes'], tf_labels):
        patch.set_facecolor(colors.get(tf, 'gray'))
        patch.set_alpha(0.7)
    
    ax_quality.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
    ax_quality.set_title('Quality Score Distribution by Timeframe', fontsize=14, fontweight='bold')
    ax_quality.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Metrics comparison
    ax_metrics = fig.add_subplot(gs[2, :])
    
    metrics_data = {
        'Bounce Mean': [],
        'Trade Profit Mean': [],
        'Quality Mean': []
    }
    
    for tf, (data, _) in valid_results.items():
        metrics_data['Bounce Mean'].append(data['bounce_strength'].mean())
        metrics_data['Trade Profit Mean'].append(data['trade_profit'].mean())
        metrics_data['Quality Mean'].append(data['quality_score'].mean())
    
    x = np.arange(len(tf_labels))
    width = 0.25
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        ax_metrics.bar(x + i*width, values, width, label=metric, alpha=0.8)
    
    ax_metrics.set_xlabel('Timeframe', fontsize=12, fontweight='bold')
    ax_metrics.set_ylabel('Mean Value', fontsize=12, fontweight='bold')
    ax_metrics.set_title('Component Means by Timeframe', fontsize=14, fontweight='bold')
    ax_metrics.set_xticks(x + width)
    ax_metrics.set_xticklabels(tf_labels)
    ax_metrics.legend()
    ax_metrics.grid(True, alpha=0.3, axis='y')
    ax_metrics.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.savefig(f'{output_dir}/multi_timeframe_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison visualization saved to: {output_dir}/multi_timeframe_comparison.png")
    plt.close()


def generate_timeframe_report(results_dict, output_dir='analysis_output/multi_timeframe'):
    """Generate text report of multi-timeframe analysis."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_path = Path(output_dir) / 'multi_timeframe_quality_report.txt'
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-TIMEFRAME QUALITY SCORE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Timeframes tested: {', '.join(results_dict.keys())}\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Timeframe':<12} {'Samples':<10} {'Bounce':<12} {'Trade':<12} {'Quality':<12}\n")
        f.write("-"*80 + "\n")
        
        for tf, (data, metadata) in results_dict.items():
            if data is not None:
                f.write(f"{tf:<12} {len(data):<10} "
                       f"{data['bounce_strength'].mean():<12.4f} "
                       f"{data['trade_profit'].mean():<12.4f} "
                       f"{data['quality_score'].mean():<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED ANALYSIS BY TIMEFRAME\n")
        f.write("="*80 + "\n")
        
        for tf, (data, metadata) in results_dict.items():
            if data is not None:
                f.write(f"\n{tf} TIMEFRAME:\n")
                f.write("-"*80 + "\n")
                
                f.write(f"Samples: {len(data):,}\n")
                f.write(f"Date range: {data['date'].min()} to {data['date'].max()}\n\n")
                
                # Components
                f.write("Bounce Strength:\n")
                f.write(f"  Mean:   {data['bounce_strength'].mean():.4f}\n")
                f.write(f"  Median: {data['bounce_strength'].median():.4f}\n")
                f.write(f"  Std:    {data['bounce_strength'].std():.4f}\n")
                f.write(f"  At max: {(data['bounce_strength'] >= 0.95).sum()/len(data)*100:.1f}%\n")
                f.write(f"  Status: {'⚠️ SATURATED' if data['bounce_strength'].mean() > 0.8 else '✅ Good'}\n\n")
                
                f.write("Trade Profit:\n")
                f.write(f"  Mean:   {data['trade_profit'].mean():.4f}\n")
                f.write(f"  Median: {data['trade_profit'].median():.4f}\n")
                f.write(f"  Std:    {data['trade_profit'].std():.4f}\n")
                f.write(f"  Win%:   {(data['trade_profit'] > 0).sum()/len(data)*100:.1f}%\n")
                f.write(f"  Status: {'⚠️ NEGATIVE' if data['trade_profit'].mean() < 0 else '✅ Positive'}\n\n")
                
                f.write("Quality Score:\n")
                f.write(f"  Mean:   {data['quality_score'].mean():.4f}\n")
                f.write(f"  Median: {data['quality_score'].median():.4f}\n")
                f.write(f"  Std:    {data['quality_score'].std():.4f}\n")
                f.write(f"  Range:  [{data['quality_score'].min():.4f}, {data['quality_score'].max():.4f}]\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("VALIDATION STATUS\n")
        f.write("="*80 + "\n\n")
        
        for tf, (data, metadata) in results_dict.items():
            if data is not None:
                f.write(f"{tf}:\n")
                
                checks = [
                    (data['bounce_strength'].mean() < 0.8, "Bounce not saturated"),
                    (data['bounce_strength'].std() > 0.2, "Bounce has variance"),
                    (data['trade_profit'].mean() > 0, "Trade profit positive"),
                    (data['quality_score'].std() > 0.2, "Quality has variance"),
                ]
                
                for passed, check_name in checks:
                    status = "✅" if passed else "❌"
                    f.write(f"  {status} {check_name}\n")
                
                f.write("\n")
    
    print(f"\n✅ Report saved to: {output_path}")


async def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("🚀 MULTI-TIMEFRAME QUALITY SCORE VALIDATION")
    print("="*80)
    print("\nTesting quality score calculation across:")
    print("   • 1h timeframe")
    print("   • 4h timeframe")
    print("   • 24h timeframe")
    print("\n" + "="*80)
    
    # Timeframes to test
    timeframes = ['1h', '4h', '24h']
    results = {}
    
    # Collect data for each timeframe
    for tf in timeframes:
        data, metadata = await collect_timeframe_data(tf)
        results[tf] = (data, metadata)
    
    # Analyze results
    print("\n" + "="*80)
    print("📊 ANALYSIS")
    print("="*80)
    
    analyze_timeframe_results(results)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📈 GENERATING VISUALIZATIONS")
    print("="*80)
    
    visualize_multi_timeframe_comparison(results)
    
    # Generate report
    print("\n" + "="*80)
    print("📝 GENERATING REPORT")
    print("="*80)
    
    generate_timeframe_report(results)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ MULTI-TIMEFRAME VALIDATION COMPLETE")
    print("="*80)
    print("\nResults:")
    for tf, (data, _) in results.items():
        if data is not None:
            print(f"   {tf}: {len(data):,} samples collected ✅")
        else:
            print(f"   {tf}: Failed ❌")
    
    print("\nOutputs:")
    print("   • Data: data_cache/sr_ml_training/multi_timeframe/")
    print("   • Visualization: analysis_output/multi_timeframe/multi_timeframe_comparison.png")
    print("   • Report: analysis_output/multi_timeframe/multi_timeframe_quality_report.txt")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    asyncio.run(main())

