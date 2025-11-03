"""
Quality Score Investigation Script

Investigates the quality_score calculation and distribution to identify potential issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_training_data():
    """Load the training data."""
    data_path = Path('data_cache/sr_ml_training/sr_quality_training_data.parquet')
    metadata_path = Path('data_cache/sr_ml_training/sr_quality_training_data_metadata.json')
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")
    
    data = pd.read_parquet(data_path)
    
    # Load metadata if available
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return data, metadata


def analyze_quality_score_distribution(data):
    """Analyze quality score distribution for red flags."""
    print("\n" + "="*80)
    print("📊 QUALITY SCORE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    quality = data['quality_score']
    
    # Basic statistics
    print(f"\n📈 Basic Statistics:")
    print(f"   Total samples: {len(quality):,}")
    print(f"   Mean:          {quality.mean():.4f}")
    print(f"   Median:        {quality.median():.4f}")
    print(f"   Std Dev:       {quality.std():.4f}")
    print(f"   Min:           {quality.min():.4f}")
    print(f"   Max:           {quality.max():.4f}")
    
    # Percentiles
    print(f"\n📊 Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"   {p}th percentile: {quality.quantile(p/100):.4f}")
    
    # Distribution shape
    print(f"\n🔍 Distribution Shape:")
    print(f"   Skewness:      {quality.skew():.4f}")
    print(f"   Kurtosis:      {quality.kurtosis():.4f}")
    
    # Red flags
    print(f"\n🚨 RED FLAG CHECKS:")
    
    # 1. Binary-like distribution
    binary_threshold = 0.1
    at_min = (quality <= binary_threshold).sum()
    at_max = (quality >= 1.0 - binary_threshold).sum()
    binary_pct = (at_min + at_max) / len(quality) * 100
    
    print(f"\n   1. Binary-like distribution:")
    print(f"      Samples at min (≤{binary_threshold}): {at_min} ({at_min/len(quality)*100:.1f}%)")
    print(f"      Samples at max (≥{1-binary_threshold}): {at_max} ({at_max/len(quality)*100:.1f}%)")
    print(f"      Total binary: {binary_pct:.1f}%")
    if binary_pct > 50:
        print(f"      ❌ WARNING: Binary-like distribution ({binary_pct:.1f}% at extremes)")
    else:
        print(f"      ✅ OK: Good continuous distribution")
    
    # 2. Narrow distribution
    iqr = quality.quantile(0.75) - quality.quantile(0.25)
    print(f"\n   2. Distribution width:")
    print(f"      IQR (25th-75th): {iqr:.4f}")
    print(f"      Range: {quality.max() - quality.min():.4f}")
    if iqr < 0.1:
        print(f"      ❌ WARNING: Very narrow distribution (IQR < 0.1)")
    else:
        print(f"      ✅ OK: Good spread")
    
    # 3. Check for mode at specific values
    print(f"\n   3. Suspicious concentrations:")
    for val in [0.0, 0.2, 0.5, 1.0]:
        count = (np.abs(quality - val) < 0.01).sum()
        pct = count / len(quality) * 100
        print(f"      Samples ≈ {val}: {count} ({pct:.1f}%)")
        if pct > 20:
            print(f"         ❌ WARNING: {pct:.1f}% concentrated at {val}")
    
    # 4. Variance check
    if quality.std() < 0.05:
        print(f"\n   4. ❌ WARNING: Very low variance (std={quality.std():.4f})")
    else:
        print(f"\n   4. ✅ OK: Adequate variance (std={quality.std():.4f})")
    
    return quality


def plot_quality_distribution(data, output_dir='analysis_output'):
    """Create visualization of quality score distribution."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    quality = data['quality_score']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histogram
    axes[0, 0].hist(quality, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(quality.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {quality.mean():.3f}')
    axes[0, 0].axvline(quality.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {quality.median():.3f}')
    axes[0, 0].set_xlabel('Quality Score', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Quality Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box plot
    axes[0, 1].boxplot(quality, vert=True)
    axes[0, 1].set_ylabel('Quality Score', fontsize=12)
    axes[0, 1].set_title('Quality Score Box Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. CDF
    sorted_quality = np.sort(quality)
    cdf = np.arange(1, len(sorted_quality) + 1) / len(sorted_quality)
    axes[1, 0].plot(sorted_quality, cdf, linewidth=2, color='steelblue')
    axes[1, 0].set_xlabel('Quality Score', fontsize=12)
    axes[1, 0].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1, 0].set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Density plot
    axes[1, 1].hist(quality, bins=100, density=True, alpha=0.5, color='steelblue', edgecolor='black')
    quality.plot(kind='kde', ax=axes[1, 1], linewidth=2, color='red')
    axes[1, 1].set_xlabel('Quality Score', fontsize=12)
    axes[1, 1].set_ylabel('Density', fontsize=12)
    axes[1, 1].set_title('Quality Score Density', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'quality_score_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Distribution plot saved to: {output_path}")
    plt.close()


def analyze_component_scores(data, output_dir='analysis_output'):
    """Analyze the components of quality score."""
    print("\n" + "="*80)
    print("🧩 QUALITY SCORE COMPONENTS ANALYSIS")
    print("="*80)
    
    components = ['bounce_strength', 'hold_strength', 'trade_profit']
    weights = [0.35, 0.35, 0.30]
    
    print(f"\nFormula: quality_score = bounce_strength * 0.35 + hold_strength * 0.35 + trade_profit * 0.30")
    print(f"\nComponent statistics:")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (comp, weight) in enumerate(zip(components, weights)):
        if comp in data.columns:
            values = data[comp]
            print(f"\n   {comp} (weight={weight}):")
            print(f"      Mean:   {values.mean():.4f}")
            print(f"      Median: {values.median():.4f}")
            print(f"      Std:    {values.std():.4f}")
            print(f"      Min:    {values.min():.4f}")
            print(f"      Max:    {values.max():.4f}")
            
            # Plot
            axes[i].hist(values, bins=50, edgecolor='black', alpha=0.7, color='coral')
            axes[i].axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.3f}')
            axes[i].set_xlabel(comp.replace('_', ' ').title(), fontsize=11)
            axes[i].set_ylabel('Frequency', fontsize=11)
            axes[i].set_title(f'{comp.replace("_", " ").title()} (weight={weight})', fontsize=12, fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Quality score in the 4th subplot
    axes[3].hist(data['quality_score'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[3].axvline(data['quality_score'].mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {data["quality_score"].mean():.3f}')
    axes[3].set_xlabel('Quality Score', fontsize=11)
    axes[3].set_ylabel('Frequency', fontsize=11)
    axes[3].set_title('Final Quality Score', fontsize=12, fontweight='bold')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'quality_components.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Component analysis plot saved to: {output_path}")
    plt.close()


def analyze_feature_correlations(data, output_dir='analysis_output'):
    """Analyze correlations between features and quality score."""
    print("\n" + "="*80)
    print("🔗 FEATURE-QUALITY CORRELATIONS")
    print("="*80)
    
    # Get feature columns
    feature_cols = [c for c in data.columns if c.startswith('feature_')]
    
    if not feature_cols:
        print("\n   ❌ No feature columns found!")
        return
    
    print(f"\n   Total features: {len(feature_cols)}")
    
    # Calculate correlations
    correlations = data[feature_cols].corrwith(data['quality_score']).abs().sort_values(ascending=False)
    
    # Top correlations
    print(f"\n   📊 Top 20 Features Correlated with Quality Score:")
    print(f"   {'Rank':<6} {'Feature':<45} {'|Correlation|':<15}")
    print(f"   {'-'*70}")
    
    for i, (feat, corr) in enumerate(correlations.head(20).items(), 1):
        feat_name = feat.replace('feature_', '')
        print(f"   {i:<6} {feat_name:<45} {corr:<15.4f}")
    
    # Red flag check
    print(f"\n   🚨 Correlation Quality Check:")
    strong_corr = (correlations > 0.3).sum()
    weak_corr = (correlations < 0.1).sum()
    
    print(f"      Strong correlations (>0.3): {strong_corr} features")
    print(f"      Weak correlations (<0.1):   {weak_corr} features")
    
    if correlations.iloc[0] < 0.2:
        print(f"      ❌ WARNING: Strongest correlation is only {correlations.iloc[0]:.4f}")
        print(f"         Quality score may not be predictable from features!")
    else:
        print(f"      ✅ OK: Top correlation is {correlations.iloc[0]:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot of top 20
    top_20 = correlations.head(20)
    axes[0].barh(range(len(top_20)), top_20.values, color='steelblue')
    axes[0].set_yticks(range(len(top_20)))
    axes[0].set_yticklabels([f.replace('feature_', '') for f in top_20.index], fontsize=9)
    axes[0].set_xlabel('|Correlation| with Quality Score', fontsize=12)
    axes[0].set_title('Top 20 Features by Correlation', fontsize=14, fontweight='bold')
    axes[0].axvline(0.3, color='red', linestyle='--', linewidth=2, label='Strong (0.3)')
    axes[0].axvline(0.5, color='green', linestyle='--', linewidth=2, label='Very Strong (0.5)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()
    
    # Distribution of all correlations
    axes[1].hist(correlations, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1].axvline(correlations.mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {correlations.mean():.3f}')
    axes[1].axvline(0.3, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Strong threshold (0.3)')
    axes[1].set_xlabel('|Correlation|', fontsize=12)
    axes[1].set_ylabel('Number of Features', fontsize=12)
    axes[1].set_title('Distribution of Feature Correlations', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'feature_correlations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Correlation plot saved to: {output_path}")
    plt.close()
    
    return correlations


def analyze_forward_vs_backward(data):
    """Check if quality is based on forward or backward data."""
    print("\n" + "="*80)
    print("⏰ FORWARD vs BACKWARD ANALYSIS")
    print("="*80)
    
    print(f"\n   Quality Score Components (from code analysis):")
    print(f"      ✅ bounce_strength:  Measured on FUTURE data (forward_days window)")
    print(f"      ✅ hold_strength:    Measured on FUTURE data (does level hold?)")
    print(f"      ✅ trade_profit:     Simulated trade on FUTURE data")
    print(f"\n   🎯 VERDICT: Quality score is FORWARD-LOOKING (predictive)")
    
    # Check correlation with historical features
    historical_features = [
        'feature_touch_count',
        'feature_strength', 
        'feature_age_bars',
        'feature_failure_count'
    ]
    
    print(f"\n   Correlation with HISTORICAL features:")
    for feat in historical_features:
        if feat in data.columns:
            corr = data[feat].corr(data['quality_score'])
            print(f"      {feat.replace('feature_', ''):<20}: {corr:.4f}")
    
    # This will show if quality is just historical strength
    if 'feature_strength' in data.columns:
        strength_corr = abs(data['feature_strength'].corr(data['quality_score']))
        if strength_corr > 0.7:
            print(f"\n   ⚠️  WARNING: Very high correlation with historical strength ({strength_corr:.3f})")
            print(f"       Quality may be too dependent on past, not future!")
        else:
            print(f"\n   ✅ OK: Quality is NOT just historical strength (corr={strength_corr:.3f})")


def check_untested_levels(data):
    """Check for untested levels (quality = 0.2)."""
    print("\n" + "="*80)
    print("🔍 UNTESTED LEVELS ANALYSIS")
    print("="*80)
    
    # The code filters out quality_score == 0.2 (untested levels)
    untested_count = (data['quality_score'] == 0.2).sum()
    
    print(f"\n   Samples with quality_score == 0.2 (untested): {untested_count}")
    
    if untested_count > 0:
        print(f"   ❌ WARNING: {untested_count} untested levels in dataset!")
        print(f"      These levels were NEVER HIT in forward window")
        print(f"      They should have been filtered out!")
    else:
        print(f"   ✅ OK: No untested levels (all filtered correctly)")
    
    # Check other suspicious values
    print(f"\n   Other suspicious concentrations:")
    for val in [0.0, 0.3, 0.5]:
        count = (np.abs(data['quality_score'] - val) < 0.001).sum()
        if count > len(data) * 0.05:  # More than 5%
            print(f"      {count} samples at exactly {val} ({count/len(data)*100:.1f}%)")


def generate_summary_report(data, correlations, output_dir='analysis_output'):
    """Generate a text summary report."""
    output_path = Path(output_dir) / 'quality_score_investigation_report.txt'
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SR QUALITY SCORE INVESTIGATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # 1. Formula
        f.write("1. QUALITY SCORE FORMULA\n")
        f.write("-"*80 + "\n")
        f.write("quality_score = bounce_strength * 0.35 + hold_strength * 0.35 + trade_profit * 0.30\n\n")
        f.write("Components:\n")
        f.write("  - bounce_strength (35%): Future price bounce after hitting level\n")
        f.write("  - hold_strength (35%):   How long level holds before breaking\n")
        f.write("  - trade_profit (30%):    Simulated trade P&L (1% SL, 2% TP)\n\n")
        f.write("✅ FORWARD-LOOKING: Uses future data (forward_days window)\n")
        f.write("✅ NOT backward-looking (not based on historical touches)\n\n")
        
        # 2. Distribution
        f.write("\n2. DISTRIBUTION STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total samples:      {len(data):,}\n")
        f.write(f"Mean:               {data['quality_score'].mean():.4f}\n")
        f.write(f"Median:             {data['quality_score'].median():.4f}\n")
        f.write(f"Std Dev:            {data['quality_score'].std():.4f}\n")
        f.write(f"Min:                {data['quality_score'].min():.4f}\n")
        f.write(f"Max:                {data['quality_score'].max():.4f}\n")
        f.write(f"IQR:                {data['quality_score'].quantile(0.75) - data['quality_score'].quantile(0.25):.4f}\n\n")
        
        # 3. Top features
        f.write("\n3. TOP FEATURES CORRELATED WITH QUALITY\n")
        f.write("-"*80 + "\n")
        for i, (feat, corr) in enumerate(correlations.head(15).items(), 1):
            f.write(f"{i:2d}. {feat.replace('feature_', ''):<40} {corr:.4f}\n")
        
        # 4. Red flags
        f.write("\n\n4. RED FLAGS CHECK\n")
        f.write("-"*80 + "\n")
        
        quality = data['quality_score']
        
        # Binary distribution
        at_extremes = ((quality <= 0.1).sum() + (quality >= 0.9).sum()) / len(quality) * 100
        f.write(f"❌ Binary distribution: {at_extremes:.1f}% at extremes\n")
        if at_extremes > 50:
            f.write("   WARNING: Too binary!\n")
        
        # Narrow distribution
        iqr = quality.quantile(0.75) - quality.quantile(0.25)
        f.write(f"{'❌' if iqr < 0.1 else '✅'} Distribution width: IQR = {iqr:.4f}\n")
        
        # Low correlation
        max_corr = correlations.iloc[0] if len(correlations) > 0 else 0
        f.write(f"{'❌' if max_corr < 0.2 else '✅'} Max correlation: {max_corr:.4f}\n")
        
        # Variance
        f.write(f"{'❌' if quality.std() < 0.05 else '✅'} Variance: std = {quality.std():.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Report generated successfully!\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Summary report saved to: {output_path}")


def main():
    """Main investigation function."""
    print("\n" + "="*80)
    print("🔍 SR QUALITY SCORE INVESTIGATION")
    print("="*80)
    
    # Load data
    print("\n📂 Loading training data...")
    data, metadata = load_training_data()
    
    if metadata:
        print(f"\n📋 Dataset Metadata:")
        print(f"   Created: {metadata.get('created_at', 'unknown')}")
        print(f"   Samples: {metadata.get('samples', 'unknown'):,}")
        print(f"   Features: {metadata.get('feature_count', 'unknown')}")
        print(f"   Symbols: {', '.join(metadata.get('symbols', []))}")
        print(f"   Timeframes: {', '.join(metadata.get('timeframes', []))}")
    
    print(f"\n✅ Loaded {len(data):,} samples")
    print(f"   Columns: {len(data.columns)}")
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
    
    # Run analyses
    print("\n" + "="*80)
    print("RUNNING ANALYSIS...")
    print("="*80)
    
    # 1. Distribution
    analyze_quality_score_distribution(data)
    plot_quality_distribution(data)
    
    # 2. Components
    analyze_component_scores(data)
    
    # 3. Correlations
    correlations = analyze_feature_correlations(data)
    
    # 4. Forward vs backward
    analyze_forward_vs_backward(data)
    
    # 5. Untested levels
    check_untested_levels(data)
    
    # 6. Generate report
    generate_summary_report(data, correlations)
    
    print("\n" + "="*80)
    print("✅ INVESTIGATION COMPLETE!")
    print("="*80)
    print(f"\n📁 All outputs saved to: analysis_output/")
    print(f"\n📊 Generated files:")
    print(f"   - quality_score_distribution.png")
    print(f"   - quality_components.png")
    print(f"   - feature_correlations.png")
    print(f"   - quality_score_investigation_report.txt")
    
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY:")
    print("="*80)
    
    quality = data['quality_score']
    
    print(f"\n✅ Quality Formula: FORWARD-LOOKING (good!)")
    print(f"   - Uses future bounce, hold, and trade profit")
    print(f"   - NOT based on historical touches only")
    
    print(f"\n📊 Distribution:")
    print(f"   - Mean: {quality.mean():.3f}")
    print(f"   - Std:  {quality.std():.3f}")
    print(f"   - IQR:  {quality.quantile(0.75) - quality.quantile(0.25):.3f}")
    
    if correlations is not None and len(correlations) > 0:
        print(f"\n🔗 Feature Correlations:")
        print(f"   - Top correlation: {correlations.iloc[0]:.3f} ({correlations.index[0].replace('feature_', '')})")
        print(f"   - Strong (>0.3): {(correlations > 0.3).sum()} features")
        print(f"   - Weak (<0.1):   {(correlations < 0.1).sum()} features")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()

