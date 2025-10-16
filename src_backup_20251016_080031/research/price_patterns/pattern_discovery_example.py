"""
Pattern Discovery Example: Mathematical Precision in Action

This example demonstrates the core innovation of mathematical pattern precision.
It shows the difference between vague pattern concepts and exact mathematical definitions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from pattern_discovery_framework import (
    PatternDiscoveryOrchestrator,
    MomentumPersistenceDiscoverer,
    MeanReversionSpeedDiscoverer,
    VolatilityExpansionDiscoverer
)


def create_sample_price_data(n_periods: int = 1000) -> pd.DataFrame:
    """Create sample price data with embedded patterns."""
    
    np.random.seed(42)
    
    # Base random walk
    returns = np.random.normal(0.0005, 0.02, n_periods)
    
    # Add momentum periods (every 100-150 periods)
    for i in range(50, n_periods, 120):
        if i + 15 < n_periods:
            # Create momentum persistence pattern
            momentum_direction = np.random.choice([-1, 1])
            base_momentum = 0.008 * momentum_direction
            
            for j in range(15):
                if i + j < n_periods:
                    # Gradual decay of momentum
                    decay_factor = max(0.3, 1 - j * 0.05)
                    returns[i + j] = base_momentum * decay_factor + np.random.normal(0, 0.01)
    
    # Add mean reversion periods
    for i in range(80, n_periods, 200):
        if i + 20 < n_periods:
            # Create oversold/overbought condition followed by reversion
            extreme_move = 0.05 * np.random.choice([-1, 1])
            returns[i] = extreme_move
            
            # Gradual reversion over next periods
            for j in range(1, 12):
                if i + j < n_periods:
                    reversion_strength = extreme_move * -0.3 * np.exp(-j * 0.2)
                    returns[i + j] = reversion_strength + np.random.normal(0, 0.015)
    
    # Add volatility expansion periods
    for i in range(200, n_periods, 300):
        if i + 25 < n_periods:
            # Low volatility period
            for j in range(10):
                if i + j < n_periods:
                    returns[i + j] = np.random.normal(0, 0.005)  # Very low vol
            
            # Followed by high volatility
            for j in range(10, 20):
                if i + j < n_periods:
                    returns[i + j] = np.random.normal(0, 0.04)  # High vol
    
    # Generate prices
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLC data
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods)))
    volume = 1000000 + np.random.normal(0, 200000, n_periods)
    
    data = pd.DataFrame({
        'open': np.roll(prices, 1),
        'high': high,
        'low': low,
        'close': prices,
        'volume': np.maximum(volume, 100000)
    })
    
    data['open'].iloc[0] = prices[0]
    data.index = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    return data


def demonstrate_mathematical_precision():
    """Demonstrate the mathematical precision of pattern definitions."""
    
    print("🎯 MATHEMATICAL PATTERN PRECISION DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Create sample data
    print("📊 Creating sample price data with embedded patterns...")
    market_data = create_sample_price_data(1000)
    prices = market_data['close']
    print(f"   Generated {len(prices)} price points")
    print()
    
    # Initialize pattern discoverers
    momentum_discoverer = MomentumPersistenceDiscoverer()
    reversion_discoverer = MeanReversionSpeedDiscoverer()
    volatility_discoverer = VolatilityExpansionDiscoverer()
    
    print("🔬 MATHEMATICAL PATTERN DEFINITIONS:")
    print("=" * 60)
    
    # Show mathematical definitions
    patterns = [
        ("Momentum Persistence", momentum_discoverer),
        ("Mean Reversion Speed", reversion_discoverer),
        ("Volatility Expansion", volatility_discoverer)
    ]
    
    for pattern_name, discoverer in patterns:
        definition = discoverer.get_pattern_definition()
        print(f"\n📈 {pattern_name.upper()}")
        print("-" * 40)
        print(f"Description: {definition.description}")
        print(f"\nMathematical Formula:")
        print(definition.mathematical_formula)
        print(f"\nParameters: {definition.parameters}")
        print()
    
    print("\n🔍 PATTERN DISCOVERY RESULTS:")
    print("=" * 60)
    
    # Discover patterns
    results = {}
    
    for pattern_name, discoverer in patterns:
        print(f"\n📊 Discovering {pattern_name}...")
        
        try:
            result = discoverer.discover_pattern(prices)
            results[pattern_name] = result
            
            print(f"   ✅ Pattern discovered!")
            print(f"   📊 Frequency: {result.frequency:.3f} ({result.frequency*100:.1f}% of periods)")
            print(f"   🎯 Predictability: {result.predictability_score:.3f}")
            print(f"   📈 Signal/Noise: {1-result.noise_ratio:.3f}")
            print(f"   📏 Avg Duration: {result.duration_stats['mean']:.1f} periods")
            print(f"   ✅ Valid Pattern: {'Yes' if result.is_valid_pattern else 'No'}")
            
            if result.statistical_significance.get('p_value'):
                p_val = result.statistical_significance['p_value']
                significance = "Significant" if p_val < 0.05 else "Not Significant"
                print(f"   📊 Statistical: {significance} (p={p_val:.3f})")
            
        except Exception as e:
            print(f"   ❌ Discovery failed: {e}")
    
    print("\n🎯 PRECISION VS VAGUE CONCEPTS:")
    print("=" * 60)
    
    print("\n❌ VAGUE (Traditional Approach):")
    print('   "Look for momentum patterns in the data"')
    print('   "Identify mean reversion behavior"')
    print('   "Find volatility clustering"')
    print("   → Subjective, not reproducible, not ML-ready")
    
    print("\n✅ MATHEMATICALLY PRECISE (Our Approach):")
    
    for pattern_name, result in results.items():
        if result.is_valid_pattern:
            pattern_count = result.labels.sum()
            total_periods = len(result.labels)
            
            print(f"\n   {pattern_name.upper()}:")
            print(f"   → Exact formula with specific thresholds")
            print(f"   → Binary labels: {pattern_count} patterns in {total_periods} periods")
            print(f"   → ML-ready targets: [0,1,0,1,0,0,1,1,0,...]")
            print(f"   → Reproducible across datasets")
    
    print("\n📈 ML TRAINING READINESS:")
    print("=" * 60)
    
    # Show ML readiness
    valid_patterns = {name: result for name, result in results.items() if result.is_valid_pattern}
    
    if valid_patterns:
        print(f"\n✅ {len(valid_patterns)} patterns ready for ML training:")
        
        # Create ML target matrix
        ml_targets = {}
        for name, result in valid_patterns.items():
            ml_targets[name.lower().replace(' ', '_')] = result.labels
        
        if ml_targets:
            targets_df = pd.DataFrame(ml_targets)
            print(f"\n📊 ML Target Matrix Shape: {targets_df.shape}")
            print(f"   Columns: {list(targets_df.columns)}")
            print(f"\n   Sample targets (first 10 periods):")
            print(targets_df.head(10).to_string())
            
            print(f"\n🎯 Pattern Occurrence Summary:")
            for col in targets_df.columns:
                count = targets_df[col].sum()
                pct = count / len(targets_df) * 100
                print(f"   {col}: {count} occurrences ({pct:.1f}%)")
    
    else:
        print("\n❌ No valid patterns found in sample data")
        print("   Try adjusting parameters or using different data")
    
    print("\n🚀 KEY INNOVATIONS:")
    print("=" * 60)
    print("\n1. MATHEMATICAL PRECISION:")
    print("   ✅ Exact formulas instead of vague concepts")
    print("   ✅ Specific thresholds and parameters")
    print("   ✅ Reproducible across different datasets")
    
    print("\n2. ML-READY OUTPUTS:")
    print("   ✅ Binary labels (0/1) for supervised learning")
    print("   ✅ Pattern frequency and duration statistics")
    print("   ✅ Statistical significance testing")
    
    print("\n3. PATTERN VALIDATION:")
    print("   ✅ Frequency thresholds (must occur often enough)")
    print("   ✅ Predictability scores (not random noise)")
    print("   ✅ Signal-to-noise ratio analysis")
    
    print("\n4. ECONOMIC FOUNDATION:")
    print("   ✅ Patterns based on market behavior theory")
    print("   ✅ Duration and magnitude measurements")
    print("   ✅ Statistical significance testing")
    
    print("\n" + "=" * 60)
    print("✅ MATHEMATICAL PATTERN PRECISION DEMONSTRATION COMPLETE")
    print("=" * 60)


def demonstrate_pattern_comparison():
    """Compare traditional vs mathematical approach side by side."""
    
    print("\n🔄 TRADITIONAL vs MATHEMATICAL APPROACH COMPARISON")
    print("=" * 70)
    
    comparisons = [
        {
            'aspect': 'Pattern Definition',
            'traditional': '"Look for momentum patterns"',
            'mathematical': 'IF |momentum(t)| > 0.005 AND same_direction ≥70% for 10 periods THEN pattern=1'
        },
        {
            'aspect': 'Reproducibility',
            'traditional': 'Subjective - different analysts get different results',
            'mathematical': 'Exact formula - same results every time'
        },
        {
            'aspect': 'ML Applicability',
            'traditional': 'Unclear how to use for supervised learning',
            'mathematical': 'Binary labels [0,1,0,1,...] ready for ML training'
        },
        {
            'aspect': 'Validation',
            'traditional': 'Visual inspection, subjective assessment',
            'mathematical': 'Statistical significance, frequency thresholds, predictability scores'
        },
        {
            'aspect': 'Parameter Tuning',
            'traditional': 'Ad-hoc adjustments based on "feel"',
            'mathematical': 'Systematic parameter optimization with clear objectives'
        }
    ]
    
    for comp in comparisons:
        print(f"\n📊 {comp['aspect'].upper()}:")
        print(f"   ❌ Traditional: {comp['traditional']}")
        print(f"   ✅ Mathematical: {comp['mathematical']}")
    
    print(f"\n🎯 RESULT:")
    print("   Traditional: Vague, subjective, not ML-ready")
    print("   Mathematical: Precise, reproducible, ML-ready")


if __name__ == "__main__":
    demonstrate_mathematical_precision()
    demonstrate_pattern_comparison()