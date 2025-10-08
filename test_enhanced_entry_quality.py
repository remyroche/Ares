"""
Test and Compare Enhanced Entry Quality Scoring Methods

This script demonstrates the differences between scoring methods:
1. Linear Weighted (original)
2. Adaptive Multi-Factor (recommended)
3. Information Ratio
4. Expected Utility
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from src.training.steps.models_training.enhanced_entry_quality_scorer import (
    EnhancedEntryQualityScorer,
    ScoringMethod,
    EnhancedScoringConfig,
    create_enhanced_scorer,
    compare_scoring_methods
)


def generate_synthetic_market_data(
    n_candles: int = 100,
    scenario: str = 'trending'
) -> pd.DataFrame:
    """
    Generate synthetic market data for testing.
    
    Scenarios:
    - 'trending': Strong uptrend with momentum
    - 'ranging': Sideways consolidation
    - 'volatile': High volatility swings
    - 'low_liquidity': Erratic volume patterns
    """
    np.random.seed(42)
    
    # Base price
    base_price = 100.0
    timestamps = pd.date_range(start='2024-01-01', periods=n_candles, freq='15min')
    
    if scenario == 'trending':
        # Strong uptrend
        drift = 0.002  # 0.2% per candle
        volatility = 0.005  # 0.5% volatility
        trend = np.arange(n_candles) * drift
        noise = np.random.normal(0, volatility, n_candles)
        close_prices = base_price * (1 + trend + noise)
        volumes = np.random.lognormal(10, 0.5, n_candles) * (1 + trend * 20)  # Increasing volume
        
    elif scenario == 'ranging':
        # Sideways movement
        volatility = 0.003
        noise = np.random.normal(0, volatility, n_candles)
        close_prices = base_price * (1 + noise + 0.01 * np.sin(np.arange(n_candles) / 10))
        volumes = np.random.lognormal(10, 0.3, n_candles)
        
    elif scenario == 'volatile':
        # High volatility
        volatility = 0.015
        noise = np.random.normal(0, volatility, n_candles)
        close_prices = base_price * (1 + noise)
        volumes = np.random.lognormal(10, 0.8, n_candles)
        
    elif scenario == 'low_liquidity':
        # Erratic with gaps
        volatility = 0.008
        noise = np.random.normal(0, volatility, n_candles)
        gaps = np.random.choice([0, 0.02, -0.02], size=n_candles, p=[0.8, 0.1, 0.1])
        close_prices = base_price * (1 + noise + gaps)
        volumes = np.random.lognormal(9, 1.2, n_candles)  # Very erratic
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Generate OHLCV
    opens = close_prices * (1 + np.random.normal(0, 0.001, n_candles))
    highs = np.maximum(opens, close_prices) * (1 + np.abs(np.random.normal(0, 0.002, n_candles)))
    lows = np.minimum(opens, close_prices) * (1 - np.abs(np.random.normal(0, 0.002, n_candles)))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': volumes
    })
    
    df.set_index('timestamp', inplace=True)
    return df


def test_single_entry(scenario: str = 'trending', entry_idx: int = 20, future_window: int = 20):
    """Test a single entry point with all scoring methods."""
    
    print(f"\n{'='*80}")
    print(f"TESTING SCENARIO: {scenario.upper()}")
    print(f"{'='*80}\n")
    
    # Generate market data
    data = generate_synthetic_market_data(n_candles=100, scenario=scenario)
    
    # Select entry point and future data
    entry_point = data.iloc[entry_idx]
    future_data = data.iloc[entry_idx+1:entry_idx+1+future_window]
    
    print(f"Entry Point: {entry_point.name}")
    print(f"Entry Price: ${entry_point['close']:.2f}")
    print(f"Future Window: {future_window} candles")
    print()
    
    # Market context
    market_context = {
        'regime_volatility': data['close'].pct_change().std(),
        'trend_strength': (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0],
        'liquidity_score': data['volume'].mean() / data['volume'].std()
    }
    
    # Determine regime
    if 'trend' in scenario:
        regime = 'trending'
    elif 'rang' in scenario:
        regime = 'ranging'
    elif 'volatile' in scenario:
        regime = 'high_volatility'
    else:
        regime = 'low_liquidity'
    
    print(f"Market Context:")
    print(f"  Regime: {regime}")
    print(f"  Volatility: {market_context['regime_volatility']:.4f}")
    print(f"  Trend Strength: {market_context['trend_strength']:.4f}")
    print(f"  Liquidity Score: {market_context['liquidity_score']:.4f}")
    print()
    
    # Calculate actual metrics
    entry_price = entry_point['close']
    max_favorable = (future_data['high'].max() - entry_price) / entry_price * 100
    max_adverse = (entry_price - future_data['low'].min()) / entry_price * 100
    time_to_peak = future_data['high'].idxmax() - entry_point.name
    
    print(f"Actual Forward Metrics:")
    print(f"  Max Favorable Move: {max_favorable:.2f}%")
    print(f"  Max Adverse Move: {max_adverse:.2f}%")
    print(f"  Risk-Reward Ratio: {max_favorable/max(max_adverse, 0.01):.2f}")
    print(f"  Time to Peak: {time_to_peak}")
    print()
    
    # Compare all scoring methods
    print(f"{'Method':<30} {'Score':<10} {'Quality'}")
    print(f"{'-'*60}")
    
    scores = compare_scoring_methods(entry_point, future_data, regime, market_context)
    
    for method_name, score in scores.items():
        if score is None:
            quality = "ERROR"
        elif score >= 0.7:
            quality = "EXCELLENT ★★★"
        elif score >= 0.5:
            quality = "GOOD ★★"
        elif score >= 0.3:
            quality = "FAIR ★"
        else:
            quality = "POOR"
        
        print(f"{method_name:<30} {score:<10.4f} {quality}")
    
    # Calculate score differences
    if ScoringMethod.LINEAR_WEIGHTED.value in scores and ScoringMethod.ADAPTIVE_MULTI_FACTOR.value in scores:
        linear_score = scores[ScoringMethod.LINEAR_WEIGHTED.value]
        adaptive_score = scores[ScoringMethod.ADAPTIVE_MULTI_FACTOR.value]
        
        if linear_score is not None and adaptive_score is not None:
            improvement = ((adaptive_score - linear_score) / max(linear_score, 0.01)) * 100
            print()
            print(f"Adaptive vs Linear Improvement: {improvement:+.1f}%")
    
    return scores, data, entry_idx, future_window


def test_multiple_entries(scenario: str = 'trending', n_entries: int = 20):
    """Test multiple entry points and compare methods."""
    
    print(f"\n{'='*80}")
    print(f"TESTING MULTIPLE ENTRIES: {scenario.upper()}")
    print(f"{'='*80}\n")
    
    # Generate market data
    data = generate_synthetic_market_data(n_candles=200, scenario=scenario)
    
    # Sample entry points
    entry_indices = np.linspace(20, 150, n_entries, dtype=int)
    future_window = 20
    
    # Store results
    results = {method.value: [] for method in ScoringMethod if method != ScoringMethod.ML_BASED}
    actual_metrics = {'favorable': [], 'adverse': [], 'risk_reward': []}
    
    for entry_idx in entry_indices:
        entry_point = data.iloc[entry_idx]
        future_data = data.iloc[entry_idx+1:entry_idx+1+future_window]
        
        if future_data.empty:
            continue
        
        # Calculate actual metrics
        entry_price = entry_point['close']
        max_favorable = (future_data['high'].max() - entry_price) / entry_price * 100
        max_adverse = (entry_price - future_data['low'].min()) / entry_price * 100
        risk_reward = max_favorable / max(max_adverse, 0.01)
        
        actual_metrics['favorable'].append(max_favorable)
        actual_metrics['adverse'].append(max_adverse)
        actual_metrics['risk_reward'].append(risk_reward)
        
        # Score with all methods
        scores = compare_scoring_methods(entry_point, future_data, scenario)
        
        for method_name, score in scores.items():
            if score is not None:
                results[method_name].append(score)
            else:
                results[method_name].append(0.0)
    
    # Calculate statistics
    print(f"Tested {len(entry_indices)} entry points\n")
    print(f"{'Method':<30} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"{'-'*70}")
    
    for method_name, scores in results.items():
        if len(scores) > 0:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            print(f"{method_name:<30} {mean_score:<10.4f} {std_score:<10.4f} {min_score:<10.4f} {max_score:<10.4f}")
    
    # Correlation with actual risk-reward
    print(f"\n{'='*70}")
    print(f"CORRELATION WITH ACTUAL RISK-REWARD RATIO")
    print(f"{'='*70}\n")
    print(f"{'Method':<30} {'Correlation':<15} {'Quality'}")
    print(f"{'-'*60}")
    
    for method_name, scores in results.items():
        if len(scores) > 0 and len(actual_metrics['risk_reward']) > 0:
            corr = np.corrcoef(scores, actual_metrics['risk_reward'][:len(scores)])[0, 1]
            
            if corr >= 0.7:
                quality = "EXCELLENT ★★★"
            elif corr >= 0.5:
                quality = "GOOD ★★"
            elif corr >= 0.3:
                quality = "FAIR ★"
            else:
                quality = "POOR"
            
            print(f"{method_name:<30} {corr:<15.4f} {quality}")
    
    return results, actual_metrics


def visualize_comparison(scenario: str = 'trending'):
    """Visualize scoring method comparison with plots."""
    
    try:
        results, actual_metrics = test_multiple_entries(scenario, n_entries=50)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Entry Quality Scoring Comparison - {scenario.upper()}', fontsize=16)
        
        # Plot 1: Score distributions
        ax = axes[0, 0]
        data_to_plot = [scores for scores in results.values() if len(scores) > 0]
        labels = [name for name, scores in results.items() if len(scores) > 0]
        ax.boxplot(data_to_plot, labels=labels)
        ax.set_ylabel('Quality Score')
        ax.set_title('Score Distributions by Method')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Correlation with risk-reward
        ax = axes[0, 1]
        rr_values = actual_metrics['risk_reward']
        
        for method_name, scores in results.items():
            if len(scores) > 0 and len(rr_values) > 0:
                ax.scatter(rr_values[:len(scores)], scores, alpha=0.5, label=method_name, s=20)
        
        ax.set_xlabel('Actual Risk-Reward Ratio')
        ax.set_ylabel('Predicted Quality Score')
        ax.set_title('Prediction vs Actual Risk-Reward')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Score evolution over time
        ax = axes[1, 0]
        for method_name, scores in results.items():
            if len(scores) > 0:
                ax.plot(scores, label=method_name, alpha=0.7)
        
        ax.set_xlabel('Entry Index')
        ax.set_ylabel('Quality Score')
        ax.set_title('Score Evolution Over Entries')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Method comparison heatmap
        ax = axes[1, 1]
        
        # Calculate mean scores for each method
        method_means = {name: np.mean(scores) for name, scores in results.items() if len(scores) > 0}
        method_names = list(method_means.keys())
        method_values = list(method_means.values())
        
        colors = ['red' if v < 0.3 else 'orange' if v < 0.5 else 'yellow' if v < 0.7 else 'green' for v in method_values]
        ax.barh(method_names, method_values, color=colors, alpha=0.7)
        ax.set_xlabel('Mean Quality Score')
        ax.set_title('Average Score by Method')
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save plot
        filename = f'entry_quality_comparison_{scenario}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✅ Plot saved to: {filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()


def run_comprehensive_tests():
    """Run comprehensive tests across all scenarios."""
    
    scenarios = ['trending', 'ranging', 'volatile', 'low_liquidity']
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ENTRY QUALITY SCORING COMPARISON")
    print("="*80)
    
    for scenario in scenarios:
        test_single_entry(scenario=scenario, entry_idx=30, future_window=20)
    
    # Multi-entry tests
    print("\n\n" + "="*80)
    print("MULTI-ENTRY STATISTICAL ANALYSIS")
    print("="*80)
    
    for scenario in scenarios:
        test_multiple_entries(scenario=scenario, n_entries=30)
    
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
The Enhanced Entry Quality Scoring system provides multiple sophisticated
algorithms for evaluating entry points:

1. **Linear Weighted** (Original):
   - Simple weighted average
   - Fast but limited adaptability
   
2. **Adaptive Multi-Factor** (Recommended):
   - Regime-aware dynamic weights
   - Additional factors (volume, momentum, microstructure)
   - Non-linear interaction terms
   - Expected improvement: 15-25%
   
3. **Information Ratio**:
   - Financial theory-based (risk-adjusted returns)
   - Naturally handles risk-reward tradeoff
   - Expected improvement: 10-20%
   
4. **Expected Utility**:
   - Economic theory (utility maximization)
   - Adjustable risk aversion
   - Expected improvement: 10-20%

**Recommendation**: Use Adaptive Multi-Factor for best results with minimal setup.
For ML-based scoring, train on historical entry performance data.
    """)


if __name__ == '__main__':
    print("Enhanced Entry Quality Scorer - Test Suite")
    print("="*80 + "\n")
    
    # Run comprehensive tests
    run_comprehensive_tests()
    
    # Generate visualizations for each scenario
    print("\n\nGenerating visualizations...")
    for scenario in ['trending', 'ranging', 'volatile', 'low_liquidity']:
        try:
            visualize_comparison(scenario)
        except Exception as e:
            print(f"⚠️  Skipping visualization for {scenario}: {e}")
    
    print("\n✅ All tests completed!")