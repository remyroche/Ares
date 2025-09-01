#!/usr/bin/env python3
"""
Example: Diverse Lookback Period Optimization

This script demonstrates finding 2-3 lookback periods for each feature that deliver
meaningful yet significantly different information.
"""

import asyncio
import pandas as pd
import numpy as np
from src.training.diverse_lookback_optimizer import DiverseLookbackOptimizer
from src.config.diverse_lookback_config import get_diverse_lookback_config

async def demonstrate_diverse_lookback_optimization():
    """Demonstrate diverse lookback period optimization."""

    print("🎯 DIVERSE LOOKBACK PERIOD OPTIMIZATION DEMONSTRATION")
    print("=" * 65)
    print("Finding 2-3 meaningful yet different lookback periods for each feature!")
    print()

    # 1. Initialize the diverse lookback optimizer
    config = get_diverse_lookback_config()
    diverse_optimizer = DiverseLookbackOptimizer(config)

    # 2. Show the optimization objectives
    print("🎯 OPTIMIZATION OBJECTIVES:")
    print("-" * 30)
    print("1. Find 2-3 lookback periods per feature")
    print("2. Ensure meaningful signal strength (SHAP importance > 0.1)")
    print("3. Maximize information diversity (low correlation between periods)")
    print("4. Capture complementary market insights")
    print("5. Optimize for high leverage trading scenarios")
    print()

    # 3. Create sample data with multiple market regimes
    print("📈 CREATING SAMPLE DATA WITH MULTIPLE REGIMES...")
    dates = pd.date_range('2023-01-01', periods=3000, freq='1min')
    np.random.seed(42)

    # Generate realistic price data with 4 distinct regimes
    n_samples = len(dates)
    regime_length = n_samples // 4

    prices = []
    for i in range(4):
        if i == 0:  # Trending up regime
            returns = np.random.normal(0.0003, 0.001, regime_length)
        elif i == 1:  # Trending down regime
            returns = np.random.normal(-0.0003, 0.001, regime_length)
        elif i == 2:  # High volatility regime
            returns = np.random.normal(0, 0.0025, regime_length)
        else:  # Low volatility regime
            returns = np.random.normal(0, 0.0003, regime_length)

        prices.extend(1000 * np.exp(np.cumsum(returns)))

    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    data.set_index('timestamp', inplace=True)

    # Create target variable (next period returns)
    data['returns'] = data['close'].pct_change().shift(-1)

    # Create regime labels
    data['regime'] = np.repeat([0, 1, 2, 3], regime_length)

    print(f"✅ Created sample data with {len(data)} rows")
    print(f"   Columns: {list(data.columns)}")
    print(f"   Regimes: {data['regime'].unique()}")
    print()

    # 4. Show the lookback ranges for different features
    print("🔍 LOOKBACK RANGES FOR DIFFERENT FEATURES:")
    print("-" * 45)

    lookback_ranges = config["lookback_ranges"]
    for feature_name, range_config in lookback_ranges.items():
        periods = list(range(range_config["min"], range_config["max"] + 1, range_config["step"]))
        print(f"{feature_name:15} : {range_config['min']:2d}-{range_config['max']:2d} (step {range_config['step']:1d}) = {len(periods):2d} periods")
        print(f"{'':15}   {range_config['description']}")
        print(f"{'':15}   Expected insights: {', '.join(range_config['expected_insights'])}")
        print()

    # 5. Demonstrate the diverse period selection process
    print("🧠 DIVERSE PERIOD SELECTION PROCESS:")
    print("-" * 40)

    print("Step 1: Calculate all periods for each feature")
    print("   - Generate feature values for each lookback period")
    print("   - Calculate SHAP importance for each period")
    print("   - Filter meaningful periods (importance > threshold)")
    print()

    print("Step 2: Select diverse subset using greedy algorithm")
    print("   - Start with highest importance period")
    print("   - Add periods that maximize diversity")
    print("   - Ensure low correlation between selected periods")
    print()

    print("Step 3: Analyze information content and market insights")
    print("   - Determine market insight for each period")
    print("   - Analyze complementarity between periods")
    print("   - Validate regime-specific performance")
    print()

    # 6. Show example for RSI
    print("📊 EXAMPLE: RSI DIVERSE PERIOD SELECTION")
    print("-" * 40)

    # Simulate RSI period analysis
    rsi_periods = list(range(5, 51, 2))  # 5 to 50 in steps of 2
    print(f"Testing {len(rsi_periods)} RSI periods: {rsi_periods}")
    print()

    # Simulate period scores
    period_scores = [
        {"period": 7, "information_score": 0.85, "market_insight": "Short-term momentum"},
        {"period": 14, "information_score": 0.78, "market_insight": "Medium-term trend"},
        {"period": 21, "information_score": 0.72, "market_insight": "Long-term trend"},
        {"period": 28, "information_score": 0.65, "market_insight": "Major trend"},
        {"period": 35, "information_score": 0.58, "market_insight": "Market regime"},
        {"period": 42, "information_score": 0.45, "market_insight": "Long-term cycles"}
    ]

    print("Period scores (information score + market insight):")
    for score in period_scores:
        print(f"   Period {score['period']:2d}: {score['information_score']:.2f} - {score['market_insight']}")
    print()

    # Simulate diversity analysis
    print("Diversity analysis between selected periods:")
    selected_periods = [7, 14, 21]  # Example selection

    correlations = [
        {"periods": "7 vs 14", "correlation": 0.45, "diversity": 0.55},
        {"periods": "7 vs 21", "correlation": 0.32, "diversity": 0.68},
        {"periods": "14 vs 21", "correlation": 0.58, "diversity": 0.42}
    ]

    for corr in correlations:
        print(f"   {corr['periods']}: correlation={corr['correlation']:.2f}, diversity={corr['diversity']:.2f}")

    avg_diversity = np.mean([c['diversity'] for c in correlations])
    print(f"   Average diversity: {avg_diversity:.2f}")
    print()

    # 7. Show the final diverse periods for each feature
    print("🎯 FINAL DIVERSE PERIODS FOR EACH FEATURE:")
    print("-" * 45)

    # Simulate final results
    diverse_periods = {
        "RSI": {
            "selected_periods": [7, 14, 21],
            "insights": ["Short-term momentum", "Medium-term trend", "Long-term trend"],
            "diversity_score": 0.55
        },
        "MACD_fast": {
            "selected_periods": [8, 12, 16],
            "insights": ["Quick momentum", "Fast trend changes", "Short-term signals"],
            "diversity_score": 0.62
        },
        "MACD_slow": {
            "selected_periods": [24, 30, 36],
            "insights": ["Trend confirmation", "Medium-term trend", "Signal filtering"],
            "diversity_score": 0.48
        },
        "Bollinger_Bands": {
            "selected_periods": [14, 28, 42],
            "insights": ["Volatility regime", "Price extremes", "Mean reversion"],
            "diversity_score": 0.58
        },
        "ATR": {
            "selected_periods": [8, 16, 24],
            "insights": ["Quick volatility", "Medium volatility", "Long volatility"],
            "diversity_score": 0.51
        }
    }

    for feature, data in diverse_periods.items():
        print(f"{feature:15} : Periods {data['selected_periods']}")
        print(f"{'':15}   Insights: {', '.join(data['insights'])}")
        print(f"{'':15}   Diversity: {data['diversity_score']:.2f}")
        print()

    # 8. Show regime-specific diverse periods
    print("🔄 REGIME-SPECIFIC DIVERSE PERIODS:")
    print("-" * 35)

    regime_periods = {
        "regime_0_trending_up": {
            "RSI": [10, 18, 26],  # Longer periods for trend confirmation
            "MACD_fast": [10, 14, 18],
            "insight": "Trend continuation periods"
        },
        "regime_2_high_volatility": {
            "RSI": [5, 12, 19],   # Shorter periods for quick signals
            "MACD_fast": [6, 10, 14],
            "insight": "Quick response periods"
        }
    }

    for regime, periods in regime_periods.items():
        print(f"{regime}:")
        for feature, feature_periods in periods.items():
            if feature != "insight":
                print(f"   {feature}: {feature_periods}")
        print(f"   Insight: {periods['insight']}")
        print()

    # 9. Show the benefits of diverse periods
    print("✅ BENEFITS OF DIVERSE LOOKBACK PERIODS:")
    print("-" * 40)
    print("1. Information Diversity:")
    print("   - Different market insights (momentum, trend, volatility)")
    print("   - Complementary signal timing")
    print("   - Reduced redundancy in feature set")
    print()

    print("2. Robust Performance:")
    print("   - Better generalization across market regimes")
    print("   - Reduced overfitting to specific timeframes")
    print("   - More stable model performance")
    print()

    print("3. High Leverage Optimization:")
    print("   - Quick signals for fast entries/exits")
    print("   - Trend confirmation for position holding")
    print("   - Risk management through volatility periods")
    print()

    print("4. Market Regime Adaptation:")
    print("   - Regime-specific period optimization")
    print("   - Automatic adaptation to market conditions")
    print("   - Improved performance in different market states")
    print()

    # 10. Show integration with step7
    print("🔗 INTEGRATION WITH STEP 7:")
    print("-" * 30)
    print("The diverse lookback optimizer is integrated into step7:")
    print("1. Loads feature data and HMM regimes")
    print("2. Finds diverse periods for each feature")
    print("3. Analyzes diversity and information content")
    print("4. Generates regime-specific periods")
    print("5. Saves diverse lookback results")
    print()

    print("📊 OUTPUT FILES:")
    print("-" * 15)
    print("- data/diverse_lookback_optimization/")
    print("  ├── {exchange}_{symbol}_{timeframe}_diverse_lookback_periods.json")
    print("  ├── {exchange}_{symbol}_{timeframe}_diversity_analysis.json")
    print("  ├── {exchange}_{symbol}_{timeframe}_information_content.json")
    print("  └── {exchange}_{symbol}_{timeframe}_regime_specific_periods.json")
    print()

    # 11. Show example output structure
    print("📋 EXAMPLE OUTPUT STRUCTURE:")
    print("-" * 35)

    example_output = {
        "RSI": {
            "selected_periods": [7, 14, 21],
            "period_scores": [
                {"period": 7, "information_score": 0.85, "market_insight": "Short-term momentum"},
                {"period": 14, "information_score": 0.78, "market_insight": "Medium-term trend"},
                {"period": 21, "information_score": 0.72, "market_insight": "Long-term trend"}
            ],
            "diversity_metrics": {
                "diversity_score": 0.55,
                "avg_correlation": 0.45,
                "n_periods": 3
            }
        }
    }

    print("For each feature:")
    print("  - selected_periods: The 2-3 chosen lookback periods")
    print("  - period_scores: Information score and market insight for each period")
    print("  - diversity_metrics: Correlation and diversity analysis")
    print()

    print("🎉 DIVERSE LOOKBACK OPTIMIZATION DEMONSTRATION COMPLETE!")
    print()
    print("💡 KEY INSIGHTS:")
    print("- Finds 2-3 meaningful yet different lookback periods per feature")
    print("- Uses SHAP importance to ensure meaningful signal strength")
    print("- Maximizes diversity through correlation analysis")
    print("- Captures complementary market insights")
    print("- Optimizes for high leverage trading scenarios")
    print("- Provides regime-specific period optimization")

if __name__ == "__main__":
    asyncio.run(demonstrate_diverse_lookback_optimization())