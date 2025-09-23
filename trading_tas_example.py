#!/usr/bin/env python3
"""
Trading Tree Architecture Search (TAS) Example

This example demonstrates how to use the Trading TAS system for:
1. Regime exploration and qualification
2. Selection of appropriate tree-based ML models during trading
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import Trading TAS components
from src.utils.ml_common.optimization.trading_tree_architecture_search import (
    TradingTreeArchitectureSearch,
    TradingTASConfig,
    TradingObjective,
    MarketRegime,
    optimize_trading_regimes,
    select_trading_model
)


def generate_synthetic_market_data(n_days: int = 252, n_symbols: int = 5, base_volatility: float = 0.02):
    """Generate synthetic market data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

    market_data = []
    target_returns = []

    # Generate data for different market regimes
    regime_periods = [
        (0, 63, 'normal'),      # Q1: Normal market
        (63, 126, 'volatile'),   # Q2: High volatility
        (126, 189, 'trending'),  # Q3: Strong trend
        (189, 252, 'mixed')     # Q4: Mixed conditions
    ]

    for symbol in range(n_symbols):
        symbol_data = []
        symbol_returns = []

        for period_start, period_end, regime in regime_periods:
            period_length = period_end - period_start

            if regime == 'normal':
                # Normal market: moderate volatility, random walk
                price_changes = np.random.normal(0, base_volatility, period_length)
            elif regime == 'volatile':
                # High volatility: increased variance
                price_changes = np.random.normal(0, base_volatility * 2, period_length)
            elif regime == 'trending':
                # Strong trend: upward bias with noise
                trend = np.linspace(0, 0.05, period_length)  # 5% upward trend
                noise = np.random.normal(0, base_volatility * 1.5, period_length)
                price_changes = trend + noise
            else:  # mixed
                # Mixed: combination of regimes
                regime_switch = np.random.choice(['normal', 'volatile', 'trending'], period_length)
                price_changes = []
                for reg in regime_switch:
                    if reg == 'normal':
                        change = np.random.normal(0, base_volatility)
                    elif reg == 'volatile':
                        change = np.random.normal(0, base_volatility * 2)
                    else:  # trending
                        change = np.random.normal(0.001, base_volatility)
                    price_changes.append(change)

            # Convert to prices
            base_price = 100 + symbol * 10  # Different base prices for symbols
            prices = [base_price]

            for change in price_changes:
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)

            symbol_data.extend(prices[1:])  # Skip first price
            symbol_returns.extend(price_changes)

        market_data.append(symbol_data)
        target_returns.append(symbol_returns)

    # Create DataFrame with multi-level columns
    columns = pd.MultiIndex.from_tuples(
        [(f'SYMBOL_{i}', 'price') for i in range(n_symbols)],
        names=['symbol', 'metric']
    )

    df = pd.DataFrame(np.column_stack(market_data), index=dates, columns=columns)
    returns_df = pd.DataFrame(np.column_stack(target_returns), index=dates)

    return df, returns_df


def demonstrate_regime_exploration():
    """Demonstrate regime exploration and qualification using Trading TAS."""
    print("🌍 Trading TAS - Regime Exploration and Qualification")
    print("=" * 60)

    # Generate synthetic market data
    print("\n1. Generating synthetic market data...")
    market_data, target_returns = generate_synthetic_market_data(n_days=252, n_symbols=3)
    print(f"   Generated {len(market_data)} days of market data")
    print(f"   Symbols: {[col[0] for col in market_data.columns[:3]]}")

    # Configure Trading TAS for regime exploration
    print("\n2. Configuring Trading TAS for regime exploration...")
    config = TradingTASConfig(
        regime_detection_enabled=True,
        trading_objectives=[
            TradingObjective.PROFITABILITY,
            TradingObjective.ROBUSTNESS,
            TradingObjective.REGIME_STABILITY
        ],
        objective_weights=[0.4, 0.3, 0.3],
        min_regime_samples=30,
        enable_performance_tracking=True
    )

    # Run regime optimization
    print("\n3. Running regime-aware TAS optimization...")
    result = optimize_trading_regimes(market_data, target_returns.iloc[:, 0], config)

    # Display results
    print("
4. Regime Analysis Results"    print("-" * 40)

    print(f"   Total execution time: {result.execution_time:.2f} seconds")
    print(f"   Detected regimes: {len(result.regime_analysis)}")
    print(f"   Best overall architecture: {result.best_architecture.n_trees} trees, depth {result.best_architecture.max_depth}")

    # Analyze each regime
    print("
   📊 Regime Details:"    for regime_type, regime in result.regime_analysis.items():
        print(f"   • {regime_type.value.upper()}:")
        print(f"     - Confidence: {regime.confidence:.2".2f"
        print(f"     - Characteristics: {regime.characteristics}")
        if regime.optimal_architecture:
            arch = regime.optimal_architecture
            print(f"     - Optimal Architecture: {arch.n_trees} trees, depth {arch.max_depth}")
            print(f"     - Performance: {arch.overall_score:.4".4f"
        if regime.performance_metrics:
            perf = regime.performance_metrics
            print(f"     - Trading Metrics: Sharpe={perf.get('sharpe_ratio', 0):.3".3f"Rawdown={perf.get('max_drawdown', 0):.3".3f"

    # Summary metrics
    print("
5. Overall Trading Performance"    print("-" * 35)
    print(f"   Total Return: {result.total_return:.4".4f")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.4".4f")
    print(f"   Max Drawdown: {result.max_drawdown:.4".4f")
    print(f"   Win Rate: {result.win_rate:.4".4f")

    return result


def demonstrate_dynamic_model_selection():
    """Demonstrate dynamic model selection during trading."""
    print("\n" + "=" * 60)
    print("🎯 Dynamic Model Selection During Trading")
    print("=" * 60)

    # Create current market conditions
    print("\n1. Simulating current market conditions...")

    # Simulate different market scenarios
    scenarios = [
        ("High Volatility Crisis", MarketRegime.HIGH_VOLATILITY, 0.7),
        ("Strong Uptrend", MarketRegime.TRENDING_UP, 0.6),
        ("Sideways Market", MarketRegime.CONSOLIDATION, 0.8),
        ("Low Volatility", MarketRegime.LOW_VOLATILITY, 0.3)
    ]

    # Configure TAS for dynamic selection
    config = TradingTASConfig(
        adaptation_enabled=True,
        risk_adjusted_return_threshold=0.1,
        max_drawdown_threshold=0.15,
        enable_performance_tracking=True
    )

    tas = TradingTreeArchitectureSearch(config)

    # Generate training data for initial optimization
    market_data, target_returns = generate_synthetic_market_data(n_days=100, n_symbols=2)
    training_result = tas.optimize_for_trading_regimes(market_data, target_returns.iloc[:, 0])

    print("
2. Initial regime optimization completed"    print(f"   Optimized architectures for {len(training_result.regime_analysis)} regimes")

    # Test model selection for different scenarios
    print("
3. Testing dynamic model selection..."    for scenario_name, regime, risk_tolerance in scenarios:
        print(f"\n   📈 Scenario: {scenario_name}")
        print(f"      Regime: {regime.value}")
        print(f"      Risk Tolerance: {risk_tolerance}")

        # Create current market data for this scenario
        current_data = generate_current_market_data_for_regime(regime)

        # Select appropriate model
        selected_arch = tas.select_model_for_trading(
            current_data, regime, risk_tolerance
        )

        print(f"      Selected Architecture: {selected_arch.n_trees} trees, depth {selected_arch.max_depth}")
        print(f"      Splitting Strategy: {selected_arch.splitting_strategy}")
        print(f"      Overall Score: {selected_arch.overall_score:.4".4f"

        # Simulate trading performance
        performance = simulate_trading_with_architecture(
            selected_arch, current_data, regime
        )

        print(f"      Simulated Performance: Sharpe={performance['sharpe_ratio']:.3".3f"Rawdown={performance['max_drawdown']:.3".3f"

    return tas, training_result


def generate_current_market_data_for_regime(regime: MarketRegime) -> pd.DataFrame:
    """Generate current market data representing a specific regime."""
    np.random.seed(42)

    # Create 100 periods of current market data
    n_periods = 100

    if regime == MarketRegime.HIGH_VOLATILITY:
        # High volatility: large price swings
        price_changes = np.random.normal(0, 0.05, n_periods)  # 5% volatility
        base_price = 100
    elif regime == MarketRegime.TRENDING_UP:
        # Strong uptrend with moderate volatility
        trend = np.linspace(0, 0.03, n_periods)  # 3% upward trend
        noise = np.random.normal(0, 0.02, n_periods)
        price_changes = trend + noise
        base_price = 100
    elif regime == MarketRegime.CONSOLIDATION:
        # Sideways movement, low volatility
        price_changes = np.random.normal(0, 0.01, n_periods)  # 1% volatility
        base_price = 100
    elif regime == MarketRegime.LOW_VOLATILITY:
        # Very low volatility
        price_changes = np.random.normal(0, 0.005, n_periods)  # 0.5% volatility
        base_price = 100
    else:
        # Default: moderate volatility
        price_changes = np.random.normal(0, 0.02, n_periods)
        base_price = 100

    # Generate price series
    prices = [base_price]
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    return pd.DataFrame({'price': prices[1:]}, index=pd.date_range('2024-01-01', periods=n_periods, freq='1min'))


def simulate_trading_with_architecture(architecture, market_data: pd.DataFrame, regime: MarketRegime) -> Dict[str, float]:
    """Simulate trading performance with a given architecture."""
    try:
        # Create model from architecture
        from src.utils.ml_common.optimization.tree_architecture_search import TreeArchitectureSearch
        tas = TreeArchitectureSearch()

        # Generate synthetic targets based on regime
        if regime == MarketRegime.HIGH_VOLATILITY:
            # High volatility: more extreme returns
            target_returns = np.random.normal(0, 0.05, len(market_data))
        elif regime == MarketRegime.TRENDING_UP:
            # Upward trend
            target_returns = np.random.normal(0.02, 0.02, len(market_data))
        elif regime == MarketRegime.CONSOLIDATION:
            # Sideways: small returns
            target_returns = np.random.normal(0, 0.01, len(market_data))
        else:
            target_returns = np.random.normal(0, 0.02, len(market_data))

        # Create and train model
        model = tas._create_model_from_candidate(architecture, target_returns)
        X = market_data.values.reshape(-1, 1)
        y = target_returns
        model.fit(X, y)

        # Simulate predictions
        predictions = model.predict(X)

        # Calculate trading metrics
        returns = predictions * y  # Simplified trading simulation
        cumulative_returns = np.cumprod(1 + returns) - 1

        # Calculate metrics
        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_drawdown = 0

        peak = cumulative_returns[0] if len(cumulative_returns) > 0 else 0
        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = (peak - value) / (1 + peak) if peak != 0 else 0
            max_drawdown = max(max_drawdown, dd)

        win_rate = np.mean(returns > 0)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_return': np.mean(returns),
            'volatility': np.std(returns)
        }

    except Exception as e:
        print(f"Trading simulation failed: {e}")
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_trade_return': 0.0,
            'volatility': 0.0
        }


def demonstrate_real_time_adaptation():
    """Demonstrate real-time adaptation to changing market conditions."""
    print("\n" + "=" * 60)
    print("🔄 Real-Time Adaptation Demo")
    print("=" * 60)

    # Create TAS instance
    config = TradingTASConfig(
        adaptation_enabled=True,
        adaptation_interval_minutes=5,
        performance_decay_rate=0.9,
        enable_performance_tracking=True
    )

    tas = TradingTreeArchitectureSearch(config)

    # Simulate changing market conditions
    print("\n1. Simulating changing market conditions...")

    market_conditions = [
        (MarketRegime.NORMAL, "Stable market conditions"),
        (MarketRegime.HIGH_VOLATILITY, "High volatility period"),
        (MarketRegime.TRENDING_UP, "Strong upward trend"),
        (MarketRegime.CONSOLIDATION, "Sideways consolidation"),
        (MarketRegime.HIGH_VOLATILITY, "Return to volatility")
    ]

    previous_performance = {'sharpe_ratio': 1.2, 'max_drawdown': 0.05, 'win_rate': 0.65}

    for i, (regime, description) in enumerate(market_conditions):
        print(f"\n   Period {i+1}: {description}")
        print(f"   Market Regime: {regime.value}")

        # Create market data for this regime
        current_data = generate_current_market_data_for_regime(regime)

        # Simulate performance monitoring
        performance_monitor = MockPerformanceMonitor(previous_performance)

        # Adapt to new conditions
        adapted_architecture = tas.adapt_to_changing_conditions(
            MockMarketDataStream(current_data, regime),
            performance_monitor
        )

        if adapted_architecture:
            print(f"   Adapted Architecture: {adapted_architecture.n_trees} trees, depth {adapted_architecture.max_depth}")
            print(f"   Adaptation Method: {adapted_architecture.search_method}")

        # Update performance for next iteration
        # In real implementation, this would come from actual trading results
        previous_performance = {
            'sharpe_ratio': 1.0 + np.random.normal(0, 0.2),
            'max_drawdown': abs(np.random.normal(0.05, 0.02)),
            'win_rate': 0.5 + np.random.normal(0, 0.1)
        }

    print("
✅ Real-time adaptation demo completed"    print("   Demonstrated dynamic switching between:")
    print("   • Different tree architectures")
    print("   • Various market regimes")
    print("   • Risk-adjusted model selection")


class MockPerformanceMonitor:
    """Mock performance monitor for demonstration."""
    def __init__(self, performance_data):
        self.performance_data = performance_data

    def get_recent_performance(self):
        return self.performance_data


class MockMarketDataStream:
    """Mock market data stream for demonstration."""
    def __init__(self, data, regime):
        self.data = data
        self.regime = regime

    def get_current_data(self):
        return self.data

    def get_regime_data(self, regime):
        return self.data, np.random.randn(len(self.data))


def save_results_to_file(results):
    """Save demonstration results to file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'regime_exploration_results': {
            'total_regimes_detected': len(results.regime_analysis),
            'best_architecture': {
                'n_trees': results.best_architecture.n_trees,
                'max_depth': results.best_architecture.max_depth,
                'overall_score': results.best_architecture.overall_score
            },
            'trading_metrics': {
                'total_return': results.total_return,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'win_rate': results.win_rate
            }
        },
        'regime_details': {}
    }

    for regime_type, regime in results.regime_analysis.items():
        output['regime_details'][regime_type.value] = {
            'confidence': regime.confidence,
            'characteristics': regime.characteristics,
            'optimal_architecture': {
                'n_trees': regime.optimal_architecture.n_trees if regime.optimal_architecture else None,
                'max_depth': regime.optimal_architecture.max_depth if regime.optimal_architecture else None
            },
            'performance_metrics': regime.performance_metrics
        }

    # Save to file
    with open('/workspace/trading_tas_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("
📊 Results saved to: /workspace/trading_tas_results.json"    return output


if __name__ == "__main__":
    try:
        print("🚀 Trading TAS Demonstration")
        print("This demo showcases regime-aware tree architecture search for trading applications\n")

        # Demonstrate regime exploration
        regime_results = demonstrate_regime_exploration()

        # Demonstrate dynamic model selection
        tas_instance, training_results = demonstrate_dynamic_model_selection()

        # Demonstrate real-time adaptation
        demonstrate_real_time_adaptation()

        # Save comprehensive results
        save_results_to_file(regime_results)

        print("\n" + "=" * 60)
        print("✅ Trading TAS Demo Complete!")
        print("=" * 60)

        print("\n🎯 Key Benefits Demonstrated:")
        print("• Automated regime detection and characterization")
        print("• Optimal tree architecture selection for each regime")
        print("• Dynamic model adaptation during trading")
        print("• Risk-aware architecture selection")
        print("• Comprehensive trading performance evaluation")

        print("
📁 Files Generated:"        print("• /workspace/trading_tas_results.json - Detailed results")
        print("• Regime-specific model architectures")
        print("• Performance metrics and adaptation logs")

        print("
🔧 Usage in Production:"        print("1. Call optimize_trading_regimes() for initial setup")
        print("2. Use select_trading_model() for live trading")
        print("3. Enable adapt_to_changing_conditions() for real-time adaptation")
        print("4. Monitor performance and regime transitions")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()