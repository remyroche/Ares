#!/usr/bin/env python3
"""
Tree-Driven Advanced Statistics (TAS) Regime Detection Example

This example demonstrates the TAS regime detection system with full tool integration
and CLVSA architecture enhancement.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from tas_regime import TASRegimeConfig, TASRegimeDetector

def generate_sample_market_data(n_samples: int = 1000, n_features: int = 5) -> pd.DataFrame:
    """Generate sample market data for regime detection."""
    np.random.seed(42)

    # Generate base price series with different regimes
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15min')

    # Regime 1: Trending up (first 300 samples)
    regime1_prices = 100 + np.cumsum(np.random.randn(300) * 0.5)

    # Regime 2: Sideways/oscillating (samples 300-600)
    regime2_prices = regime1_prices[-1] + np.random.randn(300) * 2

    # Regime 3: Trending down (samples 600-800)
    regime3_prices = regime2_prices[-1] + np.cumsum(np.random.randn(200) * -0.3)

    # Regime 4: High volatility (last 200 samples)
    regime4_prices = regime3_prices[-1] + np.random.randn(200) * 3

    # Combine all regimes
    close_prices = np.concatenate([regime1_prices, regime2_prices, regime3_prices, regime4_prices])

    # Generate other OHLCV features
    open_prices = close_prices + np.random.randn(n_samples) * 0.1
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(n_samples) * 0.2)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(n_samples) * 0.2)
    volumes = np.random.lognormal(10, 1, n_samples)

    # Create DataFrame
    market_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })

    return market_data

def main():
    """Main example function."""
    print("🌲 TAS Regime Detection Example")
    print("=" * 50)

    try:
        # Generate sample market data
        print("📊 Generating sample market data...")
        market_data = generate_sample_market_data(1000, 5)
        timestamps = market_data['timestamp'].values

        print(f"✅ Generated {len(market_data)} samples with {len(market_data.columns)} features")

        # Create TAS configuration
        print("\n⚙️ Creating TAS configuration...")
        config = TASRegimeConfig.create_short_term_trading_config()
        config.n_regimes = 8  # Use 8 regimes for this example
        config.enable_patchtst_enhancement = True
        config.enable_statistical_methods = True
        config.enable_bootstrap_analysis = True
        config.enable_uncertainty_quantification = True

        print(f"✅ Configuration created: {config.n_regimes} regimes, {config.primary_timeframe} timeframe")

        # Initialize TAS detector
        print("\n🚀 Initializing TAS regime detector...")
        detector = TASRegimeDetector(config)

        print("✅ TAS detector initialized with full tool integration")

        # Detect regimes
        print("\n🎯 Detecting market regimes...")
        result = detector.detect_regimes(
            market_data=market_data[['open', 'high', 'low', 'close', 'volume']].values,
            timestamps=timestamps,
            optimize_performance=True,
            enable_patchtst_enhancement=True
        )

        # Analyze results
        print("\n📊 TAS Regime Detection Results:")
        print("=" * 40)
        print(f"Success: {result.success}")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        print(f"Regimes detected: {len(np.unique(result.regime_predictions))}")
        print(f"Regime distribution: {np.bincount(result.regime_predictions)}")

        # Performance metrics
        print("📈 Performance Metrics:")
        print(f"  Economic significance (mean): {np.mean(result.economic_significance_scores):.3f}")
        print(f"  Trading viability (mean): {np.mean(result.trading_viability_scores):.3f}")
        print(f"  Regime stability (mean): {np.mean(result.regime_stability_scores):.3f}")

        # Uncertainty analysis
        if result.uncertainty_estimates is not None:
            print(f"  Uncertainty (mean): {np.mean(result.uncertainty_estimates):.3f}")
            print(f"  Uncertainty (std): {np.std(result.uncertainty_estimates):.3f}")

        # Tool integration status
        print("🔧 Tool Integration Status:")
        print(f"  Hardware optimization: {'✅ Enabled' if result.metadata.get('tool_integration', {}).get('hardware', False) else '❌ Disabled'}")
        print(f"  Matrix operations: {'✅ Optimized' if result.metadata.get('tool_integration', {}).get('matrix_ops', False) else '❌ Disabled'}")
        print(f"  CLVSA enhancement: {'✅ Applied' if result.metadata.get('tool_integration', {}).get('clvsa', False) else '❌ Disabled'}")
        print(f"  Tree-based learning: {'✅ Active' if result.metadata.get('tool_integration', {}).get('tree', False) else '❌ Disabled'}")

        # Advanced analysis
        print("🎯 Advanced Analysis:")
        print(f"  Transition matrix shape: {result.transition_probabilities.shape}")

        # Regime transition analysis
        transition_matrix = result.transition_probabilities
        print(f"  Strongest transitions:")
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[1]):
                if transition_matrix[i, j] > 0.3:
                    print(f"    Regime {i} -> Regime {j}: {transition_matrix[i, j]:.2f}")

        # Save results
        print("💾 Saving results...")
        output_dir = Path("tas_regime_output")
        output_dir.mkdir(exist_ok=True)

        detector.save_results(result, output_dir / "tas_regime_results.pkl")
        print(f"✅ Results saved to {output_dir / 'tas_regime_results.pkl'}")

        # Create summary report
        summary_report = f"""
# TAS Regime Detection Summary Report

## Configuration
- Architecture: {config.primary_architecture.value}
- Regimes: {config.n_regimes}
- Timeframe: {config.primary_timeframe}
- Tree depth: {config.tree_depth}
- Estimators: {config.n_estimators}

## Results
- Success: {result.success}
- Execution time: {result.execution_time:.2f} seconds
- Regimes detected: {len(np.unique(result.regime_predictions))}

## Performance
- Economic significance: {np.mean(result.economic_significance_scores):.3f}
- Trading viability: {np.mean(result.trading_viability_scores):.3f}
- Regime stability: {np.mean(result.regime_stability_scores):.3f}

## Tool Integration
- Hardware optimization: {'Enabled' if result.metadata.get('tool_integration', {}).get('hardware', False) else 'Disabled'}
- Matrix operations: {'Optimized' if result.metadata.get('tool_integration', {}).get('matrix_ops', False) else 'Disabled'}
- CLVSA enhancement: {'Applied' if result.metadata.get('tool_integration', {}).get('clvsa', False) else 'Disabled'}
- Tree-based learning: {'Active' if result.metadata.get('tool_integration', {}).get('tree', False) else 'Disabled'}

## Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        with open(output_dir / "tas_regime_summary.md", "w") as f:
            f.write(summary_report)

        print(f"✅ Summary report saved to {output_dir / 'tas_regime_summary.md'}")

        print("🎉 TAS regime detection completed successfully!")
        print(f"   Check {output_dir} for detailed results and analysis.")

    except Exception as e:
        print(f"❌ TAS regime detection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())