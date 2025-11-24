"""
Example: Programmatic Alignment of Triple Barrier with HPO Results

This example demonstrates how to use the create_triple_barrier_from_hpo() function
to automatically align triple barrier labeling parameters with the best settings found
in meta_labeling_hpo_experiment.

The function:
1. Loads the latest HPO results from outcomes/ directory
2. Extracts profit/stop thresholds, horizon, and other parameters
3. Creates a properly configured OptimizedTripleBarrierLabeling instance
4. Falls back to sensible defaults if HPO results are not found

Usage:
    python examples/triple_barrier_hpo_alignment_example.py
"""

import pandas as pd
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    create_triple_barrier_from_hpo
)

def main():
    # Example 1: Create triple barrier labeler aligned with HPO for ETHUSDT 15m
    print("=" * 80)
    print("Example 1: Creating Triple Barrier Labeler from HPO Results")
    print("=" * 80)

    symbol = 'ETHUSDT'
    timeframe = '15m'

    # Create labeler - it will automatically find and use HPO results
    labeler, hpo_params, used_hpo = create_triple_barrier_from_hpo(
        symbol=symbol,
        timeframe=timeframe,
        binary_classification=True,      # Use binary classification (BUY/SELL only)
        transaction_cost=0.0008,          # 0.08% transaction cost
    )

    print(f"\nLabeler created:")
    print(f"  - Used HPO params: {used_hpo}")
    print(f"  - Profit take: {labeler.profit_take_multiplier:.4f} ({labeler.profit_take_multiplier*100:.2f}%)")
    print(f"  - Stop loss: {labeler.stop_loss_multiplier:.4f} ({labeler.stop_loss_multiplier*100:.2f}%)")
    print(f"  - Time barrier: {labeler.time_barrier_minutes} minutes")
    print(f"  - Max lookahead: {labeler.max_lookahead} bars")
    print(f"  - Transaction cost: {labeler.transaction_cost:.4f} ({labeler.transaction_cost*100:.2f}%)")

    if used_hpo:
        print(f"\n✅ Successfully aligned with HPO results!")
        print(f"\nHPO Parameters used:")
        for key, value in hpo_params.items():
            if key in ['profit_thr_base', 'stop_to_profit_ratio', 'horizon_bars',
                      'min_event_spacing', 'kalman_Q', 'kalman_R']:
                print(f"  - {key}: {value}")
    else:
        print(f"\nℹ️  No HPO results found - using fallback parameters")

    # Example 2: Using the labeler with market data
    print("\n" + "=" * 80)
    print("Example 2: Using the Labeler with Market Data")
    print("=" * 80)

    # Simulate some market data (in practice, load from your data source)
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
    market_data = pd.DataFrame({
        'open': 2000 + np.random.randn(1000).cumsum(),
        'high': 2000 + np.random.randn(1000).cumsum() + 5,
        'low': 2000 + np.random.randn(1000).cumsum() - 5,
        'close': 2000 + np.random.randn(1000).cumsum(),
        'volume': np.random.rand(1000) * 1000000,
    }, index=dates)

    print(f"\nMarket data shape: {market_data.shape}")
    print(f"Date range: {market_data.index[0]} to {market_data.index[-1]}")

    # Generate labels using the HPO-aligned labeler
    try:
        labels, profits, metadata = labeler.generate_labels(market_data)

        print(f"\n✅ Labels generated successfully!")
        print(f"  - Total labels: {len(labels)}")
        print(f"  - Buy signals: {(labels == 1).sum()}")
        print(f"  - Sell signals: {(labels == -1).sum()}")
        if not labeler.binary_classification:
            print(f"  - Hold signals: {(labels == 0).sum()}")

        print(f"\nProfit statistics:")
        print(f"  - Mean profit: {profits.mean():.4f}")
        print(f"  - Std profit: {profits.std():.4f}")
        print(f"  - Win rate: {(profits > 0).sum() / len(profits):.2%}")

    except Exception as e:
        print(f"\n⚠️  Note: Label generation requires proper OHLCV data structure")
        print(f"   Error: {e}")

    # Example 3: Custom fallback parameters
    print("\n" + "=" * 80)
    print("Example 3: Using Custom Fallback Parameters")
    print("=" * 80)

    # Create labeler with custom fallback parameters (used if HPO not found)
    labeler_custom, _, used_hpo = create_triple_barrier_from_hpo(
        symbol='NONEXISTENT',  # Intentionally use non-existent symbol
        timeframe='1h',
        fallback_profit_take=0.006,   # 0.6% profit target
        fallback_stop_loss=0.004,     # 0.4% stop loss
        fallback_time_barrier=60,     # 60 minutes
        fallback_max_lookahead=80,    # 80 bars
        binary_classification=True,
        transaction_cost=0.001,       # 0.1% transaction cost
    )

    print(f"\nCustom labeler created:")
    print(f"  - Used HPO params: {used_hpo}")
    print(f"  - Profit take: {labeler_custom.profit_take_multiplier:.4f}")
    print(f"  - Stop loss: {labeler_custom.stop_loss_multiplier:.4f}")
    print(f"  - Time barrier: {labeler_custom.time_barrier_minutes} minutes")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
The create_triple_barrier_from_hpo() function provides:
✓ Automatic alignment with meta_labeling_hpo_experiment results
✓ Seamless fallback to sensible defaults if HPO not found
✓ Proper parameter conversion (horizon_bars → time_barrier, etc.)
✓ Validation of financial parameters
✓ Clear logging of parameter sources

Use this function whenever you need to create a triple barrier labeler
to ensure consistency with optimized parameters discovered through HPO.
""")

if __name__ == '__main__':
    main()
