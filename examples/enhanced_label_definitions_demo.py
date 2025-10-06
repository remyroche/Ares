#!/usr/bin/env python3
"""
Enhanced Label Definitions Demonstration

This script demonstrates the enhanced label definitions for trading ML:
1. Analyst labels: "Should we trade?" based on expected PnL > fees + slippage
2. Tactician labels: Direction/magnitude based on max favorable/adverse excursion
3. Regime conditioning: Volatility-scaled thresholds
4. Risk awareness: Label 0 if trade would hit stop before target
5. Data cleaning: Remove outliers, align timestamps, de-duplicate
6. Stability checks: Recompute labels, track leakage, check OOS balance

Usage:
    python examples/enhanced_label_definitions_demo.py
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.steps.pre_training.profit_labeling.enhanced_label_definitions import (
    EnhancedLabelDefinitions,
    AnalystLabelConfig,
    TacticianLabelConfig,
    RegimeConditionedConfig,
    RiskAwareConfig,
    DataCleaningConfig,
    StabilityCheckConfig,
    TradingCosts,
    create_trading_aware_config
)

from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
    create_enhanced_analyst_labeler,
    create_enhanced_tactician_labeler
)

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


def generate_sample_market_data(n_bars: int = 1000) -> pd.DataFrame:
    """Generate sample market data for demonstration."""
    tprint_info(f"📊 Generating {n_bars} bars of sample market data")

    # Create timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=15 * n_bars)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='15min')

    # Generate OHLCV data with realistic patterns
    np.random.seed(42)  # For reproducible results

    # Start with base price
    base_price = 100.0

    # Generate price movements
    returns = np.random.normal(0, 0.02, n_bars)  # 2% volatility

    # Add some trend and cycles
    trend = np.linspace(0, 0.001, n_bars)  # Slight upward trend
    cycle = 0.005 * np.sin(2 * np.pi * np.arange(n_bars) / 100)  # Cycle every 100 bars

    total_returns = returns + trend + cycle
    prices = base_price * np.exp(np.cumsum(total_returns))

    # Create OHLC from close prices
    closes = prices
    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    # Add some spread to high/low
    spreads = np.random.uniform(0.001, 0.005, n_bars)  # 0.1% to 0.5% spread
    highs = closes * (1 + spreads)
    lows = closes * (1 - spreads)

    # Generate volume (higher when price moves more)
    volumes = np.random.uniform(1000, 10000, n_bars) * (1 + abs(total_returns) * 10)

    # Create DataFrame
    market_data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=timestamps)

    tprint_success(f"✅ Generated sample market data: {len(market_data)} bars")
    return market_data


def generate_sample_volatility_data(market_data: pd.DataFrame) -> pd.Series:
    """Generate sample volatility data for demonstration."""
    tprint_info("📈 Generating sample volatility data")

    # Simple volatility estimate based on returns
    returns = market_data['close'].pct_change()
    volatility = returns.rolling(20).std() * np.sqrt(252)  # Annualized

    # Add some regime changes
    regime_changes = np.random.choice(['low', 'normal', 'high'], size=len(volatility),
                                     p=[0.2, 0.6, 0.2])

    # Apply regime-specific volatility levels
    volatility_values = volatility.values.copy()
    for i, regime in enumerate(regime_changes):
        if regime == 'low':
            volatility_values[i] *= 0.5
        elif regime == 'high':
            volatility_values[i] *= 2.0

    volatility_series = pd.Series(volatility_values, index=market_data.index, name='volatility')

    tprint_success(f"✅ Generated volatility data with {len(volatility_series)} points")
    return volatility_series


def generate_sample_regime_data(volatility_series: pd.Series) -> pd.Series:
    """Generate sample regime data for demonstration."""
    tprint_info("🎭 Generating sample regime data")

    # Create regimes based on volatility levels
    low_threshold = volatility_series.quantile(0.33)
    high_threshold = volatility_series.quantile(0.67)

    regimes = pd.Series('normal', index=volatility_series.index)

    regimes[volatility_series <= low_threshold] = 'low_vol'
    regimes[volatility_series >= high_threshold] = 'high_vol'

    tprint_success(f"✅ Generated regime data: {regimes.value_counts().to_dict()}")
    return regimes


async def demonstrate_analyst_labels():
    """Demonstrate Analyst labels (Should we trade?)."""
    tprint("\n🎯 DEMONSTRATING ANALYST LABELS")
    tprint("=" * 50)

    # Generate sample data
    market_data = generate_sample_market_data(500)
    volatility_series = generate_sample_volatility_data(market_data)
    regime_data = generate_sample_regime_data(volatility_series)

    # Create enhanced label definitions
    labeler = EnhancedLabelDefinitions(
        analyst_config=AnalystLabelConfig(
            horizon_minutes=60,
            min_profit_threshold_usd=5.0,
            trading_costs=TradingCosts(
                maker_fee=0.001,
                taker_fee=0.002,
                slippage_pct=0.001
            ),
            enable_regime_conditioning=True,
            volatility_scaling_factor=1.0
        )
    )

    # Generate analyst labels
    analyst_labels, confidence_scores = labeler.generate_analyst_labels(
        market_data, volatility_series, regime_data
    )

    # Display results
    tprint_success("📊 Analyst Labels Results:")
    tprint_info(f"   → Total samples: {len(analyst_labels)}")
    tprint_info(f"   → Positive trades: {analyst_labels.sum()}")
    tprint_info(f"   → Positive ratio: {analyst_labels.mean():.3f}")
    tprint_info(f"   → Average confidence: {confidence_scores.mean():.3f}")

    # Show some examples
    tprint("\n📋 Sample Results (first 10):")
    sample_results = pd.DataFrame({
        'close': market_data['close'].head(10),
        'analyst_label': analyst_labels.head(10),
        'confidence': confidence_scores.head(10)
    })
    tprint(str(sample_results))

    # Check stability
    stability_results = labeler.check_label_stability(analyst_labels)
    tprint("
🔍 Stability Check:"    tprint(f"   → Is stable: {stability_results['is_stable']}")
    if stability_results['issues']:
        for issue in stability_results['issues']:
            tprint(f"   → Issue: {issue}")

    return analyst_labels, confidence_scores


async def demonstrate_tactician_labels():
    """Demonstrate Tactician labels (Direction/Magnitude)."""
    tprint("\n⚔️ DEMONSTRATING TACTICIAN LABELS")
    tprint("=" * 50)

    # Generate sample data
    market_data = generate_sample_market_data(500)
    volatility_series = generate_sample_volatility_data(market_data)
    regime_data = generate_sample_regime_data(volatility_series)

    # Create enhanced label definitions
    labeler = EnhancedLabelDefinitions(
        tactician_config=TacticianLabelConfig(
            favorable_excursion_threshold=1.0,
            adverse_excursion_threshold=-2.0,
            horizon_minutes=30,
            enable_regime_conditioning=True,
            volatility_sensitivity=1.0
        )
    )

    # Generate tactician labels
    tactician_labels, magnitude_scores = labeler.generate_tactician_labels(
        market_data, volatility_series, regime_data
    )

    # Display results
    tprint_success("📊 Tactician Labels Results:")
    tprint_info(f"   → Total samples: {len(tactician_labels)}")
    tprint_info(f"   → Valid directions: {tactician_labels.sum()}")
    tprint_info(f"   → Valid ratio: {tactician_labels.mean():.3f}")
    tprint_info(f"   → Average magnitude: {magnitude_scores.mean():.3f}")

    # Show some examples
    tprint("\n📋 Sample Results (first 10):")
    sample_results = pd.DataFrame({
        'close': market_data['close'].head(10),
        'tactician_label': tactician_labels.head(10),
        'magnitude': magnitude_scores.head(10)
    })
    tprint(str(sample_results))

    # Check stability
    stability_results = labeler.check_label_stability(tactician_labels)
    tprint("
🔍 Stability Check:"    tprint(f"   → Is stable: {stability_results['is_stable']}")
    if stability_results['issues']:
        for issue in stability_results['issues']:
            tprint(f"   → Issue: {issue}")

    return tactician_labels, magnitude_scores


async def demonstrate_enhanced_volatility_labeler():
    """Demonstrate the enhanced volatility-aware labeler."""
    tprint("\n🚀 DEMONSTRATING ENHANCED VOLATILITY-AWARE LABELER")
    tprint("=" * 60)

    # Generate sample data
    market_data = generate_sample_market_data(300)

    # Test Analyst labeler
    tprint("\n🎯 Testing Enhanced Analyst Labeler")
    analyst_labeler = create_enhanced_analyst_labeler()

    try:
        analyst_result = analyst_labeler.generate_labels(market_data)
        tprint_success("✅ Analyst labeler completed successfully")
        tprint_info(f"   → Samples: {analyst_result.n_samples}")
        tprint_info(f"   → Targets: {analyst_result.n_targets}")
        tprint_info(f"   → Processing time: {analyst_result.processing_time:.2f}s")

    except Exception as e:
        tprint_error(f"❌ Analyst labeler failed: {e}")

    # Test Tactician labeler
    tprint("\n⚔️ Testing Enhanced Tactician Labeler")
    tactician_labeler = create_enhanced_tactician_labeler()

    try:
        tactician_result = tactician_labeler.generate_labels(market_data)
        tprint_success("✅ Tactician labeler completed successfully")
        tprint_info(f"   → Samples: {tactician_result.n_samples}")
        tprint_info(f"   → Targets: {tactician_result.n_targets}")
        tprint_info(f"   → Processing time: {tactician_result.processing_time:.2f}s")

    except Exception as e:
        tprint_error(f"❌ Tactician labeler failed: {e}")


async def demonstrate_data_cleaning():
    """Demonstrate data cleaning capabilities."""
    tprint("\n🧹 DEMONSTRATING DATA CLEANING")
    tprint("=" * 40)

    # Generate sample data with some issues
    market_data = generate_sample_market_data(200)

    # Add some outliers and issues
    outlier_indices = np.random.choice(market_data.index, size=10, replace=False)

    # Add price outliers
    for idx in outlier_indices[:5]:
        market_data.loc[idx, 'close'] *= 2  # Double price
        market_data.loc[idx, 'high'] = market_data.loc[idx, 'close'] * 1.1
        market_data.loc[idx, 'low'] = market_data.loc[idx, 'close'] * 0.9

    # Add volume outliers
    for idx in outlier_indices[5:]:
        market_data.loc[idx, 'volume'] *= 10  # 10x volume

    # Add some missing data
    missing_indices = np.random.choice(market_data.index, size=5, replace=False)
    market_data.loc[missing_indices, 'volume'] = np.nan

    tprint_info(f"📊 Sample data created with {len(outlier_indices)} outliers and {len(missing_indices)} missing values")

    # Create labeler with strict cleaning config
    labeler = EnhancedLabelDefinitions(
        cleaning_config=DataCleaningConfig(
            outlier_method="iqr",
            outlier_threshold=2.0,  # Stricter threshold
            min_volume_threshold=1000.0,
            enforce_timestamp_alignment=True
        )
    )

    # Apply data cleaning
    cleaned_data = labeler._apply_data_cleaning(market_data)

    tprint_success("✅ Data cleaning completed:")
    tprint_info(f"   → Original bars: {len(market_data)}")
    tprint_info(f"   → Cleaned bars: {len(cleaned_data)}")
    tprint_info(f"   → Removed bars: {len(market_data) - len(cleaned_data)}")

    # Check for remaining issues
    remaining_outliers = len(cleaned_data[cleaned_data['close'] > market_data['close'].quantile(0.95)])
    remaining_missing = cleaned_data.isnull().sum().sum()

    tprint_info(f"   → Remaining outliers: {remaining_outliers}")
    tprint_info(f"   → Remaining missing values: {remaining_missing}")


async def main():
    """Main demonstration function."""
    tprint("🚀 ENHANCED LABEL DEFINITIONS DEMONSTRATION")
    tprint("=" * 60)
    tprint("This demo showcases the enhanced label definitions for trading ML:")
    tprint("1. Analyst labels: 'Should we trade?' based on expected PnL > costs")
    tprint("2. Tactician labels: Direction/magnitude based on excursion thresholds")
    tprint("3. Regime conditioning: Volatility-scaled thresholds")
    tprint("4. Risk awareness: Stop-loss protection")
    tprint("5. Data cleaning: Outlier removal, timestamp alignment")
    tprint("6. Stability checks: Leakage detection, balance monitoring")
    tprint("=" * 60)

    try:
        # Demonstrate analyst labels
        await demonstrate_analyst_labels()

        # Demonstrate tactician labels
        await demonstrate_tactician_labels()

        # Demonstrate enhanced volatility labeler
        await demonstrate_enhanced_volatility_labeler()

        # Demonstrate data cleaning
        await demonstrate_data_cleaning()

        tprint("\n" + "=" * 60)
        tprint("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        tprint("=" * 60)
        tprint("Key benefits of enhanced label definitions:")
        tprint("✅ Trading-aware: Labels reflect actual trading decisions")
        tprint("✅ Risk-conscious: Accounts for stop-losses and portfolio risk")
        tprint("✅ Regime-adaptive: Adjusts to different market conditions")
        tprint("✅ Quality-focused: Built-in stability and leakage checks")
        tprint("✅ Production-ready: Robust data cleaning and validation")

    except Exception as e:
        tprint_error(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())