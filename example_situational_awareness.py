#!/usr/bin/env python3
"""
Example: Situational Awareness Features

This script demonstrates the situational awareness capabilities of the Price Level Bank:
1. Getting closest 0.2% and 0.4% price levels above/below current price
2. Comprehensive situational awareness around any price point
3. Default situational awareness using latest available data
4. Integration with feature generation for immediate context

Run this to see how the system provides immediate trading context.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from typing import Dict, Any

from feature_generation.core.price_level_bank import PriceLevelBank, get_global_price_level_bank
from feature_generation.categories.support_resistance import (
    SituationalAwarenessGenerator,
    ClosestPriceLevelGenerator
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_1_closest_levels():
    """Example 1: Get closest price levels by percentage."""
    logger.info("="*70)
    logger.info("EXAMPLE 1: Closest Price Levels by Percentage")
    logger.info("="*70)

    # Initialize bank
    bank = get_global_price_level_bank()

    # Example current prices to test with
    test_prices = [50000, 52000, 48000, 50500]

    for current_price in test_prices:
        logger.info(f"\n🔍 SITUATIONAL AWARENESS AROUND ${current_price",.2f"}")
        logger.info("-" * 60)

        # Get closest levels by percentage
        closest_levels = bank.get_closest_levels_by_percentage(
            'BTCUSDT', '1h', current_price, [0.2, 0.4, 1.0]
        )

        print(f"📈 ABOVE current price (${current_price",.2f"}):")
        for level in closest_levels['above']:
            distance_pct = (level.price - current_price) / current_price * 100
            print(f"  + {level.level_pct".1f"}% level: ${level.price",.2f"} (+{distance_pct"+.2f"}%) - Sig: {level.significance_level".2f"}")

        print(f"📉 BELOW current price (${current_price",.2f"}):")
        for level in closest_levels['below']:
            distance_pct = (current_price - level.price) / current_price * 100
            print(f"  - {level.level_pct".1f"}% level: ${level.price",.2f"} (-{distance_pct"+.2f"}%) - Sig: {level.significance_level".2f"}")

def example_2_comprehensive_awareness():
    """Example 2: Comprehensive situational awareness."""
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Comprehensive Situational Awareness")
    logger.info("="*70)

    bank = get_global_price_level_bank()
    current_price = 50000  # Example price

    # Get comprehensive situational awareness
    awareness = bank.get_situational_awareness('BTCUSDT', '1h', current_price)

    logger.info(f"🎯 COMPREHENSIVE ANALYSIS FOR ${current_price",.2f"}")
    logger.info("=" * 60)

    # Current context
    print(f"Current Price: ${awareness['current_price']",.2f"}")

    # Price ranges
    print("
📏 PRICE RANGES:")
    for range_name, range_value in awareness['price_ranges'].items():
        print(f"  {range_name}: ±${range_value",.2f"}")

    # Distances to nearest levels
    print("
📊 DISTANCES TO NEAREST LEVELS:")
    distances = awareness['distances']
    for pct in [0.2, 0.4, 1.0, 2.0]:
        for direction in ['above', 'below']:
            if pct in distances[direction]:
                dist = distances[direction][pct]
                print(f"  {pct".1f"}% {direction}: ${dist['price']",.2f"} "
                      f"({dist['distance_pct']"+.2f"}%)")

    # Levels in ranges
    print("
🎚️ LEVELS WITHIN RANGES:")
    levels_in_ranges = awareness['levels_in_ranges']
    for range_name, levels in levels_in_ranges.items():
        if levels:
            avg_significance = sum(l.significance_level for l in levels) / len(levels)
            print(f"  {range_name}: {len(levels)} levels (avg sig: {avg_significance".2f"})")

    # Most significant nearby levels
    print("
🏆 MOST SIGNIFICANT NEARBY LEVELS:")
    for i, level in enumerate(awareness['significant_nearby'][:5], 1):
        distance_pct = abs(level.price - current_price) / current_price * 100
        direction = "above" if level.price > current_price else "below"
        print(f"  {i}. ${level.price",.2f"} ({distance_pct"+.2f"}% {direction}) - "
              f"Sig: {level.significance_level".2f"}")

def example_3_feature_generation():
    """Example 3: Situational awareness in feature generation."""
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Situational Awareness in Feature Generation")
    logger.info("="*70)

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=50, freq='1h')
    prices = 50000 + np.cumsum(np.random.normal(0, 200, 50))  # Random walk

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.exponential(1000, 50) + 100
    })
    df.set_index('timestamp', inplace=True)

    logger.info(f"Sample data: {len(df)} periods, price range ${df['low'].min()",.0f"} - ${df['high'].max()",.0f"}")

    # Initialize situational awareness generator
    awareness_gen = SituationalAwarenessGenerator()
    closest_02_gen = ClosestPriceLevelGenerator(0.2, 'both')
    closest_04_gen = ClosestPriceLevelGenerator(0.4, 'both')

    # Generate situational awareness features
    logger.info("Generating situational awareness features...")

    awareness_features = awareness_gen._generate_feature(
        df, symbol='BTCUSDT', timeframe='1h'
    )

    closest_02_features = closest_02_gen._generate_feature(
        df, symbol='BTCUSDT', timeframe='1h'
    )

    closest_04_features = closest_04_gen._generate_feature(
        df, symbol='BTCUSDT', timeframe='1h'
    )

    # Display results
    if awareness_features:
        logger.info("✅ Situational Awareness Features Generated:")
        for feature_name, feature_series in awareness_features.items():
            if feature_series is not None and len(feature_series.dropna()) > 0:
                latest_value = feature_series.dropna().iloc[-1]
                print(f"  • {feature_name}: {latest_value}")

    if closest_02_features:
        logger.info("✅ 0.2% Level Features:")
        for feature_name, feature_series in closest_02_features.items():
            if feature_series is not None and len(feature_series.dropna()) > 0:
                latest_value = feature_series.dropna().iloc[-1]
                print(f"  • {feature_name}: {latest_value}")

    if closest_04_features:
        logger.info("✅ 0.4% Level Features:")
        for feature_name, feature_series in closest_04_features.items():
            if feature_series is not None and len(feature_series.dropna()) > 0:
                latest_value = feature_series.dropna().iloc[-1]
                print(f"  • {feature_name}: {latest_value}")

def example_4_trading_context():
    """Example 4: Trading context and decision support."""
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: Trading Context and Decision Support")
    logger.info("="*70)

    bank = get_global_price_level_bank()
    current_price = 50000  # Example current price

    # Get situational awareness
    awareness = bank.get_situational_awareness('BTCUSDT', '1h', current_price)

    logger.info("💡 TRADING CONTEXT ANALYSIS")
    logger.info("=" * 50)

    # Risk assessment
    print("
🎯 RISK ASSESSMENT:")
    price_range_02 = awareness['price_ranges']['0.2%']
    price_range_04 = awareness['price_ranges']['0.4%']

    # Check for immediate levels
    levels_within_02pct = len(awareness['levels_in_ranges']['within_0.2%'])
    levels_within_04pct = len(awareness['levels_in_ranges']['within_0.4%'])

    print(f"  • Levels within 0.2% (±${price_range_02",.2f"}): {levels_within_02pct} levels")
    print(f"  • Levels within 0.4% (±${price_range_04",.2f"}): {levels_within_04pct} levels")

    if levels_within_02pct > 0:
        print("  ⚠️ HIGH RISK: Multiple levels very close to current price")
    elif levels_within_04pct > 2:
        print("  ⚡ MODERATE RISK: Several levels within striking distance")
    else:
        print("  ✅ LOW RISK: Clear space around current price")

    # Opportunity assessment
    print("
🎯 OPPORTUNITY ASSESSMENT:")

    # Nearest resistance (above)
    nearest_resistance = None
    nearest_resistance_dist = float('inf')
    for pct in [0.2, 0.4, 1.0]:
        if pct in awareness['distances']['above']:
            dist = awareness['distances']['above'][pct]['distance_pct']
            if dist < nearest_resistance_dist:
                nearest_resistance_dist = dist
                nearest_resistance = awareness['distances']['above'][pct]

    # Nearest support (below)
    nearest_support = None
    nearest_support_dist = float('inf')
    for pct in [0.2, 0.4, 1.0]:
        if pct in awareness['distances']['below']:
            dist = awareness['distances']['below'][pct]['distance_pct']
            if dist < nearest_support_dist:
                nearest_support_dist = dist
                nearest_support = awareness['distances']['below'][pct]

    if nearest_resistance:
        print(f"  📈 Nearest Resistance: ${nearest_resistance['price']",.2f"} "
              f"({nearest_resistance['distance_pct']"+.2f"}% above)")

    if nearest_support:
        print(f"  📉 Nearest Support: ${nearest_support['price']",.2f"} "
              f"({nearest_support['distance_pct']"+.2f"}% below)")

    # Trading recommendations
    print("
💡 TRADING RECOMMENDATIONS:")

    if nearest_resistance and nearest_resistance_dist < 0.1:  # Very close resistance
        print("  🎯 SCALPING OPPORTUNITY: Resistance very close above")
        print("  💰 Consider: Quick long position targeting resistance")

    if nearest_support and nearest_support_dist < 0.1:  # Very close support
        print("  🛡️ PROTECTION: Strong support very close below")
        print("  📊 Consider: Tight stop loss above support level")

    if levels_within_02pct == 0 and levels_within_04pct <= 2:
        print("  🚀 MOMENTUM PLAY: Clear space for price movement")
        print("  🎯 Consider: Momentum trading with wider targets")

    # Position sizing guidance
    print("
📏 POSITION SIZING GUIDANCE:")

    if nearest_support and nearest_resistance:
        risk_per_share = (nearest_resistance['price'] - nearest_support['price']) / 2
        print(f"  💵 Risk per share: ${risk_per_share",.2f"}")
        print(f"  📊 Suggested position size: Based on ${risk_per_share",.2f"} risk tolerance")

def create_sample_bank_data():
    """Create sample bank data for demonstration."""
    from feature_generation.core.price_level_bank import PriceLevelData
    import pandas as pd

    # Create sample price levels around $50,000
    base_prices = [
        49500, 49600, 49800, 49900, 50000, 50100, 50200, 50400, 50500,  # Around current price
        48000, 48500, 49000, 51000, 51500, 52000,  # Further away
        46000, 46500, 53000, 53500, 54000  # Much further
    ]

    levels = []
    for price in base_prices:
        for pct in [0.1, 0.2, 0.4, 1.0, 2.0]:
            # Add some variation to create multiple levels at similar prices
            actual_price = price + np.random.normal(0, price * 0.001)  # Small variation

            level = PriceLevelData(
                price=round(actual_price, 2),
                level_pct=round(pct, 1),
                symbol='BTCUSDT',
                timeframe='1h',
                timestamp=pd.Timestamp('2024-01-01') + pd.Timedelta(hours=np.random.randint(0, 24)),
                historical_crossings=np.random.randint(10, 100),
                historical_bounces=np.random.randint(5, 50),
                historical_volume=np.random.uniform(50000, 500000),
                historical_touch_density=np.random.uniform(0.2, 1.0),
                historical_time_decay=np.random.uniform(0.4, 0.9),
                historical_success_rate=np.random.uniform(0.5, 0.9),
                significance_level=np.random.uniform(0.6, 0.95),
                session_type=np.random.choice(['asian', 'european', 'us']),
                day_of_week=np.random.randint(0, 7),
                hour_of_day=np.random.randint(0, 24)
            )
            levels.append(level)

    # Add to bank
    bank = get_global_price_level_bank()
    level_ids = bank.add_levels(levels)
    logger.info(f"Added {len(levels)} sample levels to bank")
    return bank

def main():
    """Run all situational awareness examples."""
    logger.info("Starting Situational Awareness Examples")
    logger.info("This demonstrates immediate trading context around any price point")

    try:
        # Create sample data if bank is empty
        bank = get_global_price_level_bank()
        if bank.get_statistics()['total_levels'] == 0:
            logger.info("Creating sample data for demonstration...")
            create_sample_bank_data()

        # Run examples
        example_1_closest_levels()
        example_2_comprehensive_awareness()
        example_3_feature_generation()
        example_4_trading_context()

        logger.info("\n" + "="*70)
        logger.info("SITUATIONAL AWARENESS EXAMPLES COMPLETED")
        logger.info("="*70)

        print("\n🎯 KEY TAKEAWAYS:")
        print("• Immediate access to closest 0.2% and 0.4% price levels")
        print("• Comprehensive situational awareness around any price")
        print("• Risk assessment and trading recommendations")
        print("• Integrated into feature generation pipeline")
        print("• Available as default features in ML training")

        print("\n🔧 PRACTICAL USAGE:")
        print("1. Get situational awareness:")
        print("   python query_price_level_bank.py --default-awareness --symbol BTCUSDT")
        print("2. Check closest levels:")
        print("   python query_price_level_bank.py --closest-levels --current-price 50000")
        print("3. Full situational analysis:")
        print("   python query_price_level_bank.py --situational-awareness --current-price 50000")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()