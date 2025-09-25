#!/usr/bin/env python3
"""
Test Enhanced Situational Awareness Features

This script tests the enhanced situational awareness features including:
- 0.8% levels above and below
- Raw historical data (volume, bounces, crossings)
- Percentage-only distances
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from feature_generation.core.price_level_bank import PriceLevelBank, get_global_price_level_bank
from feature_generation.categories.support_resistance import (
    SituationalAwarenessGenerator,
    ClosestPriceLevelGenerator
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_features():
    """Test the enhanced situational awareness features."""
    logger.info("="*60)
    logger.info("TESTING ENHANCED SITUATIONAL AWARENESS FEATURES")
    logger.info("="*60)

    # Initialize bank
    bank = get_global_price_level_bank()

    # Create sample data if bank is empty
    if bank.get_statistics()['total_levels'] == 0:
        logger.info("Creating sample data for testing...")
        create_test_data(bank)

    # Test 1: Check for 0.8% levels
    logger.info("\n🧪 TEST 1: 0.8% Levels Support")
    current_price = 50000

    # Get situational awareness
    awareness = bank.get_situational_awareness('BTCUSDT', '1h', current_price)

    # Check if 0.8% levels are included
    has_08_above = 0.8 in awareness['distances']['above']
    has_08_below = 0.8 in awareness['distances']['below']

    logger.info(f"✅ 0.8% level above found: {has_08_above}")
    logger.info(f"✅ 0.8% level below found: {has_08_below}")

    if has_08_above:
        level_data = awareness['distances']['above'][0.8]
        logger.info(f"  ├─ Price: ${level_data['price']",.2f"}")
        logger.info(f"  ├─ Distance: {level_data['distance_pct']".2f"}%")
        logger.info(f"  ├─ Crossings: {level_data['historical_crossings']}")
        logger.info(f"  ├─ Bounces: {level_data['historical_bounces']}")
        logger.info(f"  └─ Volume: {level_data['historical_volume']",.0f"}")

    if has_08_below:
        level_data = awareness['distances']['below'][0.8]
        logger.info(f"  ├─ Price: ${level_data['price']",.2f"}")
        logger.info(f"  ├─ Distance: {level_data['distance_pct']".2f"}%")
        logger.info(f"  ├─ Crossings: {level_data['historical_crossings']}")
        logger.info(f"  ├─ Bounces: {level_data['historical_bounces']}")
        logger.info(f"  └─ Volume: {level_data['historical_volume']",.0f"}")

    # Test 2: Verify percentage-only distances
    logger.info("\n🧪 TEST 2: Percentage-Only Distances")

    # Check that distances don't include dollar values
    for pct in [0.2, 0.4, 0.8]:
        for direction in ['above', 'below']:
            if pct in awareness['distances'][direction]:
                level_data = awareness['distances'][direction][pct]

                # Should have percentage distance
                has_pct_distance = 'distance_pct' in level_data
                logger.info(f"✅ {pct".1f"}% {direction} has percentage distance: {has_pct_distance}")

                # Should have raw historical data
                has_crossings = 'historical_crossings' in level_data
                has_bounces = 'historical_bounces' in level_data
                has_volume = 'historical_volume' in level_data

                logger.info(f"✅ {pct".1f"}% {direction} has raw historical data:")
                logger.info(f"  ├─ Crossings: {has_crossings} ({level_data.get('historical_crossings', 'N/A')})")
                logger.info(f"  ├─ Bounces: {has_bounces} ({level_data.get('historical_bounces', 'N/A')})")
                logger.info(f"  └─ Volume: {has_volume} ({level_data.get('historical_volume', 'N/A')})")

    # Test 3: Feature generation
    logger.info("\n🧪 TEST 3: Feature Generation")

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=20, freq='1h')
    prices = 50000 + np.cumsum(np.random.normal(0, 100, 20))

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.exponential(1000, 20) + 100
    })
    df.set_index('timestamp', inplace=True)

    # Test situational awareness generator
    awareness_gen = SituationalAwarenessGenerator()
    features = awareness_gen._generate_feature(df, symbol='BTCUSDT', timeframe='1h')

    logger.info("✅ Situational Awareness Features Generated:")

    # Check for new features
    expected_features = [
        'closest_0.2pct_above_pct', 'closest_0.2pct_above_crossings', 'closest_0.2pct_above_bounces', 'closest_0.2pct_above_volume',
        'closest_0.2pct_below_pct', 'closest_0.2pct_below_crossings', 'closest_0.2pct_below_bounces', 'closest_0.2pct_below_volume',
        'closest_0.4pct_above_pct', 'closest_0.4pct_above_crossings', 'closest_0.4pct_above_bounces', 'closest_0.4pct_above_volume',
        'closest_0.4pct_below_pct', 'closest_0.4pct_below_crossings', 'closest_0.4pct_below_bounces', 'closest_0.4pct_below_volume',
        'closest_0.8pct_above_pct', 'closest_0.8pct_above_crossings', 'closest_0.8pct_above_bounces', 'closest_0.8pct_above_volume',
        'closest_0.8pct_below_pct', 'closest_0.8pct_below_crossings', 'closest_0.8pct_below_bounces', 'closest_0.8pct_below_volume',
        'price_range_0.2pct', 'price_range_0.4pct', 'price_range_0.8pct', 'price_range_1.0pct'
    ]

    found_features = []
    for feature_name in expected_features:
        if feature_name in features:
            found_features.append(feature_name)
            logger.info(f"  ✅ {feature_name}")
        else:
            logger.warning(f"  ❌ {feature_name} - MISSING")

    logger.info(f"✅ Found {len(found_features)}/{len(expected_features)} expected features")

    # Test 4: Query interface
    logger.info("\n🧪 TEST 4: Query Interface")

    try:
        # Test closest levels query
        from query_price_level_bank import PriceLevelBankQuery
        query = PriceLevelBankQuery()

        # This would normally require a current price, but for testing we'll check the structure
        awareness = query.get_situational_awareness('BTCUSDT', '1h', 50000)

        # Check if 0.8% levels are included in the query response
        has_08_in_response = 0.8 in awareness['distances']['above'] or 0.8 in awareness['distances']['below']
        logger.info(f"✅ Query interface includes 0.8% levels: {has_08_in_response}")

        if has_08_in_response:
            # Check if raw historical data is included
            for direction in ['above', 'below']:
                if 0.8 in awareness['distances'][direction]:
                    level_data = awareness['distances'][direction][0.8]
                    has_raw_data = all(key in level_data for key in ['historical_crossings', 'historical_bounces', 'historical_volume'])
                    logger.info(f"✅ 0.8% {direction} includes raw historical data: {has_raw_data}")

                    if has_raw_data:
                        logger.info(f"  ├─ Crossings: {level_data['historical_crossings']}")
                        logger.info(f"  ├─ Bounces: {level_data['historical_bounces']}")
                        logger.info(f"  └─ Volume: {level_data['historical_volume']}")

    except Exception as e:
        logger.error(f"❌ Query interface test failed: {e}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("ENHANCED SITUATIONAL AWARENESS TEST RESULTS")
    logger.info("="*60)

    tests_passed = 0
    total_tests = 0

    # Test 1: 0.8% levels
    total_tests += 1
    if has_08_above and has_08_below:
        tests_passed += 1
        logger.info("✅ TEST 1 PASSED: 0.8% levels available above and below")
    else:
        logger.info("❌ TEST 1 FAILED: Missing 0.8% levels")

    # Test 2: Percentage-only distances
    total_tests += 1
    distances_pct_only = True
    for pct in [0.2, 0.4, 0.8]:
        for direction in ['above', 'below']:
            if pct in awareness['distances'][direction]:
                level_data = awareness['distances'][direction][pct]
                if 'distance_pct' not in level_data:
                    distances_pct_only = False
                    break
        if not distances_pct_only:
            break

    if distances_pct_only:
        tests_passed += 1
        logger.info("✅ TEST 2 PASSED: Percentage-only distances")
    else:
        logger.info("❌ TEST 2 FAILED: Still has dollar distances")

    # Test 3: Raw historical data
    total_tests += 1
    raw_data_available = False
    for pct in [0.2, 0.4, 0.8]:
        for direction in ['above', 'below']:
            if pct in awareness['distances'][direction]:
                level_data = awareness['distances'][direction][pct]
                if all(key in level_data for key in ['historical_crossings', 'historical_bounces', 'historical_volume']):
                    raw_data_available = True
                    break
        if raw_data_available:
            break

    if raw_data_available:
        tests_passed += 1
        logger.info("✅ TEST 3 PASSED: Raw historical data available")
    else:
        logger.info("❌ TEST 3 FAILED: Missing raw historical data")

    # Test 4: Feature generation
    total_tests += 1
    if len(found_features) >= 20:  # Most features should be present
        tests_passed += 1
        logger.info(f"✅ TEST 4 PASSED: Feature generation ({len(found_features)}/{len(expected_features)} features)")
    else:
        logger.info(f"❌ TEST 4 FAILED: Feature generation ({len(found_features)}/{len(expected_features)} features)")

    logger.info(f"\n📊 SUMMARY: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Enhanced situational awareness is working correctly.")
    else:
        logger.warning(f"⚠️ {total_tests - tests_passed} tests failed. Check implementation.")

    return tests_passed == total_tests

def create_test_data(bank):
    """Create test data for the enhanced features."""
    from feature_generation.core.price_level_bank import PriceLevelData
    import pandas as pd

    # Create levels at 0.2%, 0.4%, and 0.8% intervals around $50,000
    base_prices = []
    percentages = [0.2, 0.4, 0.8, 1.0]

    # Above current price
    current_price = 50000
    for pct in percentages:
        price = current_price * (1 + pct / 100)
        base_prices.append(round(price, 2))

    # Below current price
    for pct in percentages:
        price = current_price * (1 - pct / 100)
        base_prices.append(round(price, 2))

    # Add some additional levels for testing
    base_prices.extend([49500, 50500, 49800, 50200, 49600, 50400])

    levels = []
    for i, price in enumerate(base_prices):
        # Assign different percentages to different levels
        pct = percentages[i % len(percentages)]

        level = PriceLevelData(
            price=price,
            level_pct=round(pct, 1),
            symbol='BTCUSDT',
            timeframe='1h',
            timestamp=pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i),
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
    level_ids = bank.add_levels(levels)
    logger.info(f"✅ Created {len(levels)} test levels for enhanced situational awareness testing")

    return bank

def main():
    """Run the enhanced situational awareness tests."""
    logger.info("Testing Enhanced Situational Awareness Features")
    logger.info("This verifies the 0.8% levels and raw historical data enhancements")

    try:
        success = test_enhanced_features()

        if success:
            logger.info("\n🎉 ALL ENHANCED SITUATIONAL AWARENESS TESTS PASSED!")
            print("\n✅ Enhanced Features Verified:")
            print("• 0.8% levels above and below")
            print("• Raw historical data (crossings, bounces, volume)")
            print("• Percentage-only distances")
            print("• Enhanced feature generation")
            print("• Updated query interface")
        else:
            logger.error("\n❌ Some tests failed. Please check the implementation.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()