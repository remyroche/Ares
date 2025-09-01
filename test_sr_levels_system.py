#!/usr/bin/env python3
"""
Test Script for Comprehensive SR Levels System

This script demonstrates:
1. SR level calculation from backtesting data
2. Continuous updates during live trading
3. Trading intelligence access to SR levels
4. Price vs VWAP comparison
5. Comprehensive level information (age, strength, volume, etc.)
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.tactician.sr_levels_manager import create_sr_levels_manager, SRLevel
from src.trading.sr_trading_intelligence import create_sr_trading_intelligence


import async def test_sr_levels_system
async def test_sr_levels_system():
    """Test the comprehensive SR levels system."""
    print("🚀 Testing Comprehensive SR Levels System")
    print("=" * 60)

    # Configuration
    config = {
        "sr_levels_manager": {
            "storage_path": "data/sr_levels",
            "max_levels": 50,
            "min_strength": 0.3,
            "proximity_threshold": 0.005
        },
        "sr_trading_intelligence": {
            "enable_real_time_updates": True,
            "update_interval_seconds": 30,
            "max_position_size": 0.1,
            "risk_tolerance": 0.02
        },
        "sr_breakout_predictor": {
            "enable_detailed_reporting": True,
            "report_directory": "reports/sr_analysis",
            "max_sr_levels": 20,
            "min_sr_strength": 0.3
        }
    }

    try:
        # Test 1: Initialize SR Levels Manager
    except Exception as e:
        pass
    except Exception as e:
        pass
        print("\\\n📋 Test 1: Initializing SR Levels Manager")
        print("-" * 40)

        sr_manager = await create_sr_levels_manager(config)
        if not sr_manager:
    pass
    pass
            print("❌ Failed to initialize SR Levels Manager")
            return False

        print("✅ SR Levels Manager initialized successfully")

        # Test 2: Generate sample market data
        print("\\\n📊 Test 2: Generating Sample Market Data")
        print("-" * 40)

        sample_data = generate_sample_market_data()
        print(f"✅ Generated sample data: {len(sample_data)} rows")
        print(f"   Price range: {sample_data['close'].min():.4f} - {sample_data['close'].max():.4f}")
        print(f"   Volume range: {sample_data['volume'].min():.0f} - {sample_data['volume'].max():.0f}")

        # Test 3: Calculate SR levels from backtesting data
        print("\\\n🔍 Test 3: Calculating SR Levels from Backtesting Data")
        print("-" * 40)

        sr_levels_result = await sr_manager.calculate_sr_levels_from_backtest(sample_data, "1m")

        support_count = len(sr_levels_result.get("support_levels", []))
        resistance_count = len(sr_levels_result.get("resistance_levels", []))

        print(f"✅ Calculated SR levels:")
        print(f"   Support levels: {support_count}")
        print(f"   Resistance levels: {resistance_count}")
        print(f"   Total levels: {support_count + resistance_count}")

        # Test 3.5: Test individual detection methods
        print("\\\n🔍 Test 3.5: Testing Individual Detection Methods")
        print("-" * 40)

        detection_methods = ["fractal", "volume", "pivot", "atr"]
        method_results = {}

        for method in detection_methods:
    pass
    pass
            try:
                method_result = await sr_manager.calculate_sr_levels_with_method(
                    sample_data, method, "both"
    except Exception as e:
        pass
    except Exception as e:
        pass
                )
                method_results[method] = method_result

                support_count = len(method_result.get("support_levels", []))
                resistance_count = len(method_result.get("resistance_levels", []))

                print(f"   {method.upper()} method:")
                print(f"     Support levels: {support_count}")
                print(f"     Resistance levels: {resistance_count}")
                print(f"     Total: {support_count + resistance_count}")

            except Exception as e:
                print(f"   {method.upper()} method: ❌ Failed - {e}")

        # Show method comparison
        print(f"\\\n   Method Comparison:")
        for method, result in method_results.items():
    pass
    pass
            total_levels = len(result.get("support_levels", [])) + len(result.get("resistance_levels", []))
            print(f"     {method}: {total_levels} levels")

        # Display sample levels
        if support_count > 0:
    pass
    pass
            sample_support = sr_levels_result["support_levels"][0]
            print(f"\\\n   Sample Support Level:")
            print(f"     Price: {sample_support.price:.4f}")
            print(f"     Strength: {sample_support.strength:.3f}")
            print(f"     Method: {sample_support.method}")
            print(f"     Data Source: {sample_support.data_source}")
            print(f"     Quality Score: {sample_support.calculate_quality_score():.3f}")

        if resistance_count > 0:
    pass
    pass
            sample_resistance = sr_levels_result["resistance_levels"][0]
            print(f"\\\n   Sample Resistance Level:")
            print(f"     Price: {sample_resistance.price:.4f}")
            print(f"     Strength: {sample_resistance.strength:.3f}")
            print(f"     Method: {sample_resistance.method}")
            print(f"     Data Source: {sample_resistance.data_source}")
            print(f"     Quality Score: {sample_resistance.calculate_quality_score():.3f}")

        # Test 4: Test continuous updates with live data
        print("\\\n🔄 Test 4: Testing Continuous Updates with Live Data")
        print("-" * 40)

        # Simulate live price updates
        current_price = sample_data['close'].iloc[-1]
        current_volume = sample_data['volume'].iloc[-1]
        current_time = datetime.now()

        print(f"   Current price: {current_price:.4f}")
        print(f"   Current volume: {current_volume:.0f}")

        # Update levels with live data
        update_result = await sr_manager.update_levels_with_live_data(
            current_price, current_volume, current_time
        )

        print(f"✅ Updated SR levels:")
        print(f"   Support touches: {update_result.get('support_touches', 0)}")
        print(f"   Resistance touches: {update_result.get('resistance_touches', 0)}")
        print(f"   New levels created: {update_result.get('new_levels_created', 0)}")
        print(f"   Levels removed: {update_result.get('levels_removed', 0)}")

        # Test 5: Get SR levels for trading intelligence
        print("\\\n🧠 Test 5: Getting SR Levels for Trading Intelligence")
        print("-" * 40)

        trading_levels = sr_manager.get_sr_levels_for_trading(current_price, include_metadata=True)

        print(f"✅ Trading intelligence data:")
        print(f"   Current price: {trading_levels.get('current_price', 0):.4f}")

        nearest_support = trading_levels.get("nearest_support")
        if nearest_support:
    pass
    pass
            print(f"   Nearest support: {nearest_support['price']:.4f}")
            print(f"     Strength: {nearest_support['strength']:.3f}")
            print(f"     Proximity: {nearest_support['proximity']:.4f}")
            print(f"     Touch count: {nearest_support['touch_count']}")
            print(f"     Age (hours): {nearest_support['age_hours']:.1f}")
            print(f"     Quality score: {nearest_support['quality_score']:.3f}")

        nearest_resistance = trading_levels.get("nearest_resistance")
        if nearest_resistance:
    pass
    pass
            print(f"   Nearest resistance: {nearest_resistance['price']:.4f}")
            print(f"     Strength: {nearest_resistance['strength']:.3f}")
            print(f"     Proximity: {nearest_resistance['proximity']:.4f}")
            print(f"     Touch count: {nearest_resistance['touch_count']}")
            print(f"     Age (hours): {nearest_resistance['age_hours']:.1f}")
            print(f"     Quality score: {nearest_resistance['quality_score']:.3f}")

        # Test 6: Test Trading Intelligence System
        print("\\\n🎯 Test 6: Testing Trading Intelligence System")
        print("-" * 40)

        trading_intelligence = await create_sr_trading_intelligence(config)
        if not trading_intelligence:
    pass
    pass
            print("❌ Failed to initialize Trading Intelligence")
            return False

        print("✅ Trading Intelligence initialized successfully")

        # Get comprehensive trading data
        comprehensive_data = trading_intelligence.get_sr_levels_for_trading(current_price)

        print(f"✅ Comprehensive trading data:")

        # Trading intelligence
        ti = comprehensive_data.get("trading_intelligence", {})
        print(f"   Market position: {ti.get('market_position', 'unknown')}")
        print(f"   Trend direction: {ti.get('trend_direction', 'unknown')}")
        print(f"   Volatility assessment: {ti.get('volatility_assessment', 'unknown')}")
        print(f"   Risk level: {ti.get('risk_level', 'unknown')}")

        # Risk assessment
        risk = comprehensive_data.get("risk_assessment", {})
        print(f"   Overall risk: {risk.get('overall_risk', 'unknown')}")
        print(f"   Risk score: {risk.get('risk_score', 0):.3f}")
        if risk.get("risk_factors"):
    pass
    pass
            print(f"   Risk factors: {', '.join(risk['risk_factors'])}")

        # Position recommendations
        recommendations = comprehensive_data.get("position_recommendations", [])
        print(f"   Position recommendations: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:3]):  # Show first 3
            print(f"     {i+1}. {rec.get('action', 'unknown')} @ {rec.get('entry_price', 0):.4f}")
            print(f"        Confidence: {rec.get('confidence', 0):.3f}")
            print(f"        Reason: {rec.get('reason', 'No reason provided')}")

        # Test 7: Test Price vs VWAP Comparison
        print("\\\n⚖️ Test 7: Testing Price vs VWAP Comparison")
        print("-" * 40)

        # Create sample price and VWAP levels for comparison
        price_levels = [
            SRLevel(price=100.0, level_type="support", method="fractal", data_source="price", timestamp=datetime.now(), strength=0.8),
            SRLevel(price=105.0, level_type="resistance", method="fractal", data_source="price", timestamp=datetime.now(), strength=0.7),
            SRLevel(price=95.0, level_type="support", method="volume", data_source="price", timestamp=datetime.now(), strength=0.6)
        ]

        vwap_levels = [
            SRLevel(price=100.2, level_type="support", method="fractal", data_source="vwap", timestamp=datetime.now(), strength=0.9),
            SRLevel(price=104.8, level_type="resistance", method="fractal", data_source="vwap", timestamp=datetime.now(), strength=0.8),
            SRLevel(price=95.1, level_type="support", method="volume", data_source="vwap", timestamp=datetime.now(), strength=0.7)
        ]

        comparison = sr_manager.compare_price_vs_vwap_predictions(price_levels, vwap_levels)

        print(f"✅ Price vs VWAP comparison:")
        print(f"   Price levels: {comparison['level_counts']['price']['total']}")
        print(f"   VWAP levels: {comparison['level_counts']['vwap']['total']}")
        print(f"   Overlap: {comparison['overlap_analysis']['overlap_count']} levels ({comparison['overlap_analysis']['overlap_rate']:.1%})")

        # Quality metrics
        price_quality = comparison['quality_metrics']['price']
        vwap_quality = comparison['quality_metrics']['vwap']
        print(f"   Price quality: {price_quality['avg_quality']:.3f}")
        print(f"   VWAP quality: {vwap_quality['avg_quality']:.3f}")

        # Recommendations
        if comparison.get('recommendations'):
    pass
    pass
            print(f"   Recommendations:")
            for rec in comparison['recommendations']:
    pass
    pass
                print(f"     - {rec}")

        # Test 8: Test Persistent Storage
        print("\\\n💾 Test 8: Testing Persistent Storage")
        print("-" * 40)

        # Save levels
        await sr_manager.save_levels()
        print("✅ SR levels saved to storage")

        # Create new manager instance to test loading
        new_sr_manager = await create_sr_levels_manager(config)
        if new_sr_manager:
    pass
    pass
            print("✅ New SR manager created and loaded existing levels")
            print(f"   Loaded {len(new_sr_manager.support_levels)} support levels")
            print(f"   Loaded {len(new_sr_manager.resistance_levels)} resistance levels")
        else:
            print("❌ Failed to create new SR manager")

        # Test 9: Performance Summary
        print("\\\n📈 Test 9: Performance Summary")
        print("-" * 40)

        print("✅ All tests completed successfully!")
        print(f"   Total SR levels: {len(sr_manager.support_levels) + len(sr_manager.resistance_levels)}")
        print(f"   Support levels: {len(sr_manager.support_levels)}")
        print(f"   Resistance levels: {len(sr_manager.resistance_levels)}")
        print(f"   Update count: {sr_manager.update_count}")
        print(f"   Last update: {sr_manager.last_update}")

        # Cleanup
        await trading_intelligence.shutdown()
        print("\\\n🧹 Cleanup completed")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_sample_market_data() -> pd.DataFrame:
    pass
    pass
    """Generate sample market data for testing."""
    # Generate 1000 data points
    np.random.seed(42)  # For reproducible results

    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')

    # Generate realistic price data with trends and volatility
    base_price = 100.0
    trend = np.linspace(0, 20, 1000)  # Upward trend
    noise = np.random.normal(0, 0.5, 1000)
    volatility = np.random.normal(0, 0.02, 1000)

    close_prices = base_price + trend + noise + volatility

    # Generate OHLC data
    data = []
    for i in range(1000):
    pass
    pass
        close = close_prices[i]
        high = close + abs(np.random.normal(0, 0.01))
        low = close - abs(np.random.normal(0, 0.01))
        open_price = close + np.random.normal(0, 0.005)

        # Ensure logical OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Generate volume (higher during price movements)
        price_change = abs(close - open_price)
        volume = np.random.exponential(1000) + price_change * 10000

        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data, index=dates)

    # Add VWAP column
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    return df


async def main():
    """Main test function."""
    print("🧪 SR Levels System Test Suite")
    print("=" * 60)

    success = await test_sr_levels_system()

    if success:
    pass
    pass
        print("\\\n🎉 All tests passed! The SR levels system is working correctly.")
    else:
        print("\\\n💥 Some tests failed. Please check the error messages above.")

    return success


if __name__ == "__main__":
    pass
    pass
    asyncio.run(main())