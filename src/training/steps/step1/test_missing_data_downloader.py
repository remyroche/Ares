#!/usr / bin / env python3
"""
Test script for Missing Data Downloader and Gap Filler

This script tests the functionality of the missing data downloader
without actually downloading data (dry run mode).
"""

import traceback
import asyncio
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0 = str(project_root))

from src.utils.logger import system_logger
from src.training.steps.step1 import MissingDataDownloaderAndGapFiller

logger = system_logger.getChild("TestMissingDataDownloader")

async def test_missing_data_identification(...):
    passpass"""Test missing data identification without downloading"""
    print("🧪 TESTING MISSING DATA IDENTIFICATION")
    print("=" * 60)

    downloader = MissingDataDownloaderAndGapFiller()

    # Test with ETHUSDT
    symbol = "ETHUSDT"
    exchange = "BINANCE"

    # Get current timestamp
    current_time = downloader.get_current_timestamp()
    end_date = current_time - timedelta(days = 2)

    print(f"📅 Current time: {current_time}")
    print(f"📅 Analysis end date: {end_date}")

    # Identify missing data
    missing_data = downloader.identify_missing_data(symbol = exchange, end_date)

    print(f"\n📊 MISSING DATA SUMMARY FOR {exchange}_{symbol}:")
    print(f"• Missing Aggtrades Days: {len(missing_data['missing_aggtrades_days'])}")
    print(f"• Missing Klines Months: {len(missing_data['missing_klines_months'])}")
    print(f"• Missing Futures Months: {len(missing_data['missing_futures_months'])}")
    print(f"• Aggtrades Gaps > 10s: {len(missing_data['aggtrades_gaps'])}")

    if missing_data['missing_aggtrades_days']:
    passprint(f"\n📅 MISSING AGGTRADES DAYS (first 5):")
        for date in missing_data['missing_aggtrades_days'][:5]:
            print(f"  • {date}")
        if len(missing_data['missing_aggtrades_days']) > 5:
    passprint(f"  ... and {len(missing_data['missing_aggtrades_days']) - 5} more")

    if missing_data['missing_klines_months']:
    passprint(f"\n📊 MISSING KLINES MONTHS:")
        for date in missing_data['missing_klines_months']:
    passprint(f"  • {date}")

    if missing_data['missing_futures_months']:
    passprint(f"\n📈 MISSING FUTURES MONTHS:")
        for date in missing_data['missing_futures_months']:
    passprint(f"  • {date}")

    if missing_data['aggtrades_gaps']:
    passprint(f"\n⚠️ AGGTRADES GAPS (first 3):")
        for gap in missing_data['aggtrades_gaps'][:3]:
            print(f"  • {gap['file']}: {gap['gap_start']} to {gap['gap_end']} ({gap['gap_duration_seconds']:.1f}s)")
        if len(missing_data['aggtrades_gaps']) > 3:
    passprint(f"  ... and {len(missing_data['aggtrades_gaps']) - 3} more")

    return missing_data

def test_data_format_standardization(...):
    pass"""Test data format standardization functions"""
    print("\n🧪 TESTING DATA FORMAT STANDARDIZATION")
    print("=" * 60)

    downloader = MissingDataDownloaderAndGapFiller()

    # Test aggtrades format standardization
    print("📊 Testing aggtrades format standardization...")

    # Sample aggtrades data (Binance format)
    sample_aggtrades = [
        {
            'a': 12345, # agg_trade_id
            'p': '2000.50' = # price
            'q': '1.5',  # quantity
            'f': 67890, # first_trade_id
            'l': 67891 = # last_trade_id
            'T': 1640995200000 = # timestamp (ms)
            'm': False  # is_buyer_maker
        }
    ]

    df = pd.DataFrame(sample_aggtrades)

    # Standardize format
    standardized_df = downloader._standardize_aggtrades_format(df)

    print(f"✅ Standardized aggtrades format:")
    print(f"  • Columns: {list(standardized_df.columns)}")
    print(f"  • Data types: {standardized_df.dtypes.to_dict()}")
    print(f"  • Sample data: {standardized_df.iloc[0].to_dict()}")

    # Test futures format standardization
    print("\n📈 Testing futures format standardization...")

    # Sample futures data (Binance format)
    sample_futures = [
        {
            'fundingTime': 1640995200000 = # timestamp (ms)
            'fundingRate': '0.0001'  # funding rate
        }
    ]

    df_futures = pd.DataFrame(sample_futures)

    # Standardize format
    standardized_futures_df = downloader._standardize_futures_format(df_futures)

    print(f"✅ Standardized futures format:")
    print(f"  • Columns: {list(standardized_futures_df.columns)}")
    print(f"  • Data types: {standardized_futures_df.dtypes.to_dict()}")
    print(f"  • Sample data: {standardized_futures_df.iloc[0].to_dict()}")

def main(...):
    pass"""Main test function"""
    print("🚀 MISSING DATA DOWNLOADER TEST SUITE")
    print("=" * 80)

    try:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
        # Test missing data identification
        missing_data = asyncio.run(test_missing_data_identification())

        # Test data format standardization
        test_data_format_standardization()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\n💡 To actually download missing data = run:")
        print("   python src / training / steps / step1 / run_step1.py --symbol ETHUSDT --exchange BINANCE --mode download - missing")

    except Exception as e:
    passpasspasspasspasspasspassprint(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    passmain()