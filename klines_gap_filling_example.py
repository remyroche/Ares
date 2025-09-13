#!/usr/bin/env python3
"""
Klines/OHLCV Gap Filling Example

This script demonstrates how to use the gap filling functionality
specifically for klines/OHLCV data in the Ares project.

Gap filling automatically detects missing time periods in your klines data
and downloads the missing data from exchanges to ensure complete datasets.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.data.quality.data_cleaning import DataCleaner, handle_missing_values_intelligently
from src.utils.logger import system_logger

logger = system_logger.getChild('klines_gap_filling')

def create_sample_klines_with_gaps():
    """Create sample klines data with intentional gaps for demonstration."""
    # Create 1-minute klines data for 1 hour
    start_time = datetime(2024, 1, 1, 10, 0, 0)
    timestamps = []

    # Create timestamps with gaps
    current_time = start_time
    for i in range(60):  # 60 minutes
        timestamps.append(int(current_time.timestamp()))

        if i == 10:  # Skip 5 minutes (5 gaps)
            current_time += timedelta(minutes=6)
        elif i == 25:  # Skip 10 minutes (10 gaps)
            current_time += timedelta(minutes=11)
        elif i == 45:  # Skip 2 minutes (2 gaps)
            current_time += timedelta(minutes=3)
        else:
            current_time += timedelta(minutes=1)

    # Create OHLCV data
    np.random.seed(42)
    data = {
        'timestamp': timestamps,
        'open': np.random.uniform(50000, 51000, len(timestamps)),
        'high': np.random.uniform(50500, 51500, len(timestamps)),
        'low': np.random.uniform(49500, 50500, len(timestamps)),
        'close': np.random.uniform(50000, 51000, len(timestamps)),
        'volume': np.random.uniform(100, 1000, len(timestamps))
    }

    df = pd.DataFrame(data)

    # Ensure OHLC logic (high >= max(open,close), low <= min(open,close))
    for idx in df.index:
        df.loc[idx, 'high'] = max(df.loc[idx, ['open', 'close', 'high']].max(), df.loc[idx, 'high'])
        df.loc[idx, 'low'] = min(df.loc[idx, ['open', 'close', 'low']].min(), df.loc[idx, 'low'])

    return df

async def demonstrate_gap_filling():
    """Demonstrate gap filling on klines data."""
    logger.info("🚀 Starting Klines Gap Filling Demonstration")

    # Create sample data with gaps
    logger.info("📊 Creating sample klines data with intentional gaps...")
    original_data = create_sample_klines_with_gaps()

    logger.info(f"   Original data: {len(original_data)} rows")
    logger.info(f"   Time range: {datetime.fromtimestamp(original_data['timestamp'].min())} to {datetime.fromtimestamp(original_data['timestamp'].max())}")

    # Initialize data cleaner for klines
    logger.info("🔧 Initializing DataCleaner for klines data...")
    cleaner = DataCleaner(data_type='klines')

    # Analyze gaps before filling
    logger.info("🔍 Analyzing gaps in original data...")
    gaps_before = cleaner._analyze_gaps(original_data, 'timestamp')
    logger.info(f"   Found {len(gaps_before)} gaps before filling:")
    for gap in gaps_before:
        logger.info(f"     - {gap.gap_type.value.upper()} gap: {gap.gap_size} seconds missing")

    # Method 1: Use the convenience function
    logger.info("\n📋 Method 1: Using handle_missing_values_intelligently() function")
    logger.info("   This function automatically handles gap filling for klines data")

    try:
        filled_data_method1 = await handle_missing_values_intelligently(
            data=original_data.copy(),
            timestamp_column='timestamp',
            symbol='BTCUSDT',          # Required for gap filling
            exchange='binance',        # Required for gap filling
            timeframe='1m'             # Required for proper gap detection
        )

        logger.info(f"   ✅ Method 1 result: {len(filled_data_method1)} rows")
        if len(filled_data_method1) > len(original_data):
            logger.info(f"   📈 Added {len(filled_data_method1) - len(original_data)} rows via gap filling")

    except Exception as e:
        logger.warning(f"   ⚠️ Method 1 failed: {e}")
        filled_data_method1 = original_data.copy()

    # Method 2: Use the DataCleaner class directly
    logger.info("\n📋 Method 2: Using DataCleaner class with clean_dataframe() method")
    logger.info("   This provides more control over the cleaning process")

    try:
        filled_data_method2 = await cleaner.clean_dataframe(
            data=original_data.copy(),
            handle_missing_values=True,    # Enable gap filling
            timestamp_column='timestamp',
            symbol='BTCUSDT',             # Required for gap filling
            exchange='binance',           # Required for gap filling
            timeframe='1m'                # Required for proper gap detection
        )

        if filled_data_method2 is not None:
            logger.info(f"   ✅ Method 2 result: {len(filled_data_method2)} rows")
            if len(filled_data_method2) > len(original_data):
                logger.info(f"   📈 Added {len(filled_data_method2) - len(original_data)} rows via gap filling")
        else:
            logger.warning("   ⚠️ Method 2 returned None")
            filled_data_method2 = original_data.copy()

    except Exception as e:
        logger.warning(f"   ⚠️ Method 2 failed: {e}")
        filled_data_method2 = original_data.copy()

    # Method 3: Manual gap filling with custom logic
    logger.info("\n📋 Method 3: Manual gap filling with custom control")
    logger.info("   This gives you full control over the gap filling process")

    try:
        # Analyze gaps first
        gaps = cleaner._analyze_gaps(original_data, 'timestamp')
        logger.info(f"   Found {len(gaps)} gaps to process")

        # Process each gap type differently
        filled_data_method3 = original_data.copy()

        for gap in gaps:
            logger.info(f"   Processing {gap.gap_type.value} gap: {gap.gap_size} seconds")

            if gap.gap_type == GapType.SMALL:
                # For small gaps, you might want custom logic
                logger.info("     → Small gap: applying custom forward fill logic")
                filled_data_method3 = cleaner._handle_small_gap(
                    filled_data_method3, gap, 'timestamp'
                )

            elif gap.gap_type in [GapType.MEDIUM, GapType.LARGE]:
                # For medium/large gaps, attempt download
                logger.info("     → Medium/Large gap: attempting data download")
                try:
                    filled_data_method3 = cleaner._handle_large_gap_with_download(
                        filled_data_method3, gap, 'timestamp',
                        'BTCUSDT', 'binance', '1m'
                    )
                except Exception as e:
                    logger.warning(f"     ⚠️ Download failed, using fallback: {e}")
                    filled_data_method3 = cleaner._handle_small_gap(
                        filled_data_method3, gap, 'timestamp'
                    )

            elif gap.gap_type == GapType.CRITICAL:
                # For critical gaps, use UnifiedGapFiller
                logger.info("     → Critical gap: using UnifiedGapFiller")
                filled_data_method3 = cleaner._handle_critical_gap(
                    filled_data_method3, gap, 'timestamp',
                    'BTCUSDT', 'binance', '1m'
                )

        logger.info(f"   ✅ Method 3 result: {len(filled_data_method3)} rows")

    except Exception as e:
        logger.warning(f"   ⚠️ Method 3 failed: {e}")
        filled_data_method3 = original_data.copy()

    # Compare results
    logger.info("\n📊 Comparison of Methods:")
    logger.info(f"   Original data:     {len(original_data)} rows")
    logger.info(f"   Method 1 result:   {len(filled_data_method1)} rows")
    logger.info(f"   Method 2 result:   {len(filled_data_method2)} rows")
    logger.info(f"   Method 3 result:   {len(filled_data_method3)} rows")

    # Verify no gaps remain in final data
    final_gaps = cleaner._analyze_gaps(filled_data_method1, 'timestamp')
    logger.info(f"   Remaining gaps in Method 1: {len(final_gaps)}")

    return {
        'original': original_data,
        'method1': filled_data_method1,
        'method2': filled_data_method2,
        'method3': filled_data_method3
    }

def integrate_into_pipeline_example():
    """Example of how to integrate gap filling into your data processing pipeline."""
    logger.info("\n🔧 Integration Example: Adding Gap Filling to Data Pipeline")

    code_example = '''
# Example: Integrating gap filling into your data processing pipeline

from src.utils.data.quality.data_cleaning import DataCleaner

def process_klines_data(symbol, exchange, timeframe, raw_data):
    """
    Process klines data with integrated gap filling.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        exchange: Exchange name (e.g., 'binance')
        timeframe: Timeframe (e.g., '1m', '5m', '1h')
        raw_data: Raw DataFrame with klines data

    Returns:
        Processed DataFrame with gaps filled
    """

    # Initialize cleaner for klines data
    cleaner = DataCleaner(data_type='klines')

    # Clean and fill gaps in one step
    cleaned_data = await cleaner.clean_dataframe(
        data=raw_data,
        handle_missing_values=True,  # This enables gap filling
        timestamp_column='timestamp',
        symbol=symbol,               # Required for gap filling
        exchange=exchange,           # Required for gap filling
        timeframe=timeframe          # Required for proper gap detection
    )

    if cleaned_data is None:
        logger.error("Data cleaning failed")
        return raw_data

    logger.info(f"Processed {len(cleaned_data)} rows of {symbol} {timeframe} data")
    return cleaned_data

# Usage in your pipeline:
# processed_data = process_klines_data('BTCUSDT', 'binance', '1m', raw_klines_df)
'''

    logger.info("💡 Integration Code Example:")
    logger.info(code_example)

    return code_example

def configure_klines_gap_thresholds():
    """Demonstrate how to configure gap thresholds specifically for different klines timeframes."""
    logger.info("\n⚙️ Klines Gap Threshold Configuration Guide")

    configuration_guide = '''
# Gap Threshold Configuration for Klines Data

Gap thresholds determine when the system considers a time gap significant enough to trigger
data re-download vs. using fallback methods like forward fill.

## Default Klines Thresholds (from data_cleaning.py):
- SMALL: 60 seconds (1 minute) - Forward fill
- MEDIUM: 300 seconds (5 minutes) - Download attempt, fallback to forward fill
- LARGE: 1800 seconds (30 minutes) - Download attempt, fallback to forward fill
- CRITICAL: 3600 seconds (1 hour) - Use UnifiedGapFiller

## Customizing Thresholds for Different Timeframes:

from src.utils.data.quality.data_cleaning import DataCleaner, GapType

# For high-frequency 1-minute klines (more sensitive to gaps)
cleaner_1m = DataCleaner(data_type='klines')
cleaner_1m.gap_thresholds = {
    GapType.SMALL: 60,      # 1 minute
    GapType.MEDIUM: 180,    # 3 minutes
    GapType.LARGE: 600,     # 10 minutes
    GapType.CRITICAL: 1800  # 30 minutes
}

# For 5-minute klines (less sensitive)
cleaner_5m = DataCleaner(data_type='klines')
cleaner_5m.gap_thresholds = {
    GapType.SMALL: 300,     # 5 minutes
    GapType.MEDIUM: 900,    # 15 minutes
    GapType.LARGE: 3600,    # 1 hour
    GapType.CRITICAL: 7200  # 2 hours
}

# For hourly klines (even less sensitive)
cleaner_1h = DataCleaner(data_type='klines')
cleaner_1h.gap_thresholds = {
    GapType.SMALL: 3600,    # 1 hour
    GapType.MEDIUM: 7200,   # 2 hours
    GapType.LARGE: 14400,   # 4 hours
    GapType.CRITICAL: 86400 # 1 day
}

## Example: Processing different timeframes with appropriate thresholds

def process_multi_timeframe_klines(raw_data_1m, raw_data_5m, raw_data_1h):
    """Process klines data for multiple timeframes with appropriate gap thresholds."""

    # 1-minute data - very sensitive to gaps
    cleaner_1m = DataCleaner(data_type='klines')
    cleaner_1m.gap_thresholds = {
        GapType.SMALL: 60, GapType.MEDIUM: 180,
        GapType.LARGE: 600, GapType.CRITICAL: 1800
    }

    clean_1m = await cleaner_1m.clean_dataframe(
        raw_data_1m, handle_missing_values=True,
        timestamp_column='timestamp', symbol='BTCUSDT',
        exchange='binance', timeframe='1m'
    )

    # 5-minute data - moderately sensitive
    cleaner_5m = DataCleaner(data_type='klines')
    cleaner_5m.gap_thresholds = {
        GapType.SMALL: 300, GapType.MEDIUM: 900,
        GapType.LARGE: 3600, GapType.CRITICAL: 7200
    }

    clean_5m = await cleaner_5m.clean_dataframe(
        raw_data_5m, handle_missing_values=True,
        timestamp_column='timestamp', symbol='BTCUSDT',
        exchange='binance', timeframe='5m'
    )

    # Hourly data - least sensitive
    cleaner_1h = DataCleaner(data_type='klines')
    cleaner_1h.gap_thresholds = {
        GapType.SMALL: 3600, GapType.MEDIUM: 7200,
        GapType.LARGE: 14400, GapType.CRITICAL: 86400
    }

    clean_1h = await cleaner_1h.clean_dataframe(
        raw_data_1h, handle_missing_values=True,
        timestamp_column='timestamp', symbol='BTCUSDT',
        exchange='binance', timeframe='1h'
    )

    return clean_1m, clean_5m, clean_1h

## Understanding Gap Types:
- SMALL: Minor gaps filled with forward/backward fill (fastest, least accurate)
- MEDIUM: Attempt download, fallback to fill (balanced approach)
- LARGE: Attempt download, fallback to fill (more aggressive downloading)
- CRITICAL: Use UnifiedGapFiller for large gaps (most comprehensive but slowest)

## Recommendations:
1. Lower thresholds for higher frequency data (1m, 5m)
2. Higher thresholds for lower frequency data (1h, 1d)
3. Test thresholds on your specific dataset to balance completeness vs. processing time
4. Monitor gap filling statistics to optimize thresholds over time
'''

    logger.info(configuration_guide)

    # Demonstrate current klines thresholds
    logger.info("📊 Current Klines Gap Thresholds:")
    demo_cleaner = DataCleaner(data_type='klines')
    for gap_type, threshold in demo_cleaner.gap_thresholds.items():
        logger.info(f"   {gap_type.value.upper()}: {threshold} seconds ({threshold//60} minutes)")

    return configuration_guide

async def main():
    try:
        # Run the demonstration
        results = await demonstrate_gap_filling()

        # Show integration example
        integration_example = integrate_into_pipeline_example()

        # Show configuration guide
        config_guide = configure_klines_gap_thresholds()

        logger.info("\n✅ Gap filling demonstration completed successfully!")

    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
