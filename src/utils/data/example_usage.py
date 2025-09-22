#!/usr/bin/env python3
"""
Example Usage of Historical Data Pipeline

This script demonstrates how to use the historical data pipeline
for downloading, processing, and managing Binance klines data.
"""

import asyncio
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data.historical_data_pipeline import HistoricalDataPipeline
from src.utils.data.klines_parquet import get_klines_manager
from src.utils.tprint import tprint
from src.utils.logger import system_logger


async def example_complete_pipeline():
    """Example: Run complete pipeline for ETHUSDT."""
    tprint("🚀 Example: Complete Pipeline for ETHUSDT")
    tprint("=" * 50)
    
    # Initialize pipeline
    pipeline = HistoricalDataPipeline("historical_data")
    
    # Run complete pipeline
    results = await pipeline.run_complete_pipeline(
        symbol="ETHUSDT",
        years=1,  # Use 1 year for example
        target_intervals=["5m", "15m", "30m", "1h"]
    )
    
    if results["pipeline_success"]:
        tprint("✅ Pipeline completed successfully!")
        tprint(f"Steps completed: {results['steps_completed']}")

        # Show summary
        for step, summary in results["summary"].items():
            tprint(f"\n{step}: {summary}")
    else:
        tprint(f"❌ Pipeline failed: {results['errors']}")
    
    return results


def example_data_access():
    """Example: Access and analyze data."""
    tprint("\n📊 Example: Data Access and Analysis")
    tprint("=" * 50)
    
    # Get data manager
    manager = get_klines_manager("historical_data")
    
    # List available data
    available_data = manager.list_available_data()
    tprint(f"Available data: {available_data}")

    if "ETHUSDT" in available_data:
        # Get data info
        info = manager.get_data_info("ETHUSDT", "1m", "raw")
        tprint(f"\nRaw data info: {info}")

        if info["available"]:
            # Read a sample of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # Last 7 days

            data = manager.read_data(
                "ETHUSDT", "1m", start_date, end_date, "raw"
            )

            if data is not None:
                tprint(f"\nSample data shape: {data.shape}")
                tprint(f"Columns: {list(data.columns)}")
                tprint(f"Date range: {data.index.min()} to {data.index.max()}")

                # Show basic statistics
                if 'close' in data.columns:
                    tprint(f"\nClose price stats:")
                    tprint(f"  Min: {data['close'].min():.2f}")
                    tprint(f"  Max: {data['close'].max():.2f}")
                    tprint(f"  Mean: {data['close'].mean():.2f}")

                if 'volume' in data.columns:
                    tprint(f"\nVolume stats:")
                    tprint(f"  Min: {data['volume'].min():.2f}")
                    tprint(f"  Max: {data['volume'].max():.2f}")
                    tprint(f"  Mean: {data['volume'].mean():.2f}")
            else:
                tprint("❌ Could not read data")
        else:
            tprint("❌ No raw data available")
    else:
        tprint("❌ No ETHUSDT data found")


def example_gap_detection():
    """Example: Gap detection and filling."""
    tprint("\n🔍 Example: Gap Detection and Filling")
    tprint("=" * 50)
    
    from src.utils.data.gap_detector import GapDetector
    
    # Initialize gap detector
    detector = GapDetector("historical_data")
    
    # Detect gaps
    gaps = detector.detect_gaps("ETHUSDT", "1m", max_gap_minutes=1)
    
    if gaps:
        tprint(f"Found {len(gaps)} gaps:")
        for i, gap in enumerate(gaps, 1):
            tprint(f"  Gap {i}: {gap['gap_start']} to {gap['gap_end']} ({gap['gap_minutes']:.1f} minutes)")
    else:
        tprint("✅ No gaps found")


def example_feature_engineering():
    """Example: Feature engineering."""
    tprint("\n🔧 Example: Feature Engineering")
    tprint("=" * 50)
    
    from src.utils.data.feature_engineer import FeatureEngineer
    
    # Initialize feature engineer
    engineer = FeatureEngineer("historical_data")
    
    # Process data
    results = engineer.process_symbol_data(
        "ETHUSDT", "1m", ["5m", "15m", "30m", "1h"]
    )
    
    if results["success"]:
        tprint("✅ Feature engineering completed!")
        tprint(f"Source records: {results['source_records']}")
        tprint(f"Featured records: {results['featured_records']}")

        # Show resampling results
        for interval, result in results["resampling_results"].items():
            if result["success"]:
                tprint(f"  {interval}: {result['records']} records")
            else:
                tprint(f"  {interval}: Failed - {result.get('error', 'Unknown error')}")
    else:
        tprint(f"❌ Feature engineering failed: {results.get('error', 'Unknown error')}")


def example_data_management():
    """Example: Data management operations."""
    tprint("\n📁 Example: Data Management")
    tprint("=" * 50)
    
    manager = get_klines_manager("historical_data")
    
    # Get detailed statistics
    stats = manager.get_data_statistics("ETHUSDT", "1m", "raw")
    
    if stats.get("available"):
        tprint(f"Data statistics:")
        tprint(f"  Files: {stats['files_count']}")
        tprint(f"  Records: {stats['total_records']:,}")
        tprint(f"  Size: {stats['file_size_mb']:.2f} MB")
        tprint(f"  Columns: {len(stats['columns'])}")

        if stats.get('price_range'):
            price_range = stats['price_range']
            tprint(f"  Price range: {price_range['min']:.2f} - {price_range['max']:.2f}")

        if stats.get('volume_stats'):
            volume_stats = stats['volume_stats']
            tprint(f"  Volume range: {volume_stats['min']:.2f} - {volume_stats['max']:.2f}")
    else:
        tprint("❌ No data available for statistics")


async def main():
    """Main example function."""
    tprint("📚 Historical Data Pipeline Examples")
    tprint("=" * 60)
    
    try:
        # Example 1: Complete pipeline
        await example_complete_pipeline()
        
        # Example 2: Data access
        example_data_access()
        
        # Example 3: Gap detection
        example_gap_detection()
        
        # Example 4: Feature engineering
        example_feature_engineering()
        
        # Example 5: Data management
        example_data_management()
        
        tprint("\n🎉 All examples completed!")

    except Exception as e:
        tprint(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
