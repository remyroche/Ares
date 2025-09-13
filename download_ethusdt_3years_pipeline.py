#!/usr/bin/env python3
"""
ETHUSDT 3-Year 1m Klines Data Download Pipeline

This script demonstrates how to use the enhanced klines_downloading_processing.py
pipeline to download 3 years of ETHUSDT 1m data with comprehensive quality checks.

Usage:
    python download_ethusdt_3years_pipeline.py

Or with custom parameters:
    python -c "
    import asyncio
    from src.training.steps.data_collection.klines_downloading_processing import run_ethusdt_3year_pipeline

    async def main():
        results = await run_ethusdt_3year_pipeline(
            data_dir='historical_data',
            interval='1m',
            max_gap_minutes=1
        )
        print(f'Pipeline success: {results[\"pipeline_success\"]}')

    asyncio.run(main())
    "
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.data.historical_data_pipeline import run_ethusdt_3year_pipeline


async def main():
    """Main function to run the ETHUSDT 3-year pipeline."""
    print("🚀 ETHUSDT 3-Year Complete Klines Data Download Pipeline")
    print("=" * 60)
    print("Includes: Download → Gap Detection → Duplicate Analysis → Feature Engineering")
    print("Note: Keeping all columns including taker_buy_base, taker_buy_quote, and year")
    print()

    # Run the complete pipeline
    results = await run_ethusdt_3year_pipeline(
        data_dir="historical_data",
        target_intervals=["5m", "15m", "30m", "1h"],
        max_gap_minutes=1,
        api_key="",  # Add your API key here if needed
        api_secret=""  # Add your API secret here if needed
    )

    print("\n" + "=" * 60)
    print("🎯 PIPELINE COMPLETED")
    print("=" * 60)

    if results["pipeline_success"]:
        print("✅ SUCCESS: All pipeline steps completed successfully!")
    else:
        print("❌ FAILURE: Some pipeline steps failed.")
        print(f"   Completed steps: {len(results['steps_completed'])}")
        print(f"   Errors: {len(results['errors'])}")
        print(f"   Warnings: {len(results['warnings'])}")

    # Print detailed results
    print("\n📊 DETAILED RESULTS:")
    print(f"Symbol: {results['symbol']}")
    print(f"Years: {results['years']}")
    print(f"Interval: {results['interval']}")
    print(f"Completion Time: {results.get('completion_time', 'N/A')}")

    if results.get('summary'):
        print("\n📈 STEP SUMMARIES:")

        if 'download' in results['summary']:
            download_info = results['summary']['download']
            print(f"  📥 Download: {download_info.get('message', 'Completed')}")

        if 'column_removal' in results['summary']:
            col_info = results['summary']['column_removal']
            print(f"  🧹 Column Removal: {col_info.get('message', 'Completed')}")

        if 'gap_handling' in results['summary']:
            gap_info = results['summary']['gap_handling']
            print(f"  🔍 Gap Handling: {gap_info.get('message', 'Completed')}")

        if 'duplicate_handling' in results['summary']:
            dup_info = results['summary']['duplicate_handling']
            print(f"  🔍 Duplicate Handling: {dup_info.get('message', 'Completed')}")

        if 'quality_check' in results['summary']:
            quality_info = results['summary']['quality_check']
            print(f"  ✅ Quality Check: {quality_info.get('message', 'Completed')}")

    return results


if __name__ == "__main__":
    # Run the pipeline
    asyncio.run(main())
