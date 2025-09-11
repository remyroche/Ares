#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Standalone Data Collection Pipeline

This script provides a standalone entry point for executing the unified data collection pipeline
with all 12 steps: download, conversion, validation, preparation, feature engineering, 
resampling, gap filling, quality check, integration, storage, monitoring, and export.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.data_collection.sub_pipeline import (
    execute_full_data_collection_pipeline, 
    ExecutionMode,
    SubPipelineConfig
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Data Collection Pipeline - Complete data collection with 12 steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full data collection for ETHUSDT on Binance
  python standalone_data_collection.py --symbol ETHUSDT --exchange BINANCE

  # Light mode data collection
  python standalone_data_collection.py --symbol BTCUSDT --exchange BINANCE --mode light

  # Blank mode for testing
  python standalone_data_collection.py --symbol ADAUSDT --exchange BINANCE --mode blank

  # Custom data directory and timeframes
  python standalone_data_collection.py --symbol ETHUSDT --exchange BINANCE --data-dir custom_data --timeframes 5m 15m 1h
        """
    )
    
    parser.add_argument(
        '--symbol', 
        type=str, 
        default='ETHUSDT',
        help='Trading symbol (default: ETHUSDT)'
    )
    
    parser.add_argument(
        '--exchange', 
        type=str, 
        default='BINANCE',
        choices=['BINANCE', 'COINBASE', 'KRAKEN'],
        help='Exchange name (default: BINANCE)'
    )
    
    parser.add_argument(
        '--timeframe', 
        type=str, 
        default='1m',
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
        help='Base timeframe (default: 1m)'
    )
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='data_cache',
        help='Data directory (default: data_cache)'
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='full',
        choices=['full', 'light', 'blank'],
        help='Execution mode (default: full)'
    )
    
    parser.add_argument(
        '--lookback-days', 
        type=int, 
        default=30,
        help='Number of days to look back (default: 30)'
    )
    
    parser.add_argument(
        '--timeframes', 
        nargs='+',
        default=['5m', '15m', '30m', '1h'],
        help='Target timeframes for resampling (default: 5m 15m 30m 1h)'
    )
    
    parser.add_argument(
        '--add-technical-indicators', 
        action='store_true',
        help='Add technical indicators during data preparation'
    )
    
    parser.add_argument(
        '--force-rerun', 
        action='store_true',
        help='Force rerun even if data already exists'
    )
    
    parser.add_argument(
        '--parallel-processing', 
        action='store_true',
        default=True,
        help='Enable parallel processing (default: True)'
    )
    
    parser.add_argument(
        '--max-workers', 
        type=int, 
        default=4,
        help='Maximum number of workers for parallel processing (default: 4)'
    )
    
    return parser.parse_args()

def print_pipeline_info(args):
    """Print pipeline information."""
    tprint("=" * 80)
    tprint("🚀 UNIFIED DATA COLLECTION PIPELINE")
    tprint("=" * 80)
    tprint(f"📊 Symbol: {args.symbol}")
    tprint(f"🏢 Exchange: {args.exchange}")
    tprint(f"⏰ Base Timeframe: {args.timeframe}")
    tprint(f"📁 Data Directory: {args.data_dir}")
    tprint(f"🔧 Execution Mode: {args.mode.upper()}")
    tprint(f"📅 Lookback Days: {args.lookback_days}")
    tprint(f"⏱️ Target Timeframes: {', '.join(args.timeframes)}")
    tprint(f"📈 Technical Indicators: {'Yes' if args.add_technical_indicators else 'No'}")
    tprint(f"🔄 Force Rerun: {'Yes' if args.force_rerun else 'No'}")
    tprint(f"⚡ Parallel Processing: {'Yes' if args.parallel_processing else 'No'}")
    tprint(f"👥 Max Workers: {args.max_workers}")
    tprint("=" * 80)
    tprint("📋 Pipeline Steps:")
    tprint("   1. Data Download - Download raw data from exchanges")
    tprint("   2. Data Conversion - Convert data formats and standardize")
    tprint("   3. Data Validation - Validate data quality and integrity")
    tprint("   4. Data Preparation - Prepare data for further processing")
    tprint("   5. Feature Engineering - Limited feature engineering (price returns, volume returns)")
    tprint("   6. Data Resampling - Resample to multiple timeframes")
    tprint("   7. Gap Filling - Detect and fill data gaps")
    tprint("   8. Data Quality Check - Comprehensive quality assessment")
    tprint("   9. Data Integration - Integrate multiple data sources with backwards compatibility")
    tprint("  10. Data Storage - Store processed data")
    tprint("  11. Data Monitoring - Monitor data collection process")
    tprint("  12. Data Export - Export data in various formats")
    tprint("=" * 80)

async def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Print pipeline information
    print_pipeline_info(args)
    
    # Convert mode string to enum
    mode_map = {
        'full': ExecutionMode.FULL,
        'light': ExecutionMode.LIGHT,
        'blank': ExecutionMode.BLANK
    }
    execution_mode = mode_map[args.mode]
    
    try:
        tprint("🔄 Starting data collection pipeline...")
        
        # Execute the complete pipeline
        result = await execute_full_data_collection_pipeline(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            data_dir=args.data_dir,
            mode=execution_mode,
            lookback_days=args.lookback_days,
            target_timeframes=args.timeframes,
            add_technical_indicators=args.add_technical_indicators,
            force_rerun=args.force_rerun,
            parallel_processing=args.parallel_processing,
            max_workers=args.max_workers
        )
        
        # Print results
        pipeline_summary = result.get('pipeline_summary', {})
        sub_pipeline_results = result.get('sub_pipeline_results', [])
        
        tprint("=" * 80)
        tprint("✅ DATA COLLECTION PIPELINE COMPLETED")
        tprint("=" * 80)
        tprint(f"📊 Total Steps Executed: {pipeline_summary.get('total_steps', 0)}")
        tprint(f"✅ Successful Steps: {pipeline_summary.get('successful_steps', 0)}")
        tprint(f"❌ Failed Steps: {pipeline_summary.get('failed_steps', 0)}")
        tprint(f"⏱️ Total Duration: {pipeline_summary.get('total_duration', 0):.2f} seconds")
        tprint("=" * 80)
        
        # Print individual step results
        if sub_pipeline_results:
            tprint("📋 Step Results:")
            for step_result in sub_pipeline_results:
                status_emoji = "✅" if step_result.status.value == "completed" else "❌"
                tprint(f"   {status_emoji} {step_result.sub_pipeline_name}: {step_result.status.value}")
                if step_result.duration_seconds:
                    tprint(f"      Duration: {step_result.duration_seconds:.2f}s")
                if step_result.output_files:
                    tprint(f"      Output Files: {', '.join(step_result.output_files)}")
        
        tprint("=" * 80)
        tprint("🎉 Data collection pipeline completed successfully!")
        tprint("=" * 80)
        
        return True
        
    except Exception as e:
        tprint("=" * 80)
        tprint("❌ DATA COLLECTION PIPELINE FAILED")
        tprint("=" * 80)
        tprint(f"Error: {str(e)}")
        tprint("=" * 80)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the async main function
    success = asyncio.run(main())
    sys.exit(0 if success else 1)