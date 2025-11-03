#!/usr/bin/env python3
"""
Enhanced Klines Data Collection Script
Collects historical klines data with gap detection, filling, quality filtering, and resampling.

Usage:
    python3 run_binance_data_collection.py [--exchange EXCHANGE] [--symbol SYMBOL] [--interval INTERVAL] [--years YEARS]

Examples:
    python3 run_binance_data_collection.py --exchange binance
    python3 run_binance_data_collection.py --exchange binance --symbol ETHUSDT --interval 1m --years 4
    python3 run_binance_data_collection.py --exchange okx --symbol BTCUSDT --interval 5m --years 2
"""

import asyncio
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
    EnhancedKlinesProcessingPipeline,
    PipelineConfig,
    ResamplingConfig
)
from src.trading.execution.exchange_interface import ExchangeInterface


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced Klines Data Collection with Gap Filling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --exchange binance
  %(prog)s --exchange binance --symbol ETHUSDT --interval 1m --years 4
  %(prog)s --exchange okx --symbol BTCUSDT --interval 5m --years 2
        """
    )
    
    parser.add_argument(
        '--exchange',
        type=str,
        default='binance',
        choices=['binance', 'okx', 'gateio', 'bingx', 'mexc'],
        help='Exchange to collect data from (default: binance)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='ETHUSDT',
        help='Trading symbol (default: ETHUSDT)'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        default='1m',
        choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'],
        help='Kline interval (default: 1m)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        default=4,
        help='Number of years of historical data to collect (default: 4)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='historical_data',
        help='Base directory for data storage (default: historical_data)'
    )
    
    parser.add_argument(
        '--no-gap-filling',
        action='store_true',
        help='Disable gap filling'
    )
    
    parser.add_argument(
        '--no-resampling',
        action='store_true',
        help='Disable resampling to higher timeframes'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force fresh download, ignore existing data'
    )
    
    return parser.parse_args()


async def main():
    """Run data collection with command-line arguments."""
    
    args = parse_args()
    
    print("=" * 80)
    print(f"🚀 ENHANCED KLINES DATA COLLECTION - {args.exchange.upper()}")
    print("=" * 80)
    print()
    
    # Configuration from command-line arguments
    EXCHANGE = args.exchange
    SYMBOL = args.symbol
    INTERVAL = args.interval
    YEARS = args.years
    
    print(f"📊 Configuration:")
    print(f"   - Exchange: {EXCHANGE}")
    print(f"   - Symbol: {SYMBOL}")
    print(f"   - Interval: {INTERVAL}")
    print(f"   - Lookback: {YEARS} years")
    print(f"   - Data Directory: {args.data_dir}")
    print()
    
    # Configure pipeline with command-line options
    pipeline_config = PipelineConfig(
        data_dir=args.data_dir,
        exchange=EXCHANGE,
        enable_logging=True,
        enable_gap_filling=not args.no_gap_filling,
        enable_resampling=not args.no_resampling,
        enable_duplicate_handling=True,
        enable_quality_validation=True,
        batch_compatible=True,
        force_download=args.force_download,
        max_gap_minutes=1
    )
    
    # Configure resampling for multiple timeframes
    resampling_config = ResamplingConfig(
        target_intervals=['5m', '15m', '30m', '1h'],  # Resample to these timeframes
        method='ohlc',
        preserve_volume=True,
        resample_older_than_days=1,  # Resample all data older than 1 day
        enable_auto_resampling=True
    )
    
    print(f"⚙️  Pipeline Configuration:")
    print(f"   - Pipeline: EnhancedKlinesProcessingPipeline")
    print(f"   - Mode: {'Force Download' if args.force_download else 'Incremental (gap filling)'}")
    print(f"   - Target: {YEARS} years of data")
    print(f"   - Gap Detection: ✅")
    print(f"   - Gap Filling: {'❌ Disabled' if args.no_gap_filling else '✅ Enabled (downloads missing periods only)'}")
    print(f"   - Quality Filtering: ✅")
    
    # Handle resampling display
    if args.no_resampling:
        print(f"   - Resampling: ❌ Disabled")
    else:
        intervals_str = ', '.join(resampling_config.target_intervals)
        print(f"   - Resampling: ✅ Enabled (to {intervals_str})")
    
    print(f"   - Data Directory: {pipeline_config.data_dir}")
    print()
    
    # Create exchange interface for Binance
    # Note: Public market data (klines) doesn't require API credentials
    exchange_config = {
        'exchange_type': EXCHANGE,
        'api_key': None,      # Not required for public klines data
        'api_secret': None,   # Not required for public klines data
        'testnet': False,     # Use production data
        'rate_limits': {}
    }
    
    exchange_interface = ExchangeInterface(exchange_config)
    
    try:
        # Connect to exchange
        print(f"🔗 Connecting to {EXCHANGE.upper()}...")
        try:
            await exchange_interface.connect()
            print(f"✅ Connected to {EXCHANGE.upper()}")
        except Exception as e:
            print(f"⚠️  Connection warning: {e}")
            print(f"📝 Continuing with public data access...")
        print()
        
        # Create pipeline
        print(f"🔧 Initializing enhanced klines processing pipeline...")
        pipeline = EnhancedKlinesProcessingPipeline(pipeline_config)
        print(f"✅ Pipeline initialized")
        print()
        
        # Process klines data
        print(f"🚀 Starting data collection and processing...")
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        results = await pipeline.process_klines_data(
            symbol=SYMBOL,
            interval=INTERVAL,
            years=YEARS,
            exchange_interface=exchange_interface,
            resampling_config=resampling_config,
            max_gap_minutes=pipeline_config.max_gap_minutes,
            create_consolidated=True,
            batch_id=f"binance_{SYMBOL.lower()}_{YEARS}y_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Print results
        print()
        print("=" * 80)
        print("✅ DATA COLLECTION COMPLETED")
        print("=" * 80)
        print()
        print(f"📊 Results Summary:")
        print(f"   - Pipeline Success: {results['pipeline_success']}")
        print(f"   - Data Quality: {results['data_quality']}")
        print(f"   - Final Data Shape: {results['final_data_shape']}")
        print()
        
        if 'stored_files' in results:
            print(f"💾 Stored Files:")
            for file_path in results['stored_files']:
                print(f"   - {file_path}")
            print()
        
        if 'resampled_intervals' in results and results['resampled_intervals']:
            print(f"🔄 Resampled Intervals:")
            for interval in results['resampled_intervals']:
                print(f"   - {interval}")
            print()
        
        if 'gaps_filled' in results:
            print(f"🔧 Gaps Filled: {results['gaps_filled']}")
        
        if 'quality_metrics' in results:
            print(f"📈 Quality Metrics:")
            for key, value in results['quality_metrics'].items():
                print(f"   - {key}: {value}")
            print()
        
        print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Disconnect from exchange
        await exchange_interface.disconnect()
        print(f"✅ Disconnected from {EXCHANGE.upper()}")
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR IN DATA COLLECTION")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        
        import traceback
        traceback.print_exc()
        
        # Ensure cleanup
        try:
            await exchange_interface.disconnect()
        except:
            pass
        
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

