#!/usr/bin/env python3
"""
Binance Data Collection Script
Collects 4 years of historical klines data with gap detection, filling, quality filtering, and resampling.
"""

import asyncio
import sys
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


async def main():
    """Run Binance data collection for 4 years."""
    
    print("=" * 80)
    print("🚀 BINANCE DATA COLLECTION - 4 YEARS")
    print("=" * 80)
    print()
    
    # Configuration
    EXCHANGE = "binance"
    SYMBOL = "ETHUSDT"
    INTERVAL = "1m"
    YEARS = 4
    
    print(f"📊 Configuration:")
    print(f"   - Exchange: {EXCHANGE}")
    print(f"   - Symbol: {SYMBOL}")
    print(f"   - Interval: {INTERVAL}")
    print(f"   - Lookback: {YEARS} years")
    print()
    
    # Configure pipeline with all requested features enabled
    pipeline_config = PipelineConfig(
        data_dir="data_cache",  # Use data_cache as base directory per memory
        exchange=EXCHANGE,
        enable_logging=True,
        enable_gap_filling=True,      # ✅ Gap filling enabled
        enable_resampling=True,        # ✅ Resampling enabled
        enable_duplicate_handling=True, # Quality filtering
        enable_quality_validation=True, # ✅ Quality filtering enabled
        batch_compatible=True,
        max_gap_minutes=1              # ✅ Gap detection threshold
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
    print(f"   - Gap Detection: ✅ (max gap: {pipeline_config.max_gap_minutes} minutes)")
    print(f"   - Gap Filling: ✅")
    print(f"   - Quality Filtering: ✅")
    print(f"   - Resampling: ✅ (to {', '.join(resampling_config.target_intervals)})")
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

