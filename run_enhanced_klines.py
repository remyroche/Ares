#!/usr/bin/env python3
"""
Simple script to run the enhanced klines processing pipeline for ETHUSDT, 1 year, BingX
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def main():
    """Run the enhanced klines processing pipeline."""
    try:
        # Import the pipeline components directly
        from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
            EnhancedKlinesProcessingPipeline,
            PipelineConfig,
            ResamplingConfig
        )
        
        print("🚀 Starting Enhanced Klines Processing Pipeline for ETHUSDT, 1 year, BingX")
        
        # Configure pipeline (lightweight: no gap filling, no heavy quality validation)
        pipeline_config = PipelineConfig(
            data_dir="historical_data",
            exchange="bingx",
            enable_logging=True,
            enable_gap_filling=False,
            enable_resampling=True,
            enable_duplicate_handling=True,
            enable_quality_validation=False,
            batch_compatible=True
        )
        
        # Configure resampling: always resample regardless of recency
        resampling_config = ResamplingConfig(
            target_intervals=['5m', '15m', '30m', '1h'],
            method='ohlc',
            preserve_volume=True,
            resample_older_than_days=0,
            enable_auto_resampling=True
        )
        
        # Create pipeline
        pipeline = EnhancedKlinesProcessingPipeline(pipeline_config)
        
        print("📊 Processing data using simplified interface...")
        
        # Process data using simplified interface
        results = await pipeline.process_klines_data_simple(
            exchange="bingx",  # Exchange name
            asset="ETH",       # Asset (will create ETHUSDT symbol)
            lookback_period="1y",  # Lookback period: 1 year
            interval="1m",     # Data interval
            api_key="",        # Your API key (not needed for public data)
            api_secret="",     # Your API secret (not needed for public data)
            use_testnet=False,  # Use mainnet for public data
            resampling_config=resampling_config,
            batch_id="ethusdt_1y_bingx"
        )
        
        print(f"\n🎉 Simple processing completed: {results['pipeline_success']}")
        print(f"📊 Data quality: {results['data_quality']}")
        print(f"📈 Final shape: {results['final_data_shape']}")
        print(f"💾 Stored files: {results['stored_files']}")
        print(f"🔄 Resampled intervals: {results['resampled_intervals']}")
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
