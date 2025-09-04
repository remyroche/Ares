#!/usr/bin/env python3
"""Step 3: Market Analysis Pipeline.

This module provides the main interface for market analysis with:
1. HMM regime discovery and clustering
2. Regime data splitting and labeling
3. Feature engineering and selection
4. Advanced matrix operations
"""

import asyncio
import sys
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.market_analysis import (
    run_market_analysis_pipeline,
    run_enhanced_market_analysis_pipeline,
    MarketAnalysisPipelineOrchestrator,
)
from src.training.steps.market_analysis.enhanced_logging_metrics import enhanced_logger
from src.training.steps.market_analysis.progress_monitor import progress_monitor

async def main():
    """Main function to run market analysis pipeline with enhanced logging."""
    # Configuration
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"
    
    # Market analysis parameters
    config = {
        'force_rerun': True,
        'hmm_clustering': True,
        'regime_splitting': True,
        'feature_engineering': True,
        'matrix_operations': True,
        'feature_selection': True,
        'random_state': 42,
    }
    
    # Start enhanced logging
    correlation_id = f"market_analysis_{symbol}_{exchange}_{int(time.time())}"
    enhanced_logger.start_pipeline(symbol, exchange, correlation_id)
    
    enhanced_logger.logger.info("▶️ Step 3: Market Analysis Pipeline")
    enhanced_logger.logger.info("=" * 80)
    enhanced_logger.logger.info(f"⚙️ Configuration:")
    enhanced_logger.logger.info(f"   Symbol: {symbol}")
    enhanced_logger.logger.info(f"   Exchange: {exchange}")
    enhanced_logger.logger.info(f"   Timeframe: {timeframe}")
    enhanced_logger.logger.info(f"   Data directory: {data_dir}")
    enhanced_logger.logger.info(f"   HMM clustering: {config['hmm_clustering']}")
    enhanced_logger.logger.info(f"   Regime splitting: {config['regime_splitting']}")
    enhanced_logger.logger.info(f"   Feature engineering: {config['feature_engineering']}")
    enhanced_logger.logger.info("=" * 80)
    
    # Run market analysis pipeline
    start_time = time.time()
    
    try:
        # Use enhanced market analysis pipeline with comprehensive validation
        success = await run_enhanced_market_analysis_pipeline(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            **config
        )
        
        total_time = time.time() - start_time
        
        if success:
            enhanced_logger.logger.info("\n✓ MARKET ANALYSIS COMPLETED")
            enhanced_logger.logger.info("=" * 80)
            enhanced_logger.logger.info("✓ All market analysis steps completed:")
            enhanced_logger.logger.info("   ✓ HMM regime discovery and clustering")
            enhanced_logger.logger.info("   ✓ Regime data splitting and labeling")
            enhanced_logger.logger.info("   ✓ Feature engineering and selection")
            enhanced_logger.logger.info("   ✓ Advanced matrix operations")
            enhanced_logger.logger.info(f"⏱️ Total execution time: {total_time:.2f} seconds")
            enhanced_logger.logger.info("=" * 80)
            
            # Save configuration for future reference
            config_file = Path(data_dir) / f"market_analysis_config_{symbol}_{timeframe}.json"
            with open(config_file, 'w') as f:
                json.dump({
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'config': config,
                    'execution_time': total_time,
                    'success': True,
                    'correlation_id': correlation_id
                }, f, indent=2)
            
            enhanced_logger.logger.info(f"💾 Configuration saved to: {config_file}")
            
        else:
            enhanced_logger.logger.error("\n❌ MARKET ANALYSIS FAILED!")
            enhanced_logger.logger.error("=" * 80)
            enhanced_logger.logger.error("❌ Please check the logs for error details")
            enhanced_logger.logger.error(f"⏱️ Total execution time: {total_time:.2f} seconds")
            enhanced_logger.logger.error("=" * 80)
            
    except Exception as e:
        total_time = time.time() - start_time
        error_message = str(e)
        enhanced_logger.logger.error(f"\n❌ MARKET ANALYSIS FAILED WITH EXCEPTION: {error_message}")
        enhanced_logger.logger.error("=" * 80)
        enhanced_logger.logger.error(f"⏱️ Total execution time: {total_time:.2f} seconds")
        enhanced_logger.logger.error("=" * 80)
        
        # End enhanced logging and progress monitoring with failure
        progress_monitor.stop_monitoring()
        enhanced_logger.end_pipeline(success=False, error_message=error_message)
        raise

if __name__ == "__main__":
    # Run the market analysis pipeline
    asyncio.run(main())