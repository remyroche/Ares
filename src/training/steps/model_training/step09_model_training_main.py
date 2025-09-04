#!/usr/bin/env python3
"""Step 9: Model Training Pipeline.

This module provides the main interface for model training with:
1. HMM-based training
2. Unified regime intelligence
3. Analyst creation and enhancement
4. Tactician specialist training
"""

import asyncio
import sys
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import core decorators for enhanced validation and error handling
from src.core.decorators import (
    handles_errors,
    retry,
    timeout,
    log_execution_time,
    traced,
    validates,
)
from src.utils.compat import handle_errors
from src.utils.logger import system_logger
from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
    safe_json_dump,
    ensure_directory,
)

from src.training.steps.model_training import run_model_training_pipeline
from src.utils.pipeline_validation_utils import get_pipeline_validation_summary

@handles_errors(
    fallback=False,
    log_level="ERROR",
    include_traceback=True
)
@retry(
    max_attempts=2,
    backoff_factor=1.5,
    exceptions=(ConnectionError, TimeoutError)
)
@timeout(seconds=7200)  # 2 hour timeout
@log_execution_time
@traced
@validates(strict=True)
async def main():
    """Main function to run model training pipeline with enhanced validation and error handling."""
    logger = system_logger.getChild("ModelTrainingMain")
    
    print("🚀 Step 9: Model Training Pipeline")
    print("=" * 80)
    
    # Configuration with validation
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"
    
    # Validate data directory exists
    ensure_directory(data_dir)
    
    # Model training parameters
    config = {
        'force_rerun': True,
        'hmm_training': True,
        'regime_intelligence': True,
        'analyst_creation': True,
        'analyst_enhancement': True,
        'ensemble_creation': True,
        'tactician_training': True,
        'random_state': 42,
    }
    
    logger.info(f"Starting model training with configuration: {config}")
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: {exchange}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Data directory: {data_dir}")
    print(f"   HMM training: {config['hmm_training']}")
    print(f"   Regime intelligence: {config['regime_intelligence']}")
    print(f"   Analyst creation: {config['analyst_creation']}")
    print(f"   Tactician training: {config['tactician_training']}")
    print("=" * 80)
    
    # Run model training pipeline
    start_time = time.time()
    
    try:
        success = await run_model_training_pipeline(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            **config
        )
        
        total_time = time.time() - start_time
        
        if success:
            # Get validation summary
            validation_summary = get_pipeline_validation_summary()
            
            print("\n🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("✅ All model training steps completed:")
            print("   ✅ HMM-based training")
            print("   ✅ Unified regime intelligence")
            print("   ✅ Analyst creation and enhancement")
            print("   ✅ Ensemble creation")
            print("   ✅ Tactician specialist training")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            print("📊 VALIDATION SUMMARY:")
            print(f"   Total Validations: {validation_summary['total_validations']}")
            print(f"   Passed: {validation_summary['passed']}")
            print(f"   Failed: {validation_summary['failed']}")
            print(f"   Success Rate: {validation_summary['success_rate']:.2%}")
            print("=" * 80)
            
            # Save configuration for future reference with enhanced error handling
            config_file = Path(data_dir) / f"model_training_config_{symbol}_{timeframe}.json"
            config_data = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'config': config,
                'execution_time': total_time,
                'success': True,
                'timestamp': format_datetime(get_current_datetime()),
                'pipeline_version': 'enhanced_v1.0'
            }
            
            try:
                safe_json_dump(config_data, config_file, indent=2)
                print(f"💾 Configuration saved to: {config_file}")
                logger.info(f"Configuration saved successfully to {config_file}")
            except Exception as e:
                logger.warning(f"Failed to save configuration: {e}")
                print(f"⚠️ Warning: Could not save configuration: {e}")
            
        else:
            print("\n❌ MODEL TRAINING FAILED!")
            print("=" * 80)
            print("❌ Please check the logs for error details")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            logger.error("Model training pipeline failed")
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n💥 MODEL TRAINING FAILED WITH EXCEPTION: {e}")
        print("=" * 80)
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print("=" * 80)
        logger.exception(f"Model training failed with exception: {e}")
        raise

if __name__ == "__main__":
    # Run the model training pipeline
    asyncio.run(main())