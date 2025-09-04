#!/usr/bin/env python3
"""Enhanced Step 1: Data Collection Pipeline.

This module provides the main interface for data collection with comprehensive
validation, monitoring, and error handling:
1. Raw data collection from exchanges
2. Data quality validation with enhanced checks
3. Unified data loading with integrity validation
4. Data conversion and preprocessing with monitoring
5. Comprehensive error handling and recovery
6. Performance monitoring and logging
"""

import asyncio
import sys
from pathlib import Path
import time
import json
import logging
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import enhanced pipeline components
from src.utils.compat import handle_errors
from src.utils.common_operations_simple import (
    get_current_datetime,
    format_datetime,
    safe_file_exists,
    safe_json_dump,
    ensure_directory
)

# Import existing pipeline components
from src.training.steps.data_collection import run_data_collection_pipeline

@handle_errors(exceptions=(Exception,), default_return=False, context="data_collection_main")
async def main():
    """Enhanced main function to run data collection pipeline with comprehensive validation."""
    
    # Setup logging
    logger = logging.getLogger("data_collection_main")
    logger.setLevel(logging.INFO)
    
    print("🚀 ENHANCED Step 1: Data Collection Pipeline")
    print("=" * 80)
    print(f"📅 Started at: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}")
    
    # Configuration with enhanced validation
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"
    
    # Enhanced data collection parameters
    config = {
        'force_rerun': True,
        'quality_checks': True,
        'validate_data': True,
        'convert_format': True,
        'random_state': 42,
        'enable_validation': True,
        'enable_monitoring': True,
        'enable_checkpoints': True,
        'validation_level': 'critical',
    }
    
    print(f"📊 Enhanced Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: {exchange}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Data directory: {data_dir}")
    print(f"   Force rerun: {config['force_rerun']}")
    print(f"   Quality checks: {config['quality_checks']}")
    print(f"   Enhanced validation: {config['enable_validation']}")
    print(f"   Monitoring enabled: {config['enable_monitoring']}")
    print(f"   Checkpoints enabled: {config['enable_checkpoints']}")
    print("=" * 80)
    
    # Ensure data directory exists
    ensure_directory(data_dir)
    
    # Run enhanced data collection pipeline
    start_time = time.time()
    pipeline_id = f"data_collection_{symbol}_{exchange}_{timeframe}_{int(start_time)}"
    
    try:
        # Pre-execution validation
        await _validate_prerequisites(symbol, exchange, timeframe, data_dir, config)
        
        # Execute data collection with monitoring
        success = await _execute_data_collection_with_monitoring(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            pipeline_id=pipeline_id,
            **config
        )
        
        total_time = time.time() - start_time
        
        if success:
            print("\n🎉 ENHANCED DATA COLLECTION COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("✅ All enhanced data collection steps completed:")
            print("   ✅ Prerequisites validation")
            print("   ✅ Raw data collection from exchange")
            print("   ✅ Enhanced data quality validation")
            print("   ✅ Unified data loading with integrity checks")
            print("   ✅ Data conversion and preprocessing with monitoring")
            print("   ✅ Output validation and verification")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
            # Save enhanced configuration and results
            await _save_enhanced_results(
                symbol, exchange, timeframe, data_dir, config, 
                total_time, success, pipeline_id
            )
            
        else:
            print("\n❌ ENHANCED DATA COLLECTION FAILED!")
            print("=" * 80)
            print("❌ Please check the logs for error details")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
            # Save failure information
            await _save_failure_info(
                symbol, exchange, timeframe, data_dir, config, 
                total_time, pipeline_id
            )
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n💥 ENHANCED DATA COLLECTION FAILED WITH EXCEPTION: {e}")
        print("=" * 80)
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print("=" * 80)
        
        # Save exception information
        await _save_exception_info(
            symbol, exchange, timeframe, data_dir, config, 
            total_time, str(e), pipeline_id
        )
        raise
    
    return success


@handle_errors(exceptions=(Exception,), default_return=None, context="validate_prerequisites")
async def _validate_prerequisites(symbol: str, exchange: str, timeframe: str, data_dir: str, config: Dict[str, Any]) -> None:
    """Validate prerequisites before data collection."""
    logger = logging.getLogger("data_collection_main.validate_prerequisites")
    
    print("🔍 Validating prerequisites...")
    
    # Check data directory
    if not safe_file_exists(data_dir):
        ensure_directory(data_dir)
        logger.info(f"Created data directory: {data_dir}")
    
    # Validate configuration
    required_config_keys = ['force_rerun', 'quality_checks', 'validate_data', 'convert_format']
    for key in required_config_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Check for existing data if not forcing rerun
    if not config.get('force_rerun', True):
        existing_files = [
            f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet",
            f"{data_dir}/volume_{exchange}_{symbol}_consolidated.parquet"
        ]
        
        existing_count = sum(1 for file in existing_files if safe_file_exists(file))
        if existing_count > 0:
            logger.info(f"Found {existing_count} existing data files, skipping collection")
            return
    
    logger.info("Prerequisites validation completed successfully")
    print("✅ Prerequisites validation passed")


@handle_errors(exceptions=(Exception,), default_return=False, context="execute_data_collection")
async def _execute_data_collection_with_monitoring(
    symbol: str, exchange: str, timeframe: str, data_dir: str, 
    pipeline_id: str, **config: Dict[str, Any]
) -> bool:
    """Execute data collection with comprehensive monitoring."""
    logger = logging.getLogger("data_collection_main.execute_data_collection")
    
    print("🔄 Executing enhanced data collection...")
    
    # Record pipeline start
    logger.info(f"Starting data collection pipeline: {pipeline_id}")
    
    try:
        # Execute the main data collection pipeline
        success = await run_data_collection_pipeline(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            **config
        )
        
        if success:
            # Post-execution validation
            await _validate_outputs(symbol, exchange, timeframe, data_dir)
            logger.info(f"Data collection pipeline completed successfully: {pipeline_id}")
        else:
            logger.error(f"Data collection pipeline failed: {pipeline_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Data collection pipeline exception: {e}")
        raise


@handle_errors(exceptions=(Exception,), default_return=None, context="validate_outputs")
async def _validate_outputs(symbol: str, exchange: str, timeframe: str, data_dir: str) -> None:
    """Validate data collection outputs."""
    logger = logging.getLogger("data_collection_main.validate_outputs")
    
    print("🔍 Validating data collection outputs...")
    
    # Expected output files
    expected_files = [
        f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet",
        f"{data_dir}/volume_{exchange}_{symbol}_consolidated.parquet"
    ]
    
    # Check file existence and basic properties
    for file_path in expected_files:
        if not safe_file_exists(file_path):
            raise FileNotFoundError(f"Expected output file not found: {file_path}")
        
        # Check file size
        file_size = Path(file_path).stat().st_size
        if file_size == 0:
            raise ValueError(f"Output file is empty: {file_path}")
        
        logger.info(f"Validated output file: {file_path} ({file_size} bytes)")
    
    logger.info("Output validation completed successfully")
    print("✅ Output validation passed")


@handle_errors(exceptions=(Exception,), default_return=None, context="save_enhanced_results")
async def _save_enhanced_results(
    symbol: str, exchange: str, timeframe: str, data_dir: str,
    config: Dict[str, Any], execution_time: float, success: bool, pipeline_id: str
) -> None:
    """Save enhanced results and configuration."""
    logger = logging.getLogger("data_collection_main.save_results")
    
    # Enhanced results data
    results_data = {
        'pipeline_id': pipeline_id,
        'symbol': symbol,
        'exchange': exchange,
        'timeframe': timeframe,
        'data_dir': data_dir,
        'config': config,
        'execution_time': execution_time,
        'success': success,
        'timestamp': format_datetime(get_current_datetime()),
        'enhanced_features': {
            'validation_enabled': config.get('enable_validation', False),
            'monitoring_enabled': config.get('enable_monitoring', False),
            'checkpoints_enabled': config.get('enable_checkpoints', False)
        }
    }
    
    # Save configuration and results
    config_file = Path(data_dir) / f"enhanced_data_collection_results_{symbol}_{timeframe}.json"
    safe_json_dump(results_data, config_file, indent=2)
    
    logger.info(f"Enhanced results saved to: {config_file}")
    print(f"💾 Enhanced results saved to: {config_file}")


@handle_errors(exceptions=(Exception,), default_return=None, context="save_failure_info")
async def _save_failure_info(
    symbol: str, exchange: str, timeframe: str, data_dir: str,
    config: Dict[str, Any], execution_time: float, pipeline_id: str
) -> None:
    """Save failure information for debugging."""
    logger = logging.getLogger("data_collection_main.save_failure")
    
    failure_data = {
        'pipeline_id': pipeline_id,
        'symbol': symbol,
        'exchange': exchange,
        'timeframe': timeframe,
        'data_dir': data_dir,
        'config': config,
        'execution_time': execution_time,
        'success': False,
        'timestamp': format_datetime(get_current_datetime()),
        'error_type': 'pipeline_failure'
    }
    
    failure_file = Path(data_dir) / f"data_collection_failure_{symbol}_{timeframe}.json"
    safe_json_dump(failure_data, failure_file, indent=2)
    
    logger.error(f"Failure information saved to: {failure_file}")


@handle_errors(exceptions=(Exception,), default_return=None, context="save_exception_info")
async def _save_exception_info(
    symbol: str, exchange: str, timeframe: str, data_dir: str,
    config: Dict[str, Any], execution_time: float, exception: str, pipeline_id: str
) -> None:
    """Save exception information for debugging."""
    logger = logging.getLogger("data_collection_main.save_exception")
    
    exception_data = {
        'pipeline_id': pipeline_id,
        'symbol': symbol,
        'exchange': exchange,
        'timeframe': timeframe,
        'data_dir': data_dir,
        'config': config,
        'execution_time': execution_time,
        'success': False,
        'timestamp': format_datetime(get_current_datetime()),
        'error_type': 'exception',
        'exception': exception
    }
    
    exception_file = Path(data_dir) / f"data_collection_exception_{symbol}_{timeframe}.json"
    safe_json_dump(exception_data, exception_file, indent=2)
    
    logger.error(f"Exception information saved to: {exception_file}")

if __name__ == "__main__":
    # Run the data collection pipeline
    asyncio.run(main())