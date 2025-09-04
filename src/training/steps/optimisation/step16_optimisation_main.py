#!/usr/bin/env python3
"""Step 16: Enhanced Optimization Pipeline.

This module provides the main interface for optimization with comprehensive validation,
error handling, and data protection:
1. Confidence calibration with validation
2. Final parameters optimization with data quality checks
3. Parameter optimization wrapper with decorators
4. Comprehensive logging and monitoring
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
import time
import json
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import enhanced utilities and decorators
from src.utils.common_operations import (
    format_datetime, get_current_datetime, safe_file_exists, 
    ensure_directory, safe_json_dump, safe_json_load
)
from src.utils.data_quality_framework import DataQualityFramework
from src.utils.logger import system_logger
from src.core.decorators import handles_errors, validates, traced, log_execution_time
from src.training.steps.optimisation import run_optimisation_pipeline

# Initialize logger
logger = system_logger.getChild('OptimisationMain')

class OptimisationPipelineValidator:
    """Comprehensive validator for optimisation pipeline."""
    
    def __init__(self):
        self.dq_framework = DataQualityFramework()
        self.logger = logger.getChild('Validator')
    
    @validates()
    async def validate_input_parameters(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Validate input parameters for optimisation pipeline."""
        self.logger.info("🔍 Validating input parameters...")
        
        # Validate symbol format
        if not symbol or not isinstance(symbol, str) or len(symbol) < 3:
            self.logger.error(f"❌ Invalid symbol: {symbol}")
            return False
        
        # Validate exchange
        valid_exchanges = ['BINANCE', 'MEXC', 'GATEIO']
        if exchange not in valid_exchanges:
            self.logger.error(f"❌ Invalid exchange: {exchange}. Valid: {valid_exchanges}")
            return False
        
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        if timeframe not in valid_timeframes:
            self.logger.error(f"❌ Invalid timeframe: {timeframe}. Valid: {valid_timeframes}")
            return False
        
        # Validate data directory
        if not safe_file_exists(data_dir):
            self.logger.warning(f"⚠️ Data directory does not exist: {data_dir}")
            ensure_directory(data_dir)
        
        self.logger.info("✅ Input parameters validation passed")
        return True
    
    @validates()
    async def validate_data_availability(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """Validate that required data files are available."""
        self.logger.info("🔍 Validating data availability...")
        
        required_files = [
            f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet",
            f"{data_dir}/volume_{exchange}_{symbol}_consolidated.parquet"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not safe_file_exists(file_path):
                missing_files.append(file_path)
            else:
                self.logger.info(f"✅ Data file available: {file_path}")
        
        if missing_files:
            self.logger.error(f"❌ Missing required data files: {missing_files}")
            return False
        
        self.logger.info("✅ Data availability validation passed")
        return True
    
    @validates()
    async def validate_previous_step_outputs(self, symbol: str, exchange: str) -> bool:
        """Validate that previous step outputs are available."""
        self.logger.info("🔍 Validating previous step outputs...")
        
        # Check for tactician specialist models
        tactician_files = [
            f"models/{symbol}_{exchange}_tactician_specialist.pkl",
            f"models/{symbol}_{exchange}_analyst_ensemble.pkl"
        ]
        
        missing_tactician = []
        for file_path in tactician_files:
            if not safe_file_exists(file_path):
                missing_tactician.append(file_path)
        
        if missing_tactician:
            self.logger.warning(f"⚠️ Some tactician files missing: {missing_tactician}")
            self.logger.warning("⚠️ Optimisation will use default parameters")
        
        self.logger.info("✅ Previous step outputs validation completed")
        return True

@handles_errors(fallback=False, context="optimisation_pipeline_execution")
@traced(span_name="run_enhanced_optimisation_pipeline")
@log_execution_time("optimisation_pipeline")
async def run_enhanced_optimisation_pipeline(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str,
    enhanced_mode: bool = False,
    **config
) -> bool:
    """Run enhanced optimisation pipeline with comprehensive validation and protection."""
    
    logger.info("🚀 Starting enhanced optimisation pipeline")
    logger.info(f"📊 Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}")
    logger.info(f"🔧 Enhanced mode: {enhanced_mode}")
    
    # Initialize validator
    validator = OptimisationPipelineValidator()
    
    # Comprehensive validation
    validation_results = await asyncio.gather(
        validator.validate_input_parameters(symbol, exchange, timeframe, data_dir),
        validator.validate_data_availability(symbol, exchange, data_dir),
        validator.validate_previous_step_outputs(symbol, exchange),
        return_exceptions=True
    )
    
    # Check validation results
    validation_passed = all(
        result is True for result in validation_results if not isinstance(result, Exception)
    )
    
    if not validation_passed:
        logger.error("❌ Validation failed - cannot proceed with optimisation")
        for i, result in enumerate(validation_results):
            if isinstance(result, Exception):
                logger.error(f"❌ Validation {i} failed with exception: {result}")
            elif result is False:
                logger.error(f"❌ Validation {i} failed")
        return False
    
    # Enhanced configuration
    enhanced_config = {
        'force_rerun': config.get('force_rerun', True),
        'confidence_calibration': config.get('confidence_calibration', True),
        'parameter_optimization': config.get('parameter_optimization', True),
        'random_state': config.get('random_state', 42),
        'enhanced_mode': enhanced_mode,
        'data_quality_checks': True,
        'comprehensive_logging': True,
        'validation_enabled': True
    }
    
    logger.info("✅ All validations passed - proceeding with optimisation")
    
    try:
        # Run the optimisation pipeline with enhanced configuration
        success = await run_optimisation_pipeline(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            **enhanced_config
        )
        
        return success
        
    except Exception as e:
        logger.exception(f"❌ Optimisation pipeline execution failed: {e}")
        return False

async def main():
    """Enhanced main function with argument parsing and comprehensive validation."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Optimization Pipeline")
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", default="BINANCE", help="Exchange name")
    parser.add_argument("--timeframe", default="1m", help="Timeframe")
    parser.add_argument("--data-dir", default="data_cache", help="Data directory")
    parser.add_argument("--enhanced-mode", action="store_true", help="Enable enhanced mode")
    parser.add_argument("--force-rerun", action="store_true", help="Force rerun")
    parser.add_argument("--confidence-calibration", action="store_true", default=True, help="Enable confidence calibration")
    parser.add_argument("--parameter-optimization", action="store_true", default=True, help="Enable parameter optimization")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    
    args = parser.parse_args()
    
    # Check for enhanced mode from environment
    enhanced_mode = args.enhanced_mode or os.environ.get("OPTIMISATION_MODE") == "enhanced"
    
    print("🚀 ENHANCED OPTIMIZATION PIPELINE")
    print("=" * 80)
    print(f"🎯 Symbol: {args.symbol}")
    print(f"🏢 Exchange: {args.exchange}")
    print(f"⏰ Timeframe: {args.timeframe}")
    print(f"📁 Data Directory: {args.data_dir}")
    print(f"🔧 Enhanced Mode: {enhanced_mode}")
    print(f"🔄 Force Rerun: {args.force_rerun}")
    print(f"🎲 Random State: {args.random_state}")
    print(f"⏰ Start Time: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Configuration
    config = {
        'force_rerun': args.force_rerun,
        'confidence_calibration': args.confidence_calibration,
        'parameter_optimization': args.parameter_optimization,
        'random_state': args.random_state,
    }
    
    # Run enhanced optimization pipeline
    start_time = time.time()
    
    try:
        success = await run_enhanced_optimisation_pipeline(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            data_dir=args.data_dir,
            enhanced_mode=enhanced_mode,
            **config
        )
        
        total_time = time.time() - start_time
        
        if success:
            print("\n🎉 ENHANCED OPTIMIZATION COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("✅ All optimization steps completed with validation:")
            print("   ✅ Input parameter validation")
            print("   ✅ Data availability validation")
            print("   ✅ Previous step outputs validation")
            print("   ✅ Confidence calibration")
            print("   ✅ Final parameters optimization")
            print("   ✅ Parameter optimization wrapper")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
            # Save enhanced configuration for future reference
            config_file = Path(args.data_dir) / f"enhanced_optimisation_config_{args.symbol}_{args.timeframe}.json"
            config_data = {
                'symbol': args.symbol,
                'exchange': args.exchange,
                'timeframe': args.timeframe,
                'data_dir': args.data_dir,
                'enhanced_mode': enhanced_mode,
                'config': config,
                'execution_time': total_time,
                'success': True,
                'timestamp': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')
            }
            
            safe_json_dump(config_data, config_file, indent=2)
            print(f"💾 Enhanced configuration saved to: {config_file}")
            
        else:
            print("\n❌ ENHANCED OPTIMIZATION FAILED!")
            print("=" * 80)
            print("❌ Please check the logs for error details")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n💥 ENHANCED OPTIMIZATION FAILED WITH EXCEPTION: {e}")
        print("=" * 80)
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print("=" * 80)
        logger.exception("Optimisation pipeline failed with exception")
        raise

if __name__ == "__main__":
    # Run the enhanced optimization pipeline
    asyncio.run(main())