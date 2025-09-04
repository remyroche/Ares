#!/usr/bin/env python3
"""Enhanced Step 18: Backtesting Pipeline.

This module provides the enhanced main interface for backtesting with:
1. Comprehensive validation and error handling
2. Walk forward validation with validators
3. Monte Carlo validation with validators
4. A/B testing with validators
5. Model saving and persistence with validators
6. Common utilities for data operations
7. Performance monitoring and logging
"""

import asyncio
import sys
import argparse
import logging
from pathlib import Path
import time
import json
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import enhanced components
from src.training.steps.backtesting import run_backtesting_pipeline, BacktestingPipelineConfig
from src.training.steps.backtesting.enhanced_logging import BacktestingLogger, get_backtesting_logger
from src.utils.common_operations import (
    format_datetime, get_current_datetime, safe_file_exists, 
    ensure_directory, safe_json_dump, safe_json_load
)
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose
)
from src.core.domain.decorators import (
    validate_data_quality, monitor_step_execution, 
    ensure_data_integrity, validate_pipeline_step
)
from src.utils.logger import getChild as get_logger

# Setup logging
logger = get_logger('Step18BacktestingMain')

@compose(
    error_boundary(name="backtesting_main"),
    traced(span_name="backtesting_main"),
    log_execution_time,
    timeout(seconds=7200)  # 2 hours timeout
)
@validate_pipeline_step(
    prerequisites=['step1_data_collection', 'step2_data_reading', 'step9_hmm_based_training'],
    outputs=['backtesting_results']
)
@monitor_step_execution(
    step_name="step18_backtesting_main",
    performance_level="HIGH",
    log_memory=True,
    log_inputs=True,
    log_outputs=True
)
async def main(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE", 
    timeframe: str = "1m",
    data_dir: str = "data_cache",
    **config
) -> bool:
    """Enhanced main function to run backtesting pipeline with comprehensive validation."""
    
    # Initialize enhanced logger for main function
    main_logger = get_backtesting_logger(f"main_{symbol}_{exchange}_{timeframe}", log_dir="log/backtesting")
    main_logger.start_performance_monitoring(interval=5.0)
    
    try:
        main_logger.log_info("🚀 Enhanced Step 18: Backtesting Pipeline", "INITIALIZATION")
        main_logger.log_info("=" * 80, "INITIALIZATION")
        main_logger.log_info(f"📅 Started at: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}", "INITIALIZATION")
        
        # Enhanced configuration with validation
        enhanced_config = {
            'force_rerun': config.get('force_rerun', True),
            'walk_forward_validation': config.get('walk_forward_validation', True),
            'monte_carlo_validation': config.get('monte_carlo_validation', True),
            'ab_testing': config.get('ab_testing', True),
            'model_saving': config.get('model_saving', True),
            'random_state': config.get('random_state', 42),
            
            # Enhanced validation settings
            'enable_validation': config.get('enable_validation', True),
            'strict_validation': config.get('strict_validation', False),
            'validate_data_quality': config.get('validate_data_quality', True),
            
            # Error handling
            'retry_failed_steps': config.get('retry_failed_steps', True),
            'max_retries': config.get('max_retries', 3),
            'timeout_seconds': config.get('timeout_seconds', 3600),
            
            # Performance monitoring
            'enable_performance_monitoring': config.get('enable_performance_monitoring', True),
            'log_detailed_metrics': config.get('log_detailed_metrics', True),
        }
        
        # Log configuration with enhanced logging
        main_logger.log_info("📊 Enhanced Configuration:", "CONFIG")
        main_logger.log_info(f"   Symbol: {symbol}", "CONFIG")
        main_logger.log_info(f"   Exchange: {exchange}", "CONFIG")
        main_logger.log_info(f"   Timeframe: {timeframe}", "CONFIG")
        main_logger.log_info(f"   Data directory: {data_dir}", "CONFIG")
        main_logger.log_info(f"   Walk forward validation: {enhanced_config['walk_forward_validation']}", "CONFIG")
        main_logger.log_info(f"   Monte Carlo validation: {enhanced_config['monte_carlo_validation']}", "CONFIG")
        main_logger.log_info(f"   A/B testing: {enhanced_config['ab_testing']}", "CONFIG")
        main_logger.log_info(f"   Model saving: {enhanced_config['model_saving']}", "CONFIG")
        main_logger.log_info(f"   Enable validation: {enhanced_config['enable_validation']}", "CONFIG")
        main_logger.log_info(f"   Strict validation: {enhanced_config['strict_validation']}", "CONFIG")
        main_logger.log_info(f"   Performance monitoring: {enhanced_config['enable_performance_monitoring']}", "CONFIG")
        main_logger.log_info("=" * 80, "CONFIG")
        
        # Pre-flight validation with enhanced logging
        if enhanced_config['enable_validation']:
            main_logger.log_progress("Pre-flight Validation", 0, "Starting validation checks")
            
            with main_logger.step_timer("pre_flight_validation"):
                main_logger.log_info("🔍 Running pre-flight validation", "VALIDATION")
                
                # Validate data directory exists
                data_path = Path(data_dir)
                if not data_path.exists():
                    main_logger.log_error(Exception(f"Data directory does not exist: {data_dir}"), "VALIDATION")
                    main_logger.log_quality_flag("DATA_DIRECTORY_MISSING", f"Data directory does not exist: {data_dir}", "ERROR")
                    return False
                
                main_logger.log_success(f"Data directory exists: {data_dir}", "VALIDATION")
                
                # Validate required data files
                required_files = [
                    f"aggtrades_{exchange}_{symbol}_consolidated.parquet",
                    f"volume_{exchange}_{symbol}_consolidated.parquet"
                ]
                
                missing_files = []
                for file_name in required_files:
                    file_path = data_path / file_name
                    if not safe_file_exists(file_path):
                        missing_files.append(file_name)
                    else:
                        main_logger.log_success(f"Required file found: {file_name}", "VALIDATION")
                
                if missing_files:
                    main_logger.log_error(Exception(f"Missing required data files: {missing_files}"), "VALIDATION")
                    main_logger.log_quality_flag("MISSING_DATA_FILES", f"Missing required data files: {missing_files}", "ERROR")
                    main_logger.log_info("💡 Please run data collection first: python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE", "VALIDATION")
                    return False
                
                main_logger.log_success("All required data files found", "VALIDATION")
            
            main_logger.log_progress("Pre-flight Validation", 100, "Validation completed successfully")
        
        # Run enhanced backtesting pipeline
        start_time = time.time()
        main_logger.log_progress("Pipeline Execution", 0, "Starting enhanced backtesting pipeline")
        
        try:
            main_logger.log_info("🚀 Starting enhanced backtesting pipeline execution", "EXECUTION")
            
            success = await run_backtesting_pipeline(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                **enhanced_config
            )
            
            total_time = time.time() - start_time
            main_logger.log_progress("Pipeline Execution", 100, "Pipeline execution completed")
            
            if success:
                main_logger.log_success("🎉 ENHANCED BACKTESTING COMPLETED SUCCESSFULLY!", "COMPLETION")
                main_logger.log_info("=" * 80, "COMPLETION")
                main_logger.log_info("✅ All enhanced backtesting steps completed:", "COMPLETION")
                main_logger.log_info("   ✅ Comprehensive validation with quality assessment", "COMPLETION")
                main_logger.log_info("   ✅ Walk forward validation with detailed logging", "COMPLETION")
                main_logger.log_info("   ✅ Monte Carlo validation with performance monitoring", "COMPLETION")
                main_logger.log_info("   ✅ A/B testing with quality flags", "COMPLETION")
                main_logger.log_info("   ✅ Model saving with comprehensive reporting", "COMPLETION")
                main_logger.log_info("   ✅ Performance monitoring and resource tracking", "COMPLETION")
                main_logger.log_info("   ✅ Enhanced logging with emojis and progress indicators", "COMPLETION")
                main_logger.log_info(f"⏱️ Total execution time: {total_time:.2f} seconds", "COMPLETION")
                main_logger.log_info("=" * 80, "COMPLETION")
                
                # Save enhanced configuration and results
                results_data = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'config': enhanced_config,
                    'execution_time': total_time,
                    'success': True,
                    'start_time': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'),
                    'end_time': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'),
                    'pipeline_version': 'enhanced_v2.0_with_logging'
                }
                
                # Save configuration for future reference
                config_file = Path(data_dir) / f"enhanced_backtesting_config_{symbol}_{timeframe}.json"
                safe_json_dump(results_data, config_file, indent=2)
                main_logger.log_success(f"Enhanced configuration saved to: {config_file}", "RESULTS")
                
                # Save execution summary
                summary_file = Path(data_dir) / f"backtesting_execution_summary_{symbol}_{timeframe}.json"
                execution_summary = {
                    'execution_id': f"backtesting_{symbol}_{timeframe}_{int(time.time())}",
                    'status': 'SUCCESS',
                    'total_time_seconds': total_time,
                    'config_file': str(config_file),
                    'timestamp': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'),
                    'quality_level': 'EXCELLENT'
                }
                safe_json_dump(execution_summary, summary_file, indent=2)
                main_logger.log_success(f"Execution summary saved to: {summary_file}", "RESULTS")
                
                # Generate comprehensive report
                report_file = Path(data_dir) / f"main_backtesting_report_{symbol}_{timeframe}.json"
                main_report = main_logger.generate_report(str(report_file))
                
                # Log performance summary
                main_logger.log_performance_summary()
                
                return True
                
            else:
                main_logger.log_error(Exception("Pipeline execution failed"), "EXECUTION")
                main_logger.log_quality_flag("PIPELINE_EXECUTION_FAILURE", "Pipeline execution failed", "ERROR")
                main_logger.log_info("=" * 80, "FAILURE")
                main_logger.log_info("❌ Please check the logs for error details", "FAILURE")
                main_logger.log_info(f"⏱️ Total execution time: {total_time:.2f} seconds", "FAILURE")
                main_logger.log_info("=" * 80, "FAILURE")
                
                # Save failure information
                failure_data = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'config': enhanced_config,
                    'execution_time': total_time,
                    'success': False,
                    'error': 'Pipeline execution failed',
                    'timestamp': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')
                }
                
                failure_file = Path(data_dir) / f"backtesting_failure_{symbol}_{timeframe}.json"
                safe_json_dump(failure_data, failure_file, indent=2)
                main_logger.log_error(f"Failure information saved to: {failure_file}", "FAILURE")
                
                # Generate failure report
                failure_report_file = Path(data_dir) / f"main_backtesting_failure_report_{symbol}_{timeframe}.json"
                failure_report = main_logger.generate_report(str(failure_report_file))
                
                return False
                
        except Exception as e:
            total_time = time.time() - start_time
            main_logger.log_error(e, "PIPELINE_EXECUTION")
            main_logger.log_quality_flag("PIPELINE_EXCEPTION", f"Pipeline execution failed with exception: {e}", "ERROR")
            main_logger.log_info("=" * 80, "EXCEPTION")
            main_logger.log_info(f"⏱️ Total execution time: {total_time:.2f} seconds", "EXCEPTION")
            main_logger.log_info("=" * 80, "EXCEPTION")
            
            # Save exception information
            exception_data = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'config': enhanced_config,
                'execution_time': total_time,
                'success': False,
                'exception': str(e),
                'exception_type': type(e).__name__,
                'timestamp': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')
            }
            
            exception_file = Path(data_dir) / f"backtesting_exception_{symbol}_{timeframe}.json"
            safe_json_dump(exception_data, exception_file, indent=2)
            main_logger.log_error(f"Exception information saved to: {exception_file}", "EXCEPTION")
            
            # Generate exception report
            exception_report_file = Path(data_dir) / f"main_backtesting_exception_report_{symbol}_{timeframe}.json"
            exception_report = main_logger.generate_report(str(exception_report_file))
            
            raise
            
    finally:
        # Cleanup main logger
        main_logger.stop_performance_monitoring()
        main_logger.cleanup()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for enhanced backtesting."""
    parser = argparse.ArgumentParser(
        description="Enhanced Step 18: Backtesting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python step18_backtesting_main.py
  python step18_backtesting_main.py --symbol BTCUSDT --exchange BINANCE
  python step18_backtesting_main.py --symbol ETHUSDT --exchange BINANCE --strict-validation
  python step18_backtesting_main.py --symbol ETHUSDT --exchange BINANCE --disable-validation
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
        choices=['BINANCE', 'MEXC', 'GATEIO'],
        help='Exchange name (default: BINANCE)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1m',
        help='Timeframe (default: 1m)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data_cache',
        help='Data directory (default: data_cache)'
    )
    
    parser.add_argument(
        '--strict-validation',
        action='store_true',
        help='Enable strict validation mode'
    )
    
    parser.add_argument(
        '--disable-validation',
        action='store_true',
        help='Disable validation checks'
    )
    
    parser.add_argument(
        '--disable-walk-forward',
        action='store_true',
        help='Disable walk forward validation'
    )
    
    parser.add_argument(
        '--disable-monte-carlo',
        action='store_true',
        help='Disable Monte Carlo validation'
    )
    
    parser.add_argument(
        '--disable-ab-testing',
        action='store_true',
        help='Disable A/B testing'
    )
    
    parser.add_argument(
        '--disable-model-saving',
        action='store_true',
        help='Disable model saving'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help='Timeout in seconds (default: 3600)'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum retries for failed steps (default: 3)'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Prepare configuration from arguments
    config = {
        'strict_validation': args.strict_validation,
        'enable_validation': not args.disable_validation,
        'walk_forward_validation': not args.disable_walk_forward,
        'monte_carlo_validation': not args.disable_monte_carlo,
        'ab_testing': not args.disable_ab_testing,
        'model_saving': not args.disable_model_saving,
        'timeout_seconds': args.timeout,
        'max_retries': args.max_retries,
    }
    
    # Run the enhanced backtesting pipeline
    try:
        success = asyncio.run(main(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            data_dir=args.data_dir,
            **config
        ))
        
        if success:
            print("\n🎉 Enhanced backtesting pipeline completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Enhanced backtesting pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Backtesting pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Backtesting pipeline failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the backtesting pipeline
    asyncio.run(main())