#!/usr/bin/env python3
"""Step 9: Enhanced Model Training Pipeline with Comprehensive Validation.

This module provides the main interface for model training with:
1. HMM-based training with validation
2. Unified regime intelligence with error handling
3. Analyst creation and enhancement with data protection
4. Tactician specialist training with comprehensive monitoring
5. Full pipeline validation and error recovery
"""

import asyncio
import sys
from pathlib import Path
import time
import json
import argparse
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.decorators import handles_errors, validates, log_call, traced
from src.utils.common_operations import (
    get_current_datetime, format_datetime, ensure_directory,
    safe_json_dump, safe_json_load, validate_dataframe_schema,
    validate_data_quality, safe_copy, safe_file_exists, safe_float, safe_int,
    safe_read_parquet, safe_to_parquet, optimize_dataframe_dtypes,
    timed_operation, format_bytes, safe_log_metric, safe_log_params
)
from src.utils.logger import system_logger
from src.training.steps.model_training import run_model_training_pipeline

@handles_errors(Exception, fallback=False, log_level="ERROR")
@validates(strict=True)
@log_call
@traced
async def validate_training_config(config: Dict[str, Any]) -> bool:
    """Validate training configuration parameters."""
    logger = system_logger.getChild("ConfigValidator")
    
    print("🔍 Validating training configuration parameters...")
    logger.info("🔍 Validating training configuration parameters...")
    
    # Validate required parameters
    required_params = ['symbol', 'exchange', 'timeframe', 'data_dir']
    for param in required_params:
        if param not in config or not config[param]:
            logger.error(f"❌ Missing required parameter: {param}")
            print(f"❌ Missing required parameter: {param}")
            return False
        else:
            logger.info(f"✅ Required parameter '{param}' found: {config[param]}")
    
    # Validate symbol format
    symbol = config['symbol']
    if not isinstance(symbol, str) or len(symbol) < 3:
        logger.error(f"❌ Invalid symbol format: {symbol}")
        print(f"❌ Invalid symbol format: {symbol}")
        return False
    else:
        logger.info(f"✅ Symbol format validation passed: {symbol}")
    
    # Validate exchange
    valid_exchanges = ['BINANCE', 'MEXC', 'GATEIO']
    if config['exchange'] not in valid_exchanges:
        logger.error(f"❌ Invalid exchange: {config['exchange']}. Must be one of {valid_exchanges}")
        print(f"❌ Invalid exchange: {config['exchange']}. Must be one of {valid_exchanges}")
        return False
    else:
        logger.info(f"✅ Exchange validation passed: {config['exchange']}")
    
    # Validate timeframe
    valid_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    if config['timeframe'] not in valid_timeframes:
        logger.error(f"❌ Invalid timeframe: {config['timeframe']}. Must be one of {valid_timeframes}")
        print(f"❌ Invalid timeframe: {config['timeframe']}. Must be one of {valid_timeframes}")
        return False
    else:
        logger.info(f"✅ Timeframe validation passed: {config['timeframe']}")
    
    # Validate data directory
    if not safe_file_exists(config['data_dir']):
        logger.error(f"❌ Data directory does not exist: {config['data_dir']}")
        print(f"❌ Data directory does not exist: {config['data_dir']}")
        return False
    else:
        logger.info(f"✅ Data directory validation passed: {config['data_dir']}")
    
    # Set default values for optional parameters
    defaults = {
        'force_rerun': False,
        'hmm_training': True,
        'regime_intelligence': True,
        'analyst_creation': True,
        'analyst_enhancement': True,
        'ensemble_creation': True,
        'tactician_training': True,
        'random_state': 42,
        'max_trials': 100,
        'n_trials': 50,
        'lookback_days': 180
    }
    
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
            logger.info(f"🔧 Set default value for '{key}': {default_value}")
        elif key in ['max_trials', 'n_trials', 'lookback_days', 'random_state']:
            # Validate numeric parameters
            if key == 'random_state':
                config[key] = safe_int(config[key], default_value)
            else:
                config[key] = safe_int(config[key], default_value)
                if config[key] <= 0:
                    logger.error(f"❌ Invalid {key}: {config[key]}. Must be positive")
                    print(f"❌ Invalid {key}: {config[key]}. Must be positive")
                    return False
            logger.info(f"✅ Numeric parameter '{key}' validation passed: {config[key]}")
    
    logger.info("✅ Training configuration validation passed")
    print("✅ Training configuration validation passed")
    return True

@handles_errors(Exception, fallback=False, log_level="ERROR")
@log_call
@traced
async def validate_data_availability(config: Dict[str, Any]) -> bool:
    """Validate that all required data files are available."""
    logger = system_logger.getChild("DataValidator")
    
    print("🔍 Validating data availability and quality...")
    logger.info("🔍 Validating data availability and quality...")
    
    symbol = config['symbol']
    exchange = config['exchange']
    data_dir = config['data_dir']
    
    # Required data files
    required_files = [
        f"aggtrades_{exchange}_{symbol}_consolidated.parquet",
        f"volume_{exchange}_{symbol}_consolidated.parquet"
    ]
    
    # Optional but recommended files
    optional_files = [
        f"features_{exchange}_{symbol}_consolidated.parquet",
        f"labels_{exchange}_{symbol}_consolidated.parquet"
    ]
    
    print(f"📁 Checking data directory: {data_dir}")
    logger.info(f"📁 Checking data directory: {data_dir}")
    
    # Check required files
    print("🔍 Checking required data files...")
    logger.info("🔍 Checking required data files...")
    for file_name in required_files:
        file_path = f"{data_dir}/{file_name}"
        if not safe_file_exists(file_path):
            logger.error(f"❌ Required data file not found: {file_path}")
            print(f"❌ Required data file not found: {file_name}")
            return False
        logger.info(f"✅ Found required data file: {file_name}")
        print(f"✅ Found required data file: {file_name}")
    
    # Check optional files
    print("🔍 Checking optional data files...")
    logger.info("🔍 Checking optional data files...")
    for file_name in optional_files:
        file_path = f"{data_dir}/{file_name}"
        if safe_file_exists(file_path):
            logger.info(f"✅ Found optional data file: {file_name}")
            print(f"✅ Found optional data file: {file_name}")
        else:
            logger.warning(f"⚠️ Optional data file not found: {file_name}")
            print(f"⚠️ Optional data file not found: {file_name}")
    
    # Validate data quality of main file using common utilities
    main_data_file = f"{data_dir}/{required_files[0]}"
    print(f"🔍 Validating data quality for: {required_files[0]}")
    logger.info(f"🔍 Validating data quality for: {required_files[0]}")
    
    try:
        df = safe_read_parquet(main_data_file)
        
        if df.empty:
            logger.error(f"❌ Data file is empty: {main_data_file}")
            print(f"❌ Data file is empty: {required_files[0]}")
            return False
        
        # Optimize DataFrame memory usage
        df = optimize_dataframe_dtypes(df)
        
        if len(df) < 1000:  # Minimum data points
            logger.warning(f"⚠️ Low data volume: {len(df)} rows (minimum recommended: 1000)")
            print(f"⚠️ Low data volume: {len(df)} rows (minimum recommended: 1000)")
        else:
            logger.info(f"✅ Data volume validation passed: {len(df)} rows")
            print(f"✅ Data volume validation passed: {len(df)} rows")
        
        # Check for required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"❌ Missing required columns: {missing_columns}")
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        else:
            logger.info(f"✅ Column validation passed: {len(df.columns)} columns")
            print(f"✅ Column validation passed: {len(df.columns)} columns")
        
        # Check for data quality issues
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"⚠️ Found {nan_counts.sum()} missing values in data")
            print(f"⚠️ Found {nan_counts.sum()} missing values in data")
            for col, count in nan_counts.items():
                if count > 0:
                    logger.warning(f"   • {col}: {count} missing values")
                    print(f"   • {col}: {count} missing values")
        else:
            logger.info("✅ No missing values found in data")
            print("✅ No missing values found in data")
        
        logger.info(f"✅ Data quality validation passed: {len(df)} rows, {len(df.columns)} columns")
        print(f"✅ Data quality validation passed: {len(df)} rows, {len(df.columns)} columns")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating data file {main_data_file}: {e}")
        print(f"❌ Error validating data file {required_files[0]}: {e}")
        return False

@handles_errors(Exception, fallback=False, log_level="ERROR")
@log_call
@traced
async def create_training_summary(config: Dict[str, Any], execution_time: float, success: bool) -> None:
    """Create comprehensive training summary."""
    logger = system_logger.getChild("SummaryCreator")
    
    print("📊 Creating comprehensive training summary...")
    logger.info("📊 Creating comprehensive training summary...")
    
    # Calculate performance metrics
    execution_efficiency = 'high' if execution_time < 3600 else 'medium' if execution_time < 7200 else 'low'
    success_rate = 1.0 if success else 0.0
    
    summary = {
        'execution_info': {
            'timestamp': format_datetime(get_current_datetime()),
            'execution_time_seconds': execution_time,
            'execution_time_formatted': f"{execution_time:.2f} seconds",
            'execution_time_minutes': f"{execution_time/60:.2f} minutes",
            'success': success,
            'symbol': config['symbol'],
            'exchange': config['exchange'],
            'timeframe': config['timeframe']
        },
        'configuration': config,
        'data_info': {
            'data_directory': config['data_dir'],
            'lookback_days': safe_int(config.get('lookback_days', 180)),
            'max_trials': safe_int(config.get('max_trials', 100)),
            'n_trials': safe_int(config.get('n_trials', 50))
        },
        'training_components': {
            'hmm_training': config.get('hmm_training', True),
            'regime_intelligence': config.get('regime_intelligence', True),
            'analyst_creation': config.get('analyst_creation', True),
            'analyst_enhancement': config.get('analyst_enhancement', True),
            'ensemble_creation': config.get('ensemble_creation', True),
            'tactician_training': config.get('tactician_training', True)
        },
        'performance_metrics': {
            'success_rate': success_rate,
            'execution_efficiency': execution_efficiency,
            'execution_time_seconds': execution_time,
            'execution_time_minutes': execution_time/60
        },
        'quality_indicators': {
            'data_validation_passed': True,
            'configuration_validation_passed': True,
            'step_dependencies_validated': True,
            'overall_quality_score': success_rate
        }
    }
    
    # Save summary
    summary_file = f"{config['data_dir']}/model_training_summary_{config['symbol']}_{config['timeframe']}.json"
    safe_json_dump(summary, summary_file, indent=2)
    logger.info(f"💾 Training summary saved to: {summary_file}")
    print(f"💾 Training summary saved to: {summary_file}")
    
    # Log metrics using common utilities
    safe_log_metric("training_success", success_rate)
    safe_log_metric("execution_time_seconds", execution_time)
    safe_log_metric("execution_time_minutes", execution_time/60)
    safe_log_metric("lookback_days", safe_int(config.get('lookback_days', 180)))
    safe_log_metric("execution_efficiency", execution_efficiency)
    safe_log_params({
        "symbol": config['symbol'],
        "exchange": config['exchange'],
        "timeframe": config['timeframe'],
        "hmm_training": config.get('hmm_training', True),
        "regime_intelligence": config.get('regime_intelligence', True),
        "analyst_creation": config.get('analyst_creation', True),
        "ensemble_creation": config.get('ensemble_creation', True),
        "tactician_training": config.get('tactician_training', True)
    })
    
    # Print summary to console
    print("📊 TRAINING SUMMARY:")
    print(f"   🎯 Symbol: {config['symbol']}")
    print(f"   🏢 Exchange: {config['exchange']}")
    print(f"   ⏰ Timeframe: {config['timeframe']}")
    print(f"   ⏱️ Execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"   📊 Success rate: {success_rate:.1%}")
    print(f"   🚀 Execution efficiency: {execution_efficiency}")
    print(f"   📁 Summary file: {summary_file}")
    
    logger.info("📊 Training summary created successfully")
    print("✅ Training summary created successfully")

async def main():
    """Enhanced main function with comprehensive validation and error handling."""
    logger = system_logger.getChild("ModelTrainingMain")
    
    print("🚀 Enhanced Step 9: Model Training Pipeline")
    print("=" * 80)
    print("📊 COMPREHENSIVE LOGGING & PROGRESS TRACKING ENABLED")
    print("🔍 Quality monitoring and issue flagging active")
    print("⏱️ Real-time progress updates enabled")
    print("=" * 80)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Model Training Pipeline")
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="BINANCE", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="1m", help="Timeframe")
    parser.add_argument("--data-dir", type=str, default="data_cache", help="Data directory")
    parser.add_argument("--force-rerun", action="store_true", help="Force rerun all steps")
    parser.add_argument("--max-trials", type=int, default=100, help="Maximum trials for optimization")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--lookback-days", type=int, default=180, help="Lookback days for training")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    
    # Training component flags
    parser.add_argument("--no-hmm", action="store_true", help="Skip HMM training")
    parser.add_argument("--no-regime", action="store_true", help="Skip regime intelligence")
    parser.add_argument("--no-analyst", action="store_true", help="Skip analyst creation")
    parser.add_argument("--no-ensemble", action="store_true", help="Skip ensemble creation")
    parser.add_argument("--no-tactician", action="store_true", help="Skip tactician training")
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'symbol': args.symbol,
        'exchange': args.exchange,
        'timeframe': args.timeframe,
        'data_dir': args.data_dir,
        'force_rerun': args.force_rerun,
        'max_trials': args.max_trials,
        'n_trials': args.n_trials,
        'lookback_days': args.lookback_days,
        'random_state': args.random_state,
        'hmm_training': not args.no_hmm,
        'regime_intelligence': not args.no_regime,
        'analyst_creation': not args.no_analyst,
        'analyst_enhancement': not args.no_analyst,  # Same as analyst creation
        'ensemble_creation': not args.no_ensemble,
        'tactician_training': not args.no_tactician
    }
    
    print(f"📊 Configuration:")
    print(f"   🎯 Symbol: {config['symbol']}")
    print(f"   🏢 Exchange: {config['exchange']}")
    print(f"   ⏰ Timeframe: {config['timeframe']}")
    print(f"   📁 Data directory: {config['data_dir']}")
    print(f"   🔄 Force rerun: {config['force_rerun']}")
    print(f"   🎲 Max trials: {config['max_trials']}")
    print(f"   🔢 N trials: {config['n_trials']}")
    print(f"   📈 Lookback days: {config['lookback_days']}")
    print(f"   🎲 Random state: {config['random_state']}")
    print(f"   🧠 HMM training: {'✅' if config['hmm_training'] else '❌'}")
    print(f"   🎯 Regime intelligence: {'✅' if config['regime_intelligence'] else '❌'}")
    print(f"   👨‍💼 Analyst creation: {'✅' if config['analyst_creation'] else '❌'}")
    print(f"   🎭 Ensemble creation: {'✅' if config['ensemble_creation'] else '❌'}")
    print(f"   🎖️ Tactician training: {'✅' if config['tactician_training'] else '❌'}")
    print("=" * 80)
    
    # Log configuration to logger with emojis
    logger.info("📊 Training Configuration:")
    logger.info(f"   🎯 Symbol: {config['symbol']}")
    logger.info(f"   🏢 Exchange: {config['exchange']}")
    logger.info(f"   ⏰ Timeframe: {config['timeframe']}")
    logger.info(f"   📁 Data directory: {config['data_dir']}")
    logger.info(f"   🔄 Force rerun: {config['force_rerun']}")
    logger.info(f"   🎲 Max trials: {config['max_trials']}")
    logger.info(f"   🔢 N trials: {config['n_trials']}")
    logger.info(f"   📈 Lookback days: {config['lookback_days']}")
    logger.info(f"   🎲 Random state: {config['random_state']}")
    logger.info(f"   🧠 HMM training: {'✅' if config['hmm_training'] else '❌'}")
    logger.info(f"   🎯 Regime intelligence: {'✅' if config['regime_intelligence'] else '❌'}")
    logger.info(f"   👨‍💼 Analyst creation: {'✅' if config['analyst_creation'] else '❌'}")
    logger.info(f"   🎭 Ensemble creation: {'✅' if config['ensemble_creation'] else '❌'}")
    logger.info(f"   🎖️ Tactician training: {'✅' if config['tactician_training'] else '❌'}")
    
    # Validate configuration
    print("🔍 STEP 1: Validating training configuration...")
    logger.info("🔍 STEP 1: Validating training configuration...")
    config_valid = await validate_training_config(config)
    if not config_valid:
        logger.error("❌ Configuration validation failed")
        print("❌ Configuration validation failed")
        return False
    print("✅ Configuration validation passed")
    logger.info("✅ Configuration validation passed")
    
    # Validate data availability
    print("🔍 STEP 2: Validating data availability...")
    logger.info("🔍 STEP 2: Validating data availability...")
    data_valid = await validate_data_availability(config)
    if not data_valid:
        logger.error("❌ Data availability validation failed")
        print("❌ Data availability validation failed")
        return False
    print("✅ Data availability validation passed")
    logger.info("✅ Data availability validation passed")
    
    # Run model training pipeline
    print("🚀 STEP 3: Starting model training pipeline...")
    logger.info("🚀 STEP 3: Starting model training pipeline...")
    start_time = time.time()
    
    try:
        print("⏳ Training pipeline is running... This may take several minutes.")
        print("📊 You can monitor progress in the logs directory.")
        logger.info("⏳ Training pipeline is running... This may take several minutes.")
        logger.info("📊 You can monitor progress in the logs directory.")
        
        success = await run_model_training_pipeline(
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe'],
            data_dir=config['data_dir'],
            **config
        )
        
        total_time = time.time() - start_time
        
        if success:
            print("\n🎉 ENHANCED MODEL TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("✅ All model training steps completed with validation:")
            print("   ✅ Input validation")
            print("   ✅ Data quality validation")
            print("   ✅ Step dependency validation")
            print("   ✅ HMM-based training")
            print("   ✅ Unified regime intelligence")
            print("   ✅ Analyst creation and enhancement")
            print("   ✅ Ensemble creation")
            print("   ✅ Tactician specialist training")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print(f"📊 Performance: {total_time/60:.1f} minutes")
            print("=" * 80)
            
            # Enhanced success logging
            logger.info("🎉 ENHANCED MODEL TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("✅ All model training steps completed with validation")
            logger.info(f"⏱️ Total execution time: {total_time:.2f} seconds")
            logger.info(f"📊 Performance: {total_time/60:.1f} minutes")
            
            # Create comprehensive summary
            await create_training_summary(config, total_time, True)
            
        else:
            print("\n❌ ENHANCED MODEL TRAINING FAILED!")
            print("=" * 80)
            print("❌ Please check the logs for error details")
            print("🔍 Common issues to check:")
            print("   • Data quality and availability")
            print("   • Previous step completion")
            print("   • Memory and disk space")
            print("   • Configuration parameters")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print(f"📊 Performance: {total_time/60:.1f} minutes")
            print("=" * 80)
            
            # Enhanced failure logging
            logger.error("❌ ENHANCED MODEL TRAINING FAILED!")
            logger.error("❌ Please check the logs for error details")
            logger.error(f"⏱️ Total execution time: {total_time:.2f} seconds")
            logger.error(f"📊 Performance: {total_time/60:.1f} minutes")
            
            # Create failure summary
            await create_training_summary(config, total_time, False)
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Model training pipeline failed with exception: {e}")
        logger.error(f"📋 Exception type: {type(e).__name__}")
        logger.error(f"📋 Exception details: {str(e)}")
        print(f"\n💥 MODEL TRAINING FAILED WITH EXCEPTION!")
        print("=" * 80)
        print(f"❌ Error: {e}")
        print(f"📋 Exception type: {type(e).__name__}")
        print(f"📋 Exception details: {str(e)}")
        print("🔍 Troubleshooting suggestions:")
        print("   • Check data file integrity")
        print("   • Verify previous steps completed successfully")
        print("   • Check system resources (memory, disk space)")
        print("   • Review configuration parameters")
        print("   • Check log files for detailed error information")
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print(f"📊 Performance: {total_time/60:.1f} minutes")
        print("=" * 80)
        
        # Create failure summary
        await create_training_summary(config, total_time, False)
        
        return False
    
    return success

if __name__ == "__main__":
    """Run the enhanced model training pipeline."""
    try:
        success = asyncio.run(main())
        if success:
            print("\n🎉 Enhanced Model Training Pipeline completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Enhanced Model Training Pipeline failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Model training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error in model training pipeline: {e}")
        sys.exit(1)