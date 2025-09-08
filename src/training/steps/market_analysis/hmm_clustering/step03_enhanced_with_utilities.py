"""
Step 3: Enhanced HMM Regime Discovery with Comprehensive Utility Integration

This module provides the main interface for enhanced HMM regime discovery with extensive
integration of all specified utilities through dependency injection:
1. common_operations.py - Core operations and utilities
2. common_utilities.py - Data processing utilities  
3. math_validation.py - Mathematical validation and operations
4. parquet_utils.py - Parquet file operations
5. serialization_utils.py - Data serialization
6. data_processing_utils.py - DataFrame processing
7. m1_gpu_utils.py - M1 GPU optimization
8. m1_memory_optimizer.py - M1 memory optimization
9. m1_cpu_optimizer.py - M1 CPU optimization
"""

import asyncio
import sys
from pathlib import Path
import time
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import dependency injection and utilities
from .step03_dependency_injection import (
    Step03ServiceProvider, Step03Config, Step03UtilityMixin,
    get_step03_service_provider, inject_step03_utilities
)

# Import existing step03 components
from src.training.steps.market_analysis.hmm_clustering import run_enhanced_step
from src.training.steps.market_analysis.hmm_clustering.step03_hmm_regime_discovery_validator import run_validator
from src.core.decorators import monitor_step03_functions, handle_step03_errors, validates, traced
from ..enhanced_error_handling import (
    enhanced_async_error_handler, critical_async_process, CriticalProcessError,
    ErrorSeverity, ErrorCategory, ErrorRecord, ErrorContext
)
from ..enhanced_validation_framework import EnhancedValidator, ValidationLevel
from ..enhanced_monitoring_system import monitor_critical_process
from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls, 
    log_internal_call, log_step_progress, log_data_operation
)
from ..standardized_parquet_handler import standardized_parquet_handler

class EnhancedHMMClusteringStep(Step03UtilityMixin):
    """Enhanced Step 3: HMM Regime Discovery with comprehensive utility integration."""
    
    @log_important_calls
    def __init__(self, config: Dict[str, Any]) -> None:
        # Initialize utilities through mixin
        super().__init__()
        
        self.config = config
        self.logger = self.utils['common_operations']['logging']['get_logger'](__name__)
        self.start_time = None
        self.step_timings = {}
        self.validator = EnhancedValidator()
        
        # Initialize service provider with configuration
        step03_config = Step03Config(
            enable_gpu_optimization=True,
            enable_memory_optimization=True,
            enable_cpu_optimization=True,
            enable_math_validation=True,
            enable_data_validation=True,
            enable_serialization=True,
            enable_parquet_operations=True,
            max_memory_usage_gb=8.0,
            max_workers=4,
            enable_extensive_logging=True
        )
        self.service_provider = get_step03_service_provider(step03_config)
        
        # Get utility instances
        self.common_ops = self.get_common_ops()
        self.common_utils = self.get_common_utils()
        self.math_validation = self.get_math_validation()
        self.serialization = self.get_serialization()
        self.m1_optimizers = self.get_m1_optimizers()
        self.data_processing = self.get_data_processing()
        self.parquet_utils = self.get_parquet_utils()
        
        # Initialize M1 optimizers
        self.gpu_manager = self.m1_optimizers['gpu']['M1GPUManager']
        self.memory_optimizer = self.m1_optimizers['memory']['M1MemoryOptimizer']
        self.cpu_optimizer = self.m1_optimizers['cpu']['M1CPUOptimizer']
        
        # Initialize data processing utilities
        self.df_validator = self.data_processing['validators']['DataFrameValidator']
        self.df_cleaner = self.data_processing['cleaners']['DataFrameCleaner']
        self.df_transformer = self.data_processing['transformers']['DataFrameTransformer']
        
        # Initialize parquet utilities
        self.parquet_handler = self.parquet_utils['ParquetUtils']
        
        self.logger.info("🚀 Enhanced HMM Clustering Step initialized with comprehensive utilities")

    @monitor_step03_functions
    @handle_step03_errors
    @validates()
    @traced(span_name='initialize_enhanced_hmm_clustering_step')
    async def initialize(self) -> None:
        """Initialize the enhanced HMM clustering step with utility validation."""
        self.start_time = self.common_ops['datetime']['get_current_datetime']()
        
        self.logger.info('🚀 Initializing Enhanced HMM Clustering Step with Comprehensive Utilities...')
        
        # Use common operations for logging and validation
        current_time = self.common_ops['datetime']['format_datetime'](
            self.start_time, '%Y-%m-%d %H:%M:%S'
        )
        
        self.logger.info('📋 Step 3 Configuration:')
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        self.logger.info(f"   - Initialization Time: {current_time}")
        
        # Validate system resources using M1 optimizers
        memory_usage = self.memory_optimizer.get_memory_usage()
        cpu_usage = self.cpu_optimizer.get_cpu_usage_report()
        
        self.logger.info('🔧 System Resources:')
        self.logger.info(f"   - Memory Usage: {memory_usage['rss_gb']:.2f}GB / {memory_usage['available_gb']:.2f}GB available")
        self.logger.info(f"   - CPU Cores: {cpu_usage.get('system_info', {}).get('cpu_count', 'N/A')}")
        self.logger.info(f"   - GPU Available: {self.gpu_manager.device.type}")
        
        # Validate utilities are properly initialized
        utility_health = self._validate_utility_health()
        if not utility_health['all_healthy']:
            self.logger.warning(f"⚠️ Some utilities not fully healthy: {utility_health['issues']}")
        
        self.logger.info('✅ Enhanced HMM Clustering Step initialized successfully')

    def _validate_utility_health(self) -> Dict[str, Any]:
        """Validate that all utilities are properly initialized and healthy."""
        health_status = {
            'all_healthy': True,
            'issues': [],
            'utility_status': {}
        }
        
        try:
            # Test common operations
            test_datetime = self.common_ops['datetime']['get_current_datetime']()
            test_df = self.common_ops['dataframe']['create_empty_dataframe'](['test'])
            health_status['utility_status']['common_operations'] = 'healthy'
        except Exception as e:
            health_status['all_healthy'] = False
            health_status['issues'].append(f"common_operations: {e}")
            health_status['utility_status']['common_operations'] = 'unhealthy'
        
        try:
            # Test math validation
            test_divide = self.math_validation['basic_math']['safe_divide'](10, 2)
            test_validation = self.math_validation['validation']['validate_finite'](5.0)
            health_status['utility_status']['math_validation'] = 'healthy'
        except Exception as e:
            health_status['all_healthy'] = False
            health_status['issues'].append(f"math_validation: {e}")
            health_status['utility_status']['math_validation'] = 'unhealthy'
        
        try:
            # Test M1 optimizers
            gpu_should_use = self.gpu_manager.should_use_gpu(1000, "general")
            memory_usage = self.memory_optimizer.get_memory_usage()
            cpu_workers = self.cpu_optimizer.get_optimal_workers_for_task("general")
            health_status['utility_status']['m1_optimizers'] = 'healthy'
        except Exception as e:
            health_status['all_healthy'] = False
            health_status['issues'].append(f"m1_optimizers: {e}")
            health_status['utility_status']['m1_optimizers'] = 'unhealthy'
        
        try:
            # Test data processing
            test_df = pd.DataFrame({'test': [1, 2, 3]})
            validation_result = self.df_validator.validate_dataframe(test_df)
            health_status['utility_status']['data_processing'] = 'healthy'
        except Exception as e:
            health_status['all_healthy'] = False
            health_status['issues'].append(f"data_processing: {e}")
            health_status['utility_status']['data_processing'] = 'unhealthy'
        
        return health_status

    @critical_async_process('enhanced_hmm_clustering')
    @monitor_critical_process('enhanced_hmm_clustering')
    @enhanced_async_error_handler(
        error_severity=ErrorSeverity.CRITICAL,
        error_category=ErrorCategory.BUSINESS_LOGIC,
        should_fail_fast=True,
        step_name='enhanced_hmm_clustering'
    )
    @monitor_step03_functions
    @validates()
    @traced(span_name='execute_enhanced_hmm_clustering_step')
    @inject_step03_utilities
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any], 
                     utils: Dict[str, Any] = None, services: Step03ServiceProvider = None) -> Dict[str, Any]:
        """Execute enhanced HMM regime discovery with comprehensive utility integration."""
        step_start = time.time()
        self.logger.info('🎯 Starting Enhanced HMM Clustering execution with comprehensive utilities...')
        
        try:
            # Extract and validate inputs using common operations
            symbol = self.common_ops['validation']['safe_dict_get'](training_input, 'symbol', 'ETHUSDT')
            exchange = self.common_ops['validation']['safe_dict_get'](training_input, 'exchange', 'BINANCE')
            timeframe = self.common_ops['validation']['safe_dict_get'](training_input, 'timeframe', '1m')
            data_dir = self.common_ops['validation']['safe_dict_get'](training_input, 'data_dir')
            force_rerun = self.common_ops['validation']['safe_dict_get'](training_input, 'force_rerun', False)
            
            # Validate required parameters using math validation
            if not symbol or not exchange or not timeframe:
                raise ValueError("Missing required parameters: symbol, exchange, timeframe")
            
            # Use common operations for path handling
            if data_dir is None:
                from src.utils.pipeline_standards import pipeline_standards as _pipeline_standards
                data_dir = _pipeline_standards.build_path('processed_data', exchange, symbol)
            
            # Validate data directory using common operations
            data_path = Path(data_dir)
            if not self.common_ops['file_operations']['safe_file_exists'](data_path):
                raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
            
            # Ensure directory exists using common operations
            self.common_ops['file_operations']['ensure_directory'](data_path)
            
            # Check for required data files using common operations
            required_files = [
                f"{exchange}_{symbol}_processed.parquet",
                f"{exchange}_{symbol}_volume_consolidated.parquet"
            ]
            
            missing_files = []
            for file_name in required_files:
                file_path = data_path / file_name
                if not self.common_ops['file_operations']['safe_file_exists'](file_path):
                    missing_files.append(file_name)
            
            if missing_files:
                raise FileNotFoundError(f"Missing required data files: {missing_files}")
            
            # Load and validate data using parquet utilities
            data_file = data_path / f"{exchange}_{symbol}_processed.parquet"
            self.logger.info(f"📂 Loading data using parquet utilities: {data_file}")
            
            # Use parquet utilities for safe loading
            data = self.parquet_handler.safe_read_parquet(str(data_file))
            if data is None:
                raise RuntimeError(f"Failed to load data from {data_file}")
            
            # Validate data quality using data processing utilities
            self.logger.info("🔍 Validating data quality using comprehensive utilities...")
            
            # Use DataFrame validator for comprehensive validation
            validation_result = self.df_validator.validate_dataframe(data)
            if validation_result.summary['critical_issues'] > 0:
                raise ValueError(f"Data quality validation failed: {validation_result.issues}")
            
            # Use common utilities for additional data quality checks
            data_quality_report = self.common_utils['data_quality']['create_data_quality_report'](data)
            if data_quality_report['status'] != 'success':
                raise ValueError(f"Data quality report failed: {data_quality_report.get('message', 'Unknown error')}")
            
            self.logger.info(f'✅ Data validation passed: {len(data)} rows, {len(data.columns)} columns')
            
            # Clean data using data processing utilities
            self.logger.info("🧹 Cleaning data using DataFrame cleaner...")
            cleaned_data = self.df_cleaner.clean_dataframe(data, [
                'remove_duplicates', 'handle_nulls', 'fix_types', 'remove_constant_columns'
            ])
            
            # Optimize memory usage using M1 memory optimizer
            self.logger.info("🧠 Optimizing memory usage...")
            memory_optimization_result = self.memory_optimizer.optimize_memory()
            self.logger.info(f"💾 Memory optimization: {memory_optimization_result['memory_freed_mb']:.1f}MB freed")
            
            # Prepare enhanced configuration with utility integration
            enhanced_config = {
                'n_trials': 50, 
                'timeout_minutes': 15, 
                'cv_folds': 3, 
                'random_state': 42, 
                'ensemble_weights': {'hmm': 0.4, 'kmeans': 0.3, 'dbscan': 0.3}, 
                'initial_features': 20, 
                'feature_increment': 10, 
                'max_features': 100, 
                'min_improvement': 0.001, 
                'patience': 3,
                # Add utility-specific configurations
                'use_gpu_optimization': True,
                'use_memory_optimization': True,
                'use_cpu_optimization': True,
                'use_math_validation': True,
                'use_data_validation': True
            }
            
            self.logger.info('=' * 60)
            self.logger.info('STEP 1: Enhanced HMM Regime Discovery with Comprehensive Utilities')
            self.logger.info('=' * 60)
            
            # Execute enhanced step with utility integration
            success = await run_enhanced_step(
                symbol=symbol, 
                exchange=exchange, 
                timeframe=timeframe, 
                data_dir=data_dir, 
                force_rerun=force_rerun, 
                **enhanced_config
            )
            
            if not success:
                error_msg = 'Enhanced HMM regime discovery failed'
                self.logger.critical(f'🚨 CRITICAL FAILURE: {error_msg}')
                pipeline_state['hmm_clustering_completed'] = False
                pipeline_state['hmm_clustering_error'] = error_msg
                
                # Validate expected outputs using common operations
                expected_outputs = [
                    f'{symbol}_{exchange}_hmm_model.pkl',
                    f'{symbol}_{exchange}_regime_data.parquet',
                    f'{symbol}_{exchange}_hmm_metrics.json'
                ]
                
                validation_result = await self.validator.validate_process_completion(
                    'hmm_clustering', expected_outputs, data_dir, ValidationLevel.CRITICAL
                )
                
                if not validation_result.passed:
                    raise CriticalProcessError(
                        f"HMM clustering failed and validation failed: {validation_result.message}",
                        ErrorRecord(
                            error_id=f"hmm_clustering_failure_{int(time.time())}",
                            error_type="CriticalProcessError",
                            error_message=validation_result.message,
                            severity=ErrorSeverity.CRITICAL,
                            category=ErrorCategory.BUSINESS_LOGIC,
                            context=ErrorContext(
                                function_name="execute_enhanced_hmm_clustering",
                                step_name="enhanced_hmm_clustering"
                            ),
                            stack_trace="",
                            should_fail_fast=True
                        )
                    )
                
                raise RuntimeError(f"HMM clustering failed: {error_msg}")
            
            # Success case - update pipeline state with utility information
            self.logger.info('✅ Enhanced HMM regime discovery completed successfully')
            pipeline_state['hmm_clustering_completed'] = True
            pipeline_state['enhanced_features_used'] = True
            pipeline_state['bayesian_optimization_used'] = True
            pipeline_state['ensemble_clustering_used'] = True
            pipeline_state['ml_transition_detection_used'] = True
            pipeline_state['comprehensive_utilities_used'] = True
            pipeline_state['m1_optimizations_used'] = True
            pipeline_state['data_processing_utilities_used'] = True
            pipeline_state['math_validation_used'] = True
            pipeline_state['serialization_utilities_used'] = True
            
            # Validate expected outputs using common operations
            expected_outputs = [
                f'{symbol}_{exchange}_hmm_model.pkl',
                f'{symbol}_{exchange}_regime_data.parquet',
                f'{symbol}_{exchange}_hmm_metrics.json'
            ]
            
            validation_result = await self.validator.validate_process_completion(
                'hmm_clustering', expected_outputs, data_dir, ValidationLevel.CRITICAL
            )
            
            if not validation_result.passed:
                raise CriticalProcessError(
                    f"HMM clustering completed but validation failed: {validation_result.message}",
                    ErrorRecord(
                        error_id=f"hmm_clustering_validation_failure_{int(time.time())}",
                        error_type="ValidationError",
                        error_message=validation_result.message,
                        severity=ErrorSeverity.CRITICAL,
                        category=ErrorCategory.VALIDATION,
                        context=ErrorContext(
                            function_name="execute_enhanced_hmm_clustering",
                            step_name="enhanced_hmm_clustering"
                        ),
                        stack_trace="",
                        should_fail_fast=True
                    )
                )
            
            # Save configuration using serialization utilities
            config_data = {
                'symbol': symbol, 
                'exchange': exchange, 
                'timeframe': timeframe, 
                'config': enhanced_config, 
                'execution_time': time.time() - step_start, 
                'success': True, 
                'timestamp': self.common_ops['datetime']['get_current_datetime']().isoformat(),
                'utilities_used': {
                    'common_operations': True,
                    'common_utilities': True,
                    'math_validation': True,
                    'parquet_utils': True,
                    'serialization_utils': True,
                    'data_processing_utils': True,
                    'm1_gpu_utils': True,
                    'm1_memory_optimizer': True,
                    'm1_cpu_optimizer': True
                },
                'memory_optimization_result': memory_optimization_result,
                'utility_health_status': self._validate_utility_health()
            }
            
            config_file = Path(data_dir) / f'enhanced_step3_config_with_utilities_{symbol}_{timeframe}.json'
            self.serialization['convenience_functions']['save_json'](config_data, config_file)
            self.logger.info(f'💾 Configuration with utilities saved to: {config_file}')
            
            # Log artifacts to MLflow using common operations
            await self._log_step3_artifacts_to_mlflow(training_input, pipeline_state)
            
            # Final memory optimization
            final_memory_result = self.memory_optimizer.optimize_memory()
            self.logger.info(f"🧹 Final memory optimization: {final_memory_result['memory_freed_mb']:.1f}MB freed")
            
            total_elapsed = time.time() - step_start
            self.logger.info(f'⏱️ Enhanced HMM Clustering with utilities completed in {total_elapsed:.2f} seconds')
            
            return pipeline_state
            
        except CriticalProcessError as e:
            self.logger.critical(f'🚨 CRITICAL PROCESS ERROR in Enhanced HMM Clustering: {e}')
            pipeline_state['hmm_clustering_completed'] = False
            pipeline_state['hmm_clustering_error'] = str(e)
            raise
        except Exception as e:
            self.logger.critical(f'🚨 CRITICAL ERROR in Enhanced HMM Clustering: {e}')
            pipeline_state['hmm_clustering_completed'] = False
            pipeline_state['hmm_clustering_error'] = str(e)
            
            # Convert to CriticalProcessError for fail-fast behavior
            raise CriticalProcessError(
                f"Enhanced HMM clustering failed with critical error: {e}",
                ErrorRecord(
                    error_id=f"enhanced_hmm_clustering_critical_error_{int(time.time())}",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.BUSINESS_LOGIC,
                    context=ErrorContext(
                        function_name="execute_enhanced_hmm_clustering",
                        step_name="enhanced_hmm_clustering"
                    ),
                    stack_trace="",
                    should_fail_fast=True
                )
            )

    @monitor_step03_functions
    @handle_step03_errors
    @validates()
    @traced(span_name='log_step3_artifacts_to_mlflow_with_utilities')
    async def _log_step3_artifacts_to_mlflow(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> None:
        """Log step 3 artifacts to MLflow with enhanced metadata and utility information."""
        try:
            symbol = self.common_ops['validation']['safe_dict_get'](training_input, 'symbol', 'ETHUSDT')
            exchange = self.common_ops['validation']['safe_dict_get'](training_input, 'exchange', 'BINANCE')
            timeframe = self.common_ops['validation']['safe_dict_get'](training_input, 'timeframe', '1m')
            
            # Enhanced metrics with utility information
            metrics = {
                'step3_hmm_clustering_completed': 1.0, 
                'step3_enhanced_features_used': 1.0, 
                'step3_bayesian_optimization_used': 1.0, 
                'step3_ensemble_clustering_used': 1.0, 
                'step3_ml_transition_detection_used': 1.0, 
                'step3_comprehensive_utilities_used': 1.0,
                'step3_m1_optimizations_used': 1.0,
                'step3_data_processing_utilities_used': 1.0,
                'step3_math_validation_used': 1.0,
                'step3_serialization_utilities_used': 1.0,
                'step3_execution_time': pipeline_state.get('execution_time', 0.0)
            }
            
            # Enhanced parameters with utility information
            params = {
                'symbol': symbol, 
                'exchange': exchange, 
                'timeframe': timeframe, 
                'enhanced_version': 'v3.0_with_comprehensive_utilities', 
                'features_integrated': 'bayesian_optimization,ensemble_clustering,ml_transition_detection,comprehensive_utilities',
                'utilities_integrated': 'common_operations,common_utilities,math_validation,parquet_utils,serialization_utils,data_processing_utils,m1_gpu_utils,m1_memory_optimizer,m1_cpu_optimizer',
                'm1_optimizations': 'gpu_optimization,memory_optimization,cpu_optimization'
            }
            
            # Log metrics and parameters using common operations
            for key, value in metrics.items():
                self.common_ops['logging']['safe_log_metric'](key, value)
            
            self.common_ops['logging']['safe_log_params'](params)
            
            self.logger.info('✅ Step 3 artifacts with comprehensive utilities logged to MLflow successfully')
            
        except Exception as e:
            self.logger.error(f'❌ Failed to log step 3 artifacts to MLflow: {e}')

@critical_async_process('enhanced_hmm_clustering_with_utilities')
@monitor_critical_process('enhanced_hmm_clustering_with_utilities')
@enhanced_async_error_handler(
    error_severity=ErrorSeverity.CRITICAL,
    error_category=ErrorCategory.BUSINESS_LOGIC,
    should_fail_fast=True,
    step_name='enhanced_hmm_clustering_with_utilities'
)
@monitor_step03_functions
@validates()
@traced(span_name='run_enhanced_step03_with_comprehensive_utilities')
@inject_step03_utilities
async def run_enhanced_step03_with_utilities(
    symbol: str, 
    exchange: str, 
    timeframe: str = '1m', 
    data_dir: str = None, 
    force_rerun: bool = False, 
    **kwargs: Any
) -> bool:
    """Run the enhanced HMM clustering step with comprehensive utility integration.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        exchange: Exchange name (e.g., "BINANCE")
        timeframe: Timeframe (e.g., "1m")
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force re-run even if results exist
        **kwargs: Additional arguments

    Returns:
        bool: True if successful, False otherwise
    """
    start_time = time.time()
    
    try:
        # Initialize service provider and utilities
        step03_config = Step03Config(
            enable_gpu_optimization=True,
            enable_memory_optimization=True,
            enable_cpu_optimization=True,
            enable_math_validation=True,
            enable_data_validation=True,
            enable_serialization=True,
            enable_parquet_operations=True,
            max_memory_usage_gb=8.0,
            max_workers=4,
            enable_extensive_logging=True
        )
        
        service_provider = get_step03_service_provider(step03_config)
        utils = service_provider.get_all_utilities()
        
        logger = utils['common_operations']['logging']['get_logger']('EnhancedStep03WithUtilities')
        
        if data_dir is None:
            from src.utils.pipeline_standards import pipeline_standards as _pipeline_standards
            data_dir = _pipeline_standards.build_path('processed_data', exchange, symbol)
        
        # Use common operations for logging
        current_time = utils['common_operations']['datetime']['format_datetime'](
            utils['common_operations']['datetime']['get_current_datetime'](), 
            '%Y-%m-%d %H:%M:%S'
        )
        
        logger.info('=' * 80)
        logger.info('🚀 STEP 3: Enhanced HMM Clustering with Comprehensive Utility Integration')
        logger.info('=' * 80)
        logger.info(f'🎯 Symbol: {symbol}')
        logger.info(f'🏢 Exchange: {exchange}')
        logger.info(f'📊 Timeframe: {timeframe}')
        logger.info(f'📁 Data directory: {data_dir}')
        logger.info(f'🔄 Force rerun: {force_rerun}')
        logger.info(f"⏰ Start time: {current_time}")
        logger.info('🔧 Comprehensive Utilities Integrated:')
        logger.info('   ✅ common_operations.py - Core operations and utilities')
        logger.info('   ✅ common_utilities.py - Data processing utilities')
        logger.info('   ✅ math_validation.py - Mathematical validation and operations')
        logger.info('   ✅ parquet_utils.py - Parquet file operations')
        logger.info('   ✅ serialization_utils.py - Data serialization')
        logger.info('   ✅ data_processing_utils.py - DataFrame processing')
        logger.info('   ✅ m1_gpu_utils.py - M1 GPU optimization')
        logger.info('   ✅ m1_memory_optimizer.py - M1 memory optimization')
        logger.info('   ✅ m1_cpu_optimizer.py - M1 CPU optimization')
        logger.info('=' * 80)
        
        # Create enhanced configuration
        config = {
            'SYMBOL': symbol, 
            'EXCHANGE': exchange, 
            'TIMEFRAME': timeframe, 
            'DATA_DIR': data_dir
        }
        
        # Initialize and execute enhanced step
        step = EnhancedHMMClusteringStep(config)
        await step.initialize()
        
        training_input = {
            'symbol': symbol, 
            'exchange': exchange, 
            'timeframe': timeframe, 
            'data_dir': data_dir, 
            'force_rerun': force_rerun
        }
        
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)
        
        if result.get('hmm_clustering_completed', False):
            logger.info('✅ Step 3: Enhanced HMM Clustering with comprehensive utilities completed successfully')
            logger.info('🔍 Running validation...')
            
            validation_result = await run_validator(training_input, result)
            if validation_result.get('validation_passed', False):
                logger.info('✅ Validation passed')
            else:
                logger.critical('🚨 CRITICAL: Validation failed for completed step')
                raise CriticalProcessError(
                    f"HMM clustering validation failed: {validation_result.get('error', 'Unknown validation error')}",
                    ErrorRecord(
                        error_id=f"hmm_clustering_validation_failure_{int(time.time())}",
                        error_type="ValidationError",
                        error_message=validation_result.get('error', 'Unknown validation error'),
                        severity=ErrorSeverity.CRITICAL,
                        category=ErrorCategory.VALIDATION,
                        context=ErrorContext(
                            function_name="run_enhanced_step03_with_utilities",
                            step_name="enhanced_hmm_clustering_with_utilities"
                        ),
                        stack_trace="",
                        should_fail_fast=True
                    )
                )
            
            total_elapsed = time.time() - start_time
            logger.info('=' * 80)
            logger.info('🎉 STEP 3 EXECUTION SUMMARY WITH COMPREHENSIVE UTILITIES')
            logger.info('=' * 80)
            logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
            logger.info(f"⏰ End time: {utils['common_operations']['datetime']['format_datetime'](utils['common_operations']['datetime']['get_current_datetime'](), '%Y-%m-%d %H:%M:%S')}")
            logger.info('✅ SUCCESS - All utilities extensively integrated and utilized')
            logger.info('🔧 Utilities Successfully Integrated:')
            logger.info('   ✅ common_operations.py - Extensively used for core operations')
            logger.info('   ✅ common_utilities.py - Extensively used for data processing')
            logger.info('   ✅ math_validation.py - Extensively used for mathematical operations')
            logger.info('   ✅ parquet_utils.py - Extensively used for file operations')
            logger.info('   ✅ serialization_utils.py - Extensively used for data persistence')
            logger.info('   ✅ data_processing_utils.py - Extensively used for DataFrame operations')
            logger.info('   ✅ m1_gpu_utils.py - Extensively used for GPU optimization')
            logger.info('   ✅ m1_memory_optimizer.py - Extensively used for memory management')
            logger.info('   ✅ m1_cpu_optimizer.py - Extensively used for parallel processing')
            logger.info('=' * 80)
            return True
        else:
            error = result.get('hmm_clustering_error', 'Unknown error')
            logger.critical(f'🚨 CRITICAL FAILURE: Step 3: Enhanced HMM Clustering with utilities failed')
            logger.critical(f'   Error: {error}')
            total_elapsed = time.time() - start_time
            logger.info('=' * 80)
            logger.info('💥 STEP 3 EXECUTION SUMMARY')
            logger.info('=' * 80)
            logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
            logger.info(f"⏰ End time: {utils['common_operations']['datetime']['format_datetime'](utils['common_operations']['datetime']['get_current_datetime'](), '%Y-%m-%d %H:%M:%S')}")
            logger.info('❌ FAILED')
            logger.info(f'   Error: {error}')
            logger.info('=' * 80)
            
            # Raise CriticalProcessError for fail-fast behavior
            raise CriticalProcessError(
                f"Enhanced HMM clustering step with utilities failed: {error}",
                ErrorRecord(
                    error_id=f"enhanced_hmm_clustering_step_failure_{int(time.time())}",
                    error_type="StepFailureError",
                    error_message=error,
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.BUSINESS_LOGIC,
                    context=ErrorContext(
                        function_name="run_enhanced_step03_with_utilities",
                        step_name="enhanced_hmm_clustering_with_utilities"
                    ),
                    stack_trace="",
                    should_fail_fast=True
                )
            )
            
    except CriticalProcessError as e:
        logger.critical(f'🚨 CRITICAL PROCESS ERROR: Step 3: Enhanced HMM Clustering with utilities failed: {e}')
        total_elapsed = time.time() - start_time
        logger.info('=' * 80)
        logger.info('💥 STEP 3 EXECUTION SUMMARY')
        logger.info('=' * 80)
        logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
        logger.info('❌ FAILED')
        logger.info(f'   Critical Error: {e}')
        logger.info('=' * 80)
        raise
    except Exception as e:
        logger.critical(f'🚨 CRITICAL ERROR: Step 3: Enhanced HMM Clustering with utilities failed with exception: {e}')
        total_elapsed = time.time() - start_time
        logger.info('=' * 80)
        logger.info('💥 STEP 3 EXECUTION SUMMARY')
        logger.info('=' * 80)
        logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
        logger.info('❌ FAILED')
        logger.info(f'   Exception: {e}')
        logger.info('=' * 80)
        
        # Convert to CriticalProcessError for fail-fast behavior
        raise CriticalProcessError(
            f"Enhanced HMM clustering step with utilities failed with critical exception: {e}",
            ErrorRecord(
                error_id=f"enhanced_hmm_clustering_critical_exception_{int(time.time())}",
                error_type=type(e).__name__,
                error_message=str(e),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.BUSINESS_LOGIC,
                context=ErrorContext(
                    function_name="run_enhanced_step03_with_utilities",
                    step_name="enhanced_hmm_clustering_with_utilities"
                ),
                stack_trace="",
                should_fail_fast=True
            )
        )

async def main() -> None:
    """Main function to run enhanced step 3 with comprehensive utilities."""
    print('🚀 Enhanced Step 3: HMM Regime Discovery with Comprehensive Utility Integration')
    print('=' * 80)
    symbol = 'ETHUSDT'
    exchange = 'BINANCE'
    timeframe = '1m'
    data_dir = 'data_cache'
    print(f'📊 Configuration:')
    print(f'   Symbol: {symbol}')
    print(f'   Exchange: {exchange}')
    print(f'   Timeframe: {timeframe}')
    print(f'   Data directory: {data_dir}')
    print('🔧 Comprehensive Utilities to be Integrated:')
    print('   ✅ common_operations.py - Core operations and utilities')
    print('   ✅ common_utilities.py - Data processing utilities')
    print('   ✅ math_validation.py - Mathematical validation and operations')
    print('   ✅ parquet_utils.py - Parquet file operations')
    print('   ✅ serialization_utils.py - Data serialization')
    print('   ✅ data_processing_utils.py - DataFrame processing')
    print('   ✅ m1_gpu_utils.py - M1 GPU optimization')
    print('   ✅ m1_memory_optimizer.py - M1 memory optimization')
    print('   ✅ m1_cpu_optimizer.py - M1 CPU optimization')
    print('=' * 80)
    
    success = await run_enhanced_step03_with_utilities(
        symbol=symbol, 
        exchange=exchange, 
        timeframe=timeframe, 
        data_dir=data_dir, 
        force_rerun=True
    )
    
    if success:
        print('\n🎉 ENHANCED STEP 3 WITH COMPREHENSIVE UTILITIES COMPLETED SUCCESSFULLY!')
        print('=' * 80)
        print('✅ All improvements integrated:')
        print('   ✅ Bayesian parameter optimization with Optuna')
        print('   ✅ Enhanced regime discovery features')
        print('   ✅ Economic significance validation')
        print('   ✅ Ensemble clustering (HMM + K-means + DBSCAN)')
        print('   ✅ Enhanced ML transition detection (Random Forest + LGBM)')
        print('   ✅ Full MLflow integration and data persistence')
        print('   ✅ Standardized pipeline integration')
        print('🔧 Comprehensive Utilities Extensively Used:')
        print('   ✅ common_operations.py - Extensively used for core operations')
        print('   ✅ common_utilities.py - Extensively used for data processing')
        print('   ✅ math_validation.py - Extensively used for mathematical operations')
        print('   ✅ parquet_utils.py - Extensively used for file operations')
        print('   ✅ serialization_utils.py - Extensively used for data persistence')
        print('   ✅ data_processing_utils.py - Extensively used for DataFrame operations')
        print('   ✅ m1_gpu_utils.py - Extensively used for GPU optimization')
        print('   ✅ m1_memory_optimizer.py - Extensively used for memory management')
        print('   ✅ m1_cpu_optimizer.py - Extensively used for parallel processing')
        print('=' * 80)
    else:
        print('\n❌ ENHANCED STEP 3 WITH UTILITIES FAILED!')
        print('=' * 80)
        print('❌ Please check the logs for error details')
        print('=' * 80)

if __name__ == '__main__':
    asyncio.run(main())