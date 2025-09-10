"""
Step 16 Enhanced Confidence Calibration

This module provides the main enhanced confidence calibration implementation with:
- Comprehensive optimization integration
- Fast-fail validation mechanisms
- Memory optimization
- Enhanced algorithms
- Parallel processing
- Intelligent caching
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
import time
import json
import logging
from pathlib import Path
from datetime import datetime
import gc

# Import existing utilities and core modules
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    get_current_datetime, format_datetime, safe_sleep, safe_gather,
    safe_mean, safe_std, safe_float, safe_int, validate_dataframe_schema,
    validate_data_quality, optimize_dataframe_dtypes, safe_read_parquet,
    safe_to_parquet, get_logger, setup_basic_logging
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_weighted_average,
    MathValidationError
)
from src.utils.parquet_utils import ParquetUtils, get_parquet_utils
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, cached
)
from src.core.errors import (
    ValidationError, DataIntegrityError, BusinessRuleError, AppError
)

from .step16_optimization_utilities import (
    FastFailValidator, ParameterValidator, MemoryOptimizer,
    EnhancedMatrixOperations, CalibrationQualityMetrics,
    FastFailError, ConvergenceError,
    ConvergenceConfig, CalibrationMetrics, OptimizationLevel
)

from .step16_enhanced_calibration_methods import (
    EnhancedPlattScaling, EnhancedIsotonicRegression, EnhancedTemperatureScaling
)

from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

logger = get_logger(__name__)

class EnhancedStep16ConfidenceCalibration:
    """Enhanced Step 16 Confidence Calibration with comprehensive optimizations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbol = config.get('symbol', 'UNKNOWN')
        self.exchange = config.get('exchange', 'UNKNOWN')
        self.timeframe = config.get('timeframe', 'UNKNOWN')
        
        # Initialize optimization components
        self.validator = FastFailValidator(config)
        self.param_validator = ParameterValidator()
        self.memory_optimizer = MemoryOptimizer(config.get('memory_limit_gb', 8.0))
        self.matrix_ops = EnhancedMatrixOperations(config.get('use_gpu', True))
        self.metrics_calculator = CalibrationQualityMetrics(self.matrix_ops)
        
        # Initialize enhanced calibration methods
        self.platt_scaling = EnhancedPlattScaling(config)
        self.isotonic_regression = EnhancedIsotonicRegression(config)
        self.temperature_scaling = EnhancedTemperatureScaling(config)
        
        # Initialize parquet utilities
        self.parquet_utils = get_parquet_utils()
        
        # Optimization settings
        self.optimization_level = OptimizationLevel(config.get('optimization_level', 'standard'))
        self.enable_parallel_processing = config.get('enable_parallel_processing', True)
        self.enable_caching = config.get('enable_caching', True)
        self.enable_fast_fail = config.get('enable_fast_fail', True)
        
        # Caching
        self.cache = {} if self.enable_caching else None
        
        logger.info(f"Enhanced Step 16 Confidence Calibration initialized for {self.symbol}")
    
    @handles_errors(fallback={'success': False, 'error': 'execution_failed'}, context="enhanced_step16_execution")
    @traced(span_name="enhanced_step16_confidence_calibration")
    @log_execution_time("enhanced_step16_execution")
    async def execute(self, symbol: str, exchange: str, timeframe: str, 
                     data_dir: str, **kwargs) -> Dict[str, Any]:
        """Execute enhanced confidence calibration with comprehensive optimizations."""
        try:
            logger.info(f"🚀 Starting Enhanced Step 16: Confidence Calibration for {symbol}")
            
            # Update configuration
            self.symbol = symbol
            self.exchange = exchange
            self.timeframe = timeframe
            
            # Fast-fail validation
            if self.enable_fast_fail:
                await self._validate_execution_environment(symbol, exchange, timeframe, data_dir)
            
            # Load and validate data
            specialist_data = await self._load_and_validate_specialist_data(
                symbol, exchange, timeframe, data_dir
            )
            
            if specialist_data is None:
                raise FastFailError("Failed to load specialist data")
            
            # Execute per-regime calibration
            calibration_results = await self._execute_per_regime_calibration(
                specialist_data, symbol, exchange, timeframe, data_dir
            )
            
            # Save results
            success = await self._save_calibration_results(
                calibration_results, symbol, exchange, timeframe, data_dir
            )
            
            if success:
                logger.info(f"✅ Enhanced Step 16 completed successfully for {symbol}")
                return {
                    'success': True,
                    'status': 'COMPLETED',
                    'calibration_results': calibration_results,
                    'optimization_level': self.optimization_level.value,
                    'performance_metrics': self._calculate_performance_metrics(calibration_results)
                }
            else:
                raise Exception("Failed to save calibration results")
                
        except (FastFailError, ValidationError, ConvergenceError) as e:
            logger.error(f"❌ Enhanced Step 16 failed with validation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
        except Exception as e:
            logger.error(f"❌ Enhanced Step 16 failed with unexpected error: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'UnexpectedError'
            }
        finally:
            # Cleanup
            self.memory_optimizer.cleanup_memory()
    
    async def _validate_execution_environment(self, symbol: str, exchange: str, 
                                            timeframe: str, data_dir: str) -> None:
        """Fast-fail validation of execution environment."""
        try:
            # Validate inputs
            if not symbol or not exchange or not timeframe:
                raise FastFailError("Missing required parameters: symbol, exchange, timeframe")
            
            # Validate data directory
            data_path = Path(data_dir)
            if not data_path.exists():
                raise FastFailError(f"Data directory does not exist: {data_dir}")
            
            # Check available memory
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb < 2.0:  # Minimum 2GB required
                raise FastFailError(f"Insufficient memory: {available_memory_gb:.1f}GB available")
            
            # Validate configuration
            required_config_keys = ['platt_scaling', 'isotonic_regression', 'temperature_scaling']
            for key in required_config_keys:
                if key not in self.config:
                    raise FastFailError(f"Missing configuration: {key}")
            
            logger.info("✅ Execution environment validation passed")
            
        except FastFailError:
            raise
        except Exception as e:
            raise FastFailError(f"Environment validation failed: {e}")
    
    async def _load_and_validate_specialist_data(self, symbol: str, exchange: str, 
                                               timeframe: str, data_dir: str) -> Optional[Dict[str, Any]]:
        """Load and validate specialist data with comprehensive checks."""
        try:
            # Check cache first
            cache_key = f"specialist_data_{symbol}_{exchange}_{timeframe}"
            if self.cache and cache_key in self.cache:
                logger.info("📋 Using cached specialist data")
                return self.cache[cache_key]
            
            # Load specialist data
            specialist_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_tactician_specialist_training_aggregated.json'
            
            if not specialist_path.exists():
                # Try alternative paths
                alternative_paths = [
                    Path(data_dir) / f'{exchange}_{symbol}_{timeframe}_tactician_specialist.json',
                    Path(data_dir) / 'models' / f'{symbol}_{exchange}_tactician_specialist.pkl'
                ]
                
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        specialist_path = alt_path
                        break
                else:
                    raise FastFailError(f"Specialist data not found: {specialist_path}")
            
            # Load data using safe operations
            if specialist_path.suffix == '.json':
                specialist_data = safe_json_load(specialist_path)
            else:
                # Handle other formats if needed
                raise FastFailError(f"Unsupported file format: {specialist_path.suffix}")
            
            # Validate data structure
            if not isinstance(specialist_data, dict):
                raise ValidationError("Specialist data must be a dictionary")
            
            if 'trained_specialists' not in specialist_data:
                raise ValidationError("Missing 'trained_specialists' in specialist data")
            
            # Cache the data
            if self.cache:
                self.cache[cache_key] = specialist_data
            
            logger.info(f"✅ Loaded specialist data: {len(specialist_data.get('trained_specialists', {}))} specialists")
            return specialist_data
            
        except (FastFailError, ValidationError):
            raise
        except Exception as e:
            raise FastFailError(f"Failed to load specialist data: {e}")
    
    async def _execute_per_regime_calibration(self, specialist_data: Dict[str, Any],
                                            symbol: str, exchange: str, timeframe: str,
                                            data_dir: str) -> Dict[str, Any]:
        """Execute per-regime calibration with enhanced optimization."""
        try:
            trained_specialists = specialist_data.get('trained_specialists', {})
            
            if not trained_specialists:
                raise FastFailError("No trained specialists found for calibration")
            
            # Determine regimes
            regimes = self._identify_regimes(trained_specialists)
            
            # Execute calibration for each regime
            regime_results = {}
            
            if self.enable_parallel_processing and len(regimes) > 1:
                # Parallel processing for multiple regimes
                regime_results = await self._parallel_regime_calibration(
                    regimes, trained_specialists, symbol, exchange, timeframe, data_dir
                )
            else:
                # Sequential processing
                for regime_id in regimes:
                    regime_result = await self._calibrate_single_regime(
                        regime_id, trained_specialists, symbol, exchange, timeframe, data_dir
                    )
                    regime_results[regime_id] = regime_result
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_calibration_metrics(regime_results)
            
            return {
                'regime_results': regime_results,
                'overall_metrics': overall_metrics,
                'total_regimes': len(regimes),
                'total_specialists': len(trained_specialists),
                'calibration_timestamp': datetime.now().isoformat(),
                'optimization_level': self.optimization_level.value
            }
            
        except Exception as e:
            logger.error(f"Per-regime calibration failed: {e}")
            raise
    
    def _identify_regimes(self, trained_specialists: Dict[str, Any]) -> List[str]:
        """Identify regimes from trained specialists."""
        regimes = set()
        
        for specialist_name, specialist_data in trained_specialists.items():
            # Extract regime information from specialist name or data
            if 'regime' in specialist_name.lower():
                # Extract regime from name
                parts = specialist_name.split('_')
                for part in parts:
                    if 'regime' in part.lower():
                        regime_id = part.split('regime')[-1] if 'regime' in part else part
                        regimes.add(regime_id)
                        break
            else:
                # Default regime
                regimes.add('default')
        
        return list(regimes) if regimes else ['default']
    
    async def _parallel_regime_calibration(self, regimes: List[str], trained_specialists: Dict[str, Any],
                                         symbol: str, exchange: str, timeframe: str,
                                         data_dir: str) -> Dict[str, Any]:
        """Execute parallel regime calibration."""
        try:
            # Create tasks for parallel execution
            tasks = []
            for regime_id in regimes:
                task = self._calibrate_single_regime(
                    regime_id, trained_specialists, symbol, exchange, timeframe, data_dir
                )
                tasks.append((regime_id, task))
            
            # Execute in parallel
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # Process results
            regime_results = {}
            for i, (regime_id, result) in enumerate(zip(regimes, results)):
                if isinstance(result, Exception):
                    logger.error(f"Regime {regime_id} calibration failed: {result}")
                    regime_results[regime_id] = {
                        'success': False,
                        'error': str(result),
                        'error_type': type(result).__name__
                    }
                else:
                    regime_results[regime_id] = result
            
            return regime_results
            
        except Exception as e:
            logger.error(f"Parallel regime calibration failed: {e}")
            raise
    
    async def _calibrate_single_regime(self, regime_id: str, trained_specialists: Dict[str, Any],
                                     symbol: str, exchange: str, timeframe: str,
                                     data_dir: str) -> Dict[str, Any]:
        """Calibrate a single regime with enhanced methods."""
        try:
            logger.info(f"🔧 Calibrating regime {regime_id}")
            
            # Filter specialists for this regime
            regime_specialists = self._filter_specialists_for_regime(trained_specialists, regime_id)
            
            if not regime_specialists:
                raise FastFailError(f"No specialists found for regime {regime_id}")
            
            # Execute calibration methods
            calibration_methods = {}
            
            # Platt Scaling
            try:
                platt_result = await self._execute_platt_scaling(regime_specialists, regime_id)
                calibration_methods['platt_scaling'] = platt_result
            except Exception as e:
                logger.warning(f"Platt scaling failed for regime {regime_id}: {e}")
            
            # Isotonic Regression
            try:
                isotonic_result = await self._execute_isotonic_regression(regime_specialists, regime_id)
                calibration_methods['isotonic_regression'] = isotonic_result
            except Exception as e:
                logger.warning(f"Isotonic regression failed for regime {regime_id}: {e}")
            
            # Temperature Scaling
            try:
                temperature_result = await self._execute_temperature_scaling(regime_specialists, regime_id)
                calibration_methods['temperature_scaling'] = temperature_result
            except Exception as e:
                logger.warning(f"Temperature scaling failed for regime {regime_id}: {e}")
            
            if not calibration_methods:
                raise FastFailError(f"All calibration methods failed for regime {regime_id}")
            
            # Calculate regime metrics
            regime_metrics = self._calculate_regime_metrics(calibration_methods)
            
            return {
                'regime_id': regime_id,
                'calibration_methods': calibration_methods,
                'regime_metrics': regime_metrics,
                'specialists_count': len(regime_specialists),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Single regime calibration failed for regime {regime_id}: {e}")
            return {
                'regime_id': regime_id,
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def _filter_specialists_for_regime(self, trained_specialists: Dict[str, Any], regime_id: str) -> Dict[str, Any]:
        """Filter specialists for a specific regime."""
        if regime_id == 'default':
            return trained_specialists
        
        regime_specialists = {}
        for specialist_name, specialist_data in trained_specialists.items():
            if regime_id in specialist_name.lower():
                regime_specialists[specialist_name] = specialist_data
        
        return regime_specialists
    
    async def _execute_platt_scaling(self, specialists: Dict[str, Any], regime_id: str) -> Dict[str, Any]:
        """Execute enhanced Platt scaling calibration."""
        try:
            # Extract data from specialists
            all_probabilities = []
            all_labels = []
            
            for specialist_name, specialist_data in specialists.items():
                if 'train_probabilities' in specialist_data and 'train_labels' in specialist_data:
                    all_probabilities.extend(specialist_data['train_probabilities'])
                    all_labels.extend(specialist_data['train_labels'])
            
            if not all_probabilities or not all_labels:
                raise FastFailError(f"No training data available for Platt scaling in regime {regime_id}")
            
            # Convert to numpy arrays
            probabilities = np.array(all_probabilities)
            labels = np.array(all_labels)
            
            # Execute calibration
            result = self.platt_scaling.calibrate(probabilities, labels, regime_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Platt scaling execution failed for regime {regime_id}: {e}")
            raise
    
    async def _execute_isotonic_regression(self, specialists: Dict[str, Any], regime_id: str) -> Dict[str, Any]:
        """Execute enhanced isotonic regression calibration."""
        try:
            # Extract data from specialists
            all_probabilities = []
            all_labels = []
            
            for specialist_name, specialist_data in specialists.items():
                if 'train_probabilities' in specialist_data and 'train_labels' in specialist_data:
                    all_probabilities.extend(specialist_data['train_probabilities'])
                    all_labels.extend(specialist_data['train_labels'])
            
            if not all_probabilities or not all_labels:
                raise FastFailError(f"No training data available for isotonic regression in regime {regime_id}")
            
            # Convert to numpy arrays
            probabilities = np.array(all_probabilities)
            labels = np.array(all_labels)
            
            # Execute calibration
            result = self.isotonic_regression.calibrate(probabilities, labels, regime_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Isotonic regression execution failed for regime {regime_id}: {e}")
            raise
    
    async def _execute_temperature_scaling(self, specialists: Dict[str, Any], regime_id: str) -> Dict[str, Any]:
        """Execute enhanced temperature scaling calibration."""
        try:
            # Extract data from specialists
            all_probabilities = []
            all_labels = []
            
            for specialist_name, specialist_data in specialists.items():
                if 'train_probabilities' in specialist_data and 'train_labels' in specialist_data:
                    all_probabilities.extend(specialist_data['train_probabilities'])
                    all_labels.extend(specialist_data['train_labels'])
            
            if not all_probabilities or not all_labels:
                raise FastFailError(f"No training data available for temperature scaling in regime {regime_id}")
            
            # Convert to numpy arrays
            probabilities = np.array(all_probabilities)
            labels = np.array(all_labels)
            
            # Execute calibration
            result = self.temperature_scaling.calibrate(probabilities, labels, regime_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Temperature scaling execution failed for regime {regime_id}: {e}")
            raise
    
    def _calculate_regime_metrics(self, calibration_methods: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a regime."""
        try:
            metrics = {
                'methods_used': list(calibration_methods.keys()),
                'best_method': None,
                'overall_performance': 0.0,
                'calibration_quality': 'unknown'
            }
            
            # Find best method based on reliability score
            best_reliability = 0.0
            for method_name, method_result in calibration_methods.items():
                if 'calibration_metrics' in method_result:
                    reliability = method_result['calibration_metrics'].get('reliability_score', 0.0)
                    if reliability > best_reliability:
                        best_reliability = reliability
                        metrics['best_method'] = method_name
            
            metrics['overall_performance'] = best_reliability
            
            # Determine calibration quality
            if best_reliability >= 0.9:
                metrics['calibration_quality'] = 'excellent'
            elif best_reliability >= 0.8:
                metrics['calibration_quality'] = 'good'
            elif best_reliability >= 0.7:
                metrics['calibration_quality'] = 'fair'
            else:
                metrics['calibration_quality'] = 'poor'
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate regime metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_calibration_metrics(self, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall calibration metrics across all regimes."""
        try:
            successful_regimes = [r for r in regime_results.values() if r.get('success', False)]
            
            if not successful_regimes:
                return {'error': 'No successful regime calibrations'}
            
            # Aggregate metrics
            total_regimes = len(regime_results)
            successful_count = len(successful_regimes)
            success_rate = successful_count / total_regimes
            
            # Calculate average performance
            performances = []
            for regime in successful_regimes:
                if 'regime_metrics' in regime:
                    performance = regime['regime_metrics'].get('overall_performance', 0.0)
                    performances.append(performance)
            
            avg_performance = np.mean(performances) if performances else 0.0
            
            # Determine overall quality
            if avg_performance >= 0.9 and success_rate >= 0.9:
                overall_quality = 'excellent'
            elif avg_performance >= 0.8 and success_rate >= 0.8:
                overall_quality = 'good'
            elif avg_performance >= 0.7 and success_rate >= 0.7:
                overall_quality = 'fair'
            else:
                overall_quality = 'poor'
            
            return {
                'total_regimes': total_regimes,
                'successful_regimes': successful_count,
                'success_rate': success_rate,
                'average_performance': avg_performance,
                'overall_quality': overall_quality,
                'performance_distribution': {
                    'min': min(performances) if performances else 0.0,
                    'max': max(performances) if performances else 0.0,
                    'std': np.std(performances) if performances else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate overall metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_metrics(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for the entire calibration process."""
        try:
            start_time = time.time()
            
            # Calculate timing metrics
            timing_metrics = {
                'total_calibration_time': time.time() - start_time,
                'regimes_processed': calibration_results.get('total_regimes', 0),
                'specialists_processed': calibration_results.get('total_specialists', 0)
            }
            
            # Calculate efficiency metrics
            if timing_metrics['regimes_processed'] > 0:
                timing_metrics['time_per_regime'] = timing_metrics['total_calibration_time'] / timing_metrics['regimes_processed']
            
            if timing_metrics['specialists_processed'] > 0:
                timing_metrics['time_per_specialist'] = timing_metrics['total_calibration_time'] / timing_metrics['specialists_processed']
            
            # Memory usage
            import psutil
            memory_metrics = {
                'peak_memory_usage_gb': psutil.Process().memory_info().rss / (1024**3),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3)
            }
            
            return {
                'timing_metrics': timing_metrics,
                'memory_metrics': memory_metrics,
                'optimization_level': self.optimization_level.value,
                'parallel_processing_enabled': self.enable_parallel_processing,
                'caching_enabled': self.enable_caching,
                'fast_fail_enabled': self.enable_fast_fail
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return {'error': str(e)}
    
    async def _save_calibration_results(self, calibration_results: Dict[str, Any],
                                      symbol: str, exchange: str, timeframe: str,
                                      data_dir: str) -> bool:
        """Save calibration results with comprehensive metadata."""
        try:
            # Create output directory
            output_dir = Path(data_dir) / 'training' / 'calibration_results'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main results
            results_file = output_dir / f'{exchange}_{symbol}_{timeframe}_enhanced_confidence_calibration.json'
            
            # Add metadata
            enhanced_results = {
                'calibration_results': calibration_results,
                'metadata': {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'timestamp': datetime.now().isoformat(),
                    'version': '2.0_enhanced',
                    'optimization_level': self.optimization_level.value,
                    'data_protection_enabled': True,
                    'fast_fail_enabled': self.enable_fast_fail,
                    'parallel_processing_enabled': self.enable_parallel_processing,
                    'caching_enabled': self.enable_caching
                }
            }
            
            # Save to file using safe operations
            safe_json_dump(enhanced_results, results_file, indent=2, default=str)
            
            logger.info(f"✅ Enhanced calibration results saved: {results_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save calibration results: {e}")
            return False

# Enhanced entry points
@handles_errors(fallback={'success': False, 'error': 'run_enhanced_step16_failed'}, context="run_enhanced_step16")
@traced(span_name="run_enhanced_step16")
@log_execution_time("run_enhanced_step16")
async def run_enhanced_step16(symbol: str, exchange: str, timeframe: str, 
                            data_dir: str = None, **kwargs) -> Dict[str, Any]:
    """Enhanced entry point for Step 16: Confidence Calibration."""
    if data_dir is None:
        data_dir = f'data_cache/{exchange}_{symbol}'
    
    # Default enhanced configuration
    config = {
        'symbol': symbol,
        'exchange': exchange,
        'timeframe': timeframe,
        'optimization_level': 'aggressive',
        'memory_limit_gb': 8.0,
        'use_gpu': True,
        'enable_parallel_processing': True,
        'enable_caching': True,
        'enable_fast_fail': True,
        'platt_scaling': {
            'max_iterations': 2000,
            'learning_rate': 0.01,
            'regularization': 0.01,
            'early_stopping': True,
            'validation_split': 0.2
        },
        'isotonic_regression': {
            'out_of_bounds': 'clip',
            'increasing': True,
            'cross_validation': True,
            'cv_folds': 5
        },
        'temperature_scaling': {
            'temperature_range': [0.1, 10.0],
            'optimization_method': 'multi_start',
            'cross_validation': True,
            'validation_split': 0.2
        },
        'min_samples': 100,
        'max_missing_ratio': 0.1,
        'min_class_balance': 0.05,
        **kwargs
    }
    
    logger.info(f"🚀 Starting Enhanced Step 16: Confidence Calibration for {symbol}")
    
    step = EnhancedStep16ConfidenceCalibration(config)
    result = await step.execute(symbol, exchange, timeframe, data_dir, **kwargs)
    
    if result['success']:
        logger.info(f"✅ Enhanced Step 16 completed successfully for {symbol}")
    else:
        logger.error(f"❌ Enhanced Step 16 failed for {symbol}: {result.get('error', 'Unknown error')}")
    
    return result

@handles_errors(fallback=False, context="run_step")
@traced(span_name="run_step")
@log_execution_time("run_step")
async def run_step(symbol: str, exchange: str, timeframe: str, 
                  data_dir: str = None, **kwargs) -> bool:
    """Standard entry point for Step 16: Enhanced Confidence Calibration."""
    result = await run_enhanced_step16(symbol, exchange, timeframe, data_dir, **kwargs)
    return result['success']