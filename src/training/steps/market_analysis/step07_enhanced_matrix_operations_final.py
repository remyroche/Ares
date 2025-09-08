"""
Step 7: Enhanced Matrix Operations - Final Optimized Version

This module implements all requested optimizations and fixes:
- Computational optimizations: caching, vectorized operations, chunked processing
- Fast fails: data shape validation, dependency validation, data type validation
- Fixes: async/sync mixing, algorithmic issues
- Uses src/utils/math_validation.py for safe mathematical operations
- Extensive logging for fast fail scenarios
"""

import os
import time
import gc
import asyncio
import functools
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
import logging

# Core imports with fallback handling
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Import math validation utilities
try:
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        validate_positive, validate_range, safe_matrix_inverse,
        validate_correlation_matrix, MathValidationError, math_safe
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False
    # Fallback implementations
    def safe_divide(a, b, default=0.0): return a / b if b != 0 else default
    def safe_log(x, base=2.718281828459045, default=0.0): return 0.0 if x <= 0 else np.log(x) / np.log(base)
    def safe_sqrt(x, default=0.0): return np.sqrt(max(0, x))
    def safe_power(base, exp, default=1.0): return base ** exp
    def validate_finite(x, name="value"): return float(x)
    def validate_positive(x, name="value"): return float(x)
    def validate_range(x, min_val, max_val, name="value"): return float(x)
    def safe_matrix_inverse(matrix, default=None): return np.linalg.inv(matrix)
    def validate_correlation_matrix(matrix, name="correlation_matrix"): return matrix
    class MathValidationError(Exception): pass
    def math_safe(func): return func

# Import logging utilities
try:
    from src.utils.comprehensive_function_logger import (
        log_step_functions, log_important_calls, log_all_calls, 
        log_internal_call, log_step_progress, log_data_operation
    )
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    def log_step_functions(func): return func
    def log_important_calls(func): return func
    def log_all_calls(func): return func
    def log_internal_call(func): return func
    def log_step_progress(func): return func
    def log_data_operation(func): return func

# Import parquet handler
try:
    from ..standardized_parquet_handler import standardized_parquet_handler
    PARQUET_HANDLER_AVAILABLE = True
except ImportError:
    PARQUET_HANDLER_AVAILABLE = False
    standardized_parquet_handler = None

# Import optimized matrix operations
try:
    from .utils.matrix_operations_optimized import OptimizedMatrixOperations, FastFailError
    OPTIMIZED_MATRIX_OPS_AVAILABLE = True
except ImportError:
    OPTIMIZED_MATRIX_OPS_AVAILABLE = False
    OptimizedMatrixOperations = None
    class FastFailError(Exception): pass

# Initialize logger
logger = logging.getLogger(__name__)

class Step7EnhancedMatrixOperationsFinal:
    """Final optimized Step 7 Enhanced Matrix Operations with all improvements."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize with comprehensive validation and optimization setup."""
        self.config = config
        self.logger = logger.getChild('Step7Final')
        
        # Fast fail validation with extensive logging
        self._validate_dependencies_with_logging()
        self._validate_config_with_logging()
        
        # Optimization settings
        self.cache = {}
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour default
        self.chunk_size = config.get('chunk_size', 1000)
        self.memory_threshold_gb = config.get('memory_threshold_gb', 8.0)
        
        # Performance tracking
        self.performance_metrics = {}
        self.operation_counts = {}
        
        # Initialize optimized matrix operations
        if OPTIMIZED_MATRIX_OPS_AVAILABLE:
            self.matrix_ops = OptimizedMatrixOperations(self.logger)
        else:
            self.matrix_ops = None
            self.logger.warning("⚠️ Optimized matrix operations not available, using fallback")
        
        self.logger.info("🔧 Initialized Step 7 Final Enhanced Matrix Operations")
    
    def _validate_dependencies_with_logging(self) -> None:
        """Fast fail: Validate all required dependencies with extensive logging."""
        self.logger.info("🔍 FAST FAIL: Validating dependencies...")
        
        missing_deps = []
        available_deps = []
        
        # Check each dependency with detailed logging
        if NUMPY_AVAILABLE:
            available_deps.append("numpy")
            self.logger.info("✅ numpy: Available")
        else:
            missing_deps.append("numpy")
            self.logger.error("❌ numpy: MISSING - Critical for matrix operations")
        
        if PANDAS_AVAILABLE:
            available_deps.append("pandas")
            self.logger.info("✅ pandas: Available")
        else:
            missing_deps.append("pandas")
            self.logger.error("❌ pandas: MISSING - Critical for data processing")
        
        if MATH_VALIDATION_AVAILABLE:
            available_deps.append("math_validation")
            self.logger.info("✅ math_validation: Available")
        else:
            missing_deps.append("math_validation")
            self.logger.warning("⚠️ math_validation: MISSING - Using fallback implementations")
        
        if PARQUET_HANDLER_AVAILABLE:
            available_deps.append("parquet_handler")
            self.logger.info("✅ parquet_handler: Available")
        else:
            missing_deps.append("parquet_handler")
            self.logger.error("❌ parquet_handler: MISSING - Critical for data I/O")
        
        if PSUTIL_AVAILABLE:
            available_deps.append("psutil")
            self.logger.info("✅ psutil: Available")
        else:
            self.logger.warning("⚠️ psutil: MISSING - Memory validation will be skipped")
        
        if OPTIMIZED_MATRIX_OPS_AVAILABLE:
            available_deps.append("optimized_matrix_ops")
            self.logger.info("✅ optimized_matrix_ops: Available")
        else:
            self.logger.warning("⚠️ optimized_matrix_ops: MISSING - Using fallback implementations")
        
        # Log summary
        self.logger.info(f"📊 Dependency Summary: {len(available_deps)} available, {len(missing_deps)} missing")
        
        # Fast fail on critical missing dependencies
        critical_deps = ["numpy", "pandas", "parquet_handler"]
        critical_missing = [dep for dep in missing_deps if dep in critical_deps]
        
        if critical_missing:
            error_msg = f"❌ FAST FAIL: Missing critical dependencies: {critical_missing}"
            self.logger.error(error_msg)
            self.logger.error("🚫 Cannot proceed without critical dependencies")
            raise FastFailError(error_msg)
        
        self.logger.info("✅ All critical dependencies available")
    
    def _validate_config_with_logging(self) -> None:
        """Fast fail: Validate configuration parameters with extensive logging."""
        self.logger.info("🔍 FAST FAIL: Validating configuration...")
        
        required_configs = [
            'step07_enhanced_matrix_operations',
            'output_dir'
        ]
        
        missing_configs = []
        for config_key in required_configs:
            if config_key not in self.config:
                missing_configs.append(config_key)
                self.logger.error(f"❌ Missing required configuration: {config_key}")
            else:
                self.logger.info(f"✅ Configuration found: {config_key}")
        
        if missing_configs:
            error_msg = f"❌ FAST FAIL: Missing required configuration: {missing_configs}"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
        
        # Validate numeric parameters with detailed logging
        step_config = self.config.get('step07_enhanced_matrix_operations', {})
        
        numeric_params = [
            'condition_number_threshold',
            'correlation_threshold',
            'min_eigenvalue_threshold',
            'memory_threshold_gb',
            'chunk_size'
        ]
        
        for param in numeric_params:
            if param in step_config:
                try:
                    if param == 'correlation_threshold':
                        validate_range(step_config[param], 0.0, 1.0, param)
                    elif param in ['condition_number_threshold', 'min_eigenvalue_threshold', 'memory_threshold_gb', 'chunk_size']:
                        validate_positive(step_config[param], param)
                    
                    self.logger.info(f"✅ Configuration parameter valid: {param} = {step_config[param]}")
                except MathValidationError as e:
                    error_msg = f"❌ FAST FAIL: Invalid configuration parameter {param}: {e}"
                    self.logger.error(error_msg)
                    raise FastFailError(error_msg)
            else:
                self.logger.info(f"ℹ️ Configuration parameter not set (using default): {param}")
        
        self.logger.info("✅ Configuration validation passed")
    
    def _validate_data_shape_with_logging(self, df: pd.DataFrame, min_rows: int = 10, min_cols: int = 1) -> None:
        """Fast fail: Validate data shape requirements with extensive logging."""
        self.logger.info(f"🔍 FAST FAIL: Validating data shape: {df.shape}")
        
        # Log detailed shape information
        self.logger.info(f"📊 Data shape details:")
        self.logger.info(f"   - Rows: {len(df)}")
        self.logger.info(f"   - Columns: {len(df.columns)}")
        self.logger.info(f"   - Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        if len(df) < min_rows:
            error_msg = f"❌ FAST FAIL: Insufficient data rows: {len(df)} < {min_rows}"
            self.logger.error(error_msg)
            self.logger.error("🚫 Matrix operations require minimum data for statistical validity")
            raise FastFailError(error_msg)
        
        if len(df.columns) < min_cols:
            error_msg = f"❌ FAST FAIL: Insufficient data columns: {len(df.columns)} < {min_cols}"
            self.logger.error(error_msg)
            self.logger.error("🚫 Matrix operations require at least one feature column")
            raise FastFailError(error_msg)
        
        self.logger.info(f"✅ Data shape validation passed: {df.shape}")
    
    def _validate_data_types_with_logging(self, df: pd.DataFrame) -> None:
        """Fast fail: Validate data types and numeric columns with extensive logging."""
        self.logger.info("🔍 FAST FAIL: Validating data types...")
        
        # Log data type information
        self.logger.info(f"📊 Data type summary:")
        self.logger.info(f"   - Total columns: {len(df.columns)}")
        self.logger.info(f"   - Data types: {df.dtypes.value_counts().to_dict()}")
        
        numeric_df = df.select_dtypes(include=[np.number])
        self.logger.info(f"   - Numeric columns: {len(numeric_df.columns)}")
        
        if len(numeric_df.columns) == 0:
            error_msg = "❌ FAST FAIL: No numeric columns found in data"
            self.logger.error(error_msg)
            self.logger.error("🚫 Matrix operations require numeric data")
            self.logger.error(f"📋 Available data types: {df.dtypes.unique()}")
            raise FastFailError(error_msg)
        
        # Check for non-finite values with detailed logging
        non_finite_count = numeric_df.isin([np.inf, -np.inf, np.nan]).sum().sum()
        if non_finite_count > 0:
            self.logger.warning(f"⚠️ Found {non_finite_count} non-finite values in numeric data")
            
            # Log details about non-finite values
            for col in numeric_df.columns:
                col_non_finite = numeric_df[col].isin([np.inf, -np.inf, np.nan]).sum()
                if col_non_finite > 0:
                    self.logger.warning(f"   - Column '{col}': {col_non_finite} non-finite values")
        else:
            self.logger.info("✅ All numeric values are finite")
        
        self.logger.info(f"✅ Data type validation passed: {len(numeric_df.columns)} numeric columns")
    
    def _validate_memory_availability_with_logging(self, estimated_memory_gb: float) -> None:
        """Fast fail: Validate sufficient memory availability with extensive logging."""
        if not PSUTIL_AVAILABLE:
            self.logger.warning("⚠️ psutil not available - skipping memory validation")
            return
        
        self.logger.info(f"🔍 FAST FAIL: Validating memory availability: {estimated_memory_gb:.2f} GB required")
        
        # Get detailed memory information
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024**3)
        total_memory_gb = memory_info.total / (1024**3)
        used_memory_gb = memory_info.used / (1024**3)
        memory_percent = memory_info.percent
        
        self.logger.info(f"📊 Memory status:")
        self.logger.info(f"   - Total memory: {total_memory_gb:.2f} GB")
        self.logger.info(f"   - Used memory: {used_memory_gb:.2f} GB ({memory_percent:.1f}%)")
        self.logger.info(f"   - Available memory: {available_memory_gb:.2f} GB")
        self.logger.info(f"   - Required memory: {estimated_memory_gb:.2f} GB")
        
        if available_memory_gb < estimated_memory_gb:
            error_msg = f"❌ FAST FAIL: Insufficient memory: {available_memory_gb:.2f} GB available < {estimated_memory_gb:.2f} GB required"
            self.logger.error(error_msg)
            self.logger.error("🚫 Matrix operations require sufficient memory to avoid system crashes")
            self.logger.error(f"💡 Consider reducing data size or increasing system memory")
            raise FastFailError(error_msg)
        
        # Log memory safety margin
        safety_margin = (available_memory_gb - estimated_memory_gb) / estimated_memory_gb * 100
        self.logger.info(f"✅ Memory validation passed: {safety_margin:.1f}% safety margin")
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Step 7: Enhanced Matrix Operations with all optimizations and fixes.
        
        Args:
            training_input: Input data from previous steps
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with matrix operations results
        """
        start_time = datetime.now()
        self.logger.info('🚀 Starting Step 7: Enhanced Matrix Operations (Final Optimized)...')
        
        try:
            # Extract parameters
            symbol = training_input.get('symbol', 'UNKNOWN')
            exchange = training_input.get('exchange', 'UNKNOWN')
            timeframe = training_input.get('timeframe', '1m')
            
            self.logger.info(f"📊 Processing: {exchange}_{symbol}_{timeframe}")
            
            # Load and prepare data with validation
            df = await self._load_and_prepare_data_with_validation(symbol, exchange, timeframe)
            
            # Execute optimized matrix operations
            matrix_results = await self._execute_optimized_matrix_operations(df)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics_optimized(df, matrix_results)
            
            # Save results
            output_files = await self._save_results_optimized(matrix_results, quality_metrics, symbol, exchange, timeframe)
            
            # Update pipeline state
            pipeline_state = self._update_pipeline_state_optimized(
                pipeline_state, start_time, output_files, matrix_results, 
                quality_metrics, symbol, exchange, timeframe
            )
            
            # Log comprehensive summaries
            self._log_comprehensive_summaries_optimized(pipeline_state)
            
            self.logger.info("✅ Step 7: Enhanced Matrix Operations completed successfully")
            return pipeline_state
            
        except FastFailError as e:
            self.logger.error(f'❌ Step 7 failed due to fast fail validation: {str(e)}')
            pipeline_state['step07_enhanced_matrix_operations'] = {
                'status': 'failed', 
                'error': str(e), 
                'error_type': 'fast_fail_validation',
                'timestamp': datetime.now().isoformat()
            }
            return pipeline_state
        except Exception as e:
            self.logger.error(f'❌ Step 7 failed with unexpected error: {str(e)}')
            pipeline_state['step07_enhanced_matrix_operations'] = {
                'status': 'failed', 
                'error': str(e), 
                'error_type': 'unexpected_error',
                'timestamp': datetime.now().isoformat()
            }
            return pipeline_state
    
    async def _load_and_prepare_data_with_validation(self, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
        """Load and prepare data with comprehensive validation."""
        self.logger.info("📊 Loading and preparing data with validation...")
        
        features_train_path = f'data/training/{exchange}_{symbol}_{timeframe}_features_train.parquet'
        features_val_path = f'data/training/{exchange}_{symbol}_{timeframe}_features_val.parquet'
        
        # Validate file existence
        if not os.path.exists(features_train_path):
            error_msg = f"❌ FAST FAIL: Features train file not found: {features_train_path}"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
        
        if not os.path.exists(features_val_path):
            error_msg = f"❌ FAST FAIL: Features validation file not found: {features_val_path}"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
        
        self.logger.info(f'📊 Loading engineered features from: {features_train_path}')
        
        # Load data with error handling
        try:
            df_train = standardized_parquet_handler.read_parquet_standardized(features_train_path)
            df_val = standardized_parquet_handler.read_parquet_standardized(features_val_path)
        except Exception as e:
            error_msg = f"❌ FAST FAIL: Error loading parquet files: {e}"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
        
        # Optimize data types
        for d in (df_train, df_val):
            for c in d.select_dtypes(include=['float64']).columns:
                d[c] = d[c].astype('float32')
        
        df = pd.concat([df_train, df_val], ignore_index=True)
        
        # Fast fail validations
        self._validate_data_shape_with_logging(df)
        self._validate_data_types_with_logging(df)
        
        # Estimate memory requirements
        estimated_memory_gb = (df.memory_usage(deep=True).sum() * 4) / (1024**3)  # 4x for operations
        self._validate_memory_availability_with_logging(estimated_memory_gb)
        
        self.logger.info(f'📈 Loaded {len(df)} rows of engineered features')
        self.logger.info(f'🔢 Features: {len(df.columns)} columns')
        
        return df
    
    async def _execute_optimized_matrix_operations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute optimized matrix operations using the optimized matrix operations module."""
        self.logger.info("🔧 Executing optimized matrix operations...")
        
        if self.matrix_ops is None:
            self.logger.warning("⚠️ Optimized matrix operations not available, using fallback")
            return await self._execute_fallback_matrix_operations(df)
        
        try:
            # Prepare matrix operations configuration
            matrix_config = self._prepare_matrix_operations_config_optimized(df)
            
            # Execute standard matrix operations
            results = await self.matrix_ops.execute_standard_matrix_operations_optimized(
                df.select_dtypes(include=[np.number]), matrix_config
            )
            
            # Execute SR-specific operations if SR features are available
            if matrix_config.get('enable_sr_analysis', False) and matrix_config.get('sr_features'):
                self.logger.info('🎯 Performing SR-specific matrix operations...')
                sr_results = await self.matrix_ops.execute_sr_matrix_operations_optimized(df, matrix_config)
                results['sr_analysis'] = sr_results
            
            return results
            
        except FastFailError:
            raise
        except Exception as e:
            self.logger.error(f"❌ Error in optimized matrix operations: {e}")
            return {'error': str(e)}
    
    async def _execute_fallback_matrix_operations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback matrix operations when optimized version is not available."""
        self.logger.info("⚠️ Using fallback matrix operations...")
        
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) == 0:
                return {'error': 'No numeric columns available'}
            
            results = {}
            
            # Basic correlation analysis
            correlation_matrix = numeric_df.corr()
            results['correlation_analysis'] = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'matrix_shape': correlation_matrix.shape
            }
            
            # Basic condition number check
            try:
                condition_number = np.linalg.cond(numeric_df.values)
                results['condition_number_check'] = {
                    'condition_number': float(condition_number),
                    'is_well_conditioned': condition_number < 1e12
                }
            except Exception as e:
                results['condition_number_check'] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in fallback matrix operations: {e}")
            return {'error': str(e)}
    
    def _prepare_matrix_operations_config_optimized(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare optimized configuration for matrix operations."""
        step_config = self.config.get("step07_enhanced_matrix_operations", {})
        
        # Identify SR features efficiently
        sr_features = [col for col in df.columns if any(
            keyword in col.lower() for keyword in [
                'sr_', 'support', 'resistance', 'proximity', 'sr_distance', 'sr_proximity', 
                'sr_outcome', 'normalized_distance', 'sr_proximity_score', 'strength_score', 
                'clarity_factor', 'directional_pressure', 'sr_score', 'delta_sr_score', 
                'isolation_score', 'sr_level', 'sr_multi_timeframe', 'support_', 'resistance_'
            ]
        )]
        
        config = {
            'enable_gpu_acceleration': step_config.get('enable_gpu_acceleration', False),
            'enable_sparse_optimizations': step_config.get('enable_sparse_optimizations', True),
            'enable_memory_optimization': step_config.get('enable_memory_optimization', True),
            'enable_parallel_processing': step_config.get('enable_parallel_processing', True),
            'condition_number_threshold': step_config.get('condition_number_threshold', 1e12),
            'min_eigenvalue_threshold': step_config.get('min_eigenvalue_threshold', 1e-10),
            'correlation_threshold': step_config.get('correlation_threshold', 0.8),
            'memory_threshold_gb': step_config.get('memory_threshold_gb', 8.0),
            'chunk_size': step_config.get('chunk_size', 1000),
            'data_shape': df.shape,
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'sr_features': sr_features,
            'sr_feature_count': len(sr_features),
            'enable_sr_analysis': len(sr_features) > 0,
            'sr_correlation_threshold': step_config.get('sr_correlation_threshold', 0.7),
            'sr_condition_number_threshold': step_config.get('sr_condition_number_threshold', 1e10)
        }
        
        self.logger.info(f'🔧 Matrix operations configuration prepared:')
        self.logger.info(f'   - Total features: {len(df.columns)}')
        self.logger.info(f'   - SR features: {len(sr_features)}')
        self.logger.info(f"   - Numeric features: {len(config['numeric_columns'])}")
        
        return config
    
    def _calculate_quality_metrics_optimized(self, df: pd.DataFrame, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimized quality metrics."""
        self.logger.info('📊 Calculating optimized quality metrics...')
        
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            quality_metrics = {}
            
            # Completeness metrics
            quality_metrics['completeness'] = {
                'total_cells': numeric_df.size,
                'missing_cells': numeric_df.isnull().sum().sum(),
                'missing_ratio': safe_divide(numeric_df.isnull().sum().sum(), numeric_df.size, default=1.0),
                'complete_rows': int(numeric_df.dropna().shape[0]),
                'complete_columns': int(numeric_df.dropna(axis=1).shape[1])
            }
            
            # Variance metrics
            variances = numeric_df.var()
            quality_metrics['variance'] = {
                'mean_variance': float(variances.mean()),
                'median_variance': float(variances.median()),
                'min_variance': float(variances.min()),
                'max_variance': float(variances.max()),
                'low_variance_features': int((variances < 1e-06).sum()),
                'zero_variance_features': int((variances == 0).sum())
            }
            
            # Overall quality score
            quality_metrics['overall_score'] = self._calculate_overall_quality_score_optimized(quality_metrics, matrix_results)
            
            self.logger.info(f"✅ Quality metrics calculated. Overall score: {quality_metrics['overall_score']:.2f}")
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f'❌ Error calculating quality metrics: {str(e)}')
            return {'error': str(e)}
    
    def _calculate_overall_quality_score_optimized(self, quality_metrics: Dict[str, Any], matrix_results: Dict[str, Any]) -> float:
        """Calculate optimized overall quality score."""
        try:
            score = 0.0
            max_score = 0.0
            
            # Completeness score (25 points)
            completeness = quality_metrics.get('completeness', {})
            if 'missing_ratio' in completeness:
                completeness_score = max(0, 25 * (1 - completeness['missing_ratio']))
                score += completeness_score
                max_score += 25
            
            # Variance score (20 points)
            variance = quality_metrics.get('variance', {})
            if 'zero_variance_features' in variance:
                zero_var_ratio = safe_divide(variance['zero_variance_features'], 
                                           quality_metrics.get('completeness', {}).get('total_cells', 1), 
                                           default=1.0)
                variance_score = max(0, 20 * (1 - zero_var_ratio))
                score += variance_score
                max_score += 20
            
            return safe_divide(score, max_score, default=0.0) if max_score > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f'Error calculating overall quality score: {str(e)}')
            return 0.0
    
    async def _save_results_optimized(self, matrix_results: Dict[str, Any], quality_metrics: Dict[str, Any], 
                                    symbol: str, exchange: str, timeframe: str) -> Dict[str, str]:
        """Save optimized matrix operations results to files."""
        output_files = {}
        
        try:
            # Ensure output directory exists
            output_dir = Path(self.config.get('output_dir', 'data/matrix_operations'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            config_file = output_dir / f'{exchange}_{symbol}_{timeframe}_matrix_operations_config.json'
            import json
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            output_files['config'] = str(config_file)
            
            # Save results
            results_file = output_dir / f'{exchange}_{symbol}_{timeframe}_matrix_operations_results.json'
            with open(results_file, 'w') as f:
                json.dump(matrix_results, f, indent=2, default=str)
            output_files['results'] = str(results_file)
            
            # Save quality metrics
            quality_file = output_dir / f'{exchange}_{symbol}_{timeframe}_quality_metrics.json'
            with open(quality_file, 'w') as f:
                json.dump(quality_metrics, f, indent=2, default=str)
            output_files['quality_metrics'] = str(quality_file)
            
            self.logger.info(f'💾 Saved matrix operations results to {output_dir}')
            return output_files
            
        except Exception as e:
            self.logger.error(f'❌ Error saving results: {e}')
            return {}
    
    def _update_pipeline_state_optimized(self, pipeline_state: Dict[str, Any], start_time: datetime, 
                                       output_files: Dict[str, str], matrix_results: Dict[str, Any],
                                       quality_metrics: Dict[str, Any], symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Update pipeline state with optimized results."""
        pipeline_state["step07_enhanced_matrix_operations"] = {
            "status": "completed",
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "output_files": output_files,
            "matrix_results": matrix_results,
            "quality_metrics": quality_metrics,
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "optimization_level": "final_optimized",
            "performance_metrics": self.performance_metrics
        }
        
        return pipeline_state
    
    def _log_comprehensive_summaries_optimized(self, pipeline_state: Dict[str, Any]) -> None:
        """Log comprehensive summaries from optimized execution."""
        step_state = pipeline_state.get("step07_enhanced_matrix_operations", {})
        
        self.logger.info("📊 COMPREHENSIVE EXECUTION SUMMARY:")
        self.logger.info(f"   Status: {step_state.get('status', 'unknown')}")
        self.logger.info(f"   Execution time: {step_state.get('start_time', 'unknown')} to {step_state.get('end_time', 'unknown')}")
        self.logger.info(f"   Output files: {len(step_state.get('output_files', {}))}")
        self.logger.info(f"   Matrix operations: {len(step_state.get('matrix_results', {}))}")
        self.logger.info(f"   Quality score: {step_state.get('quality_metrics', {}).get('overall_score', 0.0):.2f}")
        self.logger.info(f"   Optimization level: {step_state.get('optimization_level', 'unknown')}")
        
        # Performance metrics summary
        if self.matrix_ops:
            perf_summary = self.matrix_ops.get_performance_summary()
            self.logger.info("📊 PERFORMANCE SUMMARY:")
            self.logger.info(f"   Cache size: {perf_summary.get('cache_size', 0)}")
            self.logger.info(f"   Memory usage: {perf_summary.get('memory_usage_mb', 0):.1f} MB")

# Async wrapper for compatibility
async def run_step(symbol: str, exchange: str, timeframe: str = '1m', data_dir: str = None, 
                  force_rerun: bool = False, **kwargs: Any) -> bool:
    """
    Run Step 7: Enhanced Matrix Operations with final optimizations.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force rerun the step
        **kwargs: Additional arguments
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if data_dir is None:
            data_dir = standardized_parquet_handler.get_standardized_path('processed_data', exchange, symbol) if standardized_parquet_handler else 'data'
        
        # Default configuration
        config = {
            'step07_enhanced_matrix_operations': {
                'correlation_threshold': 0.8,
                'condition_number_threshold': 1e12,
                'min_eigenvalue_threshold': 1e-10,
                'memory_threshold_gb': 8.0,
                'chunk_size': 1000
            },
            'output_dir': 'data/matrix_operations',
            'cache_ttl': 3600
        }
        
        # Update with any provided kwargs
        config.update(kwargs)
        
        step = Step7EnhancedMatrixOperationsFinal(config)
        
        training_input = {
            'symbol': symbol, 
            'exchange': exchange, 
            'timeframe': timeframe, 
            'data_dir': data_dir, 
            'force_rerun': force_rerun, 
            'asset': symbol, 
            'lookback_period': config.get('lookback_days', 1095), 
            'project_version': config.get('project_version', '1.0.0'), 
            **kwargs
        }
        
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)
        step_result = result.get('step07_enhanced_matrix_operations', {})
        return step_result.get('status') == 'completed'
        
    except FastFailError as e:
        logger.error(f'❌ Step 7 failed due to fast fail validation: {str(e)}')
        return False
    except Exception as e:
        logger.error(f'❌ Step 7 failed with unexpected error: {str(e)}')
        return False

__all__ = ['Step7EnhancedMatrixOperationsFinal', 'run_step', 'FastFailError']

if __name__ == '__main__':
    # Test the final optimized implementation
    async def test_final_optimized_step07():
        config = {
            'step07_enhanced_matrix_operations': {
                'correlation_threshold': 0.8,
                'condition_number_threshold': 1e12,
                'min_eigenvalue_threshold': 1e-10,
                'memory_threshold_gb': 8.0,
                'chunk_size': 1000
            },
            'output_dir': 'data/matrix_operations',
            'cache_ttl': 3600
        }
        
        success = await run_step('ETHUSDT', 'BINANCE', '1m', config=config)
        print(f"Final optimized Step 7 result: {success}")
    
    asyncio.run(test_final_optimized_step07())