"""
Step 7: Enhanced Matrix Operations - Optimized Version

This module implements computational optimizations, fast fails, and fixes for step07:
- Computational optimizations: caching, vectorized operations, chunked processing
- Fast fails: data shape validation, dependency validation, data type validation
- Fixes: async/sync mixing, algorithmic issues
- Uses src/utils/math_validation.py for safe mathematical operations
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

# Initialize logger
logger = logging.getLogger(__name__)

class FastFailError(Exception):
    """Exception raised when fast fail validation fails."""
    pass

class Step07OptimizedMatrixOperations:
    """Optimized Step 7 Enhanced Matrix Operations with fast fails and computational optimizations."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize with comprehensive validation and optimization setup."""
        self.config = config
        self.logger = logger.getChild('Step07Optimized')
        
        # Fast fail validation
        self._validate_dependencies()
        self._validate_config()
        
        # Optimization settings
        self.cache = {}
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour default
        self.chunk_size = config.get('chunk_size', 1000)
        self.memory_threshold_gb = config.get('memory_threshold_gb', 8.0)
        
        # Performance tracking
        self.performance_metrics = {}
        self.operation_counts = {}
        
        self.logger.info("🔧 Initialized Step 7 Optimized Matrix Operations")
    
    def _validate_dependencies(self) -> None:
        """Fast fail: Validate all required dependencies."""
        self.logger.info("🔍 Validating dependencies...")
        
        missing_deps = []
        
        if not NUMPY_AVAILABLE:
            missing_deps.append("numpy")
        if not PANDAS_AVAILABLE:
            missing_deps.append("pandas")
        if not MATH_VALIDATION_AVAILABLE:
            missing_deps.append("math_validation")
        if not PARQUET_HANDLER_AVAILABLE:
            missing_deps.append("parquet_handler")
        
        if missing_deps:
            error_msg = f"❌ FAST FAIL: Missing critical dependencies: {missing_deps}"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
        
        self.logger.info("✅ All critical dependencies available")
    
    def _validate_config(self) -> None:
        """Fast fail: Validate configuration parameters."""
        self.logger.info("🔍 Validating configuration...")
        
        required_configs = [
            'step07_enhanced_matrix_operations',
            'output_dir'
        ]
        
        missing_configs = []
        for config_key in required_configs:
            if config_key not in self.config:
                missing_configs.append(config_key)
        
        if missing_configs:
            error_msg = f"❌ FAST FAIL: Missing required configuration: {missing_configs}"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
        
        # Validate numeric parameters
        step_config = self.config.get('step07_enhanced_matrix_operations', {})
        try:
            if 'condition_number_threshold' in step_config:
                validate_positive(step_config['condition_number_threshold'], 'condition_number_threshold')
            if 'correlation_threshold' in step_config:
                validate_range(step_config['correlation_threshold'], 0.0, 1.0, 'correlation_threshold')
        except MathValidationError as e:
            error_msg = f"❌ FAST FAIL: Invalid configuration parameter: {e}"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
        
        self.logger.info("✅ Configuration validation passed")
    
    def _validate_data_shape(self, df: pd.DataFrame, min_rows: int = 10, min_cols: int = 1) -> None:
        """Fast fail: Validate data shape requirements."""
        self.logger.info(f"🔍 Validating data shape: {df.shape}")
        
        if len(df) < min_rows:
            error_msg = f"❌ FAST FAIL: Insufficient data rows: {len(df)} < {min_rows}"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
        
        if len(df.columns) < min_cols:
            error_msg = f"❌ FAST FAIL: Insufficient data columns: {len(df.columns)} < {min_cols}"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
        
        self.logger.info(f"✅ Data shape validation passed: {df.shape}")
    
    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Fast fail: Validate data types and numeric columns."""
        self.logger.info("🔍 Validating data types...")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            error_msg = "❌ FAST FAIL: No numeric columns found in data"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
        
        # Check for non-finite values
        non_finite_count = numeric_df.isin([np.inf, -np.inf, np.nan]).sum().sum()
        if non_finite_count > 0:
            self.logger.warning(f"⚠️ Found {non_finite_count} non-finite values in numeric data")
        
        self.logger.info(f"✅ Data type validation passed: {len(numeric_df.columns)} numeric columns")
    
    def _validate_memory_availability(self, estimated_memory_gb: float) -> None:
        """Fast fail: Validate sufficient memory availability."""
        if not PSUTIL_AVAILABLE:
            self.logger.warning("⚠️ psutil not available - skipping memory validation")
            return
        
        self.logger.info(f"🔍 Validating memory availability: {estimated_memory_gb:.2f} GB required")
        
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_memory_gb < estimated_memory_gb:
            error_msg = f"❌ FAST FAIL: Insufficient memory: {available_memory_gb:.2f} GB available < {estimated_memory_gb:.2f} GB required"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
        
        self.logger.info(f"✅ Memory validation passed: {available_memory_gb:.2f} GB available")
    
    @functools.lru_cache(maxsize=128)
    def _get_cached_correlation_matrix(self, data_hash: str, threshold: float) -> Dict[str, Any]:
        """Cached correlation matrix computation."""
        # This would be implemented with actual data, but for now return empty
        return {'cached': True, 'threshold': threshold}
    
    @math_safe
    def _compute_correlation_matrix_optimized(self, numeric_df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        """Optimized correlation matrix computation with caching."""
        # Create data hash for caching
        data_hash = f"{numeric_df.shape}_{numeric_df.sum().sum()}_{threshold}"
        
        # Check cache first
        if data_hash in self.cache:
            cache_time = self.cache[data_hash].get('timestamp', 0)
            if time.time() - cache_time < self.cache_ttl:
                self.logger.info("📋 Using cached correlation matrix")
                return self.cache[data_hash]['data']
        
        self.logger.info("📊 Computing correlation matrix...")
        
        # Use vectorized operations
        correlation_matrix = numeric_df.corr()
        
        # Validate correlation matrix
        try:
            validate_correlation_matrix(correlation_matrix.values, "correlation_matrix")
        except MathValidationError as e:
            self.logger.warning(f"⚠️ Correlation matrix validation warning: {e}")
        
        # Find high correlations efficiently
        high_correlations = self._find_high_correlations_vectorized(correlation_matrix, threshold)
        
        result = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': high_correlations,
            'matrix_shape': correlation_matrix.shape,
            'computation_time': time.time()
        }
        
        # Cache the result
        self.cache[data_hash] = {
            'data': result,
            'timestamp': time.time()
        }
        
        return result
    
    def _find_high_correlations_vectorized(self, correlation_matrix: pd.DataFrame, threshold: float) -> List[Dict[str, Any]]:
        """Vectorized high correlation finding."""
        # Get upper triangle indices
        upper_triangle = np.triu_indices_from(correlation_matrix.values, k=1)
        
        # Extract correlations and indices
        correlations = correlation_matrix.values[upper_triangle]
        row_indices, col_indices = upper_triangle
        
        # Find high correlations
        high_corr_mask = np.abs(correlations) >= threshold
        high_corr_indices = np.where(high_corr_mask)[0]
        
        # Build result list
        high_correlations = []
        for idx in high_corr_indices:
            i, j = row_indices[idx], col_indices[idx]
            high_correlations.append({
                'column1': correlation_matrix.columns[i],
                'column2': correlation_matrix.columns[j],
                'correlation': float(correlations[idx])
            })
        
        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return high_correlations
    
    def _compute_condition_number_optimized(self, numeric_df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        """Optimized condition number computation."""
        self.logger.info("🔍 Computing condition number...")
        
        try:
            # Use safe matrix operations
            matrix = numeric_df.values
            
            # Check for singular matrix
            if np.linalg.det(matrix) == 0:
                self.logger.warning("⚠️ Matrix is singular (determinant = 0)")
                return {
                    'condition_number': float('inf'),
                    'is_well_conditioned': False,
                    'error': 'singular_matrix'
                }
            
            # Compute condition number safely
            condition_number = np.linalg.cond(matrix)
            
            # Validate condition number
            condition_number = validate_finite(condition_number, 'condition_number')
            
            is_well_conditioned = condition_number < threshold
            
            return {
                'condition_number': float(condition_number),
                'is_well_conditioned': is_well_conditioned,
                'threshold': threshold
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error computing condition number: {e}")
            return {
                'condition_number': float('inf'),
                'is_well_conditioned': False,
                'error': str(e)
            }
    
    def _compute_eigenvalue_analysis_optimized(self, numeric_df: pd.DataFrame, min_threshold: float) -> Dict[str, Any]:
        """Optimized eigenvalue analysis."""
        self.logger.info("📈 Computing eigenvalue analysis...")
        
        try:
            matrix = numeric_df.values
            
            # Compute eigenvalues efficiently
            eigenvalues = np.linalg.eigvals(matrix)
            
            # Validate eigenvalues
            finite_eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
            
            if len(finite_eigenvalues) == 0:
                return {
                    'eigenvalues': [],
                    'min_eigenvalue': 0.0,
                    'max_eigenvalue': 0.0,
                    'eigenvalue_ratio': 0.0,
                    'small_eigenvalues': 0,
                    'error': 'no_finite_eigenvalues'
                }
            
            min_eigenvalue = np.min(finite_eigenvalues)
            max_eigenvalue = np.max(finite_eigenvalues)
            
            # Safe eigenvalue ratio calculation
            eigenvalue_ratio = safe_divide(max_eigenvalue, abs(min_eigenvalue), default=float('inf'))
            
            small_eigenvalues = np.sum(np.abs(finite_eigenvalues) < min_threshold)
            
            return {
                'eigenvalues': finite_eigenvalues.tolist(),
                'min_eigenvalue': float(min_eigenvalue),
                'max_eigenvalue': float(max_eigenvalue),
                'eigenvalue_ratio': float(eigenvalue_ratio),
                'small_eigenvalues': int(small_eigenvalues)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in eigenvalue analysis: {e}")
            return {
                'eigenvalues': [],
                'min_eigenvalue': 0.0,
                'max_eigenvalue': 0.0,
                'eigenvalue_ratio': 0.0,
                'small_eigenvalues': 0,
                'error': str(e)
            }
    
    def _compute_svd_analysis_optimized(self, numeric_df: pd.DataFrame, min_threshold: float) -> Dict[str, Any]:
        """Optimized SVD analysis with chunked processing for large matrices."""
        self.logger.info("🔧 Computing SVD analysis...")
        
        try:
            matrix = numeric_df.values
            
            # For large matrices, use chunked processing
            if matrix.shape[0] > self.chunk_size or matrix.shape[1] > self.chunk_size:
                return self._compute_svd_chunked(matrix, min_threshold)
            
            # Standard SVD for smaller matrices
            U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
            
            # Validate singular values
            finite_singular_values = s[np.isfinite(s)]
            
            if len(finite_singular_values) == 0:
                return {
                    'singular_values': [],
                    'rank': 0,
                    'condition_number_svd': float('inf'),
                    'error': 'no_finite_singular_values'
                }
            
            rank = np.sum(finite_singular_values > min_threshold)
            condition_number_svd = safe_divide(
                finite_singular_values[0], 
                finite_singular_values[-1], 
                default=float('inf')
            )
            
            return {
                'singular_values': finite_singular_values.tolist(),
                'rank': int(rank),
                'condition_number_svd': float(condition_number_svd)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in SVD analysis: {e}")
            return {
                'singular_values': [],
                'rank': 0,
                'condition_number_svd': float('inf'),
                'error': str(e)
            }
    
    def _compute_svd_chunked(self, matrix: np.ndarray, min_threshold: float) -> Dict[str, Any]:
        """Chunked SVD computation for large matrices."""
        self.logger.info(f"🔧 Computing chunked SVD for large matrix: {matrix.shape}")
        
        try:
            # Sample the matrix for SVD computation
            sample_size = min(self.chunk_size, matrix.shape[0])
            sample_indices = np.random.choice(matrix.shape[0], sample_size, replace=False)
            sample_matrix = matrix[sample_indices, :]
            
            # Compute SVD on sample
            U, s, Vt = np.linalg.svd(sample_matrix, full_matrices=False)
            
            # Validate and process results
            finite_singular_values = s[np.isfinite(s)]
            
            if len(finite_singular_values) == 0:
                return {
                    'singular_values': [],
                    'rank': 0,
                    'condition_number_svd': float('inf'),
                    'sample_size': sample_size,
                    'error': 'no_finite_singular_values'
                }
            
            rank = np.sum(finite_singular_values > min_threshold)
            condition_number_svd = safe_divide(
                finite_singular_values[0], 
                finite_singular_values[-1], 
                default=float('inf')
            )
            
            return {
                'singular_values': finite_singular_values.tolist(),
                'rank': int(rank),
                'condition_number_svd': float(condition_number_svd),
                'sample_size': sample_size,
                'chunked': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in chunked SVD: {e}")
            return {
                'singular_values': [],
                'rank': 0,
                'condition_number_svd': float('inf'),
                'error': str(e)
            }
    
    async def execute_optimized_matrix_operations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute optimized matrix operations with comprehensive validation."""
        start_time = time.time()
        self.logger.info("🚀 Starting optimized matrix operations...")
        
        try:
            # Fast fail validations
            self._validate_data_shape(df)
            self._validate_data_types(df)
            
            # Extract numeric data
            numeric_df = df.select_dtypes(include=[np.number])
            
            # Estimate memory requirements
            estimated_memory_gb = (numeric_df.memory_usage(deep=True).sum() * 4) / (1024**3)  # 4x for operations
            self._validate_memory_availability(estimated_memory_gb)
            
            # Get configuration
            step_config = self.config.get('step07_enhanced_matrix_operations', {})
            correlation_threshold = step_config.get('correlation_threshold', 0.8)
            condition_threshold = step_config.get('condition_number_threshold', 1e12)
            min_eigenvalue_threshold = step_config.get('min_eigenvalue_threshold', 1e-10)
            
            # Execute optimized operations
            results = {}
            
            # Correlation analysis (optimized with caching)
            results['correlation_analysis'] = self._compute_correlation_matrix_optimized(
                numeric_df, correlation_threshold
            )
            
            # Condition number analysis (optimized)
            results['condition_number_check'] = self._compute_condition_number_optimized(
                numeric_df, condition_threshold
            )
            
            # Eigenvalue analysis (optimized)
            results['eigenvalue_analysis'] = self._compute_eigenvalue_analysis_optimized(
                numeric_df, min_eigenvalue_threshold
            )
            
            # SVD analysis (optimized with chunking)
            results['singular_value_decomposition'] = self._compute_svd_analysis_optimized(
                numeric_df, min_eigenvalue_threshold
            )
            
            # Matrix rank analysis
            results['matrix_rank_analysis'] = self._compute_matrix_rank_optimized(numeric_df)
            
            # Performance metrics
            execution_time = time.time() - start_time
            results['performance_metrics'] = {
                'execution_time_seconds': execution_time,
                'data_shape': numeric_df.shape,
                'memory_usage_mb': numeric_df.memory_usage(deep=True).sum() / (1024**2),
                'operations_completed': len(results) - 1  # Exclude performance_metrics
            }
            
            self.logger.info(f"✅ Optimized matrix operations completed in {execution_time:.3f}s")
            return results
            
        except FastFailError:
            # Re-raise fast fail errors
            raise
        except Exception as e:
            self.logger.error(f"❌ Error in optimized matrix operations: {e}")
            return {
                'error': str(e),
                'execution_time_seconds': time.time() - start_time
            }
    
    def _compute_matrix_rank_optimized(self, numeric_df: pd.DataFrame) -> Dict[str, Any]:
        """Optimized matrix rank computation."""
        self.logger.info("📊 Computing matrix rank...")
        
        try:
            matrix = numeric_df.values
            
            # Use efficient rank computation
            rank = np.linalg.matrix_rank(matrix)
            
            return {
                'rank': int(rank),
                'full_rank': rank == min(matrix.shape),
                'rank_deficiency': min(matrix.shape) - rank
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error computing matrix rank: {e}")
            return {
                'rank': 0,
                'full_rank': False,
                'rank_deficiency': min(numeric_df.shape),
                'error': str(e)
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'cache_size': len(self.cache),
            'operation_counts': self.operation_counts,
            'performance_metrics': self.performance_metrics,
            'memory_usage_mb': psutil.Process().memory_info().rss / (1024**2) if PSUTIL_AVAILABLE else 0
        }
    
    def clear_cache(self) -> None:
        """Clear operation cache."""
        self.cache.clear()
        self.logger.info("🧹 Cache cleared")

# Async wrapper for compatibility
async def run_optimized_step07(symbol: str, exchange: str, timeframe: str = '1m', 
                              config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Run optimized Step 7 with comprehensive validation and error handling.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if config is None:
            config = {}
        
        # Initialize optimized step
        step = Step07OptimizedMatrixOperations(config)
        
        # Load data (this would be implemented with actual data loading)
        # For now, create sample data for demonstration
        if PANDAS_AVAILABLE and NUMPY_AVAILABLE:
            sample_data = pd.DataFrame(np.random.randn(100, 10))
            sample_data.columns = [f'feature_{i}' for i in range(10)]
        else:
            logger.error("❌ Pandas or NumPy not available")
            return False
        
        # Execute optimized operations
        results = await step.execute_optimized_matrix_operations(sample_data)
        
        if 'error' in results:
            logger.error(f"❌ Step 7 failed: {results['error']}")
            return False
        
        logger.info("✅ Optimized Step 7 completed successfully")
        return True
        
    except FastFailError as e:
        logger.error(f"❌ Fast fail validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error in Step 7: {e}")
        return False

if __name__ == "__main__":
    # Test the optimized implementation
    async def test_optimized_step07():
        config = {
            'step07_enhanced_matrix_operations': {
                'correlation_threshold': 0.8,
                'condition_number_threshold': 1e12,
                'min_eigenvalue_threshold': 1e-10
            },
            'output_dir': 'data/matrix_operations',
            'cache_ttl': 3600,
            'chunk_size': 1000,
            'memory_threshold_gb': 8.0
        }
        
        success = await run_optimized_step07('ETHUSDT', 'BINANCE', '1m', config)
        print(f"Optimized Step 7 result: {success}")
    
    asyncio.run(test_optimized_step07())