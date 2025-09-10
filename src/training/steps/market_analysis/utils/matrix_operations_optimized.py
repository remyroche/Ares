"""
Optimized Matrix Operations Module for Step 7 Enhanced Matrix Operations.

This module provides optimized matrix operations with:
- Computational optimizations: caching, vectorized operations, chunked processing
- Fast fails: data validation, memory checks, dependency validation
- Safe mathematical operations using math_validation.py
- Fixed async/sync mixing and algorithmic issues
"""

import time
import gc
import asyncio
import functools
from typing import Any, Dict, List, Optional, Union
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

# Initialize logger
logger = logging.getLogger(__name__)

class FastFailError(Exception):
    """Exception raised when fast fail validation fails."""
    pass

class OptimizedMatrixOperations:
    """Optimized matrix operations with comprehensive validation and caching."""
    
    def __init__(self, logger):
        self.logger = logger
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour default
        self.operation_counts = {}
        self.performance_metrics = {}
    
    def _validate_dependencies(self) -> None:
        """Fast fail: Validate required dependencies."""
        missing_deps = []
        
        if not NUMPY_AVAILABLE:
            missing_deps.append("numpy")
        if not PANDAS_AVAILABLE:
            missing_deps.append("pandas")
        if not MATH_VALIDATION_AVAILABLE:
            missing_deps.append("math_validation")
        
        if missing_deps:
            error_msg = f"❌ FAST FAIL: Missing dependencies: {missing_deps}"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
    
    def _validate_data_shape(self, df: pd.DataFrame, min_rows: int = 10, min_cols: int = 1) -> None:
        """Fast fail: Validate data shape requirements."""
        if len(df) < min_rows:
            error_msg = f"❌ FAST FAIL: Insufficient rows: {len(df)} < {min_rows}"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
        
        if len(df.columns) < min_cols:
            error_msg = f"❌ FAST FAIL: Insufficient columns: {len(df.columns)} < {min_cols}"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
    
    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Fast fail: Validate data types and numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            error_msg = "❌ FAST FAIL: No numeric columns found"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
        
        # Check for non-finite values
        non_finite_count = numeric_df.isin([np.inf, -np.inf, np.nan]).sum().sum()
        if non_finite_count > 0:
            self.logger.warning(f"⚠️ Found {non_finite_count} non-finite values")
    
    def _validate_memory_availability(self, estimated_memory_gb: float) -> None:
        """Fast fail: Validate sufficient memory availability."""
        if not PSUTIL_AVAILABLE:
            self.logger.warning("⚠️ psutil not available - skipping memory validation")
            return
        
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_memory_gb < estimated_memory_gb:
            error_msg = f"❌ FAST FAIL: Insufficient memory: {available_memory_gb:.2f} GB < {estimated_memory_gb:.2f} GB"
            self.logger.error(error_msg)
            raise FastFailError(error_msg)
    
    @functools.lru_cache(maxsize=64)
    def _get_cached_operation(self, operation_type: str, data_hash: str, params_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached operation result."""
        cache_key = f"{operation_type}_{data_hash}_{params_hash}"
        
        if cache_key in self.cache:
            cache_time = self.cache[cache_key].get('timestamp', 0)
            if time.time() - cache_time < self.cache_ttl:
                self.logger.info(f"📋 Using cached {operation_type}")
                return self.cache[cache_key]['data']
        
        return None
    
    def _cache_operation_result(self, operation_type: str, data_hash: str, params_hash: str, result: Dict[str, Any]) -> None:
        """Cache operation result."""
        cache_key = f"{operation_type}_{data_hash}_{params_hash}"
        self.cache[cache_key] = {
            'data': result,
            'timestamp': time.time()
        }
    
    def _create_data_hash(self, df: pd.DataFrame) -> str:
        """Create hash for data caching."""
        return f"{df.shape}_{df.sum().sum():.6f}_{df.std().sum():.6f}"
    
    @math_safe
    async def execute_standard_matrix_operations_optimized(self, numeric_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimized standard matrix operations."""
        start_time = time.time()
        self.logger.info('📊 Performing optimized correlation analysis...')
        
        try:
            # Fast fail validations
            self._validate_dependencies()
            self._validate_data_shape(numeric_df)
            self._validate_data_types(numeric_df)
            
            # Estimate memory requirements
            estimated_memory_gb = (numeric_df.memory_usage(deep=True).sum() * 4) / (1024**3)
            self._validate_memory_availability(estimated_memory_gb)
            
            # Create data hash for caching
            data_hash = self._create_data_hash(numeric_df)
            correlation_threshold = config.get('correlation_threshold', 0.8)
            params_hash = f"corr_{correlation_threshold}"
            
            # Check cache first
            cached_result = self._get_cached_operation('correlation_analysis', data_hash, params_hash)
            if cached_result:
                return cached_result
            
            results = {}
            
            # Optimized correlation analysis
            correlation_matrix = numeric_df.corr()
            
            # Validate correlation matrix
            try:
                validate_correlation_matrix(correlation_matrix.values, "correlation_matrix")
            except MathValidationError as e:
                self.logger.warning(f"⚠️ Correlation matrix validation warning: {e}")
            
            # Vectorized high correlation finding
            high_correlations = self._find_high_correlations_vectorized(correlation_matrix, correlation_threshold)
            
            results['correlation_analysis'] = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'high_correlations': high_correlations,
                'matrix_shape': correlation_matrix.shape
            }
            
            # Optimized condition number check
            self.logger.info('🔍 Checking condition number...')
            condition_number = self._compute_condition_number_safe(numeric_df, config.get('condition_number_threshold', 1e12))
            results['condition_number_check'] = condition_number
            
            # Optimized eigenvalue analysis
            self.logger.info('📈 Performing eigenvalue analysis...')
            eigenvalue_results = self._compute_eigenvalue_analysis_safe(numeric_df, config.get('min_eigenvalue_threshold', 1e-10))
            results['eigenvalue_analysis'] = eigenvalue_results
            
            # Optimized SVD analysis
            self.logger.info('🔧 Performing SVD analysis...')
            svd_results = self._compute_svd_analysis_safe(numeric_df, config.get('min_eigenvalue_threshold', 1e-10))
            results['singular_value_decomposition'] = svd_results
            
            # Optimized matrix rank analysis
            self.logger.info('📊 Analyzing matrix rank...')
            rank_results = self._compute_matrix_rank_safe(numeric_df)
            results['matrix_rank_analysis'] = rank_results
            
            # Cache the results
            self._cache_operation_result('correlation_analysis', data_hash, params_hash, results)
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self.performance_metrics['standard_operations'] = {
                'execution_time_seconds': execution_time,
                'data_shape': numeric_df.shape,
                'operations_completed': len(results)
            }
            
            self.logger.info(f"✅ Standard matrix operations completed in {execution_time:.3f}s")
            return results
            
        except FastFailError:
            raise
        except Exception as e:
            self.logger.error(f'❌ Error in standard matrix operations: {e}')
            return {'error': str(e)}
    
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
    
    @math_safe
    def _compute_condition_number_safe(self, numeric_df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        """Safe condition number computation."""
        try:
            matrix = numeric_df.values
            
            # Check for singular matrix
            det = np.linalg.det(matrix)
            if abs(det) < 1e-15:
                self.logger.warning("⚠️ Matrix is near-singular")
                return {
                    'condition_number': float('inf'),
                    'is_well_conditioned': False,
                    'error': 'near_singular_matrix'
                }
            
            # Compute condition number safely
            condition_number = np.linalg.cond(matrix)
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
    
    @math_safe
    def _compute_eigenvalue_analysis_safe(self, numeric_df: pd.DataFrame, min_threshold: float) -> Dict[str, Any]:
        """Safe eigenvalue analysis."""
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
    
    @math_safe
    def _compute_svd_analysis_safe(self, numeric_df: pd.DataFrame, min_threshold: float) -> Dict[str, Any]:
        """Safe SVD analysis with chunked processing for large matrices."""
        try:
            matrix = numeric_df.values
            
            # For large matrices, use chunked processing
            if matrix.shape[0] > 1000 or matrix.shape[1] > 1000:
                return self._compute_svd_chunked_safe(matrix, min_threshold)
            
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
    
    def _compute_svd_chunked_safe(self, matrix: np.ndarray, min_threshold: float) -> Dict[str, Any]:
        """Safe chunked SVD computation for large matrices."""
        try:
            # Sample the matrix for SVD computation
            sample_size = min(1000, matrix.shape[0])
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
    
    @math_safe
    def _compute_matrix_rank_safe(self, numeric_df: pd.DataFrame) -> Dict[str, Any]:
        """Safe matrix rank computation."""
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
    
    async def execute_sr_matrix_operations_optimized(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimized SR-specific matrix operations."""
        start_time = time.time()
        self.logger.info('🎯 Executing optimized SR matrix operations...')
        
        try:
            # Fast fail validations
            self._validate_dependencies()
            self._validate_data_shape(df)
            self._validate_data_types(df)
            
            sr_features = config.get('sr_features', [])
            if not sr_features:
                return {'error': 'No SR features found'}
            
            sr_df = df[sr_features].select_dtypes(include=[np.number])
            if len(sr_df.columns) == 0:
                return {'error': 'No numeric SR features found'}
            
            # Estimate memory requirements
            estimated_memory_gb = (sr_df.memory_usage(deep=True).sum() * 4) / (1024**3)
            self._validate_memory_availability(estimated_memory_gb)
            
            self.logger.info(f'🎯 Analyzing {len(sr_df.columns)} SR features')
            results = {}
            
            # Optimized SR correlation analysis
            self.logger.info('📊 Performing SR feature correlation analysis...')
            sr_correlation_matrix = sr_df.corr()
            
            # Validate SR correlation matrix
            try:
                validate_correlation_matrix(sr_correlation_matrix.values, "sr_correlation_matrix")
            except MathValidationError as e:
                self.logger.warning(f"⚠️ SR correlation matrix validation warning: {e}")
            
            high_correlations = self._find_high_correlations_vectorized(
                sr_correlation_matrix, 
                config.get('sr_correlation_threshold', 0.7)
            )
            
            results['sr_correlation_analysis'] = {
                'correlation_matrix': sr_correlation_matrix.to_dict(),
                'high_correlations': high_correlations,
                'sr_feature_count': len(sr_df.columns)
            }
            
            # Optimized SR condition number check
            self.logger.info('🔍 Checking SR feature condition number...')
            sr_condition_number = self._compute_condition_number_safe(
                sr_df, 
                config.get('sr_condition_number_threshold', 1e10)
            )
            results['sr_condition_number'] = sr_condition_number
            
            # Optimized SR eigenvalue analysis
            self.logger.info('📈 Performing SR feature eigenvalue analysis...')
            sr_eigenvalue_results = self._compute_eigenvalue_analysis_safe(
                sr_df, 
                config.get('min_eigenvalue_threshold', 1e-10)
            )
            results['sr_eigenvalue_analysis'] = sr_eigenvalue_results
            
            # Optimized SR clustering analysis
            self.logger.info('🔧 Performing SR feature clustering analysis...')
            results['sr_clustering_analysis'] = self._analyze_sr_feature_clusters_optimized(sr_df)
            
            # Optimized SR stability analysis
            self.logger.info('📊 Analyzing SR feature stability...')
            results['sr_stability_analysis'] = self._analyze_sr_feature_stability_optimized(sr_df)
            
            # Optimized SR importance analysis
            self.logger.info('🎯 Analyzing SR feature importance...')
            results['sr_importance_analysis'] = self._analyze_sr_feature_importance_optimized(sr_df)
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self.performance_metrics['sr_operations'] = {
                'execution_time_seconds': execution_time,
                'sr_feature_count': len(sr_df.columns),
                'operations_completed': len(results)
            }
            
            self.logger.info(f"✅ SR matrix operations completed in {execution_time:.3f}s")
            return results
            
        except FastFailError:
            raise
        except Exception as e:
            self.logger.error(f'❌ Error in SR matrix operations: {e}')
            return {'error': str(e)}
    
    def _analyze_sr_feature_clusters_optimized(self, sr_df: pd.DataFrame) -> Dict[str, Any]:
        """Optimized SR feature cluster analysis."""
        try:
            correlation_matrix = sr_df.corr()
            high_corr_groups = []
            processed_features = set()
            
            # Vectorized approach for finding correlation groups
            for i, feature1 in enumerate(sr_df.columns):
                if feature1 in processed_features:
                    continue
                group = [feature1]
                processed_features.add(feature1)
                
                # Find highly correlated features
                correlations = correlation_matrix.loc[feature1, sr_df.columns[i+1:]]
                high_corr_features = correlations[correlations.abs() > 0.8].index.tolist()
                
                for feature2 in high_corr_features:
                    if feature2 not in processed_features:
                        group.append(feature2)
                        processed_features.add(feature2)
                
                if len(group) > 1:
                    high_corr_groups.append(group)
            
            return {
                'high_correlation_groups': high_corr_groups,
                'group_count': len(high_corr_groups),
                'total_grouped_features': sum(len(group) for group in high_corr_groups)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_sr_feature_stability_optimized(self, sr_df: pd.DataFrame) -> Dict[str, Any]:
        """Optimized SR feature stability analysis."""
        try:
            stability_metrics = {}
            
            # Vectorized stability calculation
            for column in sr_df.columns:
                values = sr_df[column].dropna()
                if len(values) > 1:
                    mean_val = values.mean()
                    std_val = values.std()
                    
                    # Safe coefficient of variation calculation
                    cv = safe_divide(std_val, abs(mean_val), default=float('inf'))
                    
                    # Safe range stability calculation
                    range_val = values.max() - values.min()
                    range_stability = safe_divide(1.0, 1.0 + range_val, default=0.0)
                    
                    stability_metrics[column] = {
                        'coefficient_of_variation': float(cv),
                        'range_stability': float(range_stability),
                        'mean': float(mean_val),
                        'std': float(std_val),
                        'min': float(values.min()),
                        'max': float(values.max())
                    }
            
            # Calculate overall stability metrics
            if stability_metrics:
                cvs = [metrics['coefficient_of_variation'] for metrics in stability_metrics.values()]
                range_stabilities = [metrics['range_stability'] for metrics in stability_metrics.values()]
                
                overall_stability = {
                    'mean_cv': float(np.mean(cvs)),
                    'mean_range_stability': float(np.mean(range_stabilities)),
                    'stable_features': int(np.sum(np.array(cvs) < 0.5)),
                    'unstable_features': int(np.sum(np.array(cvs) > 1.0))
                }
            else:
                overall_stability = {
                    'mean_cv': 0.0,
                    'mean_range_stability': 0.0,
                    'stable_features': 0,
                    'unstable_features': 0
                }
            
            return {
                'feature_stability': stability_metrics,
                'overall_stability': overall_stability
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_sr_feature_importance_optimized(self, sr_df: pd.DataFrame) -> Dict[str, Any]:
        """Optimized SR feature importance analysis."""
        try:
            # Vectorized variance calculation
            variances = sr_df.var()
            variance_importance = variances.sort_values(ascending=False)
            
            # Vectorized correlation importance calculation
            correlation_matrix = sr_df.corr()
            avg_correlations = correlation_matrix.abs().mean()
            correlation_importance = (1.0 / (1.0 + avg_correlations)).sort_values(ascending=False)
            
            # Combined importance calculation
            combined_importance = (variance_importance + correlation_importance) / 2
            combined_importance = combined_importance.sort_values(ascending=False)
            
            return {
                'variance_importance': variance_importance.to_dict(),
                'correlation_importance': correlation_importance.to_dict(),
                'combined_importance': combined_importance.to_dict(),
                'top_features': combined_importance.head(10).index.tolist()
            }
        except Exception as e:
            return {'error': str(e)}
    
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

__all__ = ['OptimizedMatrixOperations', 'FastFailError']