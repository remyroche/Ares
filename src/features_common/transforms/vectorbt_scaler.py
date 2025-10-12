"""
VectorBT-Enhanced Scaler

This module provides VectorBT-optimized scaling and normalization operations
for the features_common system, leveraging VectorBT's high-performance
scaling functions.

Key Features:
- VectorBT scaling functions (zscore, minmax, robust, quantile, winsorize)
- GPU acceleration support
- Memory-efficient processing
- Batch scaling operations
- Fallback to standard scalers when VectorBT is not available
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import common utilities
from ..utils import (
    TPRINT_AVAILABLE, tprint,
    VECTORBT_AVAILABLE, vbt, scale, rank, zscore, winsorize, clip, quantile,
    VECTORBT_OPTIMIZER_AVAILABLE, VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
    UnifiedVectorizationManager, get_unified_vectorization_manager,
    CUPY_AVAILABLE, cp
)

from .base_scaler import BaseScaler

logger = logging.getLogger(__name__)


class VectorBTScaler(BaseScaler):
    """
    VectorBT-optimized scaler with comprehensive scaling methods.
    
    This scaler leverages VectorBT's high-performance scaling functions
    for maximum efficiency and accuracy with enhanced optimization features.
    """
    
    def __init__(self, method: str = 'zscore', enable_gpu: bool = False, 
                 enable_batch: bool = True, memory_efficient: bool = True,
                 use_optimizer: bool = True, use_unified_manager: bool = True, **kwargs):
        """
        Initialize VectorBT scaler with enhanced optimization.
        
        Args:
            method: Scaling method ('zscore', 'minmax', 'robust', 'quantile', 'winsorize', 'rank', 'clip', 'robust_zscore', 'adaptive', 'quantile_robust')
            enable_gpu: Enable GPU acceleration if available
            enable_batch: Enable batch processing optimization
            memory_efficient: Enable memory optimization
            use_optimizer: Whether to use VectorBTRollingOptimizer
            use_unified_manager: Whether to use UnifiedVectorizationManager
            **kwargs: Additional parameters for the scaling method
        """
        super().__init__(use_vectorbt=True, enable_gpu=enable_gpu, 
                        use_optimizer=use_optimizer, use_unified_manager=use_unified_manager)
        self.method = method
        self.kwargs = kwargs
        self.scaling_params = {}
        self.enable_batch = enable_batch
        self.memory_efficient = memory_efficient
        
        # Enhanced performance tracking
        self.performance_stats.update({
            'gpu_operations': 0,
            'batch_operations': 0,
            'adaptive_scaling_decisions': 0,
            'unified_manager_operations': 0
        })
        
        if not VECTORBT_AVAILABLE:
            logger.warning("VectorBT not available, using fallback scaler")
    
    def fit_transform(self, data: pd.Series) -> pd.Series:
        """Fit scaler parameters and transform data using VectorBT with enhanced optimization."""
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [VectorBTScaler] Starting fit_transform with method={self.method} on {len(data)} samples", color="cyan")
        
        self.performance_stats['total_operations'] += 1
        self._log_info(f"🔧 [VectorBTScaler] Fitting {self.method} scaler on {len(data)} samples")
        
        # Validate input
        self._validate_numeric_input(data, "input data")
        
        # Optimize data for VectorBT processing
        if self.memory_efficient:
            if TPRINT_AVAILABLE:
                tprint("🔧 [VectorBTScaler] Optimizing data types for memory efficiency", color="blue")
            data = self._optimize_data_types(data)
        
        # Enable GPU processing if available
        if self.enable_gpu:
            if TPRINT_AVAILABLE:
                tprint("🚀 [VectorBTScaler] Enabling GPU processing", color="magenta")
            data = self._enable_gpu_processing(data)
            self.performance_stats['gpu_operations'] += 1
        
        # Remove NaN values for fitting
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            if TPRINT_AVAILABLE:
                tprint("⚠️  [VectorBTScaler] No valid data to fit, using defaults", color="yellow")
            self._log_warning("⚠️  No valid data to fit, using defaults")
            return pd.Series(np.nan, index=data.index)
        
        if VECTORBT_AVAILABLE:
            try:
                # Use UnifiedVectorizationManager if available
                if self.use_unified_manager and self.vectorization_manager is not None:
                    if TPRINT_AVAILABLE:
                        tprint("🔧 [VectorBTScaler] Using UnifiedVectorizationManager for scaling", color="green")
                    result = self._apply_unified_vectorization_scaling(clean_data)
                    self.performance_stats['unified_manager_operations'] += 1
                else:
                    if TPRINT_AVAILABLE:
                        tprint("🔧 [VectorBTScaler] Using enhanced VectorBT scaling", color="green")
                    # Use enhanced VectorBT scaling
                    result = self._apply_enhanced_vectorbt_scaling(clean_data)
                    self.performance_stats['vectorbt_operations'] += 1
                
                # Align result with original index
                result = result.reindex(data.index)
                self.fitted = True
                if TPRINT_AVAILABLE:
                    tprint(f"✅ [VectorBTScaler] Successfully fitted {self.method} scaler", color="green")
                self._log_success(f"✅ [VectorBTScaler] Fitted {self.method} scaler successfully")
                
                # Validate output
                self._check_output_validity(result, "transformed data")
                
                return result
                
            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint(f"⚠️  [VectorBTScaler] VectorBT scaling failed: {e}, using fallback", color="yellow")
                self._log_warning(f"⚠️  VectorBT scaling failed: {e}, using fallback")
                return self._fallback_fit_transform(data)
        else:
            if TPRINT_AVAILABLE:
                tprint("⚠️  [VectorBTScaler] VectorBT not available, using fallback", color="yellow")
            return self._fallback_fit_transform(data)
    
    def _apply_enhanced_vectorbt_scaling(self, data: pd.Series) -> pd.Series:
        """Apply enhanced VectorBT scaling with advanced methods."""
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [VectorBTScaler] Applying {self.method} scaling", color="blue")
        
        # Use the unified scaling method to reduce code duplication
        result, params = self._apply_scaling_method(data, self.method, **self.kwargs)
        self.scaling_params = params
        
        return result
    
    def _apply_scaling_method(self, data: pd.Series, method: str, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Apply a specific scaling method and return result with parameters."""
        if method == 'zscore':
            result = zscore(data, **kwargs)
            params = {
                'mean': data.mean(),
                'std': data.std()
            }
        elif method == 'minmax':
            result = scale(data, method='minmax', **kwargs)
            params = {
                'min': data.min(),
                'max': data.max()
            }
        elif method == 'robust':
            result = scale(data, method='robust', **kwargs)
            median = data.median()
            mad = (data - median).abs().median()
            params = {
                'median': median,
                'mad': mad
            }
        elif method == 'quantile':
            result = quantile(data, **kwargs)
            params = {
                'quantiles': data.quantile([0.25, 0.5, 0.75])
            }
        elif method == 'winsorize':
            result = winsorize(data, **kwargs)
            params = {
                'limits': kwargs.get('limits', (0.05, 0.05))
            }
        elif method == 'rank':
            result = rank(data, **kwargs)
            params = {
                'method': kwargs.get('method', 'average')
            }
        elif method == 'clip':
            result = clip(data, **kwargs)
            params = {
                'lower': kwargs.get('lower', None),
                'upper': kwargs.get('upper', None)
            }
        elif method == 'robust_zscore':
            # Enhanced robust z-score using VectorBT
            median = data.median()
            mad = (data - median).abs().median()
            result = (data - median) / (1.4826 * mad)  # 1.4826 is consistency factor for normal distribution
            params = {
                'median': median,
                'mad': mad,
                'consistency_factor': 1.4826
            }
        elif method == 'adaptive':
            # Adaptive scaling based on data characteristics
            skewness = data.skew()
            kurtosis = data.kurtosis()
            
            if abs(skewness) > 2:  # Highly skewed data
                result, params = self._apply_scaling_method(data, 'quantile', **kwargs)
                params.update({
                    'method_used': 'quantile',
                    'reason': f'skewness={skewness:.3f}'
                })
            elif kurtosis > 3:  # Heavy-tailed data
                result, params = self._apply_scaling_method(data, 'robust', **kwargs)
                params.update({
                    'method_used': 'robust',
                    'reason': f'kurtosis={kurtosis:.3f}'
                })
            else:  # Normal-like data
                result, params = self._apply_scaling_method(data, 'zscore', **kwargs)
                params.update({
                    'method_used': 'zscore',
                    'reason': f'normal-like: skewness={skewness:.3f}, kurtosis={kurtosis:.3f}'
                })
            self.performance_stats['adaptive_scaling_decisions'] += 1
        elif method == 'quantile_robust':
            # Robust quantile scaling
            q25, q75 = data.quantile([0.25, 0.75])
            result = (data - q25) / (q75 - q25 + 1e-8)  # Add small epsilon to avoid division by zero
            params = {
                'q25': q25,
                'q75': q75,
                'iqr': q75 - q25
            }
        elif method == 'winsorize_adaptive':
            # Adaptive winsorization based on data distribution
            limits = self._calculate_adaptive_winsorize_limits(data)
            result = winsorize(data, limits=limits, **kwargs)
            params = {
                'limits': limits,
                'adaptive': True
            }
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        
        return result, params
    
    def _apply_unified_vectorization_scaling(self, data: pd.Series) -> pd.Series:
        """Apply scaling using UnifiedVectorizationManager for optimal performance."""
        try:
            # Use the unified vectorization manager for scaling
            result = self.vectorization_manager.scale_data(data, method=self.method, **self.kwargs)
            
            # Store scaling parameters for transform method
            self._store_scaling_parameters_from_data(data)
            
            return result
            
        except Exception as e:
            logger.warning(f"UnifiedVectorizationManager scaling failed: {e}, using enhanced VectorBT")
            return self._apply_enhanced_vectorbt_scaling(data)
    
    def _store_scaling_parameters_from_data(self, data: pd.Series) -> None:
        """Store scaling parameters from the data for transform method."""
        # Use the unified method to get parameters
        _, params = self._apply_scaling_method(data, self.method, **self.kwargs)
        self.scaling_params = params
    
    def _calculate_adaptive_winsorize_limits(self, data: pd.Series) -> Tuple[float, float]:
        """Calculate adaptive winsorization limits based on data distribution."""
        # Use IQR-based limits for better outlier detection
        q25, q75 = data.quantile([0.25, 0.75])
        iqr = q75 - q25
        
        # Adaptive limits based on data spread
        if iqr > 0:
            # More aggressive winsorization for wide-spread data
            lower_limit = max(0.01, min(0.1, 0.05 * (iqr / data.std())))
            upper_limit = max(0.01, min(0.1, 0.05 * (iqr / data.std())))
        else:
            # Default limits for uniform data
            lower_limit = 0.05
            upper_limit = 0.05
        
        return (lower_limit, upper_limit)
    
    def _optimize_data_types(self, data: pd.Series) -> pd.Series:
        """Optimize data types for memory efficiency."""
        if self.memory_efficient and data.dtype == 'float64':
            # Check if float32 is sufficient
            if (data.min() >= np.finfo(np.float32).min and 
                data.max() <= np.finfo(np.float32).max):
                data = data.astype(np.float32)
                self.performance_stats['memory_optimizations'] += 1
        return data
    
    def _enable_gpu_processing(self, data: pd.Series) -> pd.Series:
        """Enable GPU processing if available."""
        if self.enable_gpu and CUPY_AVAILABLE:
            try:
                gpu_data = cp.asarray(data.values)
                return pd.Series(gpu_data, index=data.index)
            except Exception as e:
                logger.warning(f"GPU processing failed: {e}")
                return data
        return data
    
    def transform(self, data: pd.Series) -> pd.Series:
        """Transform new data using fitted parameters."""
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [VectorBTScaler] Starting transform with method={self.method} on {len(data)} samples", color="cyan")
        
        self._validate_fitted()
        
        if VECTORBT_AVAILABLE and self.fitted:
            try:
                # Use UnifiedVectorizationManager if available
                if self.use_unified_manager and self.vectorization_manager is not None:
                    if TPRINT_AVAILABLE:
                        tprint("🔧 [VectorBTScaler] Using UnifiedVectorizationManager for transform", color="green")
                    result = self._transform_with_unified_manager(data)
                    self.performance_stats['unified_manager_operations'] += 1
                else:
                    if TPRINT_AVAILABLE:
                        tprint("🔧 [VectorBTScaler] Using VectorBT for transform", color="green")
                    # Use VectorBT scaling with fitted parameters
                    result = self._transform_with_vectorbt(data)
                    self.performance_stats['vectorbt_operations'] += 1
                
                # Validate output
                self._check_output_validity(result, "transformed data")
                
                if TPRINT_AVAILABLE:
                    tprint("✅ [VectorBTScaler] Transform completed successfully", color="green")
                return result
                
            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint(f"⚠️  [VectorBTScaler] VectorBT transform failed: {e}, using fallback", color="yellow")
                self._log_warning(f"⚠️  VectorBT transform failed: {e}, using fallback")
                return self._fallback_transform(data)
        else:
            if TPRINT_AVAILABLE:
                tprint("⚠️  [VectorBTScaler] VectorBT not available or not fitted, using fallback", color="yellow")
            return self._fallback_transform(data)
    
    def _transform_with_unified_manager(self, data: pd.Series) -> pd.Series:
        """Transform using UnifiedVectorizationManager."""
        try:
            # Use the unified vectorization manager for transform
            return self.vectorization_manager.scale_data(data, method=self.method, **self.kwargs)
        except Exception as e:
            logger.warning(f"UnifiedVectorizationManager transform failed: {e}, using VectorBT")
            return self._transform_with_vectorbt(data)
    
    def _transform_with_vectorbt(self, data: pd.Series) -> pd.Series:
        """Transform using basic VectorBT functions."""
        if self.method == 'zscore':
            mean = self.scaling_params['mean']
            std = self.scaling_params['std']
            result = (data - mean) / std
        elif self.method == 'minmax':
            min_val = self.scaling_params['min']
            max_val = self.scaling_params['max']
            result = (data - min_val) / (max_val - min_val)
        elif self.method == 'robust':
            median = self.scaling_params['median']
            mad = self.scaling_params['mad']
            result = (data - median) / mad
        elif self.method == 'quantile':
            # For quantile scaling, we need to use the fitted quantiles
            result = quantile(data, **self.kwargs)
        elif self.method == 'winsorize':
            result = winsorize(data, **self.kwargs)
        elif self.method == 'rank':
            result = rank(data, **self.kwargs)
        elif self.method == 'clip':
            result = clip(data, **self.kwargs)
        else:
            raise ValueError(f"Unsupported scaling method: {self.method}")
        
        return result
    
    def _fallback_fit_transform(self, data: pd.Series) -> pd.Series:
        """Fallback fit_transform using standard methods."""
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [VectorBTScaler] Using fallback fit_transform for method={self.method}", color="yellow")
        
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            if TPRINT_AVAILABLE:
                tprint("⚠️  [VectorBTScaler] No valid data for fallback, returning NaN", color="yellow")
            return pd.Series(np.nan, index=data.index)
        
        try:
            if self.method == 'zscore':
                mean = clean_data.mean()
                std = clean_data.std()
                if std == 0 or np.isnan(std):
                    if TPRINT_AVAILABLE:
                        tprint("⚠️  [VectorBTScaler] Zero std in zscore fallback, using zeros", color="yellow")
                    result = pd.Series(0, index=data.index)
                else:
                    result = (data - mean) / std
                self.scaling_params = {'mean': mean, 'std': std}
            elif self.method == 'minmax':
                min_val = clean_data.min()
                max_val = clean_data.max()
                if max_val == min_val or np.isnan(max_val) or np.isnan(min_val):
                    if TPRINT_AVAILABLE:
                        tprint("⚠️  [VectorBTScaler] Invalid min/max in minmax fallback, using zeros", color="yellow")
                    result = pd.Series(0, index=data.index)
                else:
                    result = (data - min_val) / (max_val - min_val)
                self.scaling_params = {'min': min_val, 'max': max_val}
            elif self.method == 'robust':
                median = clean_data.median()
                mad = (clean_data - median).abs().median()
                if mad == 0 or np.isnan(mad):
                    if TPRINT_AVAILABLE:
                        tprint("⚠️  [VectorBTScaler] Zero MAD in robust fallback, using zeros", color="yellow")
                    result = pd.Series(0, index=data.index)
                else:
                    result = (data - median) / mad
                self.scaling_params = {'median': median, 'mad': mad}
            else:
                # For other methods, use simple z-score as fallback
                if TPRINT_AVAILABLE:
                    tprint(f"⚠️  [VectorBTScaler] Unsupported method {self.method}, using zscore fallback", color="yellow")
                mean = clean_data.mean()
                std = clean_data.std()
                if std == 0 or np.isnan(std):
                    result = pd.Series(0, index=data.index)
                else:
                    result = (data - mean) / std
                self.scaling_params = {'mean': mean, 'std': std}
            
            self.fitted = True
            if TPRINT_AVAILABLE:
                tprint("✅ [VectorBTScaler] Fallback fit_transform completed", color="green")
            return result
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint(f"❌ [VectorBTScaler] Fallback fit_transform failed: {e}", color="red")
            # Ultimate fallback - return zeros
            return pd.Series(0, index=data.index)
    
    def _fallback_transform(self, data: pd.Series) -> pd.Series:
        """Fallback transform using fitted parameters."""
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [VectorBTScaler] Using fallback transform for method={self.method}", color="yellow")
        
        try:
            if self.method == 'zscore':
                mean = self.scaling_params.get('mean', 0)
                std = self.scaling_params.get('std', 1)
                if std == 0 or np.isnan(std):
                    if TPRINT_AVAILABLE:
                        tprint("⚠️  [VectorBTScaler] Zero std in zscore fallback transform, using zeros", color="yellow")
                    return pd.Series(0, index=data.index)
                return (data - mean) / std
            elif self.method == 'minmax':
                min_val = self.scaling_params.get('min', 0)
                max_val = self.scaling_params.get('max', 1)
                if max_val == min_val or np.isnan(max_val) or np.isnan(min_val):
                    if TPRINT_AVAILABLE:
                        tprint("⚠️  [VectorBTScaler] Invalid min/max in minmax fallback transform, using zeros", color="yellow")
                    return pd.Series(0, index=data.index)
                return (data - min_val) / (max_val - min_val)
            elif self.method == 'robust':
                median = self.scaling_params.get('median', 0)
                mad = self.scaling_params.get('mad', 1)
                if mad == 0 or np.isnan(mad):
                    if TPRINT_AVAILABLE:
                        tprint("⚠️  [VectorBTScaler] Zero MAD in robust fallback transform, using zeros", color="yellow")
                    return pd.Series(0, index=data.index)
                return (data - median) / mad
            else:
                # Fallback to z-score
                if TPRINT_AVAILABLE:
                    tprint(f"⚠️  [VectorBTScaler] Unsupported method {self.method} in fallback transform, using zscore", color="yellow")
                mean = self.scaling_params.get('mean', 0)
                std = self.scaling_params.get('std', 1)
                if std == 0 or np.isnan(std):
                    return pd.Series(0, index=data.index)
                return (data - mean) / std
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint(f"❌ [VectorBTScaler] Fallback transform failed: {e}", color="red")
            # Ultimate fallback - return zeros
            return pd.Series(0, index=data.index)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for persistence."""
        return {
            'method': self.method,
            'kwargs': self.kwargs,
            'scaling_params': self.scaling_params,
            'fitted': self.fitted
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore scaler state from persistence."""
        self.method = state.get('method', 'zscore')
        self.kwargs = state.get('kwargs', {})
        self.scaling_params = state.get('scaling_params', {})
        self.fitted = state.get('fitted', False)


class VectorBTBatchScaler:
    """
    VectorBT-optimized batch scaler for processing multiple features efficiently.
    
    This scaler can process multiple features simultaneously using VectorBT's
    batch processing capabilities with enhanced optimization.
    """
    
    def __init__(self, method: str = 'zscore', enable_gpu: bool = False, 
                 memory_efficient: bool = True, enable_parallel: bool = True,
                 use_optimizer: bool = True, use_unified_manager: bool = True, **kwargs):
        """
        Initialize VectorBT batch scaler with enhanced optimization.
        
        Args:
            method: Scaling method
            enable_gpu: Enable GPU acceleration if available
            memory_efficient: Enable memory optimization
            enable_parallel: Enable parallel processing
            use_optimizer: Whether to use VectorBTRollingOptimizer
            use_unified_manager: Whether to use UnifiedVectorizationManager
            **kwargs: Additional parameters
        """
        self.method = method
        self.kwargs = kwargs
        self.scalers = {}
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.memory_efficient = memory_efficient
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
        self.use_optimizer = use_optimizer and VECTORBT_OPTIMIZER_AVAILABLE
        self.use_unified_manager = use_unified_manager and VECTORBT_OPTIMIZER_AVAILABLE
        
        # Initialize optimization components
        if self.use_optimizer:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.enable_gpu,
                enable_parallel=self.enable_parallel,
                memory_efficient=self.memory_efficient
            )
        else:
            self.rolling_optimizer = None
        
        if self.use_unified_manager:
            self.vectorization_manager = get_unified_vectorization_manager()
        else:
            self.vectorization_manager = None
        
        # Enhanced performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'optimizer_operations': 0,
            'unified_manager_operations': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'memory_optimizations': 0,
            'parallel_operations': 0,
            'total_operations': 0
        }
        
        if not VECTORBT_AVAILABLE:
            logger.warning("VectorBT not available, using fallback batch scaler")
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform multiple features using VectorBT batch processing with optimization."""
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [VectorBTBatchScaler] Starting batch fit_transform with method={self.method} on {data.shape[0]}x{data.shape[1]} data", color="cyan")
        
        self.performance_stats['total_operations'] += 1
        
        if not VECTORBT_AVAILABLE:
            if TPRINT_AVAILABLE:
                tprint("⚠️  [VectorBTBatchScaler] VectorBT not available, using fallback", color="yellow")
            return self._fallback_fit_transform(data)
        
        # Optimize DataFrame for VectorBT processing
        if self.memory_efficient:
            if TPRINT_AVAILABLE:
                tprint("🔧 [VectorBTBatchScaler] Optimizing DataFrame types for memory efficiency", color="blue")
            data = self._optimize_dataframe_types(data)
        
        # Enable GPU processing if available
        if self.enable_gpu:
            if TPRINT_AVAILABLE:
                tprint("🚀 [VectorBTBatchScaler] Enabling GPU processing for batch operations", color="magenta")
            data = self._enable_gpu_dataframe_processing(data)
            self.performance_stats['gpu_operations'] += 1
        
        try:
            # Use UnifiedVectorizationManager if available
            if self.use_unified_manager and self.vectorization_manager is not None:
                if TPRINT_AVAILABLE:
                    tprint("🔧 [VectorBTBatchScaler] Using UnifiedVectorizationManager for batch scaling", color="green")
                result = self._apply_unified_batch_scaling(data)
                self.performance_stats['unified_manager_operations'] += 1
            else:
                if TPRINT_AVAILABLE:
                    tprint("🔧 [VectorBTBatchScaler] Using enhanced VectorBT batch scaling", color="green")
                # Use enhanced VectorBT batch scaling
                result = self._apply_enhanced_vectorbt_batch_scaling(data)
                self.performance_stats['vectorbt_operations'] += 1
            
            # Store scaling parameters for each column
            self._store_scaling_parameters(data)
            
            self.performance_stats['batch_operations'] += 1
            
            if TPRINT_AVAILABLE:
                tprint("✅ [VectorBTBatchScaler] Batch fit_transform completed successfully", color="green")
            return result
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint(f"⚠️  [VectorBTBatchScaler] VectorBT batch scaling failed: {e}, using fallback", color="yellow")
            logger.warning(f"VectorBT batch scaling failed: {e}, using fallback")
            return self._fallback_fit_transform(data)
    
    def _apply_enhanced_vectorbt_batch_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply enhanced VectorBT batch scaling with advanced methods."""
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [VectorBTBatchScaler] Applying {self.method} batch scaling", color="blue")
        
        if self.method == 'adaptive':
            # Adaptive scaling for each column
            result = data.copy()
            for col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    # Create a temporary VectorBTScaler for adaptive scaling
                    temp_scaler = VectorBTScaler('adaptive', **self.kwargs)
                    scaled_data, _ = temp_scaler._apply_scaling_method(col_data, 'adaptive', **self.kwargs)
                    result[col] = scaled_data.reindex(data[col].index)
            return result
        else:
            # Use VectorBT batch functions for non-adaptive methods
            if self.method == 'zscore':
                return zscore(data, **self.kwargs)
            elif self.method == 'minmax':
                return scale(data, method='minmax', **self.kwargs)
            elif self.method == 'robust':
                return scale(data, method='robust', **self.kwargs)
            elif self.method == 'quantile':
                return quantile(data, **self.kwargs)
            elif self.method == 'winsorize':
                return winsorize(data, **self.kwargs)
            elif self.method == 'rank':
                return rank(data, **self.kwargs)
            elif self.method == 'clip':
                return clip(data, **self.kwargs)
            elif self.method == 'robust_zscore':
                # Enhanced robust z-score for batch processing
                median = data.median()
                mad = (data - median).abs().median()
                return (data - median) / (1.4826 * mad)
            elif self.method == 'quantile_robust':
                # Robust quantile scaling for batch processing
                q25 = data.quantile(0.25)
                q75 = data.quantile(0.75)
                return (data - q25) / (q75 - q25 + 1e-8)
            else:
                raise ValueError(f"Unsupported scaling method: {self.method}")
    
    def _apply_unified_batch_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply batch scaling using UnifiedVectorizationManager for optimal performance."""
        try:
            # Use the unified vectorization manager for batch scaling
            result = self.vectorization_manager.scale_data(data, method=self.method, **self.kwargs)
            return result
        except Exception as e:
            logger.warning(f"UnifiedVectorizationManager batch scaling failed: {e}, using enhanced VectorBT")
            return self._apply_enhanced_vectorbt_batch_scaling(data)
    
    def _store_scaling_parameters(self, data: pd.DataFrame) -> None:
        """Store scaling parameters for each column."""
        for col in data.columns:
            # Create a temporary VectorBTScaler to get parameters for each column
            temp_scaler = VectorBTScaler(self.method, **self.kwargs)
            _, params = temp_scaler._apply_scaling_method(data[col], self.method, **self.kwargs)
            self.scalers[col] = params
    
    def _optimize_dataframe_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame types for memory efficiency."""
        if self.memory_efficient:
            optimized_data = data.copy()
            for column in optimized_data.columns:
                if optimized_data[column].dtype == 'float64':
                    if (optimized_data[column].min() >= np.finfo(np.float32).min and 
                        optimized_data[column].max() <= np.finfo(np.float32).max):
                        optimized_data[column] = optimized_data[column].astype(np.float32)
                        self.performance_stats['memory_optimizations'] += 1
            return optimized_data
        return data
    
    def _enable_gpu_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enable GPU DataFrame processing if available."""
        if self.enable_gpu and CUPY_AVAILABLE:
            try:
                gpu_data = {}
                for column in data.columns:
                    gpu_data[column] = cp.asarray(data[column].values)
                return pd.DataFrame(gpu_data, index=data.index)
            except Exception as e:
                logger.warning(f"GPU DataFrame processing failed: {e}")
                return data
        return data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted parameters."""
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [VectorBTBatchScaler] Starting batch transform with method={self.method} on {data.shape[0]}x{data.shape[1]} data", color="cyan")
        
        if not self.scalers:
            if TPRINT_AVAILABLE:
                tprint("❌ [VectorBTBatchScaler] Batch scaler must be fitted before transform", color="red")
            raise ValueError("Batch scaler must be fitted before transform")
        
        if not VECTORBT_AVAILABLE:
            if TPRINT_AVAILABLE:
                tprint("⚠️  [VectorBTBatchScaler] VectorBT not available, using fallback", color="yellow")
            return self._fallback_transform(data)
        
        try:
            # Use VectorBT batch scaling with fitted parameters
            if self.method == 'zscore':
                result = data.copy()
                for col in data.columns:
                    if col in self.scalers:
                        mean = self.scalers[col]['mean']
                        std = self.scalers[col]['std']
                        if std != 0:
                            result[col] = (data[col] - mean) / std
                        else:
                            result[col] = 0
            elif self.method == 'minmax':
                result = data.copy()
                for col in data.columns:
                    if col in self.scalers:
                        min_val = self.scalers[col]['min']
                        max_val = self.scalers[col]['max']
                        if max_val != min_val:
                            result[col] = (data[col] - min_val) / (max_val - min_val)
                        else:
                            result[col] = 0
            elif self.method == 'robust':
                result = data.copy()
                for col in data.columns:
                    if col in self.scalers:
                        median = self.scalers[col]['median']
                        mad = self.scalers[col]['mad']
                        if mad != 0:
                            result[col] = (data[col] - median) / mad
                        else:
                            result[col] = 0
            else:
                # For other methods, use VectorBT directly
                if self.method == 'quantile':
                    result = quantile(data, **self.kwargs)
                elif self.method == 'winsorize':
                    result = winsorize(data, **self.kwargs)
                elif self.method == 'rank':
                    result = rank(data, **self.kwargs)
                elif self.method == 'clip':
                    result = clip(data, **self.kwargs)
                else:
                    raise ValueError(f"Unsupported scaling method: {self.method}")
            
            if TPRINT_AVAILABLE:
                tprint("✅ [VectorBTBatchScaler] Batch transform completed successfully", color="green")
            return result
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint(f"⚠️  [VectorBTBatchScaler] VectorBT batch transform failed: {e}, using fallback", color="yellow")
            logger.warning(f"VectorBT batch transform failed: {e}, using fallback")
            return self._fallback_transform(data)
    
    def _fallback_fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback batch fit_transform using standard methods."""
        result = data.copy()
        
        for col in data.columns:
            scaler = VectorBTScaler(self.method, **self.kwargs)
            result[col] = scaler.fit_transform(data[col])
            self.scalers[col] = scaler.scaling_params
        
        return result
    
    def _fallback_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback batch transform using fitted parameters."""
        result = data.copy()
        
        for col in data.columns:
            if col in self.scalers:
                scaler = VectorBTScaler(self.method, **self.kwargs)
                scaler.scaling_params = self.scalers[col]
                scaler.fitted = True
                result[col] = scaler.transform(data[col])
            else:
                result[col] = data[col]  # Keep original if not fitted
        
        return result


# These functions are now available through the base_scaler module's create_optimized_scaler functions


# Available scaling methods
VECTORBT_SCALING_METHODS = [
    'zscore', 'minmax', 'robust', 'quantile', 
    'winsorize', 'rank', 'clip', 'robust_zscore', 
    'adaptive', 'quantile_robust', 'winsorize_adaptive'
] if VECTORBT_AVAILABLE else []


def get_available_scaling_methods() -> List[str]:
    """Get list of available scaling methods."""
    if VECTORBT_AVAILABLE:
        return VECTORBT_SCALING_METHODS
    else:
        return ['zscore', 'minmax', 'robust']  # Fallback methods


# Performance stats functions are now available through the scaler instances directly