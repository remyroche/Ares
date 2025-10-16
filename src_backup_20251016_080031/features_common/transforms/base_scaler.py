"""
Enhanced Base Scaler Interface with All Optimizations

Provides a shared interface for all scaling and normalization operations
across feature_generation and feature_engineering_roadmap systems with
comprehensive optimization including VectorBT, caching, performance monitoring,
and intelligent fallback mechanisms.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

# Import common utilities
from ..utils import (
    TPRINT_AVAILABLE, tprint,
    VECTORBT_AVAILABLE, vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply,
    scale, rank, zscore, winsorize, clip, quantile,
    VECTORBT_OPTIMIZER_AVAILABLE, VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
    UnifiedVectorizationManager, get_unified_vectorization_manager,
    MATH_VALIDATION_AVAILABLE, safe_divide, check_for_inf_nan, validate_numeric_array, is_valid_number
)

# Import mixins for comprehensive optimization
from ..mixins import (
    OptimizationMixin, PerformanceMixin, VectorBTMixin, 
    ValidationMixin, CachingMixin, MonitoringMixin
)

# Import unified VectorBT manager
from ..vectorbt import get_unified_vectorbt_manager

# Import configuration
from ..config import get_unified_config

# Check if VectorBT scaler is available
try:
    from .vectorbt_scaler import VectorBTScaler, VectorBTBatchScaler
    VECTORBT_SCALER_AVAILABLE = True
except ImportError:
    VECTORBT_SCALER_AVAILABLE = False
    VectorBTScaler = None
    VectorBTBatchScaler = None

logger = logging.getLogger(__name__)

class BaseScaler(ABC, OptimizationMixin, PerformanceMixin, VectorBTMixin, ValidationMixin, CachingMixin, MonitoringMixin):
    """
    Enhanced base class for all scaling/transformation operations with comprehensive optimization.
    
    This interface ensures consistency between feature_generation's normalization
    and feature_engineering_roadmap's transform systems while providing comprehensive
    optimization including VectorBT, caching, performance monitoring, and intelligent fallback.
    
    All scalers must implement:
    - fit_transform: Fit parameters and transform data
    - transform: Transform new data using fitted parameters
    - get_state: Serialize state for persistence
    - set_state: Restore state from persistence
    """
    
    def __init__(self, use_vectorbt: bool = True, enable_gpu: bool = False, vectorbt_threshold: int = 100,
                 use_optimizer: bool = True, use_unified_manager: bool = True, **kwargs):
        """
        Initialize the enhanced scaler with all optimizations.
        
        Args:
            use_vectorbt: Whether to use VectorBT optimizations
            enable_gpu: Whether to enable 
            vectorbt_threshold: Minimum data size for VectorBT optimization
            use_optimizer: Whether to use VectorBTRollingOptimizer
            use_unified_manager: Whether to use UnifiedVectorizationManager
            **kwargs: Additional configuration parameters
        """
        # Initialize all mixins first
        super().__init__()
        
        # Get unified configuration
        self.config = get_unified_config()
        
        # Scaler state
        self.fitted = False
        self.scaling_params = {}
        
        # Legacy compatibility
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.enable_gpu = False  # GPU support removed
        self.vectorbt_threshold = vectorbt_threshold
        self.use_optimizer = use_optimizer and VECTORBT_OPTIMIZER_AVAILABLE
        self.use_unified_manager = use_unified_manager and VECTORBT_OPTIMIZER_AVAILABLE
        
        # Initialize unified VectorBT manager
        self.vectorbt_manager = get_unified_vectorbt_manager()
        
        # Initialize optimization components
        if self.use_optimizer:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.enable_gpu,
                enable_parallel=True,
                memory_efficient=True
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
            'gpu_accelerations': 0,
            'pandas_fallbacks': 0,
            'memory_optimizations': 0,
            'total_operations': 0
        }
        
        # Enable all optimizations by default
        self.enable_optimization()
        self.enable_performance_monitoring()
    
    def _should_use_vectorbt(self, data: pd.Series) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [BaseScaler] Checking VectorBT usage: use_vectorbt={self.use_vectorbt}, data_size={len(data)}, threshold={self.vectorbt_threshold}", color="cyan")
        return (self.use_vectorbt and 
                len(data) >= self.vectorbt_threshold and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """
        Perform VectorBT rolling operation with fallback to pandas.
        
        Args:
            data: Input data series
            operation: Operation type ('mean', 'std', 'var', 'min', 'max', 'sum')
            window: Rolling window size
            **kwargs: Additional parameters
            
        Returns:
            Result of rolling operation
        """
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [BaseScaler] Performing VectorBT rolling operation: {operation} with window={window}", color="cyan")
        
        if not self._should_use_vectorbt(data):
            if TPRINT_AVAILABLE:
                tprint("⚠️  [BaseScaler] VectorBT not suitable, using pandas fallback", color="yellow")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        self.performance_stats['total_operations'] += 1
        
        # Use VectorBTRollingOptimizer if available
        if self.use_optimizer and self.rolling_optimizer is not None:
            try:
                self.performance_stats['optimizer_operations'] += 1
                if operation == 'mean':
                    return self.rolling_optimizer.rolling_mean(data, window=window, **kwargs)
                elif operation == 'std':
                    return self.rolling_optimizer.rolling_std(data, window=window, **kwargs)
                elif operation == 'var':
                    return self.rolling_optimizer.rolling_var(data, window=window, **kwargs)
                elif operation == 'min':
                    return self.rolling_optimizer.rolling_min(data, window=window, **kwargs)
                elif operation == 'max':
                    return self.rolling_optimizer.rolling_max(data, window=window, **kwargs)
                elif operation == 'sum':
                    return self.rolling_optimizer.rolling_sum(data, window=window, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
            except Exception as e:
                logger.warning(f"VectorBTRollingOptimizer failed: {e}, using basic VectorBT")
                # Fall through to basic VectorBT
        
        # Use basic VectorBT functions
        try:
            self.performance_stats['vectorbt_operations'] += 1
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        
        except Exception as e:
            logger.warning(f"VectorBT rolling operation failed: {e}, using pandas fallback")
            self.performance_stats['pandas_fallbacks'] += 1
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [BaseScaler] Using pandas fallback for rolling operation: {operation} with window={window}", color="yellow")
        
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            if TPRINT_AVAILABLE:
                tprint(f"❌ [BaseScaler] Unsupported operation: {operation}", color="red")
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """
        Perform VectorBT rolling apply operation with fallback to pandas.
        
        Args:
            data: Input data series
            func: Function to apply
            window: Rolling window size
            **kwargs: Additional parameters
            
        Returns:
            Result of rolling apply operation
        """
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [BaseScaler] Performing VectorBT rolling apply operation with window={window}", color="cyan")
        
        if not self._should_use_vectorbt(data):
            if TPRINT_AVAILABLE:
                tprint("⚠️  [BaseScaler] VectorBT not suitable for apply, using pandas fallback", color="yellow")
            return data.rolling(window=window).apply(func, **kwargs)
        
        self.performance_stats['total_operations'] += 1
        
        # Use VectorBTRollingOptimizer if available
        if self.use_optimizer and self.rolling_optimizer is not None:
            try:
                self.performance_stats['optimizer_operations'] += 1
                return self.rolling_optimizer.rolling_apply(data, func, window=window, **kwargs)
            except Exception as e:
                logger.warning(f"VectorBTRollingOptimizer apply failed: {e}, using basic VectorBT")
                # Fall through to basic VectorBT
        
        # Use basic VectorBT functions
        try:
            self.performance_stats['vectorbt_operations'] += 1
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            self.performance_stats['pandas_fallbacks'] += 1
            return data.rolling(window=window).apply(func, **kwargs)
    
    def fit_transform(self, data: pd.Series) -> pd.Series:
        """
        Enhanced fit_transform with all optimizations.
        
        This method provides comprehensive optimization including:
        - Data validation and sanitization
        - Automatic optimization selection
        - VectorBT acceleration when beneficial
        - Caching for repeated operations
        - Performance monitoring
        
        Args:
            data: Training data to fit and transform
            
        Returns:
            Transformed data with all optimizations applied
        """
        # Validate and sanitize input data
        sanitized_data, is_valid, warnings = self.validate_and_sanitize(data, "input data")
        if not is_valid and warnings:
            self._log_warning(f"Data validation warnings: {warnings}")
        
        # Use cached operation if available
        if hasattr(self, 'cached_operation'):
            return self.cached_operation(self._fit_transform_impl, sanitized_data)
        else:
            return self._fit_transform_impl(sanitized_data)
    
    def _fit_transform_impl(self, data: pd.Series) -> pd.Series:
        """Implementation of fit_transform with all optimizations."""
        # Use auto-optimization for the core operation
        return self.auto_optimize_operation(self._core_fit_transform, data)
    
    @abstractmethod
    def _core_fit_transform(self, data: pd.Series) -> pd.Series:
        """
        Core fit_transform implementation (to be implemented by subclasses).
        
        Args:
            data: Training data to fit and transform
            
        Returns:
            Transformed data
        """
        pass
    
    def transform(self, data: pd.Series) -> pd.Series:
        """
        Enhanced transform with all optimizations.
        
        This method provides comprehensive optimization including:
        - Data validation and sanitization
        - Automatic optimization selection
        - VectorBT acceleration when beneficial
        - Caching for repeated operations
        - Performance monitoring
        
        Args:
            data: New data to transform
            
        Returns:
            Transformed data with all optimizations applied
            
        Raises:
            ValueError: If scaler has not been fitted
        """
        # Validate that scaler has been fitted
        self._validate_fitted()
        
        # Validate and sanitize input data
        sanitized_data, is_valid, warnings = self.validate_and_sanitize(data, "input data")
        if not is_valid and warnings:
            self._log_warning(f"Data validation warnings: {warnings}")
        
        # Use cached operation if available
        if hasattr(self, 'cached_operation'):
            return self.cached_operation(self._transform_impl, sanitized_data)
        else:
            return self._transform_impl(sanitized_data)
    
    def _transform_impl(self, data: pd.Series) -> pd.Series:
        """Implementation of transform with all optimizations."""
        # Use auto-optimization for the core operation
        return self.auto_optimize_operation(self._core_transform, data)
    
    @abstractmethod
    def _core_transform(self, data: pd.Series) -> pd.Series:
        """
        Core transform implementation (to be implemented by subclasses).
        
        Args:
            data: New data to transform
            
        Returns:
            Transformed data
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get current state for persistence.
        
        Returns:
            Dictionary containing all state needed to restore this scaler
        """
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore scaler state from persistence.
        
        Args:
            state: State dictionary from get_state()
        """
        pass
    
    def is_fitted(self) -> bool:
        """
        Check if scaler has been fitted.
        
        Returns:
            True if fitted, False otherwise
        """
        return self.fitted
    
    def _validate_fitted(self) -> None:
        """
        Validate that scaler has been fitted before transforming.
        
        Raises:
            ValueError: If scaler has not been fitted
        """
        if not self.fitted:
            error_msg = (
                f"{self.__class__.__name__} must be fitted before calling transform(). "
                "Call fit_transform() first."
            )
            if TPRINT_AVAILABLE:
                tprint(f"❌ {error_msg}", color="red", bold=True)
            raise ValueError(error_msg)
    
    def _log_info(self, message: str) -> None:
        """Log info message using tprint if available, otherwise standard logging."""
        if TPRINT_AVAILABLE:
            tprint(message, color="cyan")
        else:
            logger.info(message)
    
    def _log_success(self, message: str) -> None:
        """Log success message using tprint if available."""
        if TPRINT_AVAILABLE:
            tprint(message, color="green")
        else:
            logger.info(message)
    
    def _log_warning(self, message: str) -> None:
        """Log warning message using tprint if available."""
        if TPRINT_AVAILABLE:
            tprint(message, color="yellow")
        else:
            logger.warning(message)
    
    def _log_error(self, message: str) -> None:
        """Log error message using tprint if available."""
        if TPRINT_AVAILABLE:
            tprint(message, color="red")
        else:
            logger.error(message)
    
    def _validate_numeric_input(self, data: pd.Series, name: str = "input") -> None:
        """
        Validate that input data is numeric.
        
        Args:
            data: Data to validate
            name: Name of the data for error messages
        """
        if MATH_VALIDATION_AVAILABLE:
            try:
                validate_numeric_array(data.values, name)
            except Exception as e:
                self._log_warning(f"Validation warning for {name}: {e}")
    
    def _safe_divide(self, numerator: pd.Series, denominator: float, 
                     default: float = 0.0) -> pd.Series:
        """
        Safely divide series by scalar, handling zero/inf/nan.
        
        Args:
            numerator: Numerator series
            denominator: Denominator scalar
            default: Default value for invalid results
            
        Returns:
            Result of division with safe handling
        """
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [BaseScaler] Performing safe division with denominator={denominator}", color="blue")
        
        if MATH_VALIDATION_AVAILABLE:
            # Use math_validation's safe_divide
            result = pd.Series(
                safe_divide(numerator.values, denominator, default=default),
                index=numerator.index
            )
        else:
            # Fallback implementation
            if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
                if TPRINT_AVAILABLE:
                    tprint(f"⚠️  [BaseScaler] Invalid denominator ({denominator}), using default value", color="yellow")
                return pd.Series(default, index=numerator.index)
            result = numerator / denominator
            result = result.replace([np.inf, -np.inf], default).fillna(default)
        
        # Validate result
        if result.isna().any():
            if TPRINT_AVAILABLE:
                tprint(f"⚠️  [BaseScaler] NaN values detected in division result, filling with default", color="yellow")
            result = result.fillna(default)
        
        return result
    
    def _check_output_validity(self, data: pd.Series, name: str = "output") -> None:
        """
        Check output for inf/nan values.
        
        Args:
            data: Data to check
            name: Name of the data for error messages
        """
        if MATH_VALIDATION_AVAILABLE:
            try:
                check_for_inf_nan(data.values, name)
            except Exception as e:
                self._log_warning(f"Output validation warning for {name}: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        if TPRINT_AVAILABLE:
            tprint("🔧 [BaseScaler] Getting performance statistics", color="cyan")
        
        stats = self.performance_stats.copy()
        
        # Add optimizer stats if available
        if self.use_optimizer and self.rolling_optimizer is not None:
            optimizer_stats = self.rolling_optimizer.get_performance_stats()
            stats.update({
                'optimizer_' + k: v for k, v in optimizer_stats.items()
            })
        
        # Add unified manager stats if available
        if self.use_unified_manager and self.vectorization_manager is not None:
            manager_stats = self.vectorization_manager.get_performance_stats()
            stats.update({
                'manager_' + k: v for k, v in manager_stats.items()
            })
        
        # Calculate efficiency metrics
        if stats['total_operations'] > 0:
            stats['optimizer_usage_rate'] = stats.get('optimizer_operations', 0) / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats.get('vectorbt_operations', 0) / stats['total_operations']
            stats['pandas_fallback_rate'] = stats.get('pandas_fallbacks', 0) / stats['total_operations']
        else:
            stats['optimizer_usage_rate'] = 0
            stats['vectorbt_usage_rate'] = 0
            stats['pandas_fallback_rate'] = 0
        
        return stats
    
    def reset_performance_stats(self) -> None:
        """Reset all performance statistics."""
        if TPRINT_AVAILABLE:
            tprint("🔧 [BaseScaler] Resetting performance statistics", color="cyan")
        
        self.performance_stats = {
            'vectorbt_operations': 0,
            'optimizer_operations': 0,
            'unified_manager_operations': 0,
            'gpu_accelerations': 0,
            'pandas_fallbacks': 0,
            'memory_optimizations': 0,
            'total_operations': 0
        }
        
        # Reset optimizer stats if available
        if self.use_optimizer and self.rolling_optimizer is not None:
            self.rolling_optimizer.reset_stats()
        
        # Reset manager stats if available
        if self.use_unified_manager and self.vectorization_manager is not None:
            self.vectorization_manager.reset_stats()

class SimpleScaler(BaseScaler):
    """
    Enhanced simple scaler implementation with all optimizations.
    
    This is a z-score normalization scaler that automatically uses
    all available optimizations including VectorBT, caching, and performance monitoring.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
    
    def _core_fit_transform(self, data: pd.Series) -> pd.Series:
        """Core fit_transform implementation with z-score normalization."""
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [SimpleScaler] Fitting on {len(data)} samples", color="cyan")
        
        # Remove NaN values for fitting
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            error_msg = "No valid data to fit scaler"
            self._log_error(f"❌ [SimpleScaler] {error_msg}")
            raise ValueError(error_msg)
        
        try:
            self.mean = float(clean_data.mean())
            self.std = float(clean_data.std())
            
            # Prevent division by zero
            if self.std == 0 or np.isnan(self.std) or np.isinf(self.std):
                error_msg = f"Invalid std value: {self.std}"
                self._log_error(f"❌ [SimpleScaler] {error_msg}")
                raise ValueError(error_msg)
            
            self.fitted = True
            if TPRINT_AVAILABLE:
                tprint(f"✅ [SimpleScaler] Fitted: mean={self.mean:.4f}, std={self.std:.4f}", color="green")
            
            # Transform the data
            result = self._core_transform(data)
            
            # Validate output
            self._check_output_validity(result, "transformed data")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to fit scaler: {e}"
            self._log_error(f"❌ [SimpleScaler] {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _core_transform(self, data: pd.Series) -> pd.Series:
        """Core transform implementation with z-score normalization."""
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [SimpleScaler] Transforming {len(data)} samples", color="cyan")
        
        if self.mean is None or self.std is None:
            error_msg = "Scaler state is invalid - mean or std is None"
            self._log_error(f"❌ [SimpleScaler] {error_msg}")
            raise ValueError(error_msg)
        
        try:
            # Use safe division
            result = self._safe_divide(data - self.mean, self.std, default=0.0)
            
            if TPRINT_AVAILABLE:
                tprint("✅ [SimpleScaler] Transform completed successfully", color="green")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to transform data: {e}"
            self._log_error(f"❌ [SimpleScaler] {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def get_state(self) -> Dict[str, Any]:
        """Get state for persistence."""
        return {
            'mean': self.mean,
            'std': self.std,
            'fitted': self.fitted
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state from persistence."""
        self.mean = state.get('mean')
        self.std = state.get('std')
        self.fitted = state.get('fitted', False)

def create_optimized_scaler(method: str = 'zscore', use_vectorbt: bool = True, 
                           use_optimizer: bool = True, use_unified_manager: bool = True, **kwargs) -> BaseScaler:
    """
    Create the best available scaler with VectorBT optimization.
    
    Args:
        method: Scaling method ('zscore', 'minmax', 'robust', etc.)
        use_vectorbt: Whether to prefer VectorBT scaler when available
        use_optimizer: Whether to use VectorBTRollingOptimizer
        use_unified_manager: Whether to use UnifiedVectorizationManager
        **kwargs: Additional parameters for the scaler
        
    Returns:
        Best available scaler instance with optimization
    """
    if use_vectorbt and VECTORBT_SCALER_AVAILABLE and VECTORBT_AVAILABLE:
        try:
            # Create VectorBTScaler with optimization
            return VectorBTScaler(method, use_optimizer=use_optimizer, 
                                use_unified_manager=use_unified_manager, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to create VectorBT scaler: {e}, using fallback")
    
    # Fallback to simple scaler with optimization
    if method == 'zscore':
        return SimpleScaler(use_optimizer=use_optimizer, use_unified_manager=use_unified_manager)
    else:
        # For other methods, use VectorBT scaler as fallback if available
        if VECTORBT_SCALER_AVAILABLE and VECTORBT_AVAILABLE:
            try:
                return VectorBTScaler(method, use_optimizer=use_optimizer, 
                                    use_unified_manager=use_unified_manager, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to create VectorBT scaler for {method}: {e}")
        
        # Ultimate fallback to simple scaler with optimization
        return SimpleScaler(use_optimizer=use_optimizer, use_unified_manager=use_unified_manager)

def create_optimized_batch_scaler(method: str = 'zscore', use_vectorbt: bool = True, 
                                 use_optimizer: bool = True, use_unified_manager: bool = True, **kwargs):
    """
    Create the best available batch scaler with VectorBT optimization.
    
    Args:
        method: Scaling method
        use_vectorbt: Whether to prefer VectorBT scaler when available
        use_optimizer: Whether to use VectorBTRollingOptimizer
        use_unified_manager: Whether to use UnifiedVectorizationManager
        **kwargs: Additional parameters for the scaler
        
    Returns:
        Best available batch scaler instance with optimization
        
    Raises:
        ImportError: If VectorBT batch scaler is not available and fallback fails
    """
    if TPRINT_AVAILABLE:
        tprint(f"🔧 [create_optimized_batch_scaler] Creating batch scaler with method={method}", color="cyan")
    
    if use_vectorbt and VECTORBT_SCALER_AVAILABLE and VECTORBT_AVAILABLE:
        try:
            scaler = VectorBTBatchScaler(method, use_optimizer=use_optimizer, 
                                       use_unified_manager=use_unified_manager, **kwargs)
            if TPRINT_AVAILABLE:
                tprint("✅ [create_optimized_batch_scaler] VectorBT batch scaler created successfully", color="green")
            return scaler
        except Exception as e:
            error_msg = f"Failed to create VectorBT batch scaler: {e}"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [create_optimized_batch_scaler] {error_msg}", color="red")
            raise ImportError(error_msg) from e
    
    # If VectorBT is not available, raise error instead of silent fallback
    error_msg = "VectorBT batch scaler not available and no fallback implemented"
    if TPRINT_AVAILABLE:
        tprint(f"❌ [create_optimized_batch_scaler] {error_msg}", color="red")
    raise ImportError(error_msg)
