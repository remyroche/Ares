"""
Base Scaler Interface with Native VectorBT Support

Provides a shared interface for all scaling and normalization operations
across feature_generation and feature_engineering_roadmap systems with
native VectorBT optimization for maximum performance.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
import warnings

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Import utility functions
try:
    from src.utils.tprint import tprint
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    
try:
    from src.utils.math_validation import (
        safe_divide,
        check_for_inf_nan,
        validate_numeric_array,
        is_valid_number
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseScaler(ABC):
    """
    Abstract base class for all scaling/transformation operations with native VectorBT support.
    
    This interface ensures consistency between feature_generation's normalization
    and feature_engineering_roadmap's transform systems while providing VectorBT optimization.
    
    All scalers must implement:
    - fit_transform: Fit parameters and transform data
    - transform: Transform new data using fitted parameters
    - get_state: Serialize state for persistence
    - set_state: Restore state from persistence
    """
    
    def __init__(self, use_vectorbt: bool = True, enable_gpu: bool = False, vectorbt_threshold: int = 1000):
        """
        Initialize the scaler with VectorBT support.
        
        Args:
            use_vectorbt: Whether to use VectorBT optimizations
            enable_gpu: Whether to enable GPU acceleration
            vectorbt_threshold: Minimum data size for VectorBT optimization
        """
        self.fitted = False
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.vectorbt_threshold = vectorbt_threshold
        
        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'gpu_accelerations': 0,
            'pandas_fallbacks': 0
        }
    
    def _should_use_vectorbt(self, data: pd.Series) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
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
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        self.performance_stats['vectorbt_operations'] += 1
        
        try:
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
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        self.performance_stats['vectorbt_operations'] += 1
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            self.performance_stats['pandas_fallbacks'] += 1
            return data.rolling(window=window).apply(func, **kwargs)
    
    @abstractmethod
    def fit_transform(self, data: pd.Series) -> pd.Series:
        """
        Fit scaler parameters on training data and transform it.
        
        Args:
            data: Training data to fit and transform
            
        Returns:
            Transformed data
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.Series) -> pd.Series:
        """
        Transform new data using previously fitted parameters.
        
        Args:
            data: New data to transform
            
        Returns:
            Transformed data
            
        Raises:
            ValueError: If scaler has not been fitted
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
    
    def _log_info(self, message: str, use_tprint: bool = True) -> None:
        """
        Log info message using tprint if available, otherwise standard logging.
        
        Args:
            message: Message to log
            use_tprint: Whether to use tprint (if available)
        """
        if use_tprint and TPRINT_AVAILABLE:
            tprint(message, color="cyan")
        else:
            logger.info(message)
    
    def _log_success(self, message: str, use_tprint: bool = True) -> None:
        """
        Log success message using tprint if available.
        
        Args:
            message: Message to log
            use_tprint: Whether to use tprint (if available)
        """
        if use_tprint and TPRINT_AVAILABLE:
            tprint(message, color="green")
        else:
            logger.info(message)
    
    def _log_warning(self, message: str, use_tprint: bool = True) -> None:
        """
        Log warning message using tprint if available.
        
        Args:
            message: Message to log
            use_tprint: Whether to use tprint (if available)
        """
        if use_tprint and TPRINT_AVAILABLE:
            tprint(message, color="yellow")
        else:
            logger.warning(message)
    
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
        if MATH_VALIDATION_AVAILABLE:
            # Use math_validation's safe_divide
            return pd.Series(
                safe_divide(numerator.values, denominator, default=default),
                index=numerator.index
            )
        else:
            # Fallback implementation
            if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
                return pd.Series(default, index=numerator.index)
            result = numerator / denominator
            result = result.replace([np.inf, -np.inf], default).fillna(default)
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


class SimpleScaler(BaseScaler):
    """
    Simple example implementation of BaseScaler for reference.
    
    This is a basic z-score normalization scaler that can serve as
    a template for implementing other scalers.
    """
    
    def __init__(self):
        super().__init__()
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
    
    def fit_transform(self, data: pd.Series) -> pd.Series:
        """Fit mean/std and transform data with enhanced logging and validation."""
        self._log_info(f"🔧 [SimpleScaler] Fitting on {len(data)} samples")
        
        # Validate input
        self._validate_numeric_input(data, "input data")
        
        # Remove NaN values for fitting
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            self._log_warning("⚠️  No valid data to fit, using defaults")
            self.mean = 0.0
            self.std = 1.0
        else:
            self.mean = float(clean_data.mean())
            self.std = float(clean_data.std())
            
            # Prevent division by zero
            if self.std == 0 or np.isnan(self.std):
                self._log_warning("⚠️  Zero std detected, using 1.0")
                self.std = 1.0
        
        self.fitted = True
        self._log_success(f"✅ [SimpleScaler] Fitted: mean={self.mean:.4f}, std={self.std:.4f}")
        
        transformed = self.transform(data)
        
        # Validate output
        self._check_output_validity(transformed, "transformed data")
        
        return transformed
    
    def transform(self, data: pd.Series) -> pd.Series:
        """Transform data using fitted mean/std with safe division."""
        self._validate_fitted()
        
        if self.mean is None or self.std is None:
            raise ValueError("Scaler state is invalid")
        
        # Use safe division
        return self._safe_divide(data - self.mean, self.std, default=0.0)
    
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


def create_optimized_scaler(method: str = 'zscore', use_vectorbt: bool = True, **kwargs) -> BaseScaler:
    """
    Create the best available scaler (VectorBT if available, otherwise fallback).
    
    Args:
        method: Scaling method ('zscore', 'minmax', 'robust', etc.)
        use_vectorbt: Whether to prefer VectorBT scaler when available
        **kwargs: Additional parameters for the scaler
        
    Returns:
        Best available scaler instance
    """
    if use_vectorbt and VECTORBT_SCALER_AVAILABLE and VECTORBT_AVAILABLE:
        try:
            return VectorBTScaler(method, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to create VectorBT scaler: {e}, using fallback")
    
    # Fallback to simple scaler
    if method == 'zscore':
        return SimpleScaler()
    else:
        # For other methods, use VectorBT scaler as fallback if available
        if VECTORBT_SCALER_AVAILABLE and VECTORBT_AVAILABLE:
            try:
                return VectorBTScaler(method, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to create VectorBT scaler for {method}: {e}")
        
        # Ultimate fallback to simple scaler
        return SimpleScaler()


def create_optimized_batch_scaler(method: str = 'zscore', use_vectorbt: bool = True, **kwargs):
    """
    Create the best available batch scaler (VectorBT if available, otherwise fallback).
    
    Args:
        method: Scaling method
        use_vectorbt: Whether to prefer VectorBT scaler when available
        **kwargs: Additional parameters for the scaler
        
    Returns:
        Best available batch scaler instance
    """
    if use_vectorbt and VECTORBT_SCALER_AVAILABLE and VECTORBT_AVAILABLE:
        try:
            return VectorBTBatchScaler(method, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to create VectorBT batch scaler: {e}, using fallback")
    
    # Fallback: create individual scalers for each column
    class FallbackBatchScaler:
        def __init__(self, method: str = 'zscore', **kwargs):
            self.method = method
            self.kwargs = kwargs
            self.scalers = {}
        
        def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
            result = data.copy()
            for col in data.columns:
                scaler = create_optimized_scaler(self.method, **self.kwargs)
                result[col] = scaler.fit_transform(data[col])
                self.scalers[col] = scaler
            return result
        
        def transform(self, data: pd.DataFrame) -> pd.DataFrame:
            result = data.copy()
            for col in data.columns:
                if col in self.scalers:
                    result[col] = self.scalers[col].transform(data[col])
                else:
                    result[col] = data[col]
            return result
    
    return FallbackBatchScaler(method, **kwargs)
