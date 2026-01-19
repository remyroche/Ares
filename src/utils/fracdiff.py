"""
Fractional Differencing Implementation with ADF-based d-parameter Selection

This module implements fractional differencing (FracDiff) following the de Prado framework
for financial time series analysis. It automatically determines the optimal fractional
differencing order using Augmented Dickey-Fuller (ADF) test statistics.

Key Features:
- ADF-based automatic d-parameter selection
- Memory-preserving fractional differencing
- Stationarity validation
- Back-transformation support
- Numba-optimized for performance

References:
- de Prado, M. (2018). Advances in Financial Machine Learning
- "Fractional differencing for financial time series" - de Prado framework
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, Dict, Any
import warnings
from pathlib import Path

# Try to import statistical packages
try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. ADF functionality will be limited.")

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(n):
        return range(n)

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success


class FracDiffTransformer:
    """
    Fractional Differencing Transformer with ADF-based d-parameter selection.
    
    Implements the de Prado framework for fractional differencing that preserves
    memory while achieving stationarity in financial time series.
    """
    
    def __init__(self, 
                 max_d: float = 1.0,
                 min_d: float = 0.0,
                 adf_threshold: float = 0.01,
                 max_lags: Optional[int] = None,
                 use_numba: bool = True):
        """
        Initialize FracDiff transformer.
        
        Args:
            max_d: Maximum fractional differencing order
            min_d: Minimum fractional differencing order
            adf_threshold: ADF p-value threshold for stationarity
            max_lags: Maximum lags for ADF test (auto-determined if None)
            use_numba: Whether to use Numba optimization
        """
        self.max_d = max_d
        self.min_d = min_d
        self.adf_threshold = adf_threshold
        self.max_lags = max_lags
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.optimal_d = None
        self.adf_stats = {}
        self.is_fitted = False
        
    def find_optimal_d(self, 
                      series: pd.Series,
                      method: str = 'binary_search',
                      tolerance: float = 0.01) -> float:
        """
        Find optimal fractional differencing order using ADF test.
        
        Args:
            series: Input time series
            method: Search method ('binary_search', 'linear', 'grid')
            tolerance: Tolerance for optimal d convergence
            
        Returns:
            Optimal d parameter
        """
        if not STATSMODELS_AVAILABLE:
            tprint_warning("⚠️ statsmodels not available, using default d=0.5")
            return 0.5
            
        tprint_info("🔍 Finding optimal fractional differencing order...")
        
        if method == 'binary_search':
            optimal_d = self._binary_search_d(series, tolerance)
        elif method == 'linear':
            optimal_d = self._linear_search_d(series, tolerance)
        elif method == 'grid':
            optimal_d = self._grid_search_d(series, tolerance)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        self.optimal_d = optimal_d
        self.is_fitted = True
        
        tprint_success(f"✅ Optimal d found: {optimal_d:.4f}")
        return optimal_d
    
    def _binary_search_d(self, series: pd.Series, tolerance: float) -> float:
        """Binary search for optimal d parameter."""
        # Subsample for speed if series is too long
        if len(series) > 5000:
            indices = np.linspace(0, len(series) - 1, 5000).astype(int)
            series = series.iloc[indices]
            tprint_info(f"⚡ Subsampling to {len(series)} samples for d search")
            
        low, high = self.min_d, self.max_d
        best_d = low
        
        while high - low > tolerance:
            mid = (low + high) / 2
            adf_p = self._get_adf_pvalue(series, mid)
            
            self.adf_stats[mid] = adf_p
            
            if adf_p < self.adf_threshold:
                # Stationary achieved, try lower d
                high = mid
                best_d = mid
            else:
                # Not stationary, need higher d
                low = mid
                
        return best_d
    
    def _linear_search_d(self, series: pd.Series, tolerance: float) -> float:
        """Linear search for optimal d parameter."""
        d_values = np.arange(self.min_d, self.max_d + tolerance, tolerance)
        best_d = self.min_d
        min_adf_p = float('inf')
        
        for d in d_values:
            adf_p = self._get_adf_pvalue(series, d)
            self.adf_stats[d] = adf_p
            
            if adf_p < self.adf_threshold and adf_p < min_adf_p:
                best_d = d
                min_adf_p = adf_p
                
        return best_d
    
    def _grid_search_d(self, series: pd.Series, tolerance: float) -> float:
        """Grid search for optimal d parameter."""
        d_values = np.arange(self.min_d, self.max_d + tolerance, tolerance/2)
        best_d = self.min_d
        min_adf_p = float('inf')
        
        for d in d_values:
            adf_p = self._get_adf_pvalue(series, d)
            self.adf_stats[d] = adf_p
            
            # Find the smallest d that achieves stationarity
            if adf_p < self.adf_threshold and d < best_d:
                best_d = d
                min_adf_p = adf_p
                
        return best_d
    
    def _get_adf_pvalue(self, series: pd.Series, d: float) -> float:
        """Get ADF p-value for fractionally differenced series."""
        try:
            fracdiff_series = self.fracdiff(series, d, drop_na=False)
            
            # Remove NaN values for ADF test
            clean_series = fracdiff_series.dropna()
            
            if len(clean_series) < 10:
                return 1.0  # Not enough data
                
            adf_result = adfuller(clean_series, maxlag=self.max_lags, autolag='AIC')
            return adf_result[1]  # p-value
            
        except Exception as e:
            tprint_warning(f"⚠️ ADF test failed for d={d}: {e}")
            return 1.0
    
    def fracdiff(self, 
                 series: pd.Series, 
                 d: Optional[float] = None,
                 drop_na: bool = True) -> pd.Series:
        """
        Apply fractional differencing to time series.
        
        Args:
            series: Input time series
            d: Fractional differencing order (uses optimal_d if None)
            drop_na: Whether to drop NaN values
            
        Returns:
            Fractionally differenced series
        """
        if d is None:
            if not self.is_fitted:
                raise ValueError("Model not fitted. Call find_optimal_d() first.")
            d = self.optimal_d
            
        if d == 0:
            return series.copy()
            
        tprint_info(f"🔄 Applying fractional differencing with d={d:.4f}")
        
        # Convert to numpy array for processing
        values = series.values.astype(np.float64)
        
        if self.use_numba:
            fracdiff_values = _fracdiff_numba(values, d)
        else:
            fracdiff_values = _fracdiff_numpy(values, d)
            
        result = pd.Series(fracdiff_values, index=series.index, name=f"{series.name}_fracdiff_{d:.3f}")
        
        if drop_na:
            result = result.dropna()
            
        return result
    
    def get_weights(self, d: float, max_lags: int = 100) -> np.ndarray:
        """
        Calculate fractional differencing weights.
        
        Args:
            d: Fractional differencing order
            max_lags: Maximum number of lags
            
        Returns:
            Array of weights
        """
        if self.use_numba:
            return _get_weights_numba(d, max_lags)
        else:
            return _get_weights_numpy(d, max_lags)
    
    def get_adf_statistics(self) -> Dict[float, float]:
        """Get ADF test statistics from d-parameter search."""
        return self.adf_stats.copy()
    
    def transform(self, series: pd.Series) -> pd.Series:
        """Transform series using optimal d parameter."""
        return self.fracdiff(series)
    
    def fit_transform(self, series: pd.Series, method: str = 'binary_search') -> pd.Series:
        """Fit and transform in one step."""
        self.find_optimal_d(series, method)
        return self.transform(series)


@njit
def _get_weights_numba(d: float, max_lags: int) -> np.ndarray:
    """Calculate fractional differencing weights using Numba."""
    weights = np.zeros(max_lags + 1)
    weights[0] = 1.0
    
    for k in range(1, max_lags + 1):
        weights[k] = -weights[k-1] * (d - k + 1) / k
        
    return weights


def _get_weights_numpy(d: float, max_lags: int) -> np.ndarray:
    """Calculate fractional differencing weights using NumPy."""
    weights = np.zeros(max_lags + 1)
    weights[0] = 1.0
    
    for k in range(1, max_lags + 1):
        weights[k] = -weights[k-1] * (d - k + 1) / k
        
    return weights


@njit(parallel=True)
def _fracdiff_numba(values: np.ndarray, d: float) -> np.ndarray:
    """Fractional differencing using Numba for performance."""
    n = len(values)
    result = np.zeros(n)
    
    # Calculate weights
    max_lags = min(n - 1, 100)  # Limit lags for performance
    weights = _get_weights_numba(d, max_lags)
    
    # Apply fractional differencing
    for i in prange(n):
        if i == 0:
            result[i] = np.nan
        else:
            fracdiff_val = 0.0
            for k in range(min(i, max_lags) + 1):
                fracdiff_val += weights[k] * values[i - k]
            result[i] = fracdiff_val
            
    return result


def _fracdiff_numpy(values: np.ndarray, d: float) -> np.ndarray:
    """Fractional differencing using NumPy."""
    n = len(values)
    result = np.full(n, np.nan)
    
    # Calculate weights
    max_lags = min(n - 1, 100)
    weights = _get_weights_numpy(d, max_lags)
    
    # Apply fractional differencing
    for i in range(1, n):
        lags_to_use = min(i, max_lags)
        fracdiff_val = np.sum(weights[:lags_to_use + 1] * values[i-lags_to_use:i+1])
        result[i] = fracdiff_val
        
    return result


def fracdiff_series(series: pd.Series,
                   d: Optional[float] = None,
                   adf_threshold: float = 0.01,
                   method: str = 'binary_search',
                   tolerance: float = 0.01) -> Tuple[pd.Series, float]:
    """
    Convenience function for fractional differencing with automatic d selection.
    
    Args:
        series: Input time series
        d: Fixed d parameter (if None, uses ADF-based selection)
        adf_threshold: ADF p-value threshold for stationarity
        method: d-parameter search method
        tolerance: Tolerance for optimal d convergence
        
    Returns:
        Tuple of (fractionally differenced series, optimal_d)
    """
    transformer = FracDiffTransformer(adf_threshold=adf_threshold)
    
    if d is not None:
        # Use fixed d parameter
        result = transformer.fracdiff(series, d)
        optimal_d = d
    else:
        # Find optimal d automatically
        optimal_d = transformer.find_optimal_d(series, method, tolerance)
        result = transformer.transform(series)
    
    return result, optimal_d


def validate_stationarity(series: pd.Series, 
                         adf_threshold: float = 0.01,
                         kpss_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Validate stationarity of a time series.
    
    Args:
        series: Time series to validate
        adf_threshold: ADF test p-value threshold
        kpss_threshold: KPSS test p-value threshold
        
    Returns:
        Dictionary with stationarity test results
    """
    if not STATSMODELS_AVAILABLE:
        return {"error": "statsmodels not available"}
    
    try:
        # ADF test
        adf_result = adfuller(series.dropna(), autolag='AIC')
        adf_statistic = adf_result[0]
        adf_pvalue = adf_result[1]
        adf_critical = adf_result[4]
        
        # KPSS test (if available)
        try:
            from statsmodels.tsa.stattools import kpss
            kpss_result = kpss(series.dropna(), regression='c')
            kpss_statistic = kpss_result[0]
            kpss_pvalue = kpss_result[1]
            kpss_critical = kpss_result[3]
        except:
            kpss_statistic = None
            kpss_pvalue = None
            kpss_critical = None
        
        is_stationary_adf = adf_pvalue < adf_threshold
        is_stationary_kpss = kpss_pvalue > kpss_threshold if kpss_pvalue is not None else None
        
        return {
            "is_stationary_adf": is_stationary_adf,
            "is_stationary_kpss": is_stationary_kpss,
            "adf_statistic": adf_statistic,
            "adf_pvalue": adf_pvalue,
            "adf_critical_values": adf_critical,
            "kpss_statistic": kpss_statistic,
            "kpss_pvalue": kpss_pvalue,
            "kpss_critical_values": kpss_critical,
            "stationarity_confirmed": is_stationary_adf and (is_stationary_kpss if is_stationary_kpss is not None else True)
        }
        
    except Exception as e:
        return {"error": str(e)}


# Export main functions
__all__ = [
    'FracDiffTransformer',
    'fracdiff_series', 
    'validate_stationarity',
    '_get_weights_numba',
    '_get_weights_numpy',
    '_fracdiff_numba',
    '_fracdiff_numpy'
]

class AdaptiveFracDiffTransformer(FracDiffTransformer):
    """
    Adaptive Fractional Differencing.
    
    Dynamically adjusts the differencing order 'd' over time based on
    a rolling ADF test to maintain stationarity with minimum information loss.
    """
    
    def __init__(self, 
                 window_size: int = 500,
                 step_size: int = 50,
                 d_lower: float = 0.0,
                 d_upper: float = 1.0,
                 d_step: float = 0.1,
                 adf_threshold: float = 0.01):
        super().__init__(max_d=d_upper, min_d=d_lower, adf_threshold=adf_threshold)
        self.window_size = window_size
        self.step_size = step_size
        self.d_grid = np.arange(d_lower, d_upper + d_step, d_step)
        
    def _check_window_stationarity(self, candidate_series_d: pd.Series, start_idx: int, end_idx: int) -> bool:
        """
        Check if a window of a series is stationary.

        Args:
            candidate_series_d: The pre-calculated fracdiff series for a specific d
            start_idx: Start index of the window
            end_idx: End index of the window

        Returns:
            True if stationary, False otherwise
        """
        # Get the window from pre-calculated series
        sub_series = candidate_series_d.iloc[start_idx:end_idx].dropna()

        if len(sub_series) < 20:
            return False # Not enough data to determine stationarity

        # Run ADF (fast)
        try:
            p_val = adfuller(sub_series, maxlag=self.max_lags, autolag='AIC')[1]
            return p_val < self.adf_threshold
        except:
            return False

    def transform_adaptive(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Apply adaptive fractional differencing.
        
        Args:
            series: Input time series
            
        Returns:
            Tuple[transformed_series, d_values_series]
        """
        if len(series) < self.window_size:
            tprint_warning("⚠️ Series shorter than window size, using fixed d=0.5")
            res, _ = fracdiff_series(series, d=0.5)
            return res, pd.Series(0.5, index=series.index)
            
        tprint_info(f"🔄 Calculating Adaptive FracDiff (window={self.window_size}, step={self.step_size})...")
        
        # 1. Pre-calculate FracDiff candidates for the grid
        candidate_series = {}
        for d in self.d_grid:
            # We use drop_na=False to keep indices aligned
            candidate_series[d] = self.fracdiff(series, d=d, drop_na=False)
            
        # 2. Rolling ADF selection with Binary Search
        d_trajectory = []
        timestamps = []
        
        values = series.values
        n = len(values)
        
        # Iterate with step size
        for i in range(self.window_size, n, self.step_size):
            # Define window
            start_idx = i - self.window_size
            end_idx = i
            
            # Binary search for minimum d that is stationary in this window
            # We are looking for the smallest index 'idx' in d_grid such that
            # check_stationarity(d_grid[idx]) is True.

            low = 0
            high = len(self.d_grid) - 1
            best_d_idx = high # Default to max if nothing found (conservative)
            found_stationary = False
            
            while low <= high:
                mid = (low + high) // 2
                d = self.d_grid[mid]
                
                if self._check_window_stationarity(candidate_series[d], start_idx, end_idx):
                    # This d is stationary, try to find a smaller one
                    best_d_idx = mid
                    found_stationary = True
                    high = mid - 1
                else:
                    # Not stationary, need higher d
                    low = mid + 1
            
            d_trajectory.append(self.d_grid[best_d_idx])
            timestamps.append(series.index[i])
            
        # 3. Interpolate d values
        # Create a series with defined d points and interpolate
        d_series = pd.Series(d_trajectory, index=timestamps).reindex(series.index)
        d_series = d_series.interpolate(method='linear').ffill().bfill()
        
        # 4. Construct adaptive series (Vectorized)
        # Find nearest d index in grid for every point in time
        d_indices = np.abs(d_series.values[:, None] - self.d_grid[None, :]).argmin(axis=1)
        
        # Stack candidates into a matrix: (n_samples, n_d_grid)
        # We ensure the order matches self.d_grid
        candidate_matrix = np.column_stack([candidate_series[d].values for d in self.d_grid])

        # Use advanced indexing to select the correct value for each time step
        # adaptive_values[i] = candidate_matrix[i, d_indices[i]]
        adaptive_values = candidate_matrix[np.arange(n), d_indices]
            
        return pd.Series(adaptive_values, index=series.index), d_series
