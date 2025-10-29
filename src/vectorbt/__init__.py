"""
VectorBT compatibility layer for VectorBT 0.28.1+ API.

This module provides a compatibility layer that exposes the API expected by the
Ares codebase while using the modern VectorBT 0.28.1+ internal functions.

The module automatically detects if VectorBT is available and provides appropriate
fallbacks when it's not.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import pandas as pd
import numpy as np
from typing import Any, Optional, Union


class MockVbtSettings:
    """Mock settings object for VectorBT compatibility."""
    def __init__(self):
        self.array_wrapper = MockSettingsDict()
        self.parallel = MockSettingsDict()
        self.threading = MockSettingsDict()


class MockSettingsDict(dict):
    """Mock settings dictionary that accepts any key/value assignment."""
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            # Return None for missing keys instead of raising KeyError
            return None

_ENV_FLAG = os.environ.get("ARES_ENABLE_VECTORBT", "").strip().lower()
_ALLOW_REAL_VECTORBT = _ENV_FLAG in {"1", "true", "yes", "on"}

# Determine the workspace root (the path entry that contains this stub).
_STUB_DIR = os.path.dirname(__file__)
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_STUB_DIR, os.pardir))

VECTORBT_AVAILABLE = False
_vectorbt_module = None

if _ALLOW_REAL_VECTORBT:
    try:
        # Search for the real vectorbt package on the rest of sys.path.
        _search_paths = [path for path in sys.path if os.path.abspath(path) != _WORKSPACE_ROOT]

        spec = importlib.machinery.PathFinder.find_spec("vectorbt", _search_paths)
        if spec is not None and spec.loader is not None:
            # Load the genuine vectorbt module
            _vectorbt_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_vectorbt_module)
            VECTORBT_AVAILABLE = True
        else:
            VECTORBT_AVAILABLE = False
    except Exception:
        VECTORBT_AVAILABLE = False
else:
    VECTORBT_AVAILABLE = False

# If VectorBT is available, create compatibility functions
if VECTORBT_AVAILABLE and _vectorbt_module:
    # Import required numba functions
    nb = _vectorbt_module.nb

    # Create a mock vbt object (the old API had this)
    class MockVbtObject:
        """Mock vbt object for compatibility."""
        def __init__(self):
            self.__version__ = getattr(_vectorbt_module, '__version__', '0.28.1')
            self._settings = None

        @property
        def settings(self):
            """Mock settings object for compatibility."""
            if self._settings is None:
                self._settings = MockVbtSettings()
            return self._settings

    vbt = MockVbtObject()

    # Helper function to convert between pandas and numpy formats
    def _ensure_2d_array(data: Union[pd.Series, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input to 2D numpy array as expected by VectorBT nb functions."""
        if isinstance(data, pd.Series):
            return data.values.reshape(-1, 1)
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return data.reshape(-1, 1)
            return data
        else:
            # Try to convert to numpy array
            arr = np.asarray(data)
            if arr.ndim == 1:
                return arr.reshape(-1, 1)
            return arr

    def _result_to_pandas(result: np.ndarray, original_input: Union[pd.Series, pd.DataFrame, np.ndarray]) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        """Convert VectorBT result back to appropriate format."""
        if isinstance(original_input, pd.Series):
            return pd.Series(result.flatten(), index=original_input.index, name=original_input.name)
        elif isinstance(original_input, pd.DataFrame):
            return pd.DataFrame(result, index=original_input.index, columns=original_input.columns)
        else:
            return result

    # Rolling functions
    def rolling_mean(data: Union[pd.Series, np.ndarray], window: int, minp: Optional[int] = None, **kwargs) -> Union[pd.Series, np.ndarray]:
        """Rolling mean using VectorBT nb functions."""
        data_2d = _ensure_2d_array(data)
        result = nb.rolling_mean_nb(data_2d, window, minp=minp)
        return _result_to_pandas(result, data)

    def rolling_std(data: Union[pd.Series, np.ndarray], window: int, minp: Optional[int] = None, ddof: int = 1, **kwargs) -> Union[pd.Series, np.ndarray]:
        """Rolling standard deviation using VectorBT nb functions."""
        data_2d = _ensure_2d_array(data)
        result = nb.rolling_std_nb(data_2d, window, minp=minp, ddof=ddof)
        return _result_to_pandas(result, data)

    def rolling_var(data: Union[pd.Series, np.ndarray], window: int, minp: Optional[int] = None, ddof: int = 1, **kwargs) -> Union[pd.Series, np.ndarray]:
        """Rolling variance using VectorBT nb functions."""
        # VectorBT doesn't have rolling_var_nb, so we use std^2
        data_2d = _ensure_2d_array(data)
        result = nb.rolling_std_nb(data_2d, window, minp=minp, ddof=ddof) ** 2
        return _result_to_pandas(result, data)

    def rolling_min(data: Union[pd.Series, np.ndarray], window: int, minp: Optional[int] = None, **kwargs) -> Union[pd.Series, np.ndarray]:
        """Rolling minimum using VectorBT nb functions."""
        data_2d = _ensure_2d_array(data)
        result = nb.rolling_min_nb(data_2d, window, minp=minp)
        return _result_to_pandas(result, data)

    def rolling_max(data: Union[pd.Series, np.ndarray], window: int, minp: Optional[int] = None, **kwargs) -> Union[pd.Series, np.ndarray]:
        """Rolling maximum using VectorBT nb functions."""
        data_2d = _ensure_2d_array(data)
        result = nb.rolling_max_nb(data_2d, window, minp=minp)
        return _result_to_pandas(result, data)

    def rolling_sum(data: Union[pd.Series, np.ndarray], window: int, minp: Optional[int] = None, **kwargs) -> Union[pd.Series, np.ndarray]:
        """Rolling sum - fallback to pandas since VectorBT doesn't have rolling_sum_nb."""
        if isinstance(data, pd.Series):
            return data.rolling(window=window, min_periods=minp).sum()
        elif isinstance(data, pd.DataFrame):
            return data.rolling(window=window, min_periods=minp).sum()
        else:
            # For numpy arrays, implement simple rolling sum
            result = np.full_like(data, np.nan, dtype=float)
            for i in range(window - 1, len(data)):
                start_idx = max(0, i - window + 1)
                result[i] = np.sum(data[start_idx:i+1])
            return result

    def rolling_apply(data: Union[pd.Series, np.ndarray], window: int, func, minp: Optional[int] = None, **kwargs) -> Union[pd.Series, np.ndarray]:
        """Rolling apply - fallback to pandas."""
        if isinstance(data, pd.Series):
            return data.rolling(window=window, min_periods=minp).apply(func, **kwargs)
        elif isinstance(data, pd.DataFrame):
            return data.rolling(window=window, min_periods=minp).apply(func, **kwargs)
        else:
            # For numpy arrays, use pandas temporarily
            temp_series = pd.Series(data)
            return temp_series.rolling(window=window, min_periods=minp).apply(func, **kwargs).values

    # Additional statistical functions - fallback to pandas/numpy
    def rolling_corr(data1: Union[pd.Series, np.ndarray], data2: Union[pd.Series, np.ndarray], window: int, **kwargs) -> Union[pd.Series, np.ndarray]:
        """Rolling correlation - fallback to pandas."""
        if isinstance(data1, pd.Series) and isinstance(data2, pd.Series):
            return data1.rolling(window=window).corr(data2, **kwargs)
        else:
            # Convert to pandas for calculation
            s1 = pd.Series(data1) if not isinstance(data1, pd.Series) else data1
            s2 = pd.Series(data2) if not isinstance(data2, pd.Series) else data2
            return s1.rolling(window=window).corr(s2, **kwargs)

    def rolling_cov(data1: Union[pd.Series, np.ndarray], data2: Union[pd.Series, np.ndarray], window: int, **kwargs) -> Union[pd.Series, np.ndarray]:
        """Rolling covariance - fallback to pandas."""
        if isinstance(data1, pd.Series) and isinstance(data2, pd.Series):
            return data1.rolling(window=window).cov(data2, **kwargs)
        else:
            # Convert to pandas for calculation
            s1 = pd.Series(data1) if not isinstance(data1, pd.Series) else data1
            s2 = pd.Series(data2) if not isinstance(data2, pd.Series) else data2
            return s1.rolling(window=window).cov(s2, **kwargs)

    # Data transformation functions - fallback to pandas/scipy
    def scale(data: Union[pd.Series, np.ndarray], **kwargs) -> Union[pd.Series, np.ndarray]:
        """Scale data - fallback to sklearn."""
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler(**kwargs)
            if isinstance(data, pd.Series):
                scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
                return pd.Series(scaled, index=data.index, name=data.name)
            elif isinstance(data, pd.DataFrame):
                scaled = scaler.fit_transform(data.values)
                return pd.DataFrame(scaled, index=data.index, columns=data.columns)
            else:
                return scaler.fit_transform(data.reshape(-1, 1)).flatten()
        except ImportError:
            # Fallback to manual standardization
            if isinstance(data, pd.Series):
                return (data - data.mean()) / data.std()
            elif isinstance(data, pd.DataFrame):
                return (data - data.mean()) / data.std()
            else:
                data_array = np.asarray(data)
                return (data_array - data_array.mean()) / data_array.std()

    def rank(data: Union[pd.Series, np.ndarray], **kwargs) -> Union[pd.Series, np.ndarray]:
        """Rank data - fallback to pandas."""
        if isinstance(data, pd.Series):
            return data.rank(**kwargs)
        elif isinstance(data, pd.DataFrame):
            return data.rank(**kwargs)
        else:
            return pd.Series(data).rank(**kwargs).values

    def zscore(data: Union[pd.Series, np.ndarray], **kwargs) -> Union[pd.Series, np.ndarray]:
        """Z-score data - same as scale."""
        return scale(data, **kwargs)

    def winsorize(data: Union[pd.Series, np.ndarray], limits: tuple = (0.05, 0.05), **kwargs) -> Union[pd.Series, np.ndarray]:
        """Winsorize data - fallback to scipy."""
        try:
            from scipy.stats.mstats import winsorize as scipy_winsorize
            if isinstance(data, pd.Series):
                winsorized = scipy_winsorize(data.values, limits=limits, **kwargs)
                return pd.Series(winsorized, index=data.index, name=data.name)
            elif isinstance(data, pd.DataFrame):
                winsorized = scipy_winsorize(data.values, limits=limits, **kwargs)
                return pd.DataFrame(winsorized, index=data.index, columns=data.columns)
            else:
                return scipy_winsorize(data, limits=limits, **kwargs)
        except ImportError:
            # Simple fallback implementation
            data_array = np.asarray(data)
            lower_limit = np.percentile(data_array, limits[0] * 100)
            upper_limit = np.percentile(data_array, (1 - limits[1]) * 100)
            winsorized = np.clip(data_array, lower_limit, upper_limit)
            if isinstance(data, pd.Series):
                return pd.Series(winsorized, index=data.index, name=data.name)
            elif isinstance(data, pd.DataFrame):
                return pd.DataFrame(winsorized, index=data.index, columns=data.columns)
            else:
                return winsorized

    def clip(data: Union[pd.Series, np.ndarray], lower: Optional[float] = None, upper: Optional[float] = None, **kwargs) -> Union[pd.Series, np.ndarray]:
        """Clip data - use numpy.clip."""
        if isinstance(data, pd.Series):
            return data.clip(lower=lower, upper=upper, **kwargs)
        elif isinstance(data, pd.DataFrame):
            return data.clip(lower=lower, upper=upper, **kwargs)
        else:
            return np.clip(data, a_min=lower, a_max=upper)

    def quantile(data: Union[pd.Series, np.ndarray], q: float, **kwargs) -> Union[pd.Series, np.ndarray]:
        """Quantile - fallback to numpy/pandas."""
        if isinstance(data, pd.Series):
            return data.quantile(q, **kwargs)
        elif isinstance(data, pd.DataFrame):
            return data.quantile(q, **kwargs)
        else:
            return np.quantile(data, q, **kwargs)

else:
    # VectorBT not available - provide dummy implementations
    VECTORBT_AVAILABLE = False

    class MockVbtObject:
        """Mock vbt object for compatibility."""
        def __init__(self):
            self.__version__ = "not_available"
            self._settings = None

        @property
        def settings(self):
            """Mock settings object for compatibility."""
            if self._settings is None:
                self._settings = MockVbtSettings()
            return self._settings

    vbt = MockVbtObject()

    # Dummy functions that raise ImportError
    def _vectorbt_not_available(*args, **kwargs):
        raise ImportError("VectorBT not available - install vectorbt and set ARES_ENABLE_VECTORBT=1")

    rolling_mean = _vectorbt_not_available
    rolling_std = _vectorbt_not_available
    rolling_var = _vectorbt_not_available
    rolling_min = _vectorbt_not_available
    rolling_max = _vectorbt_not_available
    rolling_sum = _vectorbt_not_available
    rolling_apply = _vectorbt_not_available
    rolling_corr = _vectorbt_not_available
    rolling_cov = _vectorbt_not_available
    scale = _vectorbt_not_available
    rank = _vectorbt_not_available
    zscore = _vectorbt_not_available
    winsorize = _vectorbt_not_available
    clip = _vectorbt_not_available
    quantile = _vectorbt_not_available
