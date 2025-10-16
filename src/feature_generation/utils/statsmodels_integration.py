"""
Statsmodels Integration for Advanced Time Series Analysis

This module provides integration with statsmodels for:
- ARIMA/ARMA modeling for time series forecasting
- Vector Autoregression (VAR) for multivariate analysis
- Advanced statistical tests and diagnostics
- Stationarity and cointegration testing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

# Import statsmodels with fallback
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.varmax import VARMAX
    from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
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
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# CuPy imports for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    STATSMODELS_AVAILABLE = True
    logger.info("✅ Statsmodels available for advanced time series analysis")
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("⚠️ Statsmodels not available - limited functionality")

class ModelType(Enum):
    ARIMA = "arima"
    ARMA = "arma"
    VAR = "var"
    VARMA = "varma"

@dataclass
class ARIMAResults:
    """Results from ARIMA modeling."""
    model: Any
    predictions: pd.Series
    residuals: pd.Series
    aic: float
    bic: float
    mse: float
    order: Tuple[int, int, int]
    seasonal_order: Optional[Tuple[int, int, int, int]] = None

@dataclass
class VARResults:
    """Results from VAR modeling."""
    model: Any
    fitted_values: pd.DataFrame
    residuals: pd.DataFrame
    aic: float
    bic: float
    fpe: float
    hqic: float
    lag_order: int
    granger_tests: Optional[Dict[str, Any]] = None

class StatsmodelsIntegration:
    """
    Integration class for statsmodels time series analysis.

    Benefits for your trading system:
    - Advanced forecasting capabilities
    - Multivariate relationship analysis
    - Statistical rigor for model validation
    - Stationarity testing for feature engineering
    """

    def __init__(self):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels not available")

        self.logger = logger.getChild('StatsmodelsIntegration')
        self.fitted_models = {}

    def fit_arima(self, series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1),
                  seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                  **kwargs) -> ARIMAResults:
        """
        Fit ARIMA model for time series forecasting.

        Benefits for trading:
        - Predict short-term price movements
        - Generate trading signals based on forecast errors
        - Risk management through volatility forecasting
        - Stationarity correction for ML features
        """
        try:
            # Ensure series is stationary (differencing if needed)
            if not self._is_stationary(series):
                self.logger.info("Series not stationary, applying differencing")
                series = series.diff().dropna()

            # Fit ARIMA model
            model = ARIMA(series, order=order, seasonal_order=seasonal_order, **kwargs)
            fitted_model = model.fit()

            # Generate predictions (in-sample for analysis)
            predictions = fitted_model.fittedvalues
            residuals = fitted_model.resid

            # Calculate metrics
            mse = np.mean(residuals**2)

            results = ARIMAResults(
                model=fitted_model,
                predictions=predictions,
                residuals=residuals,
                aic=fitted_model.aic,
                bic=fitted_model.bic,
                mse=mse,
                order=order,
                seasonal_order=seasonal_order
            )

            self.logger.info(f"✅ ARIMA({order}) fitted - AIC: {results.aic:.2f}, MSE: {mse:.6f}")
            return results

        except Exception as e:
            self.logger.error(f"❌ ARIMA fitting failed: {e}")
            raise

    def fit_var(self, data: pd.DataFrame, maxlags: int = 5,
                trend: str = 'c') -> VARResults:
        """
        Fit Vector Autoregression (VAR) model for multivariate analysis.

        Benefits for trading:
        - Analyze relationships between multiple assets
        - Cross-market signal generation
        - Portfolio optimization
        - Risk spillover analysis
        - Granger causality testing
        """
        try:
            # Ensure all series are stationary
            stationary_data = data.copy()
            for col in data.columns:
                if not self._is_stationary(data[col]):
                    stationary_data[col] = data[col].diff().dropna()

            # Drop NaN values from differencing
            stationary_data = stationary_data.dropna()

            # Fit VAR model
            model = VAR(stationary_data)
            lag_order = self._select_var_order(stationary_data, maxlags)

            fitted_model = model.fit(maxlags=lag_order, trend=trend)

            # Get fitted values and residuals
            fitted_values = fitted_model.fittedvalues
            residuals = fitted_model.resid

            # Perform Granger causality tests
            granger_results = self._granger_causality_matrix(stationary_data, maxlag=lag_order)

            results = VARResults(
                model=fitted_model,
                fitted_values=fitted_values,
                residuals=residuals,
                aic=fitted_model.aic,
                bic=fitted_model.bic,
                fpe=fitted_model.fpe,
                hqic=fitted_model.hqic,
                lag_order=lag_order,
                granger_tests=granger_results
            )

            self.logger.info(f"✅ VAR({lag_order}) fitted - AIC: {results.aic:.2f}")
            return results

        except Exception as e:
            self.logger.error(f"❌ VAR fitting failed: {e}")
            raise

    def seasonal_decomposition(self, series: pd.Series, model: str = 'additive',
                              period: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        Decompose time series into trend, seasonal, and residual components.

        Benefits for trading:
        - Remove seasonal patterns for better modeling
        - Identify long-term trends vs short-term fluctuations
        - Seasonal trading strategies
        - Detrending for stationarity
        """
        try:
            if period is None:
                # Auto-detect period (daily data assumption)
                period = 24  # Assume daily seasonality

            decomposition = seasonal_decompose(series, model=model, period=period)

            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed
            }

        except Exception as e:
            self.logger.error(f"❌ Seasonal decomposition failed: {e}")
            return {}

    def _is_stationary(self, series: pd.Series, significance: float = 0.05) -> bool:
        """Test if series is stationary using ADF test."""
        try:
            result = adfuller(series.dropna())
            p_value = result[1]
            return p_value < significance
        except:
            return False

    def _select_var_order(self, data: pd.DataFrame, maxlags: int) -> int:
        """Select optimal lag order for VAR model."""
        try:
            model = VAR(data)
            lag_selection = model.select_order(maxlags=maxlags)
            return lag_selection.aic  # Use AIC for selection
        except:
            return min(2, maxlags)  # Fallback

    def _granger_causality_matrix(self, data: pd.DataFrame, maxlag: int = 5) -> Dict[str, Any]:
        """Compute Granger causality matrix for all variable pairs."""
        try:
            results = {}
            variables = data.columns

            for i, var1 in enumerate(variables):
                for var2 in variables[i+1:]:
                    try:
                        test_result = grangercausalitytests(data[[var1, var2]], maxlag=maxlag, verbose=False)
                        # Extract p-values for different lags
                        p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag+1)]
                        results[f'{var1}_causes_{var2}'] = {
                            'p_values': p_values,
                            'significant_lags': [lag for lag, p in enumerate(p_values, 1) if p < 0.05]
                        }
                    except:
                        continue

            return results
        except:
            return {}

# Example usage functions
def create_arima_features(price_data: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """
    Create ARIMA-based features for ML models.

    This demonstrates how ARIMA can enhance your existing feature set.
    """
    if not STATSMODELS_AVAILABLE:
        return pd.DataFrame(index=price_data.index)

    integrator = StatsmodelsIntegration()
    features = pd.DataFrame(index=price_data.index)

    try:
        # Rolling ARIMA forecasts
        for i in range(window, len(price_data), window//4):  # Overlapping windows
            window_data = price_data.iloc[i-window:i]
            if len(window_data) < 20:
                continue

            arima_result = integrator.fit_arima(window_data['close'], order=(1, 1, 1))

            # Extract features from ARIMA
            features.loc[price_data.index[i-1], 'arima_residual'] = arima_result.residuals.iloc[-1]
            features.loc[price_data.index[i-1], 'arima_forecast_error'] = (
                price_data.iloc[i-1]['close'] - arima_result.predictions.iloc[-1]
            )

    except Exception as e:
        logger.error(f"ARIMA feature creation failed: {e}")

    return features.fillna(0)

def create_var_features(multi_asset_data: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Create VAR-based features for multivariate analysis.

    Useful for analyzing relationships between multiple trading pairs.
    """
    if not STATSMODELS_AVAILABLE:
        return pd.DataFrame(index=multi_asset_data.index)

    integrator = StatsmodelsIntegration()
    features = pd.DataFrame(index=multi_asset_data.index)

    try:
        # Rolling VAR analysis
        for i in range(window, len(multi_asset_data), window//2):
            window_data = multi_asset_data.iloc[i-window:i]
            if len(window_data) < 30:
                continue

            var_result = integrator.fit_var(window_data, maxlags=3)

            # Extract cross-market features
            if var_result.granger_tests:
                for test_name, test_data in var_result.granger_tests.items():
                    if test_data['significant_lags']:
                        features.loc[multi_asset_data.index[i-1], f'granger_{test_name}'] = 1
                    else:
                        features.loc[multi_asset_data.index[i-1], f'granger_{test_name}'] = 0

    except Exception as e:
        logger.error(f"VAR feature creation failed: {e}")

    return features.fillna(0)

class VectorBTHelper:
    """Helper class for VectorBT operations."""
    
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

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
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
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
