"""
Stage 1: IC Surface Estimation with HAC Standard Errors and Spline Fitting

This module implements the first stage of the lookback optimization system,
estimating smooth information coefficient surfaces for each feature family
with proper uncertainty quantification using HAC standard errors.
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Import configuration
from .config import LookbackOptimizationConfig, FamilyType, SplineConfig, HACConfig

# Import utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning

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

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ICSurfaceResult:
    """Result of IC surface estimation for a single family."""
    family: FamilyType
    lookbacks: np.ndarray
    ic_values: np.ndarray
    ic_errors: np.ndarray
    optimal_lookback: float
    optimal_ic: float
    optimal_ic_error: float
    spline_coefficients: Optional[np.ndarray] = None
    spline_knots: Optional[np.ndarray] = None
    r_squared: float = 0.0
    execution_time: float = 0.0
    
    # Cost-aware metrics
    cpu_costs: Optional[np.ndarray] = None
    staleness_costs: Optional[np.ndarray] = None
    uncertainty_costs: Optional[np.ndarray] = None
    adjusted_scores: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'family': self.family.value,
            'lookbacks': self.lookbacks.tolist(),
            'ic_values': self.ic_values.tolist(),
            'ic_errors': self.ic_errors.tolist(),
            'optimal_lookback': self.optimal_lookback,
            'optimal_ic': self.optimal_ic,
            'optimal_ic_error': self.optimal_ic_error,
            'spline_coefficients': self.spline_coefficients.tolist() if self.spline_coefficients is not None else None,
            'spline_knots': self.spline_knots.tolist() if self.spline_knots is not None else None,
            'r_squared': self.r_squared,
            'execution_time': self.execution_time,
            'cpu_costs': self.cpu_costs.tolist() if self.cpu_costs is not None else None,
            'staleness_costs': self.staleness_costs.tolist() if self.staleness_costs is not None else None,
            'uncertainty_costs': self.uncertainty_costs.tolist() if self.uncertainty_costs is not None else None,
            'adjusted_scores': self.adjusted_scores.tolist() if self.adjusted_scores is not None else None
        }


class HACStandardErrors:
    """Heteroskedasticity and Autocorrelation Consistent standard errors."""
    
    def __init__(self, config: HACConfig):
        self.config = config
    
    def compute_lag(self, n_obs: int) -> int:
        """Compute optimal lag for HAC estimation."""
        if self.config.lag_method == "sqrt_t":
            return min(int(np.sqrt(n_obs)), self.config.max_lag)
        elif self.config.lag_method == "fixed" and self.config.fixed_lag is not None:
            return min(self.config.fixed_lag, self.config.max_lag)
        elif self.config.lag_method == "aic":
            return self._aic_lag_selection(n_obs)
        elif self.config.lag_method == "bic":
            return self._bic_lag_selection(n_obs)
        else:
            return min(4, self.config.max_lag)  # Default conservative choice
    
    def _aic_lag_selection(self, n_obs: int) -> int:
        """Select lag using AIC criterion."""
        max_lag = min(20, n_obs // 4, self.config.max_lag)
        if max_lag < 1:
            return 1
        
        # Simple AIC-based selection (placeholder implementation)
        # In practice, this would involve fitting models with different lags
        return min(max_lag, 10)
    
    def _bic_lag_selection(self, n_obs: int) -> int:
        """Select lag using BIC criterion."""
        max_lag = min(20, n_obs // 4, self.config.max_lag)
        if max_lag < 1:
            return 1
        
        # Simple BIC-based selection (placeholder implementation)
        return min(max_lag, 8)
    
    def compute_hac_variance(self, residuals: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute HAC variance-covariance matrix."""
        n_obs, n_params = X.shape
        lag = self.compute_lag(n_obs)
        
        if lag >= n_obs:
            lag = max(1, n_obs // 4)
        
        # Compute kernel weights
        weights = self._compute_kernel_weights(lag)
        
        # Compute HAC estimator
        S = np.zeros((n_params, n_params))
        
        for j in range(lag + 1):
            if j == 0:
                # Diagonal term
                S += np.outer(residuals, residuals) * X.T @ X
            else:
                # Off-diagonal terms
                if j < n_obs:
                    gamma_j = np.outer(residuals[:-j], residuals[j:]) * (X[:-j].T @ X[j:])
                    S += weights[j] * (gamma_j + gamma_j.T)
        
        # Compute variance-covariance matrix
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            hac_vcov = XtX_inv @ S @ XtX_inv
            return np.diag(hac_vcov)
        except np.linalg.LinAlgError:
            # Fallback to OLS standard errors
            return np.var(residuals) * np.diag(np.linalg.inv(X.T @ X))
    
    def _compute_kernel_weights(self, lag: int) -> np.ndarray:
        """Compute kernel weights for HAC estimation."""
        weights = np.zeros(lag + 1)
        weights[0] = 1.0
        
        for j in range(1, lag + 1):
            if self.config.kernel == "bartlett":
                weights[j] = 1 - j / (lag + 1)
            elif self.config.kernel == "parzen":
                x = j / (lag + 1)
                if x <= 0.5:
                    weights[j] = 1 - 6 * x**2 + 6 * x**3
                else:
                    weights[j] = 2 * (1 - x)**3
            elif self.config.kernel == "quadratic":
                x = j / (lag + 1)
                weights[j] = 1 - x**2
            else:
                weights[j] = 1.0  # Uniform weights
        
        return weights


class PenalizedSpline:
    """Penalized spline fitting for IC surface estimation."""
    
    def __init__(self, config: SplineConfig):
        self.config = config
        self.spline = None
        self.knots = None
        self.coefficients = None
    
    def fit(self, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> 'PenalizedSpline':
        """Fit penalized spline to data."""
        if len(x) < self.config.min_data_points:
            raise ValueError(f"Insufficient data points: {len(x)} < {self.config.min_data_points}")
        
        # Transform to log space if configured
        if self.config.use_log_space:
            x_fit = np.log(x)
        else:
            x_fit = x
        
        # Sort data
        sort_idx = np.argsort(x_fit)
        x_sorted = x_fit[sort_idx]
        y_sorted = y[sort_idx]
        
        if weights is not None:
            weights_sorted = weights[sort_idx]
        else:
            weights_sorted = None
        
        # Select knots
        self.knots = self._select_knots(x_sorted)
        
        # Fit spline with penalty
        try:
            if self.config.degree == 3 and len(self.knots) > 0:
                # Use LSQUnivariateSpline for better control
                self.spline = LSQUnivariateSpline(
                    x_sorted, y_sorted, self.knots, 
                    k=self.config.degree, w=weights_sorted
                )
            else:
                # Use UnivariateSpline with smoothing
                self.spline = UnivariateSpline(
                    x_sorted, y_sorted, 
                    k=self.config.degree, w=weights_sorted,
                    s=self.config.penalty_weight * len(x_sorted)
                )
            
            self.coefficients = self.spline.get_coeffs()
            
        except Exception as e:
            logger.warning(f"Spline fitting failed: {e}. Falling back to linear regression.")
            # Fallback to linear regression
            self.spline = self._fit_linear_fallback(x_sorted, y_sorted, weights_sorted)
            self.knots = np.array([])
            self.coefficients = np.array([])
        
        return self
    
    def _select_knots(self, x: np.ndarray) -> np.ndarray:
        """Select knot positions for spline fitting."""
        if self.config.n_knots <= 0:
            return np.array([])
        
        n_points = len(x)
        if n_points <= self.config.n_knots + 2:
            return np.array([])
        
        # Use quantile-based knot selection
        quantiles = np.linspace(0.1, 0.9, self.config.n_knots)
        knots = np.quantile(x, quantiles)
        
        # Remove duplicate knots
        knots = np.unique(knots)
        
        # Ensure knots are within data range
        knots = knots[(knots > x.min()) & (knots < x.max())]
        
        return knots
    
    def _fit_linear_fallback(self, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> Any:
        """Fallback linear regression when spline fitting fails."""
        class LinearSpline:
            def __init__(self, x, y, weights):
                self.x = x
                self.y = y
                self.weights = weights
                self.coeffs = self._fit()
            
            def _fit(self):
                if self.weights is not None:
                    w = np.sqrt(self.weights)
                    X = np.column_stack([np.ones_like(self.x), self.x])
                    y_w = self.y * w
                    X_w = X * w[:, np.newaxis]
                    return np.linalg.lstsq(X_w, y_w, rcond=None)[0]
                else:
                    X = np.column_stack([np.ones_like(self.x), self.x])
                    return np.linalg.lstsq(X, self.y, rcond=None)[0]
            
            def __call__(self, x_new):
                return self.coeffs[0] + self.coeffs[1] * x_new
            
            def get_coeffs(self):
                return self.coeffs
        
        return LinearSpline(x, y, weights)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict IC values for given lookbacks."""
        if self.spline is None:
            raise ValueError("Spline not fitted yet")
        
        # Transform to log space if configured
        if self.config.use_log_space:
            x_pred = np.log(x)
        else:
            x_pred = x
        
        return self.spline(x_pred)
    
    def get_derivative(self, x: np.ndarray) -> np.ndarray:
        """Get derivative of spline at given points."""
        if self.spline is None:
            raise ValueError("Spline not fitted yet")
        
        # Transform to log space if configured
        if self.config.use_log_space:
            x_pred = np.log(x)
        else:
            x_pred = x
        
        return self.spline.derivative()(x_pred)
    
    def find_optimum(self, x_range: Tuple[float, float], n_points: int = 1000) -> Tuple[float, float]:
        """Find optimum lookback and IC value."""
        x_test = np.linspace(x_range[0], x_range[1], n_points)
        y_test = self.predict(x_test)
        
        # Find maximum
        max_idx = np.argmax(y_test)
        optimal_x = x_test[max_idx]
        optimal_y = y_test[max_idx]
        
        return optimal_x, optimal_y


class CostAwareScorer:
    """Cost-aware scoring system for lookback optimization."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        self.penalties = config.penalties
    
    def compute_cpu_cost(self, lookback: int, family: FamilyType) -> float:
        """Compute CPU cost for given lookback and family."""
        # Base cost depends on family type
        base_costs = {
            FamilyType.MOMENTUM: 1.0,
            FamilyType.VOLATILITY: 1.2,  # EW calculations are more expensive
            FamilyType.GK: 1.1,
            FamilyType.VWAP_ROLL: 1.3,   # VWAP calculations are expensive
            FamilyType.RSI: 1.0,
            FamilyType.AUTOCORR: 1.4     # Autocorrelation is expensive
        }
        
        base_cost = base_costs.get(family, 1.0)
        
        # Cost scales with lookback length
        if family == FamilyType.VOLATILITY:
            # EW calculations scale logarithmically
            cost = base_cost * np.log(1 + lookback)
        else:
            # Most calculations scale linearly
            cost = base_cost * lookback
        
        return cost
    
    def compute_staleness_cost(self, lookback: int, family: FamilyType) -> float:
        """Compute staleness cost for given lookback and family."""
        # Staleness increases with lookback length
        # Different families have different staleness characteristics
        staleness_factors = {
            FamilyType.MOMENTUM: 1.0,
            FamilyType.VOLATILITY: 0.8,  # EW is less stale
            FamilyType.GK: 1.0,
            FamilyType.VWAP_ROLL: 0.9,   # VWAP is less stale
            FamilyType.RSI: 1.1,         # RSI can be stale
            FamilyType.AUTOCORR: 1.2     # Autocorrelation is stale
        }
        
        factor = staleness_factors.get(family, 1.0)
        return factor * lookback
    
    def compute_uncertainty_cost(self, ic_error: float) -> float:
        """Compute uncertainty cost based on IC standard error."""
        # Uncertainty cost is proportional to standard error
        return ic_error
    
    def compute_adjusted_score(self, ic: float, lookback: int, family: FamilyType, 
                              ic_error: float) -> float:
        """Compute cost-adjusted IC score."""
        cpu_cost = self.compute_cpu_cost(lookback, family)
        staleness_cost = self.compute_staleness_cost(lookback, family)
        uncertainty_cost = self.compute_uncertainty_cost(ic_error)
        
        adjusted_score = (ic - 
                         self.penalties.lambda_cost * cpu_cost -
                         self.penalties.lambda_stale * staleness_cost -
                         self.penalties.lambda_uncertainty * uncertainty_cost)
        
        return adjusted_score


class ICSurfaceEstimator:
    """Main class for IC surface estimation."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        self.hac_estimator = HACStandardErrors(config.hac)
        self.cost_scorer = CostAwareScorer(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def estimate_surface(self, data: pd.DataFrame, target: np.ndarray, 
                        family: FamilyType, feature_name: str, 
                        quality_scorer=None) -> ICSurfaceResult:
        """Estimate IC surface for a single feature family."""
        start_time = time.time()
        
        try:
            tprint_info(f"Estimating IC surface for {family.value} family...")
            
            # Get search grid for this family
            lookbacks = np.array(self.config.search_grids.get_family_grid(family))
            if len(lookbacks) < 3:
                raise ValueError(f"Insufficient lookback points for {family.value}: {len(lookbacks)}")
            
            # Compute IC values for each lookback
            ic_values = []
            ic_errors = []
            cpu_costs = []
            staleness_costs = []
            uncertainty_costs = []
            
            for lookback in lookbacks:
                try:
                    # Generate feature for this lookback
                    feature_values = self._generate_feature(data, family, feature_name, lookback)
                    
                    if len(feature_values) < 100:  # Need sufficient data
                        ic_values.append(0.0)
                        ic_errors.append(1.0)
                        cpu_costs.append(self.cost_scorer.compute_cpu_cost(lookback, family))
                        staleness_costs.append(self.cost_scorer.compute_staleness_cost(lookback, family))
                        uncertainty_costs.append(1.0)
                        continue
                    
                    # Compute IC and standard error (with LQS if available)
                    ic, ic_error = self._compute_ic_with_hac(feature_values, target, quality_scorer)
                    
                    ic_values.append(ic)
                    ic_errors.append(ic_error)
                    cpu_costs.append(self.cost_scorer.compute_cpu_cost(lookback, family))
                    staleness_costs.append(self.cost_scorer.compute_staleness_cost(lookback, family))
                    uncertainty_costs.append(ic_error)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to compute IC for lookback {lookback}: {e}")
                    ic_values.append(0.0)
                    ic_errors.append(1.0)
                    cpu_costs.append(self.cost_scorer.compute_cpu_cost(lookback, family))
                    staleness_costs.append(self.cost_scorer.compute_staleness_cost(lookback, family))
                    uncertainty_costs.append(1.0)
            
            # Convert to numpy arrays
            ic_values = np.array(ic_values)
            ic_errors = np.array(ic_errors)
            cpu_costs = np.array(cpu_costs)
            staleness_costs = np.array(staleness_costs)
            uncertainty_costs = np.array(uncertainty_costs)
            
            # Fit spline to IC surface
            spline = PenalizedSpline(self.config.spline)
            weights = 1.0 / (ic_errors + 1e-6)  # Weight by inverse error
            
            try:
                spline.fit(lookbacks, ic_values, weights)
                
                # Find optimum
                x_range = (lookbacks.min(), lookbacks.max())
                optimal_lookback, optimal_ic = spline.find_optimum(x_range)
                
                # Compute R-squared
                predicted_ic = spline.predict(lookbacks)
                r_squared = 1 - np.sum((ic_values - predicted_ic)**2) / np.sum((ic_values - np.mean(ic_values))**2)
                
            except Exception as e:
                self.logger.warning(f"Spline fitting failed: {e}. Using grid maximum.")
                # Fallback to grid maximum
                max_idx = np.argmax(ic_values)
                optimal_lookback = lookbacks[max_idx]
                optimal_ic = ic_values[max_idx]
                r_squared = 0.0
            
            # Compute cost-adjusted scores
            adjusted_scores = np.array([
                self.cost_scorer.compute_adjusted_score(ic, lb, family, err)
                for ic, lb, err in zip(ic_values, lookbacks, ic_errors)
            ])
            
            # Find cost-adjusted optimum
            cost_adj_max_idx = np.argmax(adjusted_scores)
            cost_adj_optimal_lookback = lookbacks[cost_adj_max_idx]
            cost_adj_optimal_ic = ic_values[cost_adj_max_idx]
            
            execution_time = time.time() - start_time
            
            result = ICSurfaceResult(
                family=family,
                lookbacks=lookbacks,
                ic_values=ic_values,
                ic_errors=ic_errors,
                optimal_lookback=optimal_lookback,
                optimal_ic=optimal_ic,
                optimal_ic_error=ic_errors[np.argmin(np.abs(lookbacks - optimal_lookback))],
                spline_coefficients=spline.coefficients,
                spline_knots=spline.knots,
                r_squared=r_squared,
                execution_time=execution_time,
                cpu_costs=cpu_costs,
                staleness_costs=staleness_costs,
                uncertainty_costs=uncertainty_costs,
                adjusted_scores=adjusted_scores
            )
            
            tprint_info(f"IC surface estimation completed in {execution_time:.3f}s")
            tprint_info(f"Optimal lookback: {optimal_lookback:.1f}, IC: {optimal_ic:.4f}")
            tprint_info(f"Cost-adjusted optimal: {cost_adj_optimal_lookback:.1f}, IC: {cost_adj_optimal_ic:.4f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"IC surface estimation failed: {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            
            # Return empty result
            return ICSurfaceResult(
                family=family,
                lookbacks=np.array([]),
                ic_values=np.array([]),
                ic_errors=np.array([]),
                optimal_lookback=0.0,
                optimal_ic=0.0,
                optimal_ic_error=1.0,
                execution_time=execution_time
            )
    
    def _generate_feature(self, data: pd.DataFrame, family: FamilyType, 
                         feature_name: str, lookback: int) -> np.ndarray:
        """Generate feature values for given lookback."""
        # This is a simplified implementation
        # In practice, this would call the actual feature generation functions
        
        if family == FamilyType.MOMENTUM:
            # Simple momentum calculation
            if 'close' in data.columns:
                return data['close'].pct_change(lookback).values
            else:
                return np.zeros(len(data))
        
        elif family == FamilyType.VOLATILITY:
            # EW volatility calculation
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                alpha = 2 / (lookback + 1)
                ew_var = returns.ewm(alpha=alpha).var()
                return np.sqrt(ew_var).values
            else:
                return np.zeros(len(data))
        
        elif family == FamilyType.RSI:
            # RSI calculation
            if 'close' in data.columns:
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = self._vectorbt_rolling_operation(gain, "mean", lookback)
                avg_loss = self._vectorbt_rolling_operation(loss, "mean", lookback)
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return rsi.values
            else:
                return np.zeros(len(data))
        
        else:
            # Placeholder for other families
            return np.zeros(len(data))
    
    def _compute_ic_with_hac(self, feature: np.ndarray, target: np.ndarray, 
                           quality_scorer=None) -> Tuple[float, float]:
        """Compute IC with HAC standard errors, optionally using LQS scoring."""
        # Remove NaN values
        valid_mask = np.isfinite(feature) & np.isfinite(target)
        if np.sum(valid_mask) < 10:
            return 0.0, 1.0
        
        feature_clean = feature[valid_mask]
        target_clean = target[valid_mask]
        
        # Use LQS scoring if quality scorer is available
        if quality_scorer is not None:
            try:
                feature_series = pd.Series(feature_clean, name='feature')
                target_series = pd.Series(target_clean, name='target')
                lqs_result = quality_scorer.calculate_lqs_score(feature_series, target_series)
                
                if lqs_result and hasattr(lqs_result, 'overall_quality'):
                    # Use LQS as the primary score
                    ic = lqs_result.overall_quality
                    # Use stability as error proxy
                    ic_error = 1.0 - lqs_result.stability if hasattr(lqs_result, 'stability') else 0.1
                    return ic, ic_error
            except Exception as e:
                tprint_warning(f"LQS calculation failed, falling back to correlation: {e}")
        
        # Fallback to correlation-based IC
        ic = np.corrcoef(feature_clean, target_clean)[0, 1]
        
        if np.isnan(ic):
            return 0.0, 1.0
        
        # Compute HAC standard error
        n_obs = len(feature_clean)
        X = np.column_stack([np.ones(n_obs), feature_clean])
        y = target_clean
        
        # Fit linear regression
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ coeffs
            
            # Compute HAC variance
            hac_var = self.hac_estimator.compute_hac_variance(residuals, X)
            
            # IC standard error (simplified)
            ic_error = np.sqrt(hac_var[1] / np.var(feature_clean))
            
        except Exception:
            # Fallback to simple standard error
            ic_error = np.sqrt((1 - ic**2) / (n_obs - 2))
        
        return float(ic), float(ic_error)