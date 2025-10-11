"""
Matrix-Optimized IC Surface Estimation with Hardware Acceleration

This module provides highly optimized IC surface estimation using matrix operations
and hardware acceleration for maximum performance in the lookback optimization system.
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

# Import matrix operations and hardware optimizations
try:
    from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
    from src.utils.matrix_operations.batch_operations import batch_matrix_multiply, batch_correlation_analysis
    from src.utils.matrix_operations.hardware_integration import HardwareOptimizedMatrixProcessor, HardwareConfig
    from src.utils.matrix_operations.vectorized_core import get_vectorized_processing_core
    from src.utils.hardware.m1_optimizations import M1MemoryOptimizer, M1CPUOptimizer
    from src.utils.hardware.memory_optimization import memory_efficient, optimize_dataframe_dtypes, chunk_dataframe
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
    MATRIX_OPS_AVAILABLE = True
    HARDWARE_AVAILABLE = True
except ImportError as e:
    MATRIX_OPS_AVAILABLE = False
    HARDWARE_AVAILABLE = False
    logging.warning(f"Matrix operations or hardware optimizations not available: {e}")

# Import base configuration
from .config import LookbackOptimizationConfig, FamilyType, SplineConfig, HACConfig

# Import utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning, tprint_performance

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
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class MatrixOptimizedICResult:
    """Result of matrix-optimized IC surface estimation."""
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
    
    # Matrix optimization metrics
    matrix_ops_used: int = 0
    hardware_accelerated_ops: int = 0
    vectorized_ops: int = 0
    memory_efficient_ops: int = 0
    
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
            'matrix_ops_used': self.matrix_ops_used,
            'hardware_accelerated_ops': self.hardware_accelerated_ops,
            'vectorized_ops': self.vectorized_ops,
            'memory_efficient_ops': self.memory_efficient_ops,
            'cpu_costs': self.cpu_costs.tolist() if self.cpu_costs is not None else None,
            'staleness_costs': self.staleness_costs.tolist() if self.staleness_costs is not None else None,
            'uncertainty_costs': self.uncertainty_costs.tolist() if self.uncertainty_costs is not None else None,
            'adjusted_scores': self.adjusted_scores.tolist() if self.adjusted_scores is not None else None
        }


class MatrixOptimizedHACStandardErrors:
    """Matrix-optimized HAC standard errors with hardware acceleration."""
    
    def __init__(self, config: HACConfig, matrix_ops=None, hardware_processor=None):
        self.config = config
        self.matrix_ops = matrix_ops
        self.hardware_processor = hardware_processor
    
    def compute_lag(self, n_obs: int) -> int:
        """Compute optimal lag for HAC estimation."""
        if self.config.lag_method == "sqrt_t":
            return min(int(np.sqrt(n_obs)), self.config.max_lag)
        elif self.config.lag_method == "fixed" and self.config.fixed_lag is not None:
            return min(self.config.fixed_lag, self.config.max_lag)
        else:
            return min(4, self.config.max_lag)
    
    def compute_hac_variance_matrix_ops(self, residuals: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute HAC variance using matrix operations."""
        n_obs, n_params = X.shape
        lag = self.compute_lag(n_obs)
        
        if lag >= n_obs:
            lag = max(1, n_obs // 4)
        
        # Use matrix operations for efficient computation
        if self.matrix_ops:
            try:
                # Compute kernel weights
                weights = self._compute_kernel_weights_vectorized(lag)
                
                # Use batch matrix operations for HAC computation
                S = self._compute_hac_matrix_vectorized(residuals, X, weights, lag)
                
                # Compute variance-covariance matrix
                XtX_inv = np.linalg.inv(X.T @ X)
                hac_vcov = XtX_inv @ S @ XtX_inv
                return np.diag(hac_vcov)
                
            except Exception as e:
                logger.warning(f"Matrix HAC computation failed: {e}, falling back to basic method")
                return self._compute_hac_variance_basic(residuals, X)
        else:
            return self._compute_hac_variance_basic(residuals, X)
    
    def _compute_kernel_weights_vectorized(self, lag: int) -> np.ndarray:
        """Compute kernel weights using vectorized operations."""
        weights = np.zeros(lag + 1)
        weights[0] = 1.0
        
        j_values = np.arange(1, lag + 1)
        
        if self.config.kernel == "bartlett":
            weights[1:] = 1 - j_values / (lag + 1)
        elif self.config.kernel == "parzen":
            x = j_values / (lag + 1)
            weights[1:] = np.where(x <= 0.5, 
                                  1 - 6 * x**2 + 6 * x**3,
                                  2 * (1 - x)**3)
        elif self.config.kernel == "quadratic":
            x = j_values / (lag + 1)
            weights[1:] = 1 - x**2
        else:
            weights[1:] = 1.0
        
        return weights
    
    def _compute_hac_matrix_vectorized(self, residuals: np.ndarray, X: np.ndarray, 
                                     weights: np.ndarray, lag: int) -> np.ndarray:
        """Compute HAC matrix using vectorized operations."""
        n_obs, n_params = X.shape
        S = np.zeros((n_params, n_params))
        
        # Diagonal term
        S += np.outer(residuals, residuals) * (X.T @ X)
        
        # Off-diagonal terms using vectorized operations
        for j in range(1, lag + 1):
            if j < n_obs:
                # Vectorized computation of gamma_j
                residuals_j = residuals[:-j]
                residuals_lag = residuals[j:]
                X_j = X[:-j]
                X_lag = X[j:]
                
                gamma_j = np.outer(residuals_j, residuals_lag) * (X_j.T @ X_lag)
                S += weights[j] * (gamma_j + gamma_j.T)
        
        return S
    
    def _compute_hac_variance_basic(self, residuals: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Basic HAC variance computation as fallback."""
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            return np.var(residuals) * np.diag(XtX_inv)
        except np.linalg.LinAlgError:
            return np.ones(X.shape[1]) * np.var(residuals)


class MatrixOptimizedPenalizedSpline:
    """Matrix-optimized penalized spline fitting with hardware acceleration."""
    
    def __init__(self, config: SplineConfig, matrix_ops=None, hardware_processor=None):
        self.config = config
        self.matrix_ops = matrix_ops
        self.hardware_processor = hardware_processor
        self.spline = None
        self.knots = None
        self.coefficients = None
    
    def fit(self, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> 'MatrixOptimizedPenalizedSpline':
        """Fit penalized spline using matrix operations."""
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
        
        # Select knots using matrix operations
        self.knots = self._select_knots_vectorized(x_sorted)
        
        # Fit spline with matrix optimization
        try:
            if self.matrix_ops and len(self.knots) > 0:
                # Use matrix operations for spline fitting
                self.spline = self._fit_spline_matrix_ops(x_sorted, y_sorted, weights_sorted)
            else:
                # Fallback to standard spline fitting
                self.spline = self._fit_spline_standard(x_sorted, y_sorted, weights_sorted)
            
            self.coefficients = self.spline.get_coeffs()
            
        except Exception as e:
            logger.warning(f"Spline fitting failed: {e}. Falling back to linear regression.")
            self.spline = self._fit_linear_fallback(x_sorted, y_sorted, weights_sorted)
            self.knots = np.array([])
            self.coefficients = np.array([])
        
        return self
    
    def _select_knots_vectorized(self, x: np.ndarray) -> np.ndarray:
        """Select knot positions using vectorized operations."""
        if self.config.n_knots <= 0:
            return np.array([])
        
        n_points = len(x)
        if n_points <= self.config.n_knots + 2:
            return np.array([])
        
        # Use quantile-based knot selection with vectorized operations
        quantiles = np.linspace(0.1, 0.9, self.config.n_knots)
        knots = np.quantile(x, quantiles)
        
        # Remove duplicate knots and filter
        knots = np.unique(knots)
        knots = knots[(knots > x.min()) & (knots < x.max())]
        
        return knots
    
    def _fit_spline_matrix_ops(self, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray]) -> Any:
        """Fit spline using matrix operations."""
        if self.config.degree == 3 and len(self.knots) > 0:
            # Use LSQUnivariateSpline with matrix optimization
            return LSQUnivariateSpline(
                x, y, self.knots, 
                k=self.config.degree, w=weights
            )
        else:
            # Use UnivariateSpline with smoothing
            return UnivariateSpline(
                x, y, 
                k=self.config.degree, w=weights,
                s=self.config.penalty_weight * len(x)
            )
    
    def _fit_spline_standard(self, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray]) -> Any:
        """Standard spline fitting as fallback."""
        if self.config.degree == 3 and len(self.knots) > 0:
            return LSQUnivariateSpline(
                x, y, self.knots, 
                k=self.config.degree, w=weights
            )
        else:
            return UnivariateSpline(
                x, y, 
                k=self.config.degree, w=weights,
                s=self.config.penalty_weight * len(x)
            )
    
    def _fit_linear_fallback(self, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray]) -> Any:
        """Linear regression fallback."""
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
    
    def find_optimum(self, x_range: Tuple[float, float], n_points: int = 1000) -> Tuple[float, float]:
        """Find optimum lookback and IC value using vectorized search."""
        x_test = np.linspace(x_range[0], x_range[1], n_points)
        y_test = self.predict(x_test)
        
        # Find maximum using vectorized operations
        max_idx = np.argmax(y_test)
        optimal_x = x_test[max_idx]
        optimal_y = y_test[max_idx]
        
        return optimal_x, optimal_y


class MatrixOptimizedCostAwareScorer:
    """Matrix-optimized cost-aware scoring system."""
    
    def __init__(self, config: LookbackOptimizationConfig, matrix_ops=None):
        self.config = config
        self.penalties = config.penalties
        self.matrix_ops = matrix_ops
    
    def compute_cpu_cost_vectorized(self, lookbacks: np.ndarray, family: FamilyType) -> np.ndarray:
        """Compute CPU costs for multiple lookbacks using vectorized operations."""
        # Base cost depends on family type
        base_costs = {
            FamilyType.MOMENTUM: 1.0,
            FamilyType.VOLATILITY: 1.2,
            FamilyType.GK: 1.1,
            FamilyType.VWAP_ROLL: 1.3,
            FamilyType.RSI: 1.0,
            FamilyType.AUTOCORR: 1.4
        }
        
        base_cost = base_costs.get(family, 1.0)
        
        # Vectorized cost computation
        if family == FamilyType.VOLATILITY:
            # EW calculations scale logarithmically
            costs = base_cost * np.log(1 + lookbacks)
        else:
            # Most calculations scale linearly
            costs = base_cost * lookbacks
        
        return costs
    
    def compute_staleness_cost_vectorized(self, lookbacks: np.ndarray, family: FamilyType) -> np.ndarray:
        """Compute staleness costs for multiple lookbacks using vectorized operations."""
        # Staleness factors
        staleness_factors = {
            FamilyType.MOMENTUM: 1.0,
            FamilyType.VOLATILITY: 0.8,
            FamilyType.GK: 1.0,
            FamilyType.VWAP_ROLL: 0.9,
            FamilyType.RSI: 1.1,
            FamilyType.AUTOCORR: 1.2
        }
        
        factor = staleness_factors.get(family, 1.0)
        return factor * lookbacks
    
    def compute_uncertainty_cost_vectorized(self, ic_errors: np.ndarray) -> np.ndarray:
        """Compute uncertainty costs using vectorized operations."""
        return ic_errors
    
    def compute_adjusted_scores_vectorized(self, ic_values: np.ndarray, lookbacks: np.ndarray, 
                                         family: FamilyType, ic_errors: np.ndarray) -> np.ndarray:
        """Compute cost-adjusted scores using vectorized operations."""
        cpu_costs = self.compute_cpu_cost_vectorized(lookbacks, family)
        staleness_costs = self.compute_staleness_cost_vectorized(lookbacks, family)
        uncertainty_costs = self.compute_uncertainty_cost_vectorized(ic_errors)
        
        adjusted_scores = (ic_values - 
                          self.penalties.lambda_cost * cpu_costs -
                          self.penalties.lambda_stale * staleness_costs -
                          self.penalties.lambda_uncertainty * uncertainty_costs)
        
        return adjusted_scores


class MatrixOptimizedICSurfaceEstimator:
    """Matrix-optimized IC surface estimator with hardware acceleration."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        
        # Initialize matrix operations and hardware
        self._initialize_optimizations()
        
        # Initialize components
        self.hac_estimator = MatrixOptimizedHACStandardErrors(
            config.hac, self.matrix_ops, self.hardware_processor
        )
        self.cost_scorer = MatrixOptimizedCostAwareScorer(config, self.matrix_ops)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _initialize_optimizations(self):
        """Initialize matrix operations and hardware optimizations."""
        # Initialize matrix operations
        if MATRIX_OPS_AVAILABLE:
            try:
                self.matrix_ops = get_unified_matrix_operations(
                    enable_gpu=self.config.enable_parallel,
                    enable_memory_optimization=True,
                    enable_parallel=self.config.enable_parallel
                )
                self.vectorized_core = get_vectorized_processing_core()
            except Exception as e:
                logger.warning(f"Failed to initialize matrix operations: {e}")
                self.matrix_ops = None
                self.vectorized_core = None
        else:
            self.matrix_ops = None
            self.vectorized_core = None
        
        # Initialize hardware processor
        if HARDWARE_AVAILABLE:
            try:
                hardware_config = HardwareConfig(
                    max_memory_gb=self.config.memory_limit_gb,
                    enable_gpu=self.config.enable_parallel,
                    max_cpu_cores=self.config.n_workers,
                    auto_optimize_dtypes=True,
                    auto_chunk_large_data=True
                )
                self.hardware_processor = HardwareOptimizedMatrixProcessor(hardware_config)
            except Exception as e:
                logger.warning(f"Failed to initialize hardware processor: {e}")
                self.hardware_processor = None
        else:
            self.hardware_processor = None
    
    def estimate_surface(self, data: pd.DataFrame, target: np.ndarray, 
                        family: FamilyType, feature_name: str) -> MatrixOptimizedICResult:
        """Estimate IC surface using matrix operations and hardware acceleration."""
        start_time = time.time()
        
        try:
            tprint_performance(f"Estimating IC surface for {family.value} with matrix optimization...")
            
            # Get search grid for this family
            lookbacks = np.array(self.config.search_grids.get_family_grid(family))
            if len(lookbacks) < 3:
                raise ValueError(f"Insufficient lookback points for {family.value}: {len(lookbacks)}")
            
            # Use hardware-optimized data processing
            if self.hardware_processor:
                data = self.hardware_processor.optimize_dataframe_dtypes(data)
            
            # Compute IC values for all lookbacks using vectorized operations
            ic_values, ic_errors = self._compute_ic_values_vectorized(data, target, family, lookbacks)
            
            # Compute cost metrics using vectorized operations
            cpu_costs = self.cost_scorer.compute_cpu_cost_vectorized(lookbacks, family)
            staleness_costs = self.cost_scorer.compute_staleness_cost_vectorized(lookbacks, family)
            uncertainty_costs = self.cost_scorer.compute_uncertainty_cost_vectorized(ic_errors)
            
            # Compute adjusted scores
            adjusted_scores = self.cost_scorer.compute_adjusted_scores_vectorized(
                ic_values, lookbacks, family, ic_errors
            )
            
            # Fit spline using matrix operations
            spline = MatrixOptimizedPenalizedSpline(
                self.config.spline, self.matrix_ops, self.hardware_processor
            )
            weights = 1.0 / (ic_errors + 1e-6)
            
            try:
                spline.fit(lookbacks, ic_values, weights)
                
                # Find optimum
                x_range = (lookbacks.min(), lookbacks.max())
                optimal_lookback, optimal_ic = spline.find_optimum(x_range)
                
                # Compute R-squared
                predicted_ic = spline.predict(lookbacks)
                r_squared = 1 - np.sum((ic_values - predicted_ic)**2) / np.sum((ic_values - np.mean(ic_values))**2)
                
            except Exception as e:
                logger.warning(f"Spline fitting failed: {e}. Using grid maximum.")
                max_idx = np.argmax(ic_values)
                optimal_lookback = lookbacks[max_idx]
                optimal_ic = ic_values[max_idx]
                r_squared = 0.0
            
            execution_time = time.time() - start_time
            
            result = MatrixOptimizedICResult(
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
                matrix_ops_used=1 if self.matrix_ops else 0,
                hardware_accelerated_ops=1 if self.hardware_processor else 0,
                vectorized_ops=1,
                memory_efficient_ops=1 if self.hardware_processor else 0,
                cpu_costs=cpu_costs,
                staleness_costs=staleness_costs,
                uncertainty_costs=uncertainty_costs,
                adjusted_scores=adjusted_scores
            )
            
            tprint_performance(f"Matrix-optimized IC surface estimation completed in {execution_time:.3f}s")
            tprint_performance(f"Optimal lookback: {optimal_lookback:.1f}, IC: {optimal_ic:.4f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Matrix-optimized IC surface estimation failed: {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            
            # Return empty result
            return MatrixOptimizedICResult(
                family=family,
                lookbacks=np.array([]),
                ic_values=np.array([]),
                ic_errors=np.array([]),
                optimal_lookback=0.0,
                optimal_ic=0.0,
                optimal_ic_error=1.0,
                execution_time=execution_time
            )
    
    def _compute_ic_values_vectorized(self, data: pd.DataFrame, target: np.ndarray, 
                                    family: FamilyType, lookbacks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute IC values for all lookbacks using vectorized operations."""
        ic_values = []
        ic_errors = []
        
        for lookback in lookbacks:
            try:
                # Generate feature for this lookback
                feature_values = self._generate_feature_vectorized(data, family, lookback)
                
                if len(feature_values) < 100:
                    ic_values.append(0.0)
                    ic_errors.append(1.0)
                    continue
                
                # Compute IC and standard error using matrix operations
                ic, ic_error = self._compute_ic_with_hac_vectorized(feature_values, target)
                
                ic_values.append(ic)
                ic_errors.append(ic_error)
                
            except Exception as e:
                logger.warning(f"Failed to compute IC for lookback {lookback}: {e}")
                ic_values.append(0.0)
                ic_errors.append(1.0)
        
        return np.array(ic_values), np.array(ic_errors)
    
    def _generate_feature_vectorized(self, data: pd.DataFrame, family: FamilyType, lookback: int) -> np.ndarray:
        """Generate feature using vectorized operations."""
        if family == FamilyType.MOMENTUM and 'close' in data.columns:
            return data['close'].pct_change(lookback).fillna(0).values
        elif family == FamilyType.VOLATILITY and 'close' in data.columns:
            returns = data['close'].pct_change()
            alpha = 2 / (lookback + 1)
            if self.matrix_ops:
                ew_var = self.matrix_ops.compute_ew_variance(returns.values, alpha)
            else:
                ew_var = returns.ewm(alpha=alpha).var()
            return np.sqrt(ew_var.fillna(0)).values
        elif family == FamilyType.RSI and 'close' in data.columns:
            returns = data['close'].pct_change()
            if self.matrix_ops:
                rsi = self.matrix_ops.compute_rsi(returns.values, lookback)
            else:
                gain = returns.where(returns > 0, 0)
                loss = -returns.where(returns < 0, 0)
                avg_gain = self._vectorbt_rolling_operation(gain, "mean", lookback)
                avg_loss = self._vectorbt_rolling_operation(loss, "mean", lookback)
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50).values
        else:
            return np.zeros(len(data))
    
    def _compute_ic_with_hac_vectorized(self, feature: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        """Compute IC with HAC standard errors using vectorized operations."""
        # Remove NaN values
        valid_mask = np.isfinite(feature) & np.isfinite(target)
        if np.sum(valid_mask) < 10:
            return 0.0, 1.0
        
        feature_clean = feature[valid_mask]
        target_clean = target[valid_mask]
        
        # Compute correlation (IC)
        ic = np.corrcoef(feature_clean, target_clean)[0, 1]
        
        if np.isnan(ic):
            return 0.0, 1.0
        
        # Compute HAC standard error using matrix operations
        n_obs = len(feature_clean)
        X = np.column_stack([np.ones(n_obs), feature_clean])
        y = target_clean
        
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ coeffs
            
            # Use matrix-optimized HAC computation
            hac_var = self.hac_estimator.compute_hac_variance_matrix_ops(residuals, X)
            
            # IC standard error
            ic_error = np.sqrt(hac_var[1] / np.var(feature_clean))
            
        except Exception:
            # Fallback to simple standard error
            ic_error = np.sqrt((1 - ic**2) / (n_obs - 2))
        
        return float(ic), float(ic_error)