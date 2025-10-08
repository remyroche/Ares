"""
Optimization Strategy Abstraction Layer.

This module provides abstract base classes and interfaces for different
optimization strategies, enabling pluggable optimization algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from .constants import OPTIMIZATION_CONSTANTS

# Import tprint for enhanced logging
try:
    from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False

    def tprint(*args, **kwargs):
        print(*args, **kwargs)

    def tprint_error(*args, **kwargs):
        print("ERROR:", *args, **kwargs)

    def tprint_success(*args, **kwargs):
        print("SUCCESS:", *args, **kwargs)

    def tprint_warning(*args, **kwargs):
        print("WARNING:", *args, **kwargs)

    def tprint_debug(*args, **kwargs):
        print("DEBUG:", *args, **kwargs)


def log_info(message: str) -> None:
    """Log informational messages with timestamped printing."""
    tprint(message)


def log_success(message: str) -> None:
    """Log success messages using tprint."""
    tprint_success(message)


def log_warning(message: str) -> None:
    """Log warning messages using tprint."""
    tprint_warning(message)


def log_error(message: str) -> None:
    """Log error messages using tprint."""
    tprint_error(message)


def log_debug(message: str) -> None:
    """Log debug messages using tprint."""
    tprint_debug(message)

# Import math validation utilities for safe operations
try:
    from src.utils.math_validation import (
        safe_correlation, validate_finite, safe_mean, safe_std
    )
    from src.utils.core.common import safe_list_get
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False
    
    # Fallback implementations
    def safe_list_get(lst, index, default=None):
        try:
            value = lst[index] if lst and 0 <= index < len(lst) else default
            log_debug(f"Using fallback safe_list_get: index={index}, value={value}")
            return value
        except (IndexError, TypeError) as exc:
            log_warning(f"safe_list_get encountered an issue: {exc}. Returning default={default}")
            return default

    def safe_correlation(x, y, default=0.0):
        try:
            if len(x) != len(y) or len(x) < 2:
                log_warning("safe_correlation received insufficient data. Returning default value.")
                return default
            corr = np.corrcoef(x, y)[0, 1]
            result = corr if np.isfinite(corr) else default
            log_debug(f"Computed fallback safe_correlation: result={result}")
            return result
        except Exception as exc:
            log_error(f"safe_correlation failed with error: {exc}. Returning default={default}")
            return default


class OptimizationMethod(Enum):
    """Available optimization methods."""
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    GENETIC_ALGORITHM = "genetic_algorithm"
    TWO_STEP_GRID_TPE = "two_step_grid_tpe"
    MRMR = "mrmr"
    RANDOM_SEARCH = "random_search"


@dataclass
class OptimizationResult:
    """Standardized optimization result."""
    best_lookback_period: int
    best_score: float
    optimization_method: str
    total_trials: int
    optimization_time: float
    convergence_achieved: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'best_lookback_period': self.best_lookback_period,
            'best_score': self.best_score,
            'optimization_method': self.optimization_method,
            'total_trials': self.total_trials,
            'optimization_time': self.optimization_time,
            'convergence_achieved': self.convergence_achieved,
            'metadata': self.metadata
        }


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the optimization strategy."""
        self.config = config or {}
        self.method_name = "base_strategy"
        log_debug(f"Initialized {self.__class__.__name__} with config={self.config}")
    
    @abstractmethod
    def optimize(self, 
                 data: pd.DataFrame, 
                 feature_name: str, 
                 target_column: str,
                 **kwargs) -> OptimizationResult:
        """
        Perform optimization.
        
        Args:
            data: Input data for optimization
            feature_name: Name of the feature to optimize
            target_column: Target column for optimization
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult with optimization results
        """
        pass
    
    @abstractmethod
    def validate_inputs(self, 
                       data: pd.DataFrame, 
                       feature_name: str, 
                       target_column: str) -> Tuple[bool, str]:
        """
        Validate inputs for optimization.
        
        Args:
            data: Input data
            feature_name: Feature name
            target_column: Target column name
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this strategy."""
        default_config = {
            'min_lookback': OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK,
            'max_lookback': OPTIMIZATION_CONSTANTS.DEFAULT_MAX_LOOKBACK,
            'random_state': 42
        }
        log_debug(f"Default config for {self.__class__.__name__}: {default_config}")
        return default_config

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration."""
        self.config.update(new_config)
        log_info(f"Updated config for {self.__class__.__name__}: {self.config}")


class GridSearchStrategy(OptimizationStrategy):
    """Grid search optimization strategy."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.method_name = "grid_search"
        log_info(f"Initialized GridSearchStrategy with config={self.config}")

    def optimize(self,
                 data: pd.DataFrame,
                 feature_name: str,
                 target_column: str,
                 **kwargs) -> OptimizationResult:
        """Perform grid search optimization."""
        import time
        from sklearn.feature_selection import mutual_info_regression

        start_time = time.time()
        log_info(f"🔍 Starting grid search optimization for {feature_name}")
        log_info(f"   → Target column: {target_column}")

        # Get parameters
        min_lookback = self.config.get('min_lookback', OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK)
        max_lookback = self.config.get('max_lookback', OPTIMIZATION_CONSTANTS.DEFAULT_MAX_LOOKBACK)
        grid_size = self.config.get('grid_size', 20)

        log_debug(f"   → Lookback range: {min_lookback} to {max_lookback}")
        log_debug(f"   → Grid size: {grid_size}")

        # Create grid
        lookback_values = np.linspace(min_lookback, max_lookback, grid_size, dtype=int)

        best_score = -np.inf
        best_lookback = min_lookback
        total_trials = 0
        
        # Grid search
        for lookback in lookback_values:
            try:
                # Generate feature with current lookback
                feature_values = self._generate_feature_with_lookback(data, feature_name, lookback)
                target_values = data[target_column].values
                
                # Safely align and clean data
                if len(feature_values) == 0 or len(target_values) == 0:
                    continue
                    
                min_length = min(len(feature_values), len(target_values))
                feature_values = feature_values[:min_length]
                target_values = target_values[:min_length]
                
                # Remove NaN values with safe operations
                if MATH_VALIDATION_AVAILABLE:
                    valid_mask = np.isfinite(feature_values) & np.isfinite(target_values)
                    if not np.any(valid_mask):
                        continue
                    feature_clean = feature_values[valid_mask]
                    target_clean = target_values[valid_mask]
                else:
                    mask = ~(np.isnan(feature_values) | np.isnan(target_values))
                    if mask.sum() < 10:
                        continue
                    feature_clean = feature_values[mask]
                    target_clean = target_values[mask]
                
                # Check minimum data requirement (increased for reliability)
                min_samples = max(30, lookback * 2)
                if len(feature_clean) < min_samples:
                    continue
                
                # Calculate score with safe operations
                score = 0.0
                try:
                    mi_scores = mutual_info_regression(
                        feature_clean.reshape(-1, 1), 
                        target_clean,
                        random_state=self.config.get('random_state', 42)
                    )
                    # Safe array access
                    score = safe_list_get(mi_scores, 0, 0.0)
                except Exception:
                    score = 0.0
                
                # Fallback to safe correlation if MI failed
                if score == 0.0:
                    if MATH_VALIDATION_AVAILABLE:
                        correlation = safe_correlation(feature_clean, target_clean, default=0.0)
                    else:
                        try:
                            corr_matrix = np.corrcoef(feature_clean, target_clean)
                            if corr_matrix.shape == (2, 2):
                                correlation = corr_matrix[0, 1]
                                correlation = correlation if np.isfinite(correlation) else 0.0
                            else:
                                correlation = 0.0
                        except Exception:
                            correlation = 0.0
                    score = abs(correlation)
                
                # Validate final score
                if MATH_VALIDATION_AVAILABLE:
                    try:
                        score = validate_finite(score, "optimization_score")
                        score = max(0.0, score)  # Ensure non-negative
                    except Exception:
                        score = 0.0
                
                total_trials += 1

                if score > best_score:
                    best_score = score
                    best_lookback = lookback
                    log_debug(f"New best lookback found: {best_lookback} with score={best_score}")

            except Exception:
                log_warning(f"Skipping lookback {lookback} due to processing error.")
                continue

        optimization_time = time.time() - start_time
        log_success(
            f"Grid search completed for {feature_name} in {optimization_time:.2f}s."
            f" Best lookback={best_lookback}, score={best_score}"
        )

        return OptimizationResult(
            best_lookback_period=best_lookback,
            best_score=best_score,
            optimization_method=self.method_name,
            total_trials=total_trials,
            optimization_time=optimization_time,
            convergence_achieved=total_trials > 0,
            metadata={
                'grid_size': grid_size,
                'lookback_range': (min_lookback, max_lookback)
            }
        )
    
    def validate_inputs(self, 
                       data: pd.DataFrame, 
                       feature_name: str, 
                       target_column: str) -> Tuple[bool, str]:
        """Validate inputs for grid search."""
        if data.empty:
            message = "Data is empty"
            log_warning(message)
            return False, message

        if target_column not in data.columns:
            message = f"Target column '{target_column}' not found"
            log_warning(message)
            return False, message

        min_required_data = self.config.get('max_lookback', OPTIMIZATION_CONSTANTS.DEFAULT_MAX_LOOKBACK) * 3  # Increased for reliability
        if len(data) < min_required_data:
            message = f"Insufficient data for optimization: {len(data)} < {min_required_data}"
            log_warning(message)
            return False, message

        log_debug("Input validation passed for grid search optimization.")
        return True, ""

    def _generate_feature_with_lookback(self,
                                      data: pd.DataFrame,
                                      feature_name: str,
                                      lookback_period: int) -> np.ndarray:
        """Generate feature values with specific lookback period."""
        if feature_name in data.columns:
            log_debug(f"Generating feature '{feature_name}' using rolling mean with lookback={lookback_period}")
            return data[feature_name].rolling(window=lookback_period).mean().values
        else:
            # Default to stationary transform of close price
            log_warning(
                f"Feature '{feature_name}' not found. Falling back to stationary transform of 'close' for lookback={lookback_period}"
            )
            if 'close' not in data.columns:
                log_error("Fallback column 'close' missing from data. Returning zeros for safety.")
                return np.zeros(len(data))

            return self._compute_stationary_close_transform(data['close'], lookback_period)

    @staticmethod
    def _compute_stationary_close_transform(close_series: pd.Series, lookback_period: int) -> np.ndarray:
        """Generate a stationary fallback series using log returns or price spreads."""
        safe_close = close_series.astype(float).replace([np.inf, -np.inf], np.nan)
        if safe_close.dropna().empty:
            return np.zeros(len(close_series))

        # Prefer log returns when prices are strictly positive, otherwise fall back to percent change
        if (safe_close > 0).all():
            returns = np.log(safe_close).diff()
        else:
            returns = safe_close.pct_change()

        returns = returns.replace([np.inf, -np.inf], np.nan)

        if returns.dropna().empty:
            # As a last resort, use simple differencing
            returns = safe_close.diff()

        window = max(2, min(lookback_period, len(returns)))
        stationary = returns.rolling(window=window, min_periods=1).mean()

        return stationary.fillna(0.0).values


class OptimizationStrategyFactory:
    """Factory for creating optimization strategies."""
    
    _strategies = {
        OptimizationMethod.GRID_SEARCH: GridSearchStrategy,
        # Add more strategies as they are implemented
    }
    
    @classmethod
    def create_strategy(cls, 
                       method: OptimizationMethod, 
                       config: Optional[Dict[str, Any]] = None) -> OptimizationStrategy:
        """
        Create an optimization strategy.
        
        Args:
            method: Optimization method to use
            config: Configuration for the strategy
            
        Returns:
            OptimizationStrategy instance
            
        Raises:
            ValueError: If method is not supported
        """
        if method not in cls._strategies:
            log_error(f"Unsupported optimization method requested: {method}")
            raise ValueError(f"Unsupported optimization method: {method}")

        strategy_class = cls._strategies[method]
        log_info(f"Creating strategy {strategy_class.__name__} for method={method}")
        return strategy_class(config)

    @classmethod
    def get_available_methods(cls) -> List[OptimizationMethod]:
        """Get list of available optimization methods."""
        methods = list(cls._strategies.keys())
        log_debug(f"Available optimization methods: {methods}")
        return methods
    
    @classmethod
    def register_strategy(cls, 
                         method: OptimizationMethod, 
                         strategy_class: type) -> None:
        """
        Register a new optimization strategy.
        
        Args:
            method: Optimization method enum
            strategy_class: Strategy class to register
        """
        log_info(f"Registering strategy {strategy_class.__name__} for method={method}")
        cls._strategies[method] = strategy_class

